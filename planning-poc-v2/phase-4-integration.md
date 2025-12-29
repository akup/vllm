# Phase 4: API Integration

## Objective

Add PoC API routes to vLLM server with minimal changes to core code.

## Deliverable

Working API endpoints that can start/stop PoC and return results.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pow/init` | POST | Initialize PoC round |
| `/api/v1/pow/init/generate` | POST | Init + start generating |
| `/api/v1/pow/init/validate` | POST | Init + start validating |
| `/api/v1/pow/phase/generate` | POST | Switch to generate mode |
| `/api/v1/pow/phase/validate` | POST | Switch to validate mode |
| `/api/v1/pow/validate` | POST | Validate submitted nonces (accepts ProofBatch) |
| `/api/v1/pow/status` | GET | Get status and valid nonces |
| `/api/v1/pow/stop` | POST | Stop current round |

## Implementation

### File: `vllm/poc/routes.py`

```python
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from .config import PoCConfig, PoCState
from .data import ProofBatch

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

class PoCInitRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = -1
    node_count: int = -1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None  # Push valid batches to this URL if set

class PoCStatusResponse(BaseModel):
    state: str
    valid_nonces: List[int]
    valid_distances: List[float]
    total_checked: int
    total_valid: int
    elapsed_seconds: float
    rate_per_second: float

class PoCValidateResponse(BaseModel):
    nonces: List[int]
    distances: List[float]
    valid: List[bool]

def get_poc_manager(request: Request):
    manager = getattr(request.app.state, 'poc_manager', None)
    if manager is None:
        raise HTTPException(status_code=503, detail="PoC not enabled")
    return manager

def _create_config(body: PoCInitRequest) -> PoCConfig:
    return PoCConfig(
        block_hash=body.block_hash,
        block_height=body.block_height,
        public_key=body.public_key,
        r_target=body.r_target,
        fraud_threshold=body.fraud_threshold,
        node_id=body.node_id,
        node_count=body.node_count,
        batch_size=body.batch_size,
        seq_len=body.seq_len,
        callback_url=body.callback_url,
    )

@router.post("/init")
async def init_round(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round without starting generation"""
    manager = get_poc_manager(request)
    
    if manager.state == PoCState.GENERATING:
        raise HTTPException(status_code=400, detail="Round already in progress")
    
    config = _create_config(body)
    manager.init_round(config)
    
    return {"status": "OK", "pow_status": manager.get_status()}

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start generating"""
    manager = get_poc_manager(request)
    
    if body.node_id == -1 or body.node_count == -1:
        raise HTTPException(status_code=400, detail="Node ID and node count must be set")
    
    config = _create_config(body)
    manager.init_round(config)
    manager.start_generate()
    
    # Start background generation loop with tracking
    import asyncio
    task = asyncio.create_task(_generation_loop(manager))
    manager.set_generation_task(task)
    
    return {"status": "OK", "pow_status": manager.get_status()}

@router.post("/init/validate")
async def init_validate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start validating"""
    manager = get_poc_manager(request)
    
    config = _create_config(body)
    manager.init_round(config)
    manager.start_validate()
    
    return {"status": "OK", "pow_status": manager.get_status()}

@router.post("/phase/generate")
async def start_generate(request: Request) -> dict:
    """Switch to generate mode"""
    manager = get_poc_manager(request)
    
    if manager.config is None:
        raise HTTPException(status_code=400, detail="PoC not initialized")
    if manager.config.node_id == -1 or manager.config.node_count == -1:
        raise HTTPException(status_code=400, detail="Node ID and node count must be set")
    
    manager.start_generate()
    
    # Start background generation loop with tracking
    import asyncio
    task = asyncio.create_task(_generation_loop(manager))
    manager.set_generation_task(task)
    
    return {"status": "OK", "pow_status": manager.get_status()}

@router.post("/phase/validate")
async def start_validate(request: Request) -> dict:
    """Switch to validate mode"""
    manager = get_poc_manager(request)
    
    if manager.config is None:
        raise HTTPException(status_code=400, detail="PoC not initialized")
    
    manager.start_validate()
    
    return {"status": "OK", "pow_status": manager.get_status()}

async def _generation_loop(manager):
    """Background task that runs batches until stopped.
    
    Task is tracked by manager for proper cleanup on stop/reinit.
    Uses run_batch_async to send valid batches to callback URL if configured.
    """
    import asyncio
    try:
        while manager.state == PoCState.GENERATING:
            await manager.run_batch_async()  # Async: sends to callback if configured
            await asyncio.sleep(0)  # Yield to event loop
    except asyncio.CancelledError:
        pass  # Normal cancellation on stop/reinit

@router.post("/stop")
async def stop_round(request: Request) -> dict:
    manager = get_poc_manager(request)
    manager.stop_round()
    return {"status": "OK", "pow_status": manager.get_status()}

@router.get("/status", response_model=PoCStatusResponse)
async def get_status(request: Request) -> PoCStatusResponse:
    manager = get_poc_manager(request)
    status = manager.get_status()
    return PoCStatusResponse(**status)

@router.post("/validate", response_model=PoCValidateResponse)
async def validate_nonces(request: Request, body: ProofBatch) -> PoCValidateResponse:
    """Validate submitted nonces by recomputing distances.
    
    Accepts full ProofBatch (matching original API).
    """
    manager = get_poc_manager(request)
    
    if manager.config is None:
        raise HTTPException(status_code=400, detail="No round configured")
    
    if manager.config.block_hash != body.block_hash:
        raise HTTPException(status_code=400, detail="Block hash mismatch")
    
    distances, valid = manager.validate(body.nonces, body.public_key)
    
    return PoCValidateResponse(
        nonces=body.nonces,
        distances=distances,
        valid=valid,
    )
```

### Changes to `vllm/entrypoints/openai/api_server.py`

Minimal additions:

```python
# Near top of file, with other imports
from vllm.poc.routes import router as poc_router
from vllm.poc.manager import PoCManager

# In build_app() function, after other routers:
def build_app(args: Namespace) -> FastAPI:
    # ... existing code ...
    
    if args.enable_poc:
        app.include_router(poc_router)
    
    return app

# In init_app_state() function:
async def init_app_state(engine_client, vllm_config, state, args):
    # ... existing code ...
    
    if args.enable_poc:
        model = engine_client.engine.model_executor.driver_worker.model_runner.model
        state.poc_manager = PoCManager(model, vllm_config.model_config)
```

### Changes to `vllm/entrypoints/openai/cli_args.py`

Add arguments:

```python
# In make_arg_parser() function:
def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    # ... existing code ...
    
    # PoC arguments
    parser.add_argument(
        "--enable-poc",
        action="store_true",
        default=False,
        help="Enable Proof of Compute endpoints",
    )
    parser.add_argument(
        "--poc-batch-size",
        type=int,
        default=32,
        help="Default batch size for PoC nonce computation",
    )
    parser.add_argument(
        "--poc-seq-len",
        type=int,
        default=256,
        help="Default sequence length for PoC inputs",
    )
    
    return parser
```

## Inference Blocking

During PoC (both GENERATING and VALIDATING), inference should return 503. Add middleware:

```python
# In routes.py or as middleware in api_server.py

@router.middleware("http")
async def poc_blocking_middleware(request: Request, call_next):
    manager = getattr(request.app.state, 'poc_manager', None)
    
    # Block inference endpoints during PoC (both generating and validating)
    # Note: In Phase 6, parallel inference during VALIDATING may be enabled
    if manager and manager.state in (PoCState.GENERATING, PoCState.VALIDATING):
        if request.url.path.startswith("/v1/"):
            raise HTTPException(status_code=503, detail="PoC in progress")
    
    return await call_next(request)
```

## Directory Structure After Phase 4

```
vllm/
├── poc/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── gpu_random.py
│   ├── manager.py
│   └── routes.py
└── entrypoints/openai/
    ├── api_server.py    # Modified: include poc_router
    └── cli_args.py      # Modified: add --enable-poc args

tests/poc/
├── __init__.py
├── test_data.py
├── test_gpu_random.py
├── test_manager.py
└── test_routes.py
```

## Unit Tests

### File: `tests/poc/test_routes.py`

**Cross-check**: Compare API endpoint behavior with original:
`/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/service/routes.py`

Tests use FastAPI TestClient with mocked manager - no vLLM server required.

```python
import pytest
from unittest.mock import Mock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.poc.routes import router
from vllm.poc.config import PoCConfig, PoCState

@pytest.fixture
def mock_manager():
    """Create mock PoCManager"""
    manager = Mock()
    manager.state = PoCState.IDLE
    manager.config = None
    manager.get_status.return_value = {
        "state": "IDLE",
        "valid_nonces": [],
        "valid_distances": [],
        "total_checked": 0,
        "total_valid": 0,
        "elapsed_seconds": 0.0,
        "rate_per_second": 0.0,
    }
    return manager

@pytest.fixture
def app(mock_manager):
    """Create FastAPI app with mocked manager"""
    app = FastAPI()
    app.include_router(router)
    app.state.poc_manager = mock_manager
    return app

@pytest.fixture
def client(app):
    return TestClient(app)

class TestStatusEndpoint:
    def test_status_returns_data(self, client, mock_manager):
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "IDLE"
        assert data["valid_nonces"] == []
        assert "total_checked" in data

class TestInitEndpoint:
    def test_init_success(self, client, mock_manager):
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "hash1",
            "block_height": 100,
            "public_key": "node1",
            "r_target": 0.5,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "OK"
        mock_manager.init_round.assert_called_once()
    
    def test_init_blocked_when_generating(self, client, mock_manager):
        mock_manager.state = PoCState.GENERATING
        response = client.post("/api/v1/pow/init", json={
            "block_hash": "hash1",
            "block_height": 100,
            "public_key": "node1",
            "r_target": 0.5,
        })
        assert response.status_code == 400
        assert "already in progress" in response.json()["detail"]

class TestInitGenerateEndpoint:
    def test_requires_node_id(self, client, mock_manager):
        """Cross-check: Original requires node_id and node_count"""
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "hash1",
            "block_height": 100,
            "public_key": "node1",
            "r_target": 0.5,
            # node_id and node_count default to -1
        })
        assert response.status_code == 400
        assert "Node ID" in response.json()["detail"]
    
    def test_success_with_node_info(self, client, mock_manager):
        response = client.post("/api/v1/pow/init/generate", json={
            "block_hash": "hash1",
            "block_height": 100,
            "public_key": "node1",
            "r_target": 0.5,
            "node_id": 0,
            "node_count": 1,
        })
        assert response.status_code == 200
        mock_manager.init_round.assert_called_once()
        mock_manager.start_generate.assert_called_once()

class TestPhaseGenerateEndpoint:
    def test_requires_init_first(self, client, mock_manager):
        mock_manager.config = None
        response = client.post("/api/v1/pow/phase/generate")
        assert response.status_code == 400
        assert "not initialized" in response.json()["detail"]
    
    def test_requires_node_info(self, client, mock_manager):
        mock_manager.config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=-1,
            node_count=-1,
        )
        response = client.post("/api/v1/pow/phase/generate")
        assert response.status_code == 400
        assert "Node ID" in response.json()["detail"]

class TestPhaseValidateEndpoint:
    def test_requires_init_first(self, client, mock_manager):
        mock_manager.config = None
        response = client.post("/api/v1/pow/phase/validate")
        assert response.status_code == 400
        assert "not initialized" in response.json()["detail"]
    
    def test_success_after_init(self, client, mock_manager):
        mock_manager.config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
        response = client.post("/api/v1/pow/phase/validate")
        assert response.status_code == 200
        mock_manager.start_validate.assert_called_once()

class TestStopEndpoint:
    def test_stop_success(self, client, mock_manager):
        response = client.post("/api/v1/pow/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "OK"
        mock_manager.stop_round.assert_called_once()

class TestValidateEndpoint:
    """Tests for /validate endpoint - accepts full ProofBatch (matching original API)"""
    
    def test_requires_config(self, client, mock_manager):
        mock_manager.config = None
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "node1",
            "block_hash": "hash1",
            "block_height": 100,
            "nonces": [0, 1, 2],
            "dist": [0.3, 0.4, 0.6],
            "node_id": 0,
        })
        assert response.status_code == 400
        assert "No round configured" in response.json()["detail"]
    
    def test_block_hash_mismatch(self, client, mock_manager):
        """Cross-check: Original validates block_hash match"""
        mock_manager.config = PoCConfig(
            block_hash="correct_hash",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "node1",
            "block_hash": "wrong_hash",
            "block_height": 100,
            "nonces": [0, 1, 2],
            "dist": [0.3, 0.4, 0.6],
            "node_id": 0,
        })
        assert response.status_code == 400
        assert "Block hash mismatch" in response.json()["detail"]
    
    def test_validate_success(self, client, mock_manager):
        mock_manager.config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
        mock_manager.validate.return_value = ([0.3, 0.4, 0.6], [True, True, False])
        
        response = client.post("/api/v1/pow/validate", json={
            "public_key": "node1",
            "block_hash": "hash1",
            "block_height": 100,
            "nonces": [0, 1, 2],
            "dist": [0.3, 0.4, 0.6],
            "node_id": 0,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["nonces"] == [0, 1, 2]
        assert data["distances"] == [0.3, 0.4, 0.6]
        assert data["valid"] == [True, True, False]

class TestPoCNotEnabled:
    def test_503_when_manager_not_set(self):
        """When --enable-poc is not used, endpoints return 503"""
        app = FastAPI()
        app.include_router(router)
        # Don't set app.state.poc_manager
        client = TestClient(app)
        
        response = client.get("/api/v1/pow/status")
        assert response.status_code == 503
        assert "PoC not enabled" in response.json()["detail"]
```

## Running Tests

```bash
pytest tests/poc/test_routes.py -v
```

## Testing with curl

```bash
# Start server
vllm serve Qwen/Qwen3-0.6B --enable-poc

# Initialize and start PoC round (pull model - fetch via /status)
curl -X POST http://localhost:8000/api/v1/pow/init/generate \
  -H "Content-Type: application/json" \
  -d '{"block_hash": "abc123", "block_height": 100, "public_key": "node1", "r_target": 0.5, "node_id": 0, "node_count": 1}'

# Initialize with callback URL (push model - results POSTed to callback)
curl -X POST http://localhost:8000/api/v1/pow/init/generate \
  -H "Content-Type: application/json" \
  -d '{"block_hash": "abc123", "block_height": 100, "public_key": "node1", "r_target": 0.5, "node_id": 0, "node_count": 1, "callback_url": "http://aggregator:9000/pow"}'

# Check status (always works, even with callback)
curl http://localhost:8000/api/v1/pow/status

# Switch to validate mode
curl -X POST http://localhost:8000/api/v1/pow/phase/validate

# Stop round
curl -X POST http://localhost:8000/api/v1/pow/stop

# Validate nonces (accepts full ProofBatch - matching original API)
curl -X POST http://localhost:8000/api/v1/pow/validate \
  -H "Content-Type: application/json" \
  -d '{"public_key": "node1", "block_hash": "abc123", "block_height": 100, "nonces": [0, 1, 2], "dist": [0.3, 0.4, 0.5], "node_id": 0}'
```

### Callback URL Endpoints

When `callback_url` is set, valid batches are POSTed to:
- `{callback_url}/generated` - Valid nonces as they're found (during GENERATING)
- `{callback_url}/validated` - Validation results (during VALIDATING)

## Acceptance Criteria

- [ ] All 8 endpoints working (init, init/generate, init/validate, phase/generate, phase/validate, validate, status, stop)
- [ ] `--enable-poc` flag controls endpoint availability
- [ ] Inference blocked (503) during PoC
- [ ] Can init, generate, validate, and stop rounds
- [ ] Validation recomputes distances correctly
- [ ] fraud_threshold passed through to config
- [ ] callback_url triggers push of valid batches to external service
- [ ] Callback failures don't block generation (fire-and-forget)
- [ ] All unit tests pass: `pytest tests/poc/test_routes.py`

