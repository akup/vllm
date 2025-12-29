# Phase 3: PoC Manager

## Objective

Implement `PoCManager` that orchestrates PoC rounds using the deployed vLLM model.

## Deliverable

Working `PoCManager` that can:
1. Start/stop PoC rounds
2. Access the deployed model for forward passes
3. Generate and validate nonces
4. Track statistics

## Architecture

```
PoCManager
    │
    ├── config: PoCConfig
    ├── state: PoCState
    ├── model: Qwen3ForCausalLM (reference to deployed model)
    ├── target: Tensor (unit vector)
    ├── valid_nonces: List[int]
    ├── valid_distances: List[float]
    ├── _generation_task: Optional[asyncio.Task] (tracked for cleanup)
    ├── _callback_sender: Optional[PoCCallbackSender] (push results if configured)
    │
    └── Methods:
        ├── init_round(config) → Sets config, generates target, state = IDLE
        ├── start_generate() → Sets state = GENERATING
        ├── start_validate() → Sets state = VALIDATING
        ├── stop_round() → Cancels task, sets state = STOPPED
        ├── run_batch() → Computes distances, collects valid
        ├── run_batch_async() → Async version, sends to callback if configured
        ├── validate(nonces) → Recomputes distances for verification
        └── get_status() → Returns current state and results

PoCCallbackSender
    │
    ├── callback_url: str
    ├── r_target: float
    ├── fraud_threshold: float
    │
    └── Methods:
        ├── send_generated(batch) → POST {callback_url}/generated
        └── send_validated(batch) → POST {callback_url}/validated
```

Note: State is stateless across restarts. If vLLM restarts during a round, the round is lost.

## Model Access

vLLM loads the model once at startup. We access it via:

```python
# In api_server.py, after init_app_state:
model = engine_client.engine.model_executor.driver_worker.model_runner.model
```

The model supports `inputs_embeds` directly:
```python
# From vllm/worker/model_runner.py
hidden_states = model(
    input_ids=None,
    inputs_embeds=inputs_embeds,  # Our generated inputs
    positions=positions,
)

# Model-agnostic logits computation
# Use model.compute_logits() if available (vLLM standard)
# Fallback to model.lm_head for direct access
if hasattr(model, 'compute_logits'):
    logits = model.compute_logits(hidden_states, sampling_metadata=None)
else:
    logits = model.lm_head(hidden_states)
```

## Implementation

### File: `vllm/poc/sender.py`

```python
import asyncio
import logging
from typing import Optional, List
from dataclasses import asdict

import aiohttp

from .data import ProofBatch, ValidatedBatch

logger = logging.getLogger(__name__)


class PoCCallbackSender:
    """Sends valid batches to callback URL (async, non-blocking).
    
    Callback failures are logged but don't block generation (fire-and-forget).
    """
    
    def __init__(self, callback_url: str, r_target: float, fraud_threshold: float):
        self.callback_url = callback_url.rstrip("/")
        self.r_target = r_target
        self.fraud_threshold = fraud_threshold
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _post(self, endpoint: str, data: dict) -> bool:
        """POST data to callback URL. Returns True on success."""
        url = f"{self.callback_url}{endpoint}"
        try:
            session = await self._get_session()
            async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    logger.debug(f"Callback sent to {url}")
                    return True
                else:
                    logger.warning(f"Callback failed: {url} returned {resp.status}")
                    return False
        except Exception as e:
            logger.warning(f"Callback error for {url}: {e}")
            return False
    
    async def send_generated(self, batch: ProofBatch) -> None:
        """POST {callback_url}/generated with valid nonces (filtered by r_target)."""
        valid_batch = batch.sub_batch(self.r_target)
        if len(valid_batch) > 0:
            await self._post("/generated", asdict(valid_batch))
    
    async def send_validated(self, batch: ValidatedBatch) -> None:
        """POST {callback_url}/validated with validation results."""
        await self._post("/validated", asdict(batch))
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
```

### File: `vllm/poc/manager.py`

```python
import time
import torch
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from .config import PoCConfig, PoCState
from .data import ProofBatch
from .gpu_random import (
    generate_inputs,
    generate_permutations,
    generate_target,
    compute_distances,
)

@dataclass
class PoCStats:
    total_checked: int = 0
    total_valid: int = 0
    start_time: float = 0.0
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time > 0 else 0.0
    
    @property
    def rate(self) -> float:
        return self.total_checked / self.elapsed if self.elapsed > 0 else 0.0

class PoCManager:
    def __init__(self, model, model_config):
        self.model = model
        self.model_config = model_config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        self.state = PoCState.IDLE
        self.config: Optional[PoCConfig] = None
        self.target: Optional[torch.Tensor] = None
        self.stats = PoCStats()
        
        self.valid_nonces: List[int] = []
        self.valid_distances: List[float] = []
        self._nonce_counter = 0
        self._generation_task: Optional['asyncio.Task'] = None  # Track background task
        self._callback_sender: Optional['PoCCallbackSender'] = None  # Push to callback URL
    
    def init_round(self, config: PoCConfig) -> None:
        """Initialize round with config and generate target. Does not start generating."""
        if self.state == PoCState.GENERATING:
            raise RuntimeError("Round already in progress")
        
        # Cancel any existing generation task
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        
        self.config = config
        self.stats = PoCStats(start_time=time.time())
        self.valid_nonces = []
        self.valid_distances = []
        self._nonce_counter = config.node_id
        
        # Generate target vector for this round
        self.target = generate_target(
            config.block_hash,
            self.model_config.vocab_size,
            self.device,
        )
        
        # Create callback sender if URL provided
        if config.callback_url:
            from .sender import PoCCallbackSender
            self._callback_sender = PoCCallbackSender(
                callback_url=config.callback_url,
                r_target=config.r_target,
                fraud_threshold=config.fraud_threshold,
            )
        else:
            self._callback_sender = None
        
        self.state = PoCState.IDLE
    
    def start_generate(self) -> None:
        """Switch to GENERATING state. Call after init_round()."""
        if self.config is None:
            raise RuntimeError("Round not initialized")
        self.state = PoCState.GENERATING
    
    def start_validate(self) -> None:
        """Switch to VALIDATING state. Call after init_round()."""
        if self.config is None:
            raise RuntimeError("Round not initialized")
        # Cancel generation task if running
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        self.state = PoCState.VALIDATING
    
    def stop_round(self) -> None:
        """Stop current round and cancel any running tasks."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            self._generation_task = None
        self.state = PoCState.STOPPED
    
    def set_generation_task(self, task: 'asyncio.Task') -> None:
        """Track the background generation task for cleanup."""
        self._generation_task = task
    
    def get_next_nonces(self) -> List[int]:
        """Get next batch of nonces for this node"""
        nonces = []
        for _ in range(self.config.batch_size):
            nonces.append(self._nonce_counter)
            self._nonce_counter += self.config.node_count
        return nonces
    
    def _compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute logits from hidden states (model-agnostic).
        
        Supports different model architectures by checking available methods.
        """
        if hasattr(self.model, 'compute_logits'):
            # vLLM standard interface
            return self.model.compute_logits(hidden_states, sampling_metadata=None)
        elif hasattr(self.model, 'lm_head'):
            # Direct lm_head access (fallback)
            return self.model.lm_head(hidden_states)
        else:
            raise RuntimeError("Model does not have compute_logits or lm_head")
    
    @torch.inference_mode()
    def run_batch(self) -> ProofBatch:
        """Run one batch of nonce computation"""
        if self.state != PoCState.GENERATING:
            return ProofBatch.empty()
        
        nonces = self.get_next_nonces()
        
        # Generate inputs
        inputs_embeds = generate_inputs(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            dim=self.model_config.hidden_size,
            seq_len=self.config.seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Generate positions [0, 1, 2, ..., seq_len-1] for each batch item
        batch_size = len(nonces)
        positions = torch.arange(
            self.config.seq_len, 
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Forward pass
        hidden_states = self.model(
            input_ids=None,
            positions=positions.flatten(),
            inputs_embeds=inputs_embeds.view(-1, self.model_config.hidden_size),
        )
        
        # Reshape and get last token hidden state
        hidden_states = hidden_states.view(batch_size, self.config.seq_len, -1)
        last_hidden = hidden_states[:, -1, :]
        
        # Compute logits (model-agnostic)
        logits = self._compute_logits(last_hidden)
        
        # Generate permutations and compute distances
        permutations = generate_permutations(
            self.config.block_hash,
            self.config.public_key,
            nonces,
            self.model_config.vocab_size,
            self.device,
        )
        
        distances = compute_distances(logits.float(), permutations, self.target)
        
        # Create batch
        batch = ProofBatch(
            public_key=self.config.public_key,
            block_hash=self.config.block_hash,
            block_height=self.config.block_height,
            nonces=nonces,
            dist=distances.cpu().tolist(),
            node_id=self.config.node_id,
        )
        
        # Update stats
        self.stats.total_checked += len(nonces)
        
        # Filter valid nonces
        valid_batch = batch.sub_batch(self.config.r_target)
        self.stats.total_valid += len(valid_batch)
        self.valid_nonces.extend(valid_batch.nonces)
        self.valid_distances.extend(valid_batch.dist)
        
        return batch
    
    async def run_batch_async(self) -> ProofBatch:
        """Async version of run_batch. Sends to callback URL if configured."""
        batch = self.run_batch()
        if self._callback_sender and len(batch) > 0:
            await self._callback_sender.send_generated(batch)
        return batch
    
    @torch.inference_mode()
    def validate(self, nonces: List[int], public_key: str) -> Tuple[List[float], List[bool]]:
        """Validate nonces by recomputing distances"""
        if self.config is None:
            raise RuntimeError("No round configured")
        
        # Generate inputs for validation
        inputs_embeds = generate_inputs(
            self.config.block_hash,
            public_key,
            nonces,
            dim=self.model_config.hidden_size,
            seq_len=self.config.seq_len,
            device=self.device,
            dtype=self.dtype,
        )
        
        batch_size = len(nonces)
        positions = torch.arange(
            self.config.seq_len,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = self.model(
            input_ids=None,
            positions=positions.flatten(),
            inputs_embeds=inputs_embeds.view(-1, self.model_config.hidden_size),
        )
        
        hidden_states = hidden_states.view(batch_size, self.config.seq_len, -1)
        last_hidden = hidden_states[:, -1, :]
        logits = self._compute_logits(last_hidden)
        
        permutations = generate_permutations(
            self.config.block_hash,
            public_key,
            nonces,
            self.model_config.vocab_size,
            self.device,
        )
        
        distances = compute_distances(logits.float(), permutations, self.target)
        distances_list = distances.cpu().tolist()
        valid_list = [d < self.config.r_target for d in distances_list]
        
        return distances_list, valid_list
    
    def get_status(self) -> dict:
        return {
            "state": self.state.value,
            "valid_nonces": self.valid_nonces,
            "valid_distances": self.valid_distances,
            "total_checked": self.stats.total_checked,
            "total_valid": self.stats.total_valid,
            "elapsed_seconds": self.stats.elapsed,
            "rate_per_second": self.stats.rate,
        }
```

## Integration Point

In `api_server.py`, after model is loaded:

```python
from vllm.poc.manager import PoCManager

async def init_app_state(engine_client, vllm_config, state, args):
    # ... existing code ...
    
    if args.enable_poc:
        model = engine_client.engine.model_executor.driver_worker.model_runner.model
        state.poc_manager = PoCManager(model, vllm_config.model_config)
```

## Directory Structure After Phase 3

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
├── gpu_random.py
├── manager.py
└── sender.py      # NEW: Callback sender for push model

tests/poc/
├── __init__.py
├── test_data.py
├── test_gpu_random.py
└── test_manager.py
```

## Unit Tests

### File: `tests/poc/test_manager.py`

**Cross-check**: Compare nonce iteration logic with original:
`/home/ubuntu/workspace/gonka/mlnode/packages/pow/src/pow/compute/utils.py` (NonceIterator)

Note: `run_batch()` and `validate()` require actual GPU model - tested in Phase 5 E2E tests.

```python
import pytest
import time
import torch
from unittest.mock import Mock, MagicMock, patch
from vllm.poc.config import PoCConfig, PoCState
from vllm.poc.manager import PoCManager, PoCStats

class MockModelConfig:
    """Mock vLLM model config"""
    hidden_size = 128
    vocab_size = 1000

class TestPoCStats:
    def test_initial_state(self):
        stats = PoCStats()
        assert stats.total_checked == 0
        assert stats.total_valid == 0
        assert stats.elapsed == 0.0
        assert stats.rate == 0.0
    
    def test_elapsed_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.elapsed >= 10
        assert stats.elapsed < 11
    
    def test_rate_calculation(self):
        stats = PoCStats(total_checked=100, start_time=time.time() - 10)
        assert stats.rate >= 9
        assert stats.rate <= 11

class TestPoCManagerInit:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        # Return a CPU tensor for testing (no GPU required)
        model.parameters.return_value = iter([torch.zeros(1)])
        return model
    
    @pytest.fixture
    def manager(self, mock_model):
        return PoCManager(mock_model, MockModelConfig())
    
    def test_initial_state(self, manager):
        assert manager.state == PoCState.IDLE
        assert manager.config is None
        assert manager.target is None
        assert manager.valid_nonces == []
        assert manager.valid_distances == []

class TestPoCManagerStateTransitions:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1, device="cpu")])
        return model
    
    @pytest.fixture
    def manager(self, mock_model):
        return PoCManager(mock_model, MockModelConfig())
    
    @pytest.fixture
    def config(self):
        return PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
        )
    
    def test_init_round_sets_config(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        assert manager.state == PoCState.IDLE  # init doesn't start generating
        assert manager.config == config
    
    def test_start_generate_sets_state(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        manager.start_generate()
        assert manager.state == PoCState.GENERATING
    
    def test_start_validate_sets_state(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        manager.start_validate()
        assert manager.state == PoCState.VALIDATING
    
    def test_init_round_resets_counters(self, manager, config):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
        
        assert manager.valid_nonces == []
        assert manager.valid_distances == []
        assert manager.stats.total_checked == 0
    
    def test_init_round_raises_if_already_generating(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        with pytest.raises(RuntimeError, match="Round already in progress"):
            manager.init_round(config)
    
    def test_start_generate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_generate()
    
    def test_start_validate_requires_init(self, manager):
        with pytest.raises(RuntimeError, match="Round not initialized"):
            manager.start_validate()
    
    def test_stop_round(self, manager, config):
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        manager.stop_round()
        assert manager.state == PoCState.STOPPED

class TestPoCManagerNonceGeneration:
    """Cross-check: Nonce iteration pattern with original NonceIterator"""
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1, device="cpu")])
        return model
    
    @pytest.fixture
    def manager(self, mock_model):
        return PoCManager(mock_model, MockModelConfig())
    
    def test_single_node_nonces(self, manager):
        """Single node gets sequential nonces: 0, 1, 2, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=0,
            node_count=1,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        nonces1 = manager.get_next_nonces()
        assert nonces1 == [0, 1, 2, 3]
        
        nonces2 = manager.get_next_nonces()
        assert nonces2 == [4, 5, 6, 7]
    
    def test_multi_node_nonces_node0(self, manager):
        """Node 0 of 3 gets: 0, 3, 6, 9, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=0,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [0, 3, 6, 9]
    
    def test_multi_node_nonces_node1(self, manager):
        """Node 1 of 3 gets: 1, 4, 7, 10, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=1,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [1, 4, 7, 10]
    
    def test_multi_node_nonces_node2(self, manager):
        """Node 2 of 3 gets: 2, 5, 8, 11, ..."""
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            r_target=0.5,
            node_id=2,
            node_count=3,
            batch_size=4,
        )
        with patch('vllm.poc.manager.generate_target') as mock_target:
            mock_target.return_value = torch.randn(1000)
            manager.init_round(config)
            manager.start_generate()
        
        nonces = manager.get_next_nonces()
        assert nonces == [2, 5, 8, 11]

class TestPoCManagerStatus:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.parameters.return_value = iter([torch.zeros(1, device="cpu")])
        return model
    
    @pytest.fixture
    def manager(self, mock_model):
        return PoCManager(mock_model, MockModelConfig())
    
    def test_get_status_idle(self, manager):
        status = manager.get_status()
        assert status["state"] == "IDLE"
        assert status["valid_nonces"] == []
        assert status["valid_distances"] == []
        assert status["total_checked"] == 0
        assert status["total_valid"] == 0
    
    def test_get_status_with_data(self, manager):
        manager.valid_nonces = [1, 2, 3]
        manager.valid_distances = [0.1, 0.2, 0.3]
        manager.stats.total_checked = 100
        manager.stats.total_valid = 3
        
        status = manager.get_status()
        assert status["valid_nonces"] == [1, 2, 3]
        assert status["valid_distances"] == [0.1, 0.2, 0.3]
        assert status["total_checked"] == 100
        assert status["total_valid"] == 3
```

## Running Tests

```bash
pytest tests/poc/test_manager.py -v
```

## Acceptance Criteria

- [ ] PoCManager can access deployed model
- [ ] `start_round()` sets state and generates target
- [ ] `run_batch()` computes distances and collects valid nonces
- [ ] `validate()` recomputes distances for verification
- [ ] Statistics tracked correctly
- [ ] State transitions work (IDLE -> GENERATING -> IDLE)
- [ ] All unit tests pass: `pytest tests/poc/test_manager.py`

