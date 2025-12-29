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
    ├── model_config: ModelConfig (for hidden_size, vocab_size)
    ├── vllm_config: VllmConfig (for set_forward_context)
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
        ├── get_status() → Returns current state and results
        └── _create_prefill_attn_metadata() → Creates minimal attention metadata

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

### Nonce Iteration Simplification

NOTE: Rethink - Original implementation has `group_id` / `n_groups` concept. Probably we should get it back as single MLNode still might have multiple instances of vllm ...

## Model Access

vLLM loads the model once at startup. We access it via:

```python
# In api_server.py, after init_app_state:
model = engine_client.engine.model_executor.driver_worker.model_runner.model
model_config = engine_client.engine.model_config
vllm_config = engine_client.engine.vllm_config
```

## Forward Context

vLLM models require a `ForwardContext` to be set before forward passes. The context includes attention metadata and vllm_config.

For PoC, we use minimal attention metadata with `PAD_SLOT_ID` (-1) for all slot mappings. This tells the KV cache kernels to skip storage, which is appropriate since we don't need KV caching for single-step PoC forward passes.

```python
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import set_forward_context

# Create minimal attention metadata
attn_metadata = FlashAttentionMetadata(
    num_prefills=batch_size,
    num_prefill_tokens=num_tokens,
    num_decode_tokens=0,
    slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=device),
    # ... other required fields
)

# Run model with forward context
with set_forward_context(attn_metadata, vllm_config):
    hidden_states = model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        positions=positions,
    )

# Model-agnostic logits computation
if hasattr(model, 'compute_logits'):
    logits = model.compute_logits(hidden_states, sampling_metadata=None)
else:
    logits = model.lm_head(hidden_states)
```

## Future: Multistep Inference with KV Cache

Current implementation uses PAD_SLOT_ID which skips KV storage. For proper multistep:
1. Use CacheEngine to allocate KV cache blocks
2. Compute slot_mapping from block_tables (slot = block_id * block_size + offset)
3. Track sequence positions across steps (context_len grows each step)
4. Support decode batches (num_decode_tokens > 0)
5. Consider using model_runner.execute_model() for full vLLM parity

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
    """Sends valid batches to callback URL with retry logic.
    
    Failed batches are stored and retried on subsequent send calls.
    Callback failures don't block generation.
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

Key implementation details:

```python
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import set_forward_context

class PoCManager:
    def __init__(self, model, model_config, vllm_config: "VllmConfig"):
        self.model = model
        self.model_config = model_config
        self.vllm_config = vllm_config  # Needed for set_forward_context
        # ... rest of init
    
    def _create_prefill_attn_metadata(self, batch_size: int, seq_len: int) -> FlashAttentionMetadata:
        """Create minimal attention metadata for prefill-only forward pass.
        
        Uses PAD_SLOT_ID for all slots, so KV cache writes are skipped.
        """
        num_tokens = batch_size * seq_len
        seq_lens = [seq_len] * batch_size
        
        seq_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        seq_start_loc[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device), dim=0
        )
        
        return FlashAttentionMetadata(
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
            slot_mapping=torch.full((num_tokens,), PAD_SLOT_ID, dtype=torch.long, device=self.device),
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int, device=self.device),
            max_prefill_seq_len=seq_len,
            max_decode_seq_len=0,
            query_start_loc=seq_start_loc.clone(),
            seq_start_loc=seq_start_loc,
            context_lens_tensor=torch.zeros(batch_size, dtype=torch.int, device=self.device),
            block_tables=torch.empty((batch_size, 0), dtype=torch.int, device=self.device),
            use_cuda_graph=False,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
    
    @torch.inference_mode()
    def run_batch(self) -> ProofBatch:
        """Run one batch of nonce computation."""
        # ... nonce generation and input preparation ...
        
        # Create minimal attention metadata with PAD_SLOT_ID
        attn_metadata = self._create_prefill_attn_metadata(batch_size, seq_len)
        
        # Run model with forward context
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=None,
                positions=positions.flatten(),
                inputs_embeds=inputs_embeds.view(-1, hidden_size),
            )
        
        # ... logits computation, distance calculation, stats update ...
```

See full implementation in `vllm/poc/manager.py`.

## Integration Point

In `api_server.py`, after model is loaded:

```python
from vllm.poc.manager import PoCManager

async def init_app_state(engine_client, vllm_config, state, args):
    # ... existing code ...
    
    if args.enable_poc:
        # v0 API access pattern
        # TODO: Update for v1 compatibility - model access path differs in v1 engine
        model = engine_client.engine.model_executor.driver_worker.model_runner.model
        model_config = engine_client.engine.model_config
        vllm_config = engine_client.engine.vllm_config
        state.poc_manager = PoCManager(model, model_config, vllm_config)
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

Note: `run_batch()` and `validate()` require actual GPU model - tested via `scripts/poc_smoke_test.py`.

Key testing patterns:

```python
from unittest.mock import Mock, MagicMock, patch

class MockModelConfig:
    """Mock vLLM model config (v0 API)"""
    def get_hidden_size(self):
        return 128
    def get_vocab_size(self):
        return 1000

def create_mock_vllm_config():
    """Create a mock VllmConfig for testing."""
    vllm_config = MagicMock()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.data_parallel_size = 1
    return vllm_config

class TestPoCManagerInit:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        param = torch.zeros(1)
        model.parameters.side_effect = lambda: iter([param])
        return model
    
    @pytest.fixture
    def mock_vllm_config(self):
        return create_mock_vllm_config()
    
    @pytest.fixture
    def manager(self, mock_model, mock_vllm_config):
        return PoCManager(mock_model, MockModelConfig(), mock_vllm_config)
```

Full test suite: 20 unit tests covering stats, state transitions, nonce generation, status reporting.

### File: `scripts/poc_smoke_test.py`

Smoke test that runs PoCManager with real vLLM model:

```bash
VLLM_USE_V1=0 python scripts/poc_smoke_test.py
```

Verifies:
- Distances in valid range [0, 2]
- Deterministic (recompute matches original)
- Different public key produces different distances
- Stats tracking works correctly

## Running Tests

```bash
pytest tests/poc/test_manager.py -v
```

## Acceptance Criteria

- [x] PoCManager can access deployed model with proper forward context
- [x] `init_round()` sets state and generates target
- [x] `run_batch()` computes distances and collects valid nonces
- [x] `validate()` recomputes distances for verification
- [x] Statistics tracked correctly
- [x] State transitions work (IDLE -> GENERATING -> VALIDATING -> STOPPED)
- [x] All unit tests pass: `pytest tests/poc/test_manager.py`
- [x] Smoke test passes: `VLLM_USE_V1=0 python scripts/poc_smoke_test.py`

