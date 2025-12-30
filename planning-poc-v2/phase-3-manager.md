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

**Note:** Phase 4 updated this to use `collective_rpc` for TP/PP support. The direct model access shown below is outdated.

```python
# Phase 4 approach - via collective_rpc in PoCManager:
model_executor = engine_client.engine.model_executor
model_config = engine_client.engine.model_config
vllm_config = engine_client.engine.vllm_config

manager = PoCManager(model_executor, model_config, vllm_config)
# Uses collective_rpc internally for TP/PP support
# See vllm/poc/worker_ops.py for actual GPU execution
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

**Note:** The callback sender logic was simplified and moved inline into `routes.py` during Phase 4 integration. The separate `sender.py` file was not created - callback sending uses `aiohttp` directly in the route handlers.

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

**Note:** Phase 4 moved integration to the engine multiprocessing layer. PoCManager is created in `MQLLMEngine` and accessed via RPC, not directly in api_server.

```python
# Phase 4 approach - in vllm/engine/multiprocessing/engine.py:
# PoCManager is created lazily when first PoC request arrives
# API routes use engine_client.poc_request() to communicate via RPC

# See phase-4-integration.md for full architecture
```

## Directory Structure After Phase 3

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
├── gpu_random.py
└── manager.py

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

