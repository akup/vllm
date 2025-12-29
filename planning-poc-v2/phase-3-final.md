# Phase 3: PoC Manager - Final

## Status: Complete

## Files Created/Modified

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
├── gpu_random.py
├── manager.py      <- NEW
└── sender.py       <- NEW

tests/poc/
├── __init__.py
├── test_data.py
├── test_gpu_random.py
└── test_manager.py <- NEW

scripts/
└── poc_smoke_test.py <- NEW
```

## Implementation Summary

### vllm/poc/manager.py

Core classes:
- `PoCStats` - Dataclass tracking total_checked, total_valid, elapsed time, rate
- `PoCManager` - Orchestrates PoC rounds with model forward passes

Key methods:
- `__init__(model, model_config, vllm_config)` - Accepts vllm_config for forward context
- `init_round(config)` - Sets config, generates target, state = IDLE
- `start_generate()` / `start_validate()` / `stop_round()` - State transitions
- `run_batch()` - Computes distances using set_forward_context
- `validate(nonces, public_key)` - Recomputes distances for verification
- `get_status()` - Returns current state and results
- `_create_prefill_attn_metadata()` - Creates minimal attention metadata with PAD_SLOT_ID

### vllm/poc/sender.py

- `PoCCallbackSender` - Sends valid batches to callback URL with retry logic
- Fire-and-forget pattern - failures logged but don't block generation

### tests/poc/test_manager.py

- 20 unit tests covering stats, state transitions, nonce generation, status
- Uses mocked model and vllm_config for fast CPU-only testing
- GPU tests (run_batch, validate) skipped in unit tests, covered by smoke test

### scripts/poc_smoke_test.py

Smoke test that runs PoCManager with real Qwen3-0.6B model:
- Loads model using vLLM v0 API
- Runs multiple batches
- Validates nonce recomputation
- Verifies determinism and correctness

## Acceptance Criteria

- [x] PoCManager can access deployed model with proper forward context
- [x] `init_round()` sets state and generates target
- [x] `run_batch()` computes distances and collects valid nonces
- [x] `validate()` recomputes distances for verification
- [x] Statistics tracked correctly
- [x] State transitions work (IDLE -> GENERATING -> VALIDATING -> STOPPED)
- [x] All unit tests pass: `pytest tests/poc/test_manager.py`
- [x] Smoke test passes: `VLLM_USE_V1=0 python scripts/poc_smoke_test.py`

## Design Decisions

### Forward Context with PAD_SLOT_ID

vLLM models require `set_forward_context()` before forward passes. The context includes attention metadata with slot mappings for KV cache.

For PoC single-step inference, we use `PAD_SLOT_ID` (-1) for all slot mappings. This tells the KV cache kernels to skip storage:

```c++
// In vLLM CUDA kernels:
const int64_t slot_idx = slot_mapping[token_idx];
if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
}
```

This approach:
- Avoids allocating full KV caches
- Matches vLLM's profiling run pattern
- Works with any model architecture
- Is simpler than full model_runner integration

### vllm_config Dependency

Manager requires `vllm_config` parameter (from `llm.llm_engine.vllm_config`) for:
- `set_forward_context()` needs compilation_config.static_forward_context
- Matches vLLM's internal patterns

### Model Config Access

Uses `model_config.get_hidden_size()` and `model_config.get_vocab_size()` methods instead of direct attribute access for v0/v1 compatibility.

## Future Work (TODOs in code)

### Multistep Inference with KV Cache

Current implementation uses PAD_SLOT_ID which skips KV storage. For proper multistep:
1. Use CacheEngine to allocate KV cache blocks
2. Compute slot_mapping from block_tables (slot = block_id * block_size + offset)
3. Track sequence positions across steps (context_len grows each step)
4. Support decode batches (num_decode_tokens > 0)
5. Consider using model_runner.execute_model() for full vLLM parity

### vLLM v1 Compatibility

Current implementation uses v0 API:
- Model access: `model_executor.driver_worker.model_runner.model`
- Need to update for v1: `worker.get_model()` or `collective_rpc` pattern

## Test Results

```
pytest tests/poc/test_manager.py -v
20 passed, 5 skipped (GPU tests deferred to smoke test)

VLLM_USE_V1=0 python scripts/poc_smoke_test.py
[PASS] Distances in valid range [0, 2]
[PASS] Deterministic (recompute matches)
[PASS] Different pubkey -> different distances
[PASS] Stats tracking works
ALL CHECKS PASSED!
```

