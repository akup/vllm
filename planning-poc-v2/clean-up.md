# Production Clean-up (2026-01-02)

Simplified PoC pipeline for production deployment.

## Changes Summary

### worker_ops.py (416 -> 198 lines, -52%)

**Removed:**
- `POC_OUTPUT_DIM` constant and random lm_head projection
- `_random_lm_head_cache` and `_generate_random_lm_head()`
- `_normalize_distance_for_dim()` - k is now constant
- All transformation flags: `use_sign_flips`, `use_nonce_householder`, `use_nonce_orthogonal`, `pick_k_dims`
- `return_inputs`, `return_outputs` parameters
- Dead `artifacts` references
- `vocab_size` parameter (not needed)

**Simplified pipeline:**
```
input -> model.forward() -> normalize -> pick k dims -> Haar rotate -> distance to k-dim target
```

### config.py (39 -> 25 lines)

Removed all transformation flags - single pipeline, no options.

### manager.py (365 -> 233 lines, -36%)

- Removed artifact handling
- Removed `run_batch_with_state` optional parameters
- Layer hooks always enabled

### routes.py

- Removed experiment flags from `PoCInitRequest`
- Removed artifact encoding functions
- Simplified `/batch` endpoint

### gpu_random.py

- `generate_target()` now requires `public_key` parameter
- Target seeded by `f"{block_hash}_{public_key}_target"`

### Engine handlers

- `multiprocessing/engine.py` and `async_llm_engine.py`: removed `return_inputs`/`return_outputs` params from `run_batch_with_state` calls

### Scripts

Deleted outdated test scripts:
- `scripts/poc_e2e_test.py`
- `scripts/poc_full_e2e_test.py`

Created:
- `scripts/poc_e2e_simple_test.py` - simplified E2E test

## Production Pipeline

```
POC_PICK_K_DIMS = 64

1. Generate random inputs (block_hash, public_key, nonce)
2. Forward pass with layer hooks (per-layer normalization + Householder)
3. Normalize last hidden state to unit sphere
4. Pick k=64 dims per nonce (seeded)
5. Haar rotate in k-dim space
6. Compare to k-dim target (block_hash, public_key)
7. Distance on unit sphere
```

Target `r_target` for 10% valid rate in 64D: ~1.30

## Test Results

Smoke test: PASS
- Deterministic distances
- Different pubkey -> different distances
- Validation matches generation

## Next Steps (TODO)

Further cleanup for minimal production footprint:

### tests/
- Review `tests/poc/test_gpu_random.py` - remove tests for deprecated functions
- Remove tests that use old transformation flags

### routes.py
- Remove `/phase/generate_manual` if not needed
- Simplify callback handling if not used
- Remove unused imports

### gpu_random.py
- Remove deprecated `generate_permutations()`
- Remove deprecated `compute_distances()` (with permutation)
- Remove unused convenience wrappers:
  - `apply_sign_flips_then_normalize()`
  - `apply_householder_reflections()`
  - `slice_k_from_full()`
  - `orthogonal_transform_k()`
  - `haar_rotate_k()`
  - `fixed_project_full_to_k()`
- Keep only functions used by production pipeline

### scripts/poc_distribution/
- Evaluate if distribution analysis scripts are still needed
- Remove if not used in production

### layer_hooks.py
- Review if any simplification possible

