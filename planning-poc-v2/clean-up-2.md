# Production Clean-up Phase 2 (2026-01-02)

Continued simplification from clean-up.md.

## Changes Summary

### gpu_random.py (500+ -> 244 lines, -51%)

**Removed deprecated functions:**
- `generate_permutations()` - old permutation-based pipeline
- `compute_distances()` - old distance with permutation
- `generate_nonce_transform_vectors()` - superseded by Haar rotation
- `compute_distances_direct()` - unused
- `generate_sign_flips()` - removed from pipeline

**Removed convenience wrappers (moved to scripts/poc_distribution/):**
- `unit_normalize()`
- `apply_sign_flips_then_normalize()`
- `apply_householder_reflections()`
- `slice_k_from_full()`
- `orthogonal_transform_k()`
- `haar_rotate_k()`
- `fixed_project_full_to_k()`
- `random_pick_orthogonal_transform()`
- `generate_nonce_transform_vectors()`

**Retained core functions:**
- `_seed_from_string()`, `_murmur3_32()`, `_uniform()`, `_normal()`
- `generate_inputs()`, `generate_target()`
- `generate_householder_vector()`, `apply_householder()`
- `random_pick_indices()`, `generate_haar_orthogonal_matrices()`

### scripts/poc_distribution/gpu_random_utils.py (NEW)

Created to house analysis-specific utilities moved from gpu_random.py.
Used by `run_bubble_from_raw.py` for distribution analysis.

### tests/poc/test_gpu_random.py (400+ -> 321 lines)

**Removed tests for deprecated functions:**
- `test_permutations_determinism`
- `test_distance_range`
- `test_permutation_is_valid`
- `test_different_public_key_produces_different_permutations`
- `test_cpu_gpu_permutations_match`
- `test_nonce_transform_vectors_*`
- `test_compute_distances_direct_range`

**Added tests for new functions:**
- `test_random_pick_indices_*` (determinism, shape, range, uniqueness, CPU/GPU match)
- `test_haar_orthogonal_matrices_*` (determinism, shape, orthogonality, determinant, CPU/GPU match)

### routes.py

**Removed:**
- `/phase/generate_manual` endpoint (unused)
- `return_inputs`/`return_outputs` references in docstrings

### scripts/poc_e2e_test.py (REWRITTEN)

Comprehensive E2E test with 3x3 seed matrix:
- Tests 9 seed combinations per model without server restart
- Saves per-seed JSON with full nonces for later validation
- Determinism test: repeats first seed, compares nonces
- Independence check: verifies different seeds produce different nonces
- Fraud detection: wrong hash/pubkey tests

### Deleted Files

- `scripts/poc_e2e_simple_test.py` - replaced by comprehensive test
- `scripts/poc_distribution/collect_hidden_vectors.py` - broken, referenced removed features

## Test Results

### E2E Test (100s per seed)

| Model | Seeds | Total Checked | Total Valid | Avg Rate |
|-------|-------|---------------|-------------|----------|
| qwen | 9 | 116,736 | 24,326 | 20.8% |
| llama | 9 | 69,696 | 14,322 | 20.5% |
| qwen4b | 9 | 19,872 | 4,130 | 20.8% |

All tests PASS:
- Determinism: Same seed produces same nonces
- Independence: Different seeds produce different nonces
- Fraud detection: Wrong hash/pubkey detected

### Unit Tests

```bash
python -m pytest tests/poc/test_gpu_random.py -v
# 24 passed
```

## File Summary

| File | Before | After | Change |
|------|--------|-------|--------|
| gpu_random.py | 500+ | 244 | -51% |
| test_gpu_random.py | 400+ | 321 | -20% |
| routes.py | 415 | 395 | -5% |
| poc_e2e_test.py | N/A | 607 | new |

## Production Pipeline (unchanged)

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

