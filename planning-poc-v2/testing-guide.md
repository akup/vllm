# PoC Testing Guide

## Running Tests

### Unit Tests
```bash
cd /home/ubuntu/workspace/vllm
source .venv/bin/activate
python -m pytest tests/poc/ -v
```

### Smoke Test (quick validation with real model)
```bash
python scripts/poc_smoke_test.py
```

### Full E2E Tests (server mode with callbacks)
```bash
# Run with all configured models
python scripts/poc_e2e_test.py

# Run with specific models
python scripts/poc_e2e_test.py --models qwen
python scripts/poc_e2e_test.py --models qwen llama

# Logs saved to: logs/
#   - server_{model}.log  - vLLM server logs with PoC progress
#   - callback.log        - Callback receiver logs
#   - test_results.json   - Test results summary
```

### Full 18-Test Distribution Experiment
```bash
# Run complete distribution test (18 tests: 2 models × 3 blocks × 3 public keys)
python scripts/poc_full_e2e_test.py

# Run with specific models
python scripts/poc_full_e2e_test.py --models qwen
python scripts/poc_full_e2e_test.py --models llama

# Adjust test duration (default: 80s per test)
python scripts/poc_full_e2e_test.py --duration 40

# Monitor progress
tail -f logs/e2e_results.jsonl
```

This test validates that the per-layer normalization and random `lm_head` produce **consistent valid rates** across:
- **2 models**: Qwen/Qwen3-0.6B, unsloth/Llama-3.2-1B-Instruct
- **3 block hashes**: block_alpha, block_beta, block_gamma
- **3 public keys**: node_A, node_B, node_C

Each test runs with fresh server + callback instances. Results saved to `logs/e2e_results.jsonl`.

**Expected Results** (with `r_target=1.416`):
| Model | Valid Rate Range | Spread |
|-------|-----------------|--------|
| Qwen  | 58-65%          | ~6%    |
| Llama | 53-65%          | ~11%   |

**Key Validation**: Valid rates should be consistent across different `public_key` values within the same model/block_hash combination, confirming per-nonce randomization works correctly.

## E2E Test Phases

1. **Generation Phase**: Start server, generate nonces for 10s, collect valid batches
2. **Restart & Validate Phase**: Restart server, validate same nonces produce same distances
3. **Wrong Seed Tests**: Verify fraud detection with wrong block_hash/public_key

## r_target Estimation

The `r_target` parameter controls what fraction of nonces are considered "valid" (distance < r_target).

### Formula

From original PoW (`pow/compute/stats.py`):

```python
def estimate_R(n, P, num_samples=100000):
    """Estimate R for dimension n such that P fraction of points are within distance R.
    
    Args:
        n: Hidden dimension of model
        P: Desired probability/fraction of valid nonces (e.g., 0.1 for 10%)
    
    Returns:
        R value to use as r_target
    """
    import numpy as np
    points = np.random.normal(size=(num_samples, n))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    sample_point = points[0]
    distances = np.linalg.norm(points - sample_point, axis=1)
    sorted_distances = np.sort(distances)
    index = int(P * num_samples)
    return sorted_distances[index]
```

### Theoretical vs Empirical Values

**UPDATE (Phase 4.2)**: With per-layer normalization and random lm_head (POC_OUTPUT_DIM=8192), the distribution is now **consistent** and matches theoretical values:

| Model | Theoretical (10%) | Current (10%) | Cross-Block Spread |
|-------|------------------|---------------|-------------------|
| Qwen/Qwen3-0.6B | 1.4119 | **~1.404** | 3.5% |
| Llama-3.2-1B-Instruct | 1.4116 | **~1.407** | 2.0% |

### Empirical Calibration (UPDATED)

**Both Models** (consistent distribution with per-layer normalization):
```
 5% valid: r_target ≈ 1.400
10% valid: r_target ≈ 1.405
15% valid: r_target ≈ 1.408
```

**Note**: The old model-specific values (Qwen ~1.17, Llama ~1.38) are **obsolete**. Per-layer normalization breaks the structure that caused those compressed distributions.

### Key Insight: Per-Layer Normalization Solved Structure Problem

The initial problem was that trained models had structure that caused inconsistent distributions:
- Different block_hashes → different Householder rotations → different mean distances
- Result: 49-54% spread in valid rates across block_hashes

**Solution**: Normalize hidden states to unit sphere at each layer:
```python
def normalize_and_transform(x):
    x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    return apply_householder(x_norm, v)
```

This breaks structure accumulation because normalization is non-linear (unlike orthogonal transforms like Householder, rotations, permutations which preserve structure).

## Server Logs

With proper logging, server shows progress during generation:

```
INFO [routes.py:100] PoC generation started (r_target=1.38)
INFO [routes.py:122] Generated: 169 / 1536 (1 in 9) Time: 0.08min (2023.23 valid/min)
INFO [routes.py:122] Generated: 336 / 3104 (1 in 9) Time: 0.17min (1998.42 valid/min)
INFO [routes.py:142] PoC stopped: 336 / 3104 (1 in 9) in 0.17min (1998.08 valid/min)
```

Format: `Generated: {valid} / {total} (1 in {ratio}) Time: {elapsed}min ({rate} valid/min)`

## Troubleshooting

### "No available memory for the cache blocks"
Increase `--gpu-memory-utilization` or use smaller model.

### FP8 models fail with "RuntimeError: Error Internal"
GPU doesn't support FP8. Use non-quantized models (e.g., `unsloth/Llama-3.2-1B-Instruct` instead of FP8-Dynamic).

### Validation after restart fails
Some models may have non-deterministic behavior due to:
- Flash Attention implementation
- CUDA graph variations
- Model-specific numerical precision

Test with `--enforce-eager` flag to disable CUDA graphs for debugging.

