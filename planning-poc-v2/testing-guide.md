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

Theoretical formula assumes uniform distribution on sphere (random weights). **Trained models have structure** that compresses distances. Always use empirical values.

| Model | Theoretical (10%) | Empirical (10%) | Mean Distance |
|-------|------------------|-----------------|---------------|
| Qwen/Qwen3-0.6B | 1.4119 | **1.1739** | 1.2155 |
| Llama-3.2-1B-Instruct | 1.4116 | **1.3796** | 1.4486 |
| Random weights | 1.4119 | 1.4117 | 1.4138 |

### Empirical Calibration (Use These)

**Qwen/Qwen3-0.6B** (highly compressed distribution):
```
 5% valid: r_target = 1.166
10% valid: r_target = 1.174
20% valid: r_target = 1.184
30% valid: r_target = 1.193
50% valid: r_target = 1.209
```

**Llama-3.2-1B-Instruct** (close to theoretical):
```
 5% valid: r_target = 1.363
10% valid: r_target = 1.380
20% valid: r_target = 1.406
30% valid: r_target = 1.423
50% valid: r_target = 1.451
```

### Key Insight: Model Structure

Experiments confirmed (see `logs/experiment_report.md`):
- Random Qwen (same architecture, random weights): mean = 1.4138 (matches theory)
- Trained Qwen: mean = 1.2155 (14% lower than theory)
- Trained Llama: mean = 1.4486 (close to theory)

This proves trained weights create structure that compresses Qwen's output distribution.

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

