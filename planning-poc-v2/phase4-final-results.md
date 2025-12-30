# Phase 4: Final Results - Sign Flips Wins

**Date**: 2025-12-30  
**Status**: COMPLETE ✓

## Executive Summary

After extensive testing (36 E2E tests), **Sign Flips** is the recommended randomization approach:

| Model | Sign Flips | Layer Hooks | Winner |
|-------|------------|-------------|--------|
| Qwen 0.6B | **1.1%** spread | 3.9% spread | Sign Flips |
| Llama 1B | **0.9%** spread | 10.3% spread | Sign Flips |

**Success Criteria**: Cross-block spread < 5%  
**Sign Flips**: ✓ PASS (both models)  
**Layer Hooks**: ✗ FAIL (Llama exceeds 5%)

## Test Configuration

- **Models**: Qwen/Qwen3-0.6B, unsloth/Llama-3.2-1B-Instruct
- **Block hashes**: block_alpha, block_beta, block_gamma
- **Public keys**: node_A, node_B, node_C
- **r_target**: 1.416
- **Duration**: 40s per test
- **Total tests**: 36 (18 per mode)

## Sign Flips Results (Recommended)

```
QWEN:
  block_alpha: 57.7%, 58.5%, 58.0% -> avg=58.1%
  block_beta:  59.7%, 58.1%, 58.9% -> avg=58.9%
  block_gamma: 58.9%, 59.7%, 58.9% -> avg=59.2%
  CROSS-BLOCK SPREAD: 1.1%

LLAMA:
  block_alpha: 57.5%, 58.3%, 58.4% -> avg=58.1%
  block_beta:  58.8%, 57.3%, 58.1% -> avg=58.1%
  block_gamma: 58.7%, 59.5%, 58.8% -> avg=59.0%
  CROSS-BLOCK SPREAD: 0.9%
```

## Layer Hooks Results (Alternative)

```
QWEN:
  block_alpha: 60.2%, 60.0%, 58.9% -> avg=59.7%
  block_beta:  63.3%, 64.9%, 62.6% -> avg=63.6%
  block_gamma: 60.0%, 60.4%, 59.7% -> avg=60.0%
  CROSS-BLOCK SPREAD: 3.9%

LLAMA:
  block_alpha: 62.5%, 62.7%, 62.4% -> avg=62.5%
  block_beta:  64.4%, 64.7%, 65.3% -> avg=64.8%
  block_gamma: 55.5%, 54.2%, 53.9% -> avg=54.5%
  CROSS-BLOCK SPREAD: 10.3%  ← FAILS <5% CRITERIA
```

## Why Sign Flips Wins

1. **Better consistency**: 0.9-1.1% spread vs 3.9-10.3%
2. **Model-agnostic**: Works equally well on Qwen and Llama
3. **Simpler**: No forward hooks, no layer modifications
4. **Faster**: ~5% throughput improvement (no per-layer overhead)

## Implementation

Default configuration (`vllm/poc/config.py`):
```python
use_layer_hooks: bool = False  # Per-layer normalization (alternative)
use_sign_flips: bool = True    # Per-nonce sign flips (recommended)
```

Pipeline (`vllm/poc/worker_ops.py`):
```python
# 1. Apply per-nonce sign flips
signs = generate_sign_flips(block_hash, public_key, nonces, hidden_size, device)
last_hidden = last_hidden * signs

# 2. Normalize to unit sphere (breaks magnitude structure)
last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

# 3. Per-nonce Householder transform (8 reflections)
for r in range(8):
    last_hidden = apply_householder(last_hidden, transform_vectors[:, r, :])

# 4. Random lm_head projection
logits = last_hidden @ random_lm_head.T

# 5. Distance to target
distances = compute_distances_direct(logits, target)
```

## Test Logs

Full test logs available in:
- `logs/sign_flips_e2e.log` - Sign Flips mode (18 tests)
- `logs/layer_hooks_e2e.log` - Layer Hooks mode (18 tests)
- `logs/e2e_results.jsonl` - All 36 test results in JSON Lines format

## Conclusion

**Sign Flips is the final solution** for PoC randomization:
- Achieves < 2% cross-block spread on all tested models
- Simple implementation, no model hooks required
- Now the default configuration

