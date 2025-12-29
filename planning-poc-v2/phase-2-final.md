# Phase 2: GPU Random Generation - Final

## Status: Complete

## Files Created

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
└── gpu_random.py      <- NEW

tests/poc/
├── __init__.py
├── test_data.py
└── test_gpu_random.py <- NEW
```

## Implementation Summary

### vllm/poc/gpu_random.py

Internal helpers:
- `_seed_from_string(seed_string)` - SHA256 to uint32 seed
- `_murmur3_32(keys, seed)` - Standard murmur3-32 hash on GPU tensors
- `_uniform(seed, n, device)` - Murmur3 hash to uniform floats [0, 1)
- `_normal(seed, n, device)` - Box-Muller transform for normal distribution

Public API:
- `generate_inputs(block_hash, public_key, nonces, dim, seq_len, device, dtype)` - Returns `[batch, seq_len, dim]`
- `generate_permutations(block_hash, public_key, nonces, vocab_size, device)` - Returns `[batch, vocab_size]`
- `generate_target(block_hash, vocab_size, device, dtype)` - Returns unit vector `[vocab_size]`
- `compute_distances(logits, permutations, target)` - Returns `[batch]`

### tests/poc/test_gpu_random.py
- 6 unit tests covering determinism, different seeds, unit vector, distance range
- All tests passing

## Acceptance Criteria

- [x] All generation functions implemented with murmur3
- [x] Determinism tests pass (same seed = same output)
- [x] Different seeds produce different outputs
- [x] Target is unit vector (norm = 1)
- [x] Distances in range [0, 2]
- [x] Works on CUDA device
- [x] Portable across different GPU architectures

## Design Decisions

- **Murmur3 over torch.Generator**: torch.Generator on CUDA is not portable across GPU architectures. Murmur3 is pure math, deterministic on any device.
- **Box-Muller for normal distribution**: Standard transform from uniform to normal, using pairs of uniform randoms.
- **Argsort for permutations**: `torch.argsort(uniform_randoms)` is fully parallel on GPU, unlike sequential Fisher-Yates.
- **Int32 overflow handling**: Seeds from SHA256 can exceed signed int32 range; converted using `seed - 0x100000000` when needed.
- **Seed format**: Uses 8 hex chars (32 bits) from SHA256, sufficient entropy for murmur3 seed.

