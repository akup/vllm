# Phase 2: GPU Random Generation

## Breaking Change Notice

This implementation uses murmur3-based deterministic random generation, which produces DIFFERENT sequences than the original numpy SeedSequence implementation. This is intentional for PoC 2.0. All nodes must use this new implementation - mixing old/new nodes will cause validation failures.

## Objective

Implement GPU-native deterministic random generation for PoC 2.0 with cross-device reproducibility.

## Deliverable

Working `gpu_random.py` with determinism tests passing.

## Why Murmur3?

torch.Generator on CUDA is NOT portable across GPU architectures (A100 vs H100 vs consumer GPUs). For PoC validation where validators must reproduce bit-exact same random numbers, this is a critical flaw.

Murmur3-based generation solves this:
- Pure integer/float math operations - deterministic on any device
- Standard algorithm with well-defined specification
- Box-Muller transform for normal distribution
- Argsort for permutations (fully parallel on GPU)

## Core Functions

### File: `vllm/poc/gpu_random.py`

```python
import hashlib
import math
from typing import List

import torch


def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    c1, c2 = 0xcc9e2d51, 0x1b873593
    seed_i32 = seed if seed < 0x80000000 else seed - 0x100000000
    h = torch.full_like(keys, seed_i32, dtype=torch.int32)
    k = keys.to(torch.int32)

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)
    return (hashes.to(torch.float32) + 2147483648.0) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = _seed_from_string(seed_str)
        normal = _normal(seed, seq_len * dim, device)
        result[i] = normal.view(seq_len, dim).to(dtype)

    return result


def generate_permutations(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(nonces)
    result = torch.empty(batch_size, vocab_size, device=device, dtype=torch.int64)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_permutations"
        seed = _seed_from_string(seed_str)
        uniform = _uniform(seed, vocab_size, device)
        result[i] = torch.argsort(uniform)

    return result


def generate_target(
    block_hash: str,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    seed_str = f"{block_hash}_target"
    seed = _seed_from_string(seed_str)
    normal = _normal(seed, vocab_size, device)
    target = normal.to(dtype)
    target = target / target.norm()
    return target


def compute_distances(
    logits: torch.Tensor,
    permutations: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    logits_perm = torch.gather(logits, 1, permutations)
    logits_norm = logits_perm / logits_perm.norm(dim=1, keepdim=True)
    return (logits_norm - target.unsqueeze(0)).norm(dim=1)
```

## Determinism Test

### File: `tests/poc/test_gpu_random.py`

```python
import torch
import pytest
from vllm.poc.gpu_random import (
    generate_inputs,
    generate_permutations,
    generate_target,
    compute_distances,
)

BLOCK_HASH = "test_block_hash_12345"
PUBLIC_KEY = "test_public_key"

def test_inputs_determinism():
    device = torch.device("cuda:0")
    nonces = [1, 2, 3]

    inputs1 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, nonces, dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, nonces, dim=128, seq_len=16, device=device)

    assert torch.allclose(inputs1, inputs2)

def test_inputs_different_nonces():
    device = torch.device("cuda:0")

    inputs1 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [1], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [2], dim=128, seq_len=16, device=device)

    assert not torch.allclose(inputs1, inputs2)

def test_permutations_determinism():
    device = torch.device("cuda:0")
    nonces = [1, 2, 3]

    perm1 = generate_permutations(BLOCK_HASH, PUBLIC_KEY, nonces, vocab_size=1000, device=device)
    perm2 = generate_permutations(BLOCK_HASH, PUBLIC_KEY, nonces, vocab_size=1000, device=device)

    assert torch.equal(perm1, perm2)

def test_target_unit_vector():
    device = torch.device("cuda:0")
    target = generate_target(BLOCK_HASH, vocab_size=1000, device=device)

    assert abs(target.norm().item() - 1.0) < 1e-5

def test_distance_range():
    device = torch.device("cuda:0")
    batch_size = 10
    vocab_size = 1000

    logits = torch.randn(batch_size, vocab_size, device=device)
    perms = generate_permutations(BLOCK_HASH, PUBLIC_KEY, list(range(batch_size)), vocab_size, device)
    target = generate_target(BLOCK_HASH, vocab_size, device)

    distances = compute_distances(logits, perms, target)

    assert (distances >= 0).all()
    assert (distances <= 2).all()

def test_different_block_hash():
    device = torch.device("cuda:0")

    target1 = generate_target("hash1", vocab_size=1000, device=device)
    target2 = generate_target("hash2", vocab_size=1000, device=device)

    assert not torch.allclose(target1, target2)
```

## Directory Structure After Phase 2

```
vllm/poc/
├── __init__.py
├── config.py
├── data.py
└── gpu_random.py

tests/poc/
├── __init__.py
├── test_data.py
└── test_gpu_random.py
```

## Acceptance Criteria

- [ ] All generation functions implemented with murmur3
- [ ] Determinism tests pass (same seed = same output)
- [ ] Different seeds produce different outputs
- [ ] Target is unit vector (norm = 1)
- [ ] Distances in range [0, 2]
- [ ] Works on CUDA device
- [ ] Portable across different GPU architectures
