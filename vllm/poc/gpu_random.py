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
