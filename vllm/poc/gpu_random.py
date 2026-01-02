import hashlib
import math
from typing import List

import torch


def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xcc9e2d51, 0x1b873593
    
    # Work in int64 to handle uint32 range properly
    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

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
    hashes = _murmur3_32(indices, seed)  # Returns int64 in [0, 2^32)
    return hashes.to(torch.float32) / 4294967296.0


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
    """Generate per-nonce permutations for logit reordering.
    
    DEPRECATED: Use generate_nonce_transform_vectors + apply_householder instead.
    Hidden state Householder transforms provide better structure breaking with less memory.
    
    Kept for backwards compatibility with tests and experiments.
    """
    batch_size = len(nonces)
    result = torch.empty(batch_size, vocab_size, device=device, dtype=torch.int64)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_permutations"
        seed = _seed_from_string(seed_str)
        indices = torch.arange(vocab_size, device=device, dtype=torch.int32)
        hashes = _murmur3_32(indices, seed)
        result[i] = torch.argsort(hashes, stable=True)

    return result


def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    seed_str = f"{block_hash}_{public_key}_target"
    seed = _seed_from_string(seed_str)
    normal = _normal(seed, dim, device)
    target = normal.to(dtype)
    target = target / target.norm()
    return target


def compute_distances(
    logits: torch.Tensor,
    permutations: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute distances with logit permutation.
    
    DEPRECATED: Use compute_distances_direct instead.
    The new hidden state transform approach doesn't need permutations.
    
    Kept for backwards compatibility with tests and experiments.
    """
    logits_perm = torch.gather(logits, 1, permutations)
    logits_norm = logits_perm / logits_perm.norm(dim=1, keepdim=True)
    return (logits_norm - target.unsqueeze(0)).norm(dim=1)


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection."""
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v·x)*v
    
    Args:
        x: Input tensor of shape [..., dim]
        v: Unit vector of shape [dim] or [batch, dim]
    
    Returns:
        Transformed tensor of same shape as x
    """
    if v.dim() == 1:
        dot = (x * v).sum(dim=-1, keepdim=True)
        return x - 2 * dot * v
    else:
        dot = (x * v).sum(dim=-1, keepdim=True)
        return x - 2 * dot * v


def generate_nonce_transform_vectors(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    device: torch.device,
    num_reflections: int = 4,
) -> torch.Tensor:
    """Generate per-nonce Householder vectors for hidden state transform.
    
    Args:
        block_hash: Block hash for deterministic generation
        public_key: Public key for deterministic generation
        nonces: List of nonces
        dim: Hidden dimension size
        device: Target device
        num_reflections: Number of Householder reflections per nonce
    
    Returns:
        Tensor of shape [batch_size, num_reflections, dim]
    """
    batch_size = len(nonces)
    vectors = torch.empty(batch_size, num_reflections, dim, device=device, dtype=torch.float32)
    
    for i, nonce in enumerate(nonces):
        for r in range(num_reflections):
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_hidden_r{r}"
            vectors[i, r] = generate_householder_vector(seed_str, dim, device)
    
    return vectors


def compute_distances_direct(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute distances from normalized logits to target (no permutation).
    
    Args:
        logits: Logits tensor of shape [batch, vocab_size]
        target: Target unit vector of shape [vocab_size]
    
    Returns:
        Distances tensor of shape [batch]
    """
    logits_norm = logits / logits.norm(dim=1, keepdim=True)
    return (logits_norm - target.unsqueeze(0)).norm(dim=1)


def generate_sign_flips(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    device: torch.device,
    round_idx: int = 0,
) -> torch.Tensor:
    """Generate per-nonce random sign flips (+1 or -1) for each dimension.
    
    This breaks hidden state clustering by randomly flipping the sign of each
    dimension, which decorrelates the directional structure while preserving
    the magnitude structure after normalization.
    
    Args:
        block_hash: Block hash for deterministic generation
        public_key: Public key for deterministic generation
        nonces: List of nonces
        dim: Hidden dimension size
        device: Target device
        
    Returns:
        Tensor of shape [batch_size, dim] with values +1 or -1
    """
    batch_size = len(nonces)
    signs = torch.empty(batch_size, dim, device=device, dtype=torch.float32)
    
    for i, nonce in enumerate(nonces):
        # round_idx=0 preserves the original seed string for backwards compatibility.
        if int(round_idx) == 0:
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_signs"
        else:
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_signs_r{int(round_idx)}"
        seed = _seed_from_string(seed_str)
        # Generate uniform values and threshold at 0.5 to get +1/-1
        u = _uniform(seed, dim, device)
        signs[i] = (u > 0.5).float() * 2 - 1  # Convert to +1/-1
    
    return signs


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (seed-based).
    
    We score each dimension index by a seeded hash and take the k smallest
    scores. This yields a deterministic, per-nonce subset without replacement.
    
    Returns:
        indices: int64 tensor of shape [batch_size, k]
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)
    out = torch.empty(batch_size, k, device=device, dtype=torch.int64)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32)

    for i, nonce in enumerate(nonces):
        seed = _seed_from_string(
            f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}"
        )
        scores = _murmur3_32(all_idx, seed)  # int64
        # Take k smallest scores via topk on the negated values (O(dim log k)).
        _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False)
        out[i] = chosen.to(torch.int64)

    return out


def generate_haar_orthogonal_matrices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate per-nonce Haar-random orthogonal matrices of shape [k, k].
    
    Uses seeded Gaussian -> QR -> sign-fix to get Haar distribution over O(k).
    
    Returns:
        Q: Tensor of shape [batch_size, k, k]
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    Qs = []
    for nonce in nonces:
        seed = _seed_from_string(
            f"{block_hash}_{public_key}_nonce_{nonce}_haar_qr_{k}"
        )
        A = _normal(seed, k * k, device).view(k, k).to(dtype)

        Q, R = torch.linalg.qr(A, mode="reduced")

        # Haar sign-fix: make diag(R) positive by scaling columns of Q.
        diag = torch.diagonal(R, 0)
        signs = torch.sign(diag)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)

        Qs.append(Q)

    return torch.stack(Qs, dim=0)


def random_pick_orthogonal_transform(
    x: torch.Tensor,
    target_full: torch.Tensor,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-pick k dims per nonce, rotate by Haar-random Q[k,k], and slice target.
    
    Steps (per nonce):
      1) Pick indices I (seeded by nonce) of length k
      2) Build Haar-random orthogonal Q[k,k] (seeded by nonce)
      3) y = Q @ x[I]
      4) t = target_full[I]
    
    Args:
        x: [batch, dim] float tensor
        target_full: [dim] float tensor (seeded by block_hash elsewhere)
        block_hash/public_key/nonces: seeds
        k: picked dimension count
    
    Returns:
        y: [batch, k]
        t: [batch, k]
    """
    device = x.device
    dim = x.shape[1]
    indices = random_pick_indices(block_hash, public_key, nonces, dim, k, device)

    # Gather picked sub-vectors and target slices.
    xk = torch.gather(x, 1, indices)
    tk = target_full.to(device=device, dtype=x.dtype).unsqueeze(0).expand(x.shape[0], -1)
    tk = torch.gather(tk, 1, indices)

    # Rotate with per-nonce Haar Q.
    Q = generate_haar_orthogonal_matrices(
        block_hash, public_key, nonces, k, device, dtype=x.dtype
    )
    y = torch.bmm(Q, xk.unsqueeze(-1)).squeeze(-1)

    return y, tk


# -----------------------------------------------------------------------------
# Convenience wrappers (reuse production primitives in offline experiments)
# -----------------------------------------------------------------------------

def unit_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors to unit norm along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def apply_sign_flips_then_normalize(
    x: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    num_rounds: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match PoC semantics: sign flips first, then normalization."""
    if x.dim() != 2:
        raise ValueError(f"x must be [B,d], got shape={tuple(x.shape)}")
    if len(nonces) != x.shape[0]:
        raise ValueError(f"len(nonces) must match batch size, got {len(nonces)} vs {x.shape[0]}")
    rounds = int(num_rounds)
    if rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    signs = None
    for r in range(rounds):
        sr = generate_sign_flips(block_hash, public_key, nonces, x.shape[1], x.device, round_idx=r)
        signs = sr if signs is None else (signs * sr)
    x = x * signs.to(dtype=x.dtype)
    return unit_normalize(x, eps=eps)


def apply_householder_reflections(
    x_unit: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    num_reflections: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply per-nonce Householder reflections (full-d) and re-normalize."""
    if x_unit.dim() != 2:
        raise ValueError(f"x_unit must be [B,d], got shape={tuple(x_unit.shape)}")
    if len(nonces) != x_unit.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {x_unit.shape[0]}"
        )
    dim = x_unit.shape[1]
    tv = generate_nonce_transform_vectors(
        block_hash, public_key, nonces, dim, x_unit.device, num_reflections=num_reflections
    )  # [B,R,d]
    y = x_unit
    for r in range(tv.shape[1]):
        y = apply_householder(y, tv[:, r, :].to(dtype=y.dtype))
    return unit_normalize(y, eps=eps)


def slice_k_from_full(
    x_unit_full: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    k: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically pick k dims (per nonce) and return (xk_unit, indices)."""
    if x_unit_full.dim() != 2:
        raise ValueError(f"x_unit_full must be [B,d], got shape={tuple(x_unit_full.shape)}")
    if len(nonces) != x_unit_full.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {x_unit_full.shape[0]}"
        )
    dim = x_unit_full.shape[1]
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    idx = random_pick_indices(block_hash, public_key, nonces, dim, k, x_unit_full.device)
    xk = torch.gather(x_unit_full, 1, idx)
    xk = unit_normalize(xk, eps=eps)
    return xk, idx


def orthogonal_transform_k(
    x_unit_full: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    k: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match current PoC orthogonal mode: pick k dims per nonce and rotate by Haar Q[k,k].

    Returns:
        yk_unit: [B,k]
        indices: [B,k]
    """
    if x_unit_full.dim() != 2:
        raise ValueError(f"x_unit_full must be [B,d], got shape={tuple(x_unit_full.shape)}")
    if len(nonces) != x_unit_full.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {x_unit_full.shape[0]}"
        )
    dim = x_unit_full.shape[1]
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")
    target_full = generate_target(block_hash, public_key, dim, x_unit_full.device)
    yk, _tk = random_pick_orthogonal_transform(
        x_unit_full, target_full, block_hash, public_key, nonces, k
    )
    yk = unit_normalize(yk, eps=eps)
    idx = random_pick_indices(block_hash, public_key, nonces, dim, k, x_unit_full.device)
    return yk, idx


def haar_rotate_k(
    xk_unit: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rotate already-sliced k-vectors by per-nonce Haar Q[k,k] and re-normalize.

    Args:
        xk_unit: [B,k] float tensor (will be treated as unit vectors; re-normalized)
        block_hash/public_key/nonces: seeds for deterministic per-nonce Haar

    Returns:
        yk_unit: [B,k]
    """
    if xk_unit.dim() != 2:
        raise ValueError(f"xk_unit must be [B,k], got shape={tuple(xk_unit.shape)}")
    if len(nonces) != xk_unit.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {xk_unit.shape[0]}"
        )
    k = int(xk_unit.shape[1])
    Q = generate_haar_orthogonal_matrices(
        block_hash, public_key, nonces, k, xk_unit.device, dtype=xk_unit.dtype
    )  # [B,k,k]
    y = torch.bmm(Q, xk_unit.unsqueeze(-1)).squeeze(-1)
    return unit_normalize(y, eps=eps)


def fixed_project_full_to_k(
    x_unit_full: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project full-d unit vectors to k dims using a single fixed seeded Gaussian matrix.

    This is a *shared* (non-per-nonce) projection intended for offline analysis:
      A[d,k] ~ N(0,1) seeded by (block_hash, public_key, d, k)
      xk = normalize(x_full @ A)

    Args:
        x_unit_full: [B,d] float tensor (treated as unit vectors; normalized internally)
        block_hash/public_key: seed inputs for deterministic A
        k: output dimension

    Returns:
        xk_unit: [B,k]
    """
    if x_unit_full.dim() != 2:
        raise ValueError(f"x_unit_full must be [B,d], got shape={tuple(x_unit_full.shape)}")
    d = int(x_unit_full.shape[1])
    if k <= 0 or k > d:
        raise ValueError(f"k must be in [1, d], got k={k}, d={d}")

    x_unit_full = unit_normalize(x_unit_full, eps=eps)

    seed_str = f"{block_hash}_{public_key}_fixedproj_gaussian_raw_d{d}_k{k}"
    seed = _seed_from_string(seed_str)
    A = _normal(seed, d * int(k), x_unit_full.device).view(d, int(k)).to(dtype=x_unit_full.dtype)
    xk = x_unit_full @ A
    return unit_normalize(xk, eps=eps)
