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


def generate_orthogonal_matrix(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a random orthogonal matrix using QR decomposition.
    
    Uses a random Gaussian matrix and extracts the Q factor.
    This produces a Haar-distributed random orthogonal matrix.
    
    Args:
        seed_str: Seed string for deterministic generation
        dim: Matrix dimension (produces dim x dim matrix)
        device: Target device
    
    Returns:
        Orthogonal matrix of shape [dim, dim]
    """
    seed = _seed_from_string(seed_str)
    # Generate random Gaussian matrix
    # Need dim*dim random numbers
    gauss = _normal(seed, dim * dim, device).view(dim, dim)
    # QR decomposition - Q is orthogonal
    Q, R = torch.linalg.qr(gauss)
    # Ensure determinant is +1 (proper rotation, not reflection)
    # by flipping sign of columns where diagonal of R is negative
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q


def apply_orthogonal_transform(
    x: torch.Tensor,
    Q: torch.Tensor,
) -> torch.Tensor:
    """Apply orthogonal transformation: y = x @ Q.T
    
    Args:
        x: Input tensor of shape [..., dim]
        Q: Orthogonal matrix of shape [dim, dim]
    
    Returns:
        Transformed tensor of same shape as x
    """
    return x @ Q.T
