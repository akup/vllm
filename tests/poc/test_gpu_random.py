import torch
import pytest

from vllm.poc.gpu_random import (
    generate_inputs,
    generate_target,
    generate_householder_vector,
    apply_householder,
    random_pick_indices,
    generate_haar_orthogonal_matrices,
)

BLOCK_HASH = "test_block_hash_12345"
PUBLIC_KEY = "test_public_key"


# === Input Generation Tests ===

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


def test_different_block_hash_produces_different_inputs():
    """Different block_hash produces different input tensors"""
    device = torch.device("cuda:0")
    inputs1 = generate_inputs("hash1", PUBLIC_KEY, [0], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs("hash2", PUBLIC_KEY, [0], dim=128, seq_len=16, device=device)
    assert not torch.allclose(inputs1, inputs2)


def test_different_public_key_produces_different_inputs():
    """Different public_key produces different input tensors"""
    device = torch.device("cuda:0")
    inputs1 = generate_inputs(BLOCK_HASH, "node1", [0], dim=128, seq_len=16, device=device)
    inputs2 = generate_inputs(BLOCK_HASH, "node2", [0], dim=128, seq_len=16, device=device)
    assert not torch.allclose(inputs1, inputs2)


def test_cpu_gpu_inputs_match():
    """CPU and GPU produce identical inputs (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    inputs_cpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=cpu)
    inputs_gpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=gpu)

    # Allow small tolerance for float32->float16 conversion differences between CPU/GPU
    assert torch.allclose(inputs_cpu, inputs_gpu.cpu(), rtol=1e-3, atol=1e-3)


# === Target Generation Tests ===

def test_target_unit_vector():
    device = torch.device("cuda:0")
    target = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=device)

    assert abs(target.norm().item() - 1.0) < 1e-5


def test_different_block_hash():
    device = torch.device("cuda:0")

    target1 = generate_target("hash1", PUBLIC_KEY, dim=1000, device=device)
    target2 = generate_target("hash2", PUBLIC_KEY, dim=1000, device=device)

    assert not torch.allclose(target1, target2)


def test_cpu_gpu_target_match():
    """CPU and GPU produce identical target vectors (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    target_cpu = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=cpu)
    target_gpu = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=gpu)

    assert torch.allclose(target_cpu, target_gpu.cpu())


# === Householder Transform Tests ===

def test_householder_vector_is_unit():
    """Householder reflection vector should be unit vector"""
    device = torch.device("cuda:0")
    v = generate_householder_vector("test_seed", dim=1024, device=device)
    assert abs(v.norm().item() - 1.0) < 1e-5


def test_householder_vector_determinism():
    """Same seed produces same Householder vector"""
    device = torch.device("cuda:0")
    v1 = generate_householder_vector("test_seed", dim=1024, device=device)
    v2 = generate_householder_vector("test_seed", dim=1024, device=device)
    assert torch.allclose(v1, v2)


def test_householder_vector_different_seeds():
    """Different seeds produce different Householder vectors"""
    device = torch.device("cuda:0")
    v1 = generate_householder_vector("seed1", dim=1024, device=device)
    v2 = generate_householder_vector("seed2", dim=1024, device=device)
    assert not torch.allclose(v1, v2)


def test_apply_householder_preserves_norm():
    """Householder reflection preserves vector norm (orthogonal transform)"""
    device = torch.device("cuda:0")
    x = torch.randn(10, 1024, device=device)
    v = generate_householder_vector("test", dim=1024, device=device)
    
    x_transformed = apply_householder(x, v)
    
    # Norms should be preserved
    norms_before = x.norm(dim=1)
    norms_after = x_transformed.norm(dim=1)
    assert torch.allclose(norms_before, norms_after, rtol=1e-4)


def test_apply_householder_is_involutory():
    """Householder reflection applied twice returns original (H @ H @ x = x)"""
    device = torch.device("cuda:0")
    x = torch.randn(5, 1024, device=device)
    v = generate_householder_vector("test", dim=1024, device=device)
    
    x_once = apply_householder(x, v)
    x_twice = apply_householder(x_once, v)
    
    # Numerical precision can accumulate, so use larger tolerance
    assert torch.allclose(x, x_twice, rtol=1e-3, atol=1e-5)


def test_cpu_gpu_householder_match():
    """CPU and GPU produce identical Householder vectors"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    v_cpu = generate_householder_vector("test", dim=1024, device=cpu)
    v_gpu = generate_householder_vector("test", dim=1024, device=gpu)
    
    assert torch.allclose(v_cpu, v_gpu.cpu())


# === Random Pick Indices Tests ===

def test_random_pick_indices_determinism():
    """Same inputs produce same indices"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    idx1 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=64, device=device)
    idx2 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=64, device=device)
    
    assert torch.equal(idx1, idx2)


def test_random_pick_indices_shape():
    """random_pick_indices returns correct shape"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2, 3, 4]
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, nonces, dim=1000, k=k, device=device)
    
    assert indices.shape == (len(nonces), k)
    assert indices.dtype == torch.int64


def test_random_pick_indices_range():
    """Indices are within valid range [0, dim)"""
    device = torch.device("cuda:0")
    dim = 1000
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=dim, k=k, device=device)
    
    assert (indices >= 0).all()
    assert (indices < dim).all()


def test_random_pick_indices_uniqueness():
    """Each nonce's indices are unique (no replacement)"""
    device = torch.device("cuda:0")
    k = 64
    
    indices = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=k, device=device)
    
    # Check all indices are unique for this nonce
    unique_count = len(torch.unique(indices[0]))
    assert unique_count == k


def test_random_pick_indices_different_nonces():
    """Different nonces produce different indices"""
    device = torch.device("cuda:0")
    
    idx1 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=64, device=device)
    idx2 = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [1], dim=1000, k=64, device=device)
    
    # Different nonces should produce different index sets (with high probability)
    assert not torch.equal(idx1, idx2)


def test_random_pick_indices_invalid_k():
    """Invalid k values raise ValueError"""
    device = torch.device("cuda:0")
    
    with pytest.raises(ValueError):
        random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=0, device=device)
    
    with pytest.raises(ValueError):
        random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0], dim=1000, k=1001, device=device)


def test_cpu_gpu_random_pick_indices_match():
    """CPU and GPU produce same set of indices (order may differ due to topk impl)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    idx_cpu = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=1000, k=64, device=cpu)
    idx_gpu = random_pick_indices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=1000, k=64, device=gpu)
    
    # Compare sorted indices since topk order is implementation-defined
    for i in range(idx_cpu.shape[0]):
        cpu_sorted = torch.sort(idx_cpu[i]).values
        gpu_sorted = torch.sort(idx_gpu[i].cpu()).values
        assert torch.equal(cpu_sorted, gpu_sorted)


# === Haar Orthogonal Matrices Tests ===

def test_haar_orthogonal_matrices_determinism():
    """Same inputs produce same matrices"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    Q1 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=64, device=device)
    Q2 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=64, device=device)
    
    assert torch.allclose(Q1, Q2)


def test_haar_orthogonal_matrices_shape():
    """generate_haar_orthogonal_matrices returns correct shape"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2, 3, 4]
    k = 64
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, nonces, k=k, device=device)
    
    assert Q.shape == (len(nonces), k, k)


def test_haar_orthogonal_matrices_orthogonality():
    """Generated matrices are orthogonal (Q @ Q^T = I)"""
    device = torch.device("cuda:0")
    k = 64
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], k=k, device=device)
    
    for i in range(Q.shape[0]):
        QQt = Q[i] @ Q[i].T
        identity = torch.eye(k, device=device, dtype=Q.dtype)
        assert torch.allclose(QQt, identity, atol=1e-5)


def test_haar_orthogonal_matrices_determinant():
    """Orthogonal matrices have determinant +/- 1"""
    device = torch.device("cuda:0")
    k = 32  # Smaller k for faster determinant computation
    
    Q = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], k=k, device=device)
    
    for i in range(Q.shape[0]):
        det = torch.linalg.det(Q[i])
        assert abs(abs(det.item()) - 1.0) < 1e-4


def test_haar_orthogonal_matrices_different_nonces():
    """Different nonces produce different matrices"""
    device = torch.device("cuda:0")
    
    Q1 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0], k=64, device=device)
    Q2 = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [1], k=64, device=device)
    
    assert not torch.allclose(Q1, Q2)


def test_haar_orthogonal_matrices_invalid_k():
    """Invalid k values raise ValueError"""
    device = torch.device("cuda:0")
    
    with pytest.raises(ValueError):
        generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0], k=0, device=device)


def test_cpu_gpu_haar_matrices_match():
    """CPU and GPU produce identical Haar matrices (small numerical tolerance)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    Q_cpu = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1], k=32, device=cpu)
    Q_gpu = generate_haar_orthogonal_matrices(BLOCK_HASH, PUBLIC_KEY, [0, 1], k=32, device=gpu)
    
    # QR decomposition may have small numerical differences across devices
    assert torch.allclose(Q_cpu, Q_gpu.cpu(), rtol=1e-4, atol=1e-5)
