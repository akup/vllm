import torch
import pytest

from vllm.poc.gpu_random import (
    generate_inputs,
    generate_permutations,
    generate_target,
    compute_distances,
    generate_householder_vector,
    apply_householder,
    generate_nonce_transform_vectors,
    compute_distances_direct,
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
    target = generate_target(BLOCK_HASH, PUBLIC_KEY, dim=1000, device=device)

    assert abs(target.norm().item() - 1.0) < 1e-5


def test_distance_range():
    device = torch.device("cuda:0")
    batch_size = 10
    vocab_size = 1000

    logits = torch.randn(batch_size, vocab_size, device=device)
    perms = generate_permutations(BLOCK_HASH, PUBLIC_KEY, list(range(batch_size)), vocab_size, device)
    target = generate_target(BLOCK_HASH, PUBLIC_KEY, vocab_size, device)

    distances = compute_distances(logits, perms, target)

    assert (distances >= 0).all()
    assert (distances <= 2).all()


def test_different_block_hash():
    device = torch.device("cuda:0")

    target1 = generate_target("hash1", PUBLIC_KEY, dim=1000, device=device)
    target2 = generate_target("hash2", PUBLIC_KEY, dim=1000, device=device)

    assert not torch.allclose(target1, target2)


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


def test_permutation_is_valid():
    """Each permutation contains all indices 0 to vocab_size-1"""
    device = torch.device("cuda:0")
    vocab_size = 1000
    perms = generate_permutations(BLOCK_HASH, PUBLIC_KEY, [0], vocab_size, device)
    sorted_perm = torch.sort(perms[0]).values
    expected = torch.arange(vocab_size, device=device)
    assert torch.equal(sorted_perm, expected)


def test_different_public_key_produces_different_permutations():
    """Different public_key produces different permutations"""
    device = torch.device("cuda:0")
    perm1 = generate_permutations(BLOCK_HASH, "node1", [0], vocab_size=1000, device=device)
    perm2 = generate_permutations(BLOCK_HASH, "node2", [0], vocab_size=1000, device=device)
    assert not torch.equal(perm1, perm2)


def test_cpu_gpu_inputs_match():
    """CPU and GPU produce identical inputs (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    inputs_cpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=cpu)
    inputs_gpu = generate_inputs(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], dim=128, seq_len=16, device=gpu)

    # Allow small tolerance for float32->float16 conversion differences between CPU/GPU
    assert torch.allclose(inputs_cpu, inputs_gpu.cpu(), rtol=1e-3, atol=1e-3)


def test_cpu_gpu_permutations_match():
    """CPU and GPU produce identical permutations (cross-device reproducibility)"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    perm_cpu = generate_permutations(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], vocab_size=1000, device=cpu)
    perm_gpu = generate_permutations(BLOCK_HASH, PUBLIC_KEY, [0, 1, 2], vocab_size=1000, device=gpu)

    assert torch.equal(perm_cpu, perm_gpu.cpu())


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


def test_nonce_transform_vectors_shape():
    """generate_nonce_transform_vectors returns correct shape"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2, 3, 4]
    num_reflections = 4
    dim = 1024
    
    vectors = generate_nonce_transform_vectors(
        BLOCK_HASH, PUBLIC_KEY, nonces, dim, device, num_reflections
    )
    
    assert vectors.shape == (len(nonces), num_reflections, dim)


def test_nonce_transform_vectors_determinism():
    """Same inputs produce same transform vectors"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    v1 = generate_nonce_transform_vectors(BLOCK_HASH, PUBLIC_KEY, nonces, 1024, device, 4)
    v2 = generate_nonce_transform_vectors(BLOCK_HASH, PUBLIC_KEY, nonces, 1024, device, 4)
    
    assert torch.allclose(v1, v2)


def test_nonce_transform_vectors_different_nonces():
    """Different nonces produce different transform vectors"""
    device = torch.device("cuda:0")
    
    v1 = generate_nonce_transform_vectors(BLOCK_HASH, PUBLIC_KEY, [0], 1024, device, 4)
    v2 = generate_nonce_transform_vectors(BLOCK_HASH, PUBLIC_KEY, [1], 1024, device, 4)
    
    assert not torch.allclose(v1, v2)


def test_nonce_transform_vectors_are_unit():
    """All generated transform vectors should be unit vectors"""
    device = torch.device("cuda:0")
    nonces = [0, 1, 2]
    
    vectors = generate_nonce_transform_vectors(BLOCK_HASH, PUBLIC_KEY, nonces, 1024, device, 4)
    
    # Check each vector is unit
    for i in range(len(nonces)):
        for r in range(4):
            norm = vectors[i, r].norm().item()
            assert abs(norm - 1.0) < 1e-5


def test_compute_distances_direct_range():
    """compute_distances_direct returns distances in valid range [0, 2]"""
    device = torch.device("cuda:0")
    batch_size = 10
    vocab_size = 1000
    
    logits = torch.randn(batch_size, vocab_size, device=device)
    target = generate_target(BLOCK_HASH, PUBLIC_KEY, vocab_size, device)
    
    distances = compute_distances_direct(logits, target)
    
    assert (distances >= 0).all()
    assert (distances <= 2).all()


def test_cpu_gpu_householder_match():
    """CPU and GPU produce identical Householder vectors"""
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    
    v_cpu = generate_householder_vector("test", dim=1024, device=cpu)
    v_gpu = generate_householder_vector("test", dim=1024, device=gpu)
    
    assert torch.allclose(v_cpu, v_gpu.cpu())
