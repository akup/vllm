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

    assert torch.allclose(inputs_cpu, inputs_gpu.cpu())


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

    target_cpu = generate_target(BLOCK_HASH, vocab_size=1000, device=cpu)
    target_gpu = generate_target(BLOCK_HASH, vocab_size=1000, device=gpu)

    assert torch.allclose(target_cpu, target_gpu.cpu())
