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

