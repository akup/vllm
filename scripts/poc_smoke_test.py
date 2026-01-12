#!/usr/bin/env python3
"""
Smoke test for PoCManager with real vLLM model (artifact-based protocol).

Usage:
    VLLM_USE_V1=0 python scripts/poc_smoke_test.py

This script:
1. Loads Qwen3-0.6B model using vLLM v0
2. Creates a PoCManager with model_executor (TP/PP aware)
3. Runs generate_artifacts for specific nonces via collective_rpc
4. Verifies artifact generation and determinism
5. Reports results

Note: PoCManager is now stateless - all state management (nonce counter, stats)
is done in the API layer. This test focuses on the generate_artifacts function.
"""

import os
import sys

# Force v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM
from vllm.poc.manager import PoCManager
from vllm.poc.data import decode_vector
import numpy as np


def main():
    print("=" * 60)
    print("PoC Manager Smoke Test (artifact-based protocol)")
    print("=" * 60)
    
    # Load model
    print("\n[1/4] Loading model...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=256,
    )
    
    # Access model_executor, config, and vllm_config (v0 API)
    model_executor = llm.llm_engine.model_executor
    model_config = llm.llm_engine.model_config
    vllm_config = llm.llm_engine.vllm_config
    
    print(f"   Model: {model_config.model}")
    print(f"   Vocab size: {model_config.get_vocab_size()}")
    print(f"   Hidden size: {model_config.get_hidden_size()}")
    print(f"   Executor type: {type(model_executor).__name__}")
    
    # Create manager (now stateless)
    print("\n[2/4] Creating PoCManager (stateless)...")
    manager = PoCManager(model_executor, model_config, vllm_config)
    print(f"   Manager ready")
    
    # Test params
    block_hash = "smoke_test_block_hash_12345"
    public_key = "test_node_pubkey"
    seq_len = 32
    k_dim = 12
    
    # Run generate_artifacts
    print("\n[3/4] Running generate_artifacts (via collective_rpc)...")
    
    # Batch 1: nonces [0, 1, 2, 3]
    nonces1 = [0, 1, 2, 3]
    artifacts1 = manager.generate_artifacts(
        nonces=nonces1,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
    )
    print(f"   Batch 1: nonces={nonces1}, artifacts={len(artifacts1)}")
    
    # Batch 2: nonces [4, 5, 6, 7]
    nonces2 = [4, 5, 6, 7]
    artifacts2 = manager.generate_artifacts(
        nonces=nonces2,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
    )
    print(f"   Batch 2: nonces={nonces2}, artifacts={len(artifacts2)}")
    
    # Batch 3: nonces [8, 9, 10, 11]
    nonces3 = [8, 9, 10, 11]
    artifacts3 = manager.generate_artifacts(
        nonces=nonces3,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
    )
    print(f"   Batch 3: nonces={nonces3}, artifacts={len(artifacts3)}")
    
    all_artifacts = artifacts1 + artifacts2 + artifacts3
    
    # Decode and show first vector
    if all_artifacts:
        first_vec = decode_vector(all_artifacts[0].vector_b64)
        print(f"   First vector shape: {first_vec.shape}")
        print(f"   First vector norm: {np.linalg.norm(first_vec):.4f}")
    
    # Test determinism by regenerating same nonces
    print("\n[4/4] Testing determinism...")
    
    nonces_to_test = nonces1
    original_artifacts = artifacts1
    
    # Generate again for same nonces
    recomputed = manager.generate_artifacts(
        nonces=nonces_to_test,
        block_hash=block_hash,
        public_key=public_key,
        seq_len=seq_len,
        k_dim=k_dim,
    )
    
    print(f"   Testing {len(nonces_to_test)} nonces for determinism...")
    
    all_match = True
    for orig, recomp in zip(original_artifacts, recomputed):
        orig_vec = decode_vector(orig.vector_b64)
        recomp_vec = decode_vector(recomp.vector_b64)
        diff = np.linalg.norm(orig_vec - recomp_vec)
        match = diff < 1e-5
        all_match = all_match and match
        status = "OK" if match else "MISMATCH"
        print(f"   Nonce {orig.nonce}: diff={diff:.2e} [{status}]")
    
    # Test different public key produces different vectors
    print("\n   Testing different public key...")
    other_artifacts = manager.generate_artifacts(
        nonces=nonces_to_test,
        block_hash=block_hash,
        public_key="different_pubkey",
        seq_len=seq_len,
        k_dim=k_dim,
    )
    
    vectors_differ = False
    for orig, other in zip(original_artifacts, other_artifacts):
        orig_vec = decode_vector(orig.vector_b64)
        other_vec = decode_vector(other.vector_b64)
        diff = np.linalg.norm(orig_vec - other_vec)
        if diff > 0.01:  # Vectors should differ significantly
            vectors_differ = True
            break
    
    print(f"   Different public key -> different vectors: {vectors_differ}")
    
    # Verify vector properties
    all_vectors_valid = True
    for artifact in all_artifacts[:4]:
        vec = decode_vector(artifact.vector_b64)
        norm = np.linalg.norm(vec)
        if vec.shape[0] != k_dim:
            all_vectors_valid = False
        if norm < 0.5 or norm > 1.5:  # Normalized vectors should be ~1.0
            all_vectors_valid = False
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    checks = [
        ("Artifacts generated", len(all_artifacts) == 12),
        ("Vectors have correct k_dim", all_vectors_valid),
        ("Deterministic (recompute matches)", all_match),
        ("Different pubkey -> different vectors", vectors_differ),
        ("collective_rpc execution works", len(all_artifacts) == 12),
    ]
    
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED!")
    else:
        print("SOME CHECKS FAILED!")
        sys.exit(1)
    print("=" * 60)
    
    # Cleanup
    del llm


if __name__ == "__main__":
    main()
