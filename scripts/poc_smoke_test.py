#!/usr/bin/env python3
"""
Smoke test for PoCManager with real vLLM model (artifact-based protocol).

Usage:
    VLLM_USE_V1=0 python scripts/poc_smoke_test.py

This script:
1. Loads Qwen3-0.6B model using vLLM v0
2. Creates a PoCManager with model_executor (TP/PP aware)
3. Runs a few batches via collective_rpc
4. Verifies artifact generation
5. Reports results
"""

import os
import sys

# Force v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM
from vllm.poc.manager import PoCManager
from vllm.poc.config import PoCConfig
from vllm.poc.data import decode_vector
import numpy as np


def main():
    print("=" * 60)
    print("PoC Manager Smoke Test (artifact-based protocol)")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading model...")
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
    
    # Create manager with model_executor (handles TP/PP via collective_rpc)
    print("\n[2/5] Creating PoCManager...")
    manager = PoCManager(model_executor, model_config, vllm_config)
    print(f"   State: {manager.state.value}")
    
    # Initialize round
    print("\n[3/5] Initializing round...")
    config = PoCConfig(
        block_hash="smoke_test_block_hash_12345",
        block_height=100,
        public_key="test_node_pubkey",
        batch_size=4,
        seq_len=32,
        k_dim=12,
        node_id=0,
        node_count=1,
    )
    manager.init_round(config)
    print(f"   Block hash: {config.block_hash}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Seq len: {config.seq_len}")
    print(f"   k_dim: {config.k_dim}")
    
    # Run batches
    print("\n[4/5] Running batches (via collective_rpc)...")
    manager.start_generate()
    
    num_batches = 3
    all_artifacts = []
    all_nonces = []
    
    for i in range(num_batches):
        result = manager.run_batch()
        artifacts = result.get("artifacts", [])
        nonces = result.get("nonces", [])
        
        all_nonces.extend(nonces)
        all_artifacts.extend(artifacts)
        
        print(f"   Batch {i+1}:")
        print(f"      Nonces: {nonces}")
        print(f"      Artifacts: {len(artifacts)} vectors")
        
        # Decode and show first vector
        if artifacts:
            first_vec = decode_vector(artifacts[0].vector_b64)
            print(f"      First vector shape: {first_vec.shape}")
            print(f"      First vector norm: {np.linalg.norm(first_vec):.4f}")
    
    print(f"\n   Total processed: {manager.stats.total_processed}")
    print(f"   Elapsed: {manager.stats.elapsed:.3f}s")
    print(f"   Rate: {manager.stats.nonces_per_second:.1f} nonces/s")
    
    # Test determinism by regenerating same nonces
    print("\n[5/5] Testing determinism...")
    
    nonces_to_test = all_nonces[:4]
    original_artifacts = all_artifacts[:4]
    
    # Generate again for same nonces
    recomputed = manager.generate_artifacts(
        nonces=nonces_to_test,
        block_hash=config.block_hash,
        public_key=config.public_key,
        seq_len=config.seq_len,
        k_dim=config.k_dim,
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
        block_hash=config.block_hash,
        public_key="different_pubkey",
        seq_len=config.seq_len,
        k_dim=config.k_dim,
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
        if vec.shape[0] != config.k_dim:
            all_vectors_valid = False
        if norm < 0.5 or norm > 1.5:  # Normalized vectors should be ~1.0
            all_vectors_valid = False
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    checks = [
        ("Artifacts generated", len(all_artifacts) == num_batches * config.batch_size),
        ("Vectors have correct k_dim", all_vectors_valid),
        ("Deterministic (recompute matches)", all_match),
        ("Different pubkey -> different vectors", vectors_differ),
        ("Stats tracking works", manager.stats.total_processed == num_batches * config.batch_size + len(nonces_to_test) * 2),
        ("collective_rpc execution works", len(all_nonces) == num_batches * config.batch_size),
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
