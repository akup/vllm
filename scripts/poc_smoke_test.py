#!/usr/bin/env python3
"""
Smoke test for PoCManager with real vLLM model.

Usage:
    VLLM_USE_V1=0 python scripts/poc_smoke_test.py

This script:
1. Loads Qwen3-0.6B model using vLLM v0
2. Creates a PoCManager with model_executor (TP/PP aware)
3. Runs a few batches via collective_rpc
4. Validates nonces
5. Reports results
"""

import os
import sys

# Force v0 engine
os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM
from vllm.poc.manager import PoCManager
from vllm.poc.config import PoCConfig


def main():
    print("=" * 60)
    print("PoC Manager Smoke Test (collective_rpc mode)")
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
    print(f"   Device: {manager.device}")
    print(f"   State: {manager.state.value}")
    
    # Initialize round
    print("\n[3/5] Initializing round...")
    config = PoCConfig(
        block_hash="smoke_test_block_hash_12345",
        block_height=100,
        public_key="test_node_pubkey",
        r_target=1.5,  # Relaxed target for smoke test
        batch_size=4,
        seq_len=32,
        node_id=0,
        node_count=1,
    )
    manager.init_round(config)
    print(f"   Block hash: {config.block_hash}")
    print(f"   r_target: {config.r_target}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Seq len: {config.seq_len}")
    
    # Run batches
    print("\n[4/5] Running batches (via collective_rpc)...")
    manager.start_generate()
    
    num_batches = 3
    all_distances = []
    all_nonces = []
    
    for i in range(num_batches):
        batch = manager.run_batch()
        all_nonces.extend(batch.nonces)
        all_distances.extend(batch.dist)
        
        print(f"   Batch {i+1}:")
        print(f"      Nonces: {batch.nonces}")
        print(f"      Distances: {[f'{d:.4f}' for d in batch.dist]}")
        valid_count = sum(1 for d in batch.dist if d < config.r_target)
        print(f"      Valid (< {config.r_target}): {valid_count}/{len(batch)}")
    
    print(f"\n   Total checked: {manager.stats.total_checked}")
    print(f"   Total valid: {manager.stats.total_valid}")
    print(f"   Elapsed: {manager.stats.elapsed:.3f}s")
    print(f"   Rate: {manager.stats.rate:.1f} nonces/s")
    
    # Validate nonces
    print("\n[5/5] Validating nonces (via collective_rpc)...")
    manager.start_validate()
    
    # Pick first few nonces to validate
    nonces_to_validate = all_nonces[:4]
    original_distances = all_distances[:4]
    
    recomputed_distances, valid_flags = manager.validate(
        nonces_to_validate, 
        config.public_key
    )
    
    print(f"   Nonces: {nonces_to_validate}")
    print(f"   Original distances:   {[f'{d:.6f}' for d in original_distances]}")
    print(f"   Recomputed distances: {[f'{d:.6f}' for d in recomputed_distances]}")
    
    # Check determinism
    all_match = True
    for i, (orig, recomp) in enumerate(zip(original_distances, recomputed_distances)):
        diff = abs(orig - recomp)
        match = diff < 1e-5
        all_match = all_match and match
        status = "OK" if match else "MISMATCH"
        print(f"   Nonce {nonces_to_validate[i]}: diff={diff:.2e} [{status}]")
    
    # Test different public key produces different distances
    print("\n   Testing different public key...")
    other_distances, _ = manager.validate(nonces_to_validate, "different_pubkey")
    distances_differ = original_distances != other_distances
    print(f"   Different public key -> different distances: {distances_differ}")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    checks = [
        ("Distances in valid range [0, 2]", all(0 <= d <= 2 for d in all_distances)),
        ("Deterministic (recompute matches)", all_match),
        ("Different pubkey -> different distances", distances_differ),
        ("Stats tracking works", manager.stats.total_checked == num_batches * config.batch_size),
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
