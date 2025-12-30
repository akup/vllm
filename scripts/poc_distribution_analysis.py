#!/usr/bin/env python3
"""
Distribution analysis: Verify hidden state Householder transform breaks structure.

Uses PoCManager (which now uses hidden state transform) to collect distance samples.
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import numpy as np
from vllm import LLM
from vllm.poc.config import PoCConfig
from vllm.poc.manager import PoCManager


def main():
    print("=" * 70)
    print("Distribution Analysis: Hidden State Householder Transform")
    print("=" * 70)
    print("Theoretical: mean=1.4142, p10=1.4119")
    print()
    
    # Load model
    print("Loading Qwen/Qwen3-0.6B model...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.4,
        max_model_len=256,
        enforce_eager=True,
    )
    
    model_executor = llm.llm_engine.model_executor
    model_config = llm.llm_engine.model_config
    vllm_config = llm.llm_engine.vllm_config
    
    print(f"Hidden size: {model_config.get_hidden_size()}")
    print(f"Vocab size: {model_config.get_vocab_size()}")
    print()
    
    # Create manager
    manager = PoCManager(model_executor, model_config, vllm_config)
    
    # Collect samples from multiple seeds
    all_distances = []
    seeds = ["seed_A", "seed_B", "seed_C"]
    num_batches = 50
    batch_size = 32
    
    for seed in seeds:
        print(f"Seed: {seed}")
        manager.stop_round()
        
        config = PoCConfig(
            block_hash=seed,
            block_height=1,
            public_key="test_node",
            r_target=2.0,  # High threshold to collect all
            node_id=0,
            node_count=1,
            batch_size=batch_size,
            seq_len=256,
        )
        manager.init_round(config)
        manager.start_generate()
        
        seed_distances = []
        for _ in range(num_batches):
            batch = manager.run_batch()
            seed_distances.extend(batch.dist)
        
        d = np.array(seed_distances)
        print(f"  Samples: {len(d)}, mean={np.mean(d):.4f}, p10={np.percentile(d, 10):.4f}")
        all_distances.extend(seed_distances)
    
    manager.stop_round()
    
    # Analyze combined results
    d = np.array(all_distances)
    print()
    print("=" * 70)
    print(f"Combined Results ({len(d)} samples):")
    print("=" * 70)
    print(f"  Mean: {np.mean(d):.4f} (theoretical: 1.4142)")
    print(f"  Std:  {np.std(d):.4f}")
    print(f"  Min:  {np.min(d):.4f}")
    print(f"  Max:  {np.max(d):.4f}")
    print()
    print("  Percentiles:")
    for p in [10, 20, 30, 50, 70, 90]:
        print(f"    {p:2d}th: {np.percentile(d, p):.4f}")
    
    print()
    mean_deviation = (np.mean(d) - 1.4142) / 1.4142 * 100
    p10_deviation = (np.percentile(d, 10) - 1.4119) / 1.4119 * 100
    print(f"  Mean deviation from theoretical: {mean_deviation:+.1f}%")
    print(f"  p10 deviation from theoretical: {p10_deviation:+.1f}%")
    
    print()
    print("=" * 70)
    print("COMPARISON WITH PREVIOUS APPROACHES:")
    print("=" * 70)
    print(f"{'Approach':<35} {'Mean':<10} {'p10':<10} {'Mean Dev':<12} {'p10 Dev':<12}")
    print("-" * 70)
    print(f"{'Theoretical':<35} {'1.4142':<10} {'1.4119':<10} {'-':<12} {'-':<12}")
    print(f"{'Old (perm only, from experiment)':<35} {'1.2153':<10} {'1.1726':<10} {'-14.1%':<12} {'-16.9%':<12}")
    print(f"{'Layer+logit combined (from exp)':<35} {'1.4405':<10} {'1.1734':<10} {'+1.9%':<12} {'-16.9%':<12}")
    print(f"{'NEW (hidden Householder)':<35} {np.mean(d):<10.4f} {np.percentile(d, 10):<10.4f} {mean_deviation:+.1f}%        {p10_deviation:+.1f}%")


if __name__ == "__main__":
    main()

