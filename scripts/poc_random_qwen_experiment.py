#!/usr/bin/env python3
"""
Experiment C: Random Qwen Model Comparison.

Loads actual Qwen3 model architecture from vLLM, then randomizes all weights.
This tests the hypothesis that trained weights create structure.
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import torch
import numpy as np

from vllm import LLM
from vllm.poc.config import PoCConfig, PoCState
from vllm.poc.manager import PoCManager


def randomize_weights(model: torch.nn.Module, seed: int = 42):
    """Randomize all weights in the model in-place."""
    torch.manual_seed(seed)
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad or True:  # Randomize all params
            # Use Kaiming init for weights, zeros for biases
            if 'bias' in name:
                torch.nn.init.zeros_(param.data)
            elif len(param.shape) >= 2:
                torch.nn.init.kaiming_normal_(param.data, mode='fan_out')
            else:
                torch.nn.init.normal_(param.data, std=0.02)
            count += 1
    return count


def main():
    print("=" * 60)
    print("Experiment C: Random Qwen Model (Actual Architecture)")
    print("=" * 60)
    
    # Load Qwen model
    print("\nLoading Qwen/Qwen3-0.6B model...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.4,
        max_model_len=256,
        enforce_eager=True,  # Disable CUDA graphs for this test
    )
    
    # Access the model
    model_executor = llm.llm_engine.model_executor
    model_config = llm.llm_engine.model_config
    vllm_config = llm.llm_engine.vllm_config
    
    # Get the actual model from driver worker
    model = model_executor.driver_worker.model_runner.model
    
    print(f"Model hidden_size: {model_config.get_hidden_size()}")
    print(f"Model vocab_size: {model_config.get_vocab_size()}")
    
    # Randomize weights
    print("\nRandomizing all model weights...")
    num_params = randomize_weights(model)
    print(f"Randomized {num_params} parameter tensors")
    
    # Create PoCManager
    manager = PoCManager(model_executor, model_config, vllm_config)
    
    # Collect distances
    all_distances = []
    seeds = ["seed_A", "seed_B", "seed_C"]
    num_batches = 30
    
    for seed in seeds:
        print(f"\nSeed: {seed}")
        # Stop previous round if any
        manager.stop_round()
        
        # Reinit with new seed
        config_seed = PoCConfig(
            block_hash=seed,
            block_height=1,
            public_key="test_node",
            r_target=2.0,
            node_id=0,
            node_count=1,
            batch_size=32,
            seq_len=256,
        )
        manager.init_round(config_seed)
        manager.start_generate()
        
        seed_distances = []
        for _ in range(num_batches):
            batch = manager.run_batch()
            seed_distances.extend(batch.dist)
        
        d = np.array(seed_distances)
        print(f"  Samples: {len(d)}, mean={np.mean(d):.4f}, p10={np.percentile(d, 10):.4f}")
        all_distances.extend(seed_distances)
    
    manager.stop_round()
    
    # Analyze combined
    d = np.array(all_distances)
    print(f"\n{'='*60}")
    print(f"Combined Results ({len(d)} samples):")
    print(f"{'='*60}")
    print(f"  Mean: {np.mean(d):.4f} (theoretical: 1.4142)")
    print(f"  Std:  {np.std(d):.4f}")
    print(f"  Min:  {np.min(d):.4f}")
    print(f"  Max:  {np.max(d):.4f}")
    print()
    print("  Percentiles:")
    for p in [10, 20, 30, 50, 70, 90]:
        print(f"    {p:2d}th: {np.percentile(d, p):.4f}")
    
    p10 = np.percentile(d, 10)
    theoretical_p10 = 1.4119
    print(f"\n  Empirical r_target for 10%: {p10:.4f}")
    print(f"  Theoretical r_target for 10%: {theoretical_p10:.4f}")
    print(f"  Deviation: {(p10 - theoretical_p10) / theoretical_p10 * 100:.1f}%")
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY:")
    print(f"{'='*60}")
    print(f"  Theoretical random:     mean=1.4142, p10=1.4119")
    print(f"  Random Qwen (this):     mean={np.mean(d):.4f}, p10={p10:.4f}")
    print(f"  Trained Qwen:           mean=1.2155, p10=1.1739")
    print(f"  Trained Llama:          mean=1.4486, p10=1.3796")
    print()
    
    deviation = abs(np.mean(d) - 1.4142) / 1.4142 * 100
    if deviation < 5:
        print(f"  CONCLUSION: Random Qwen matches theoretical! (deviation: {deviation:.1f}%)")
        print("  -> Trained Qwen weights create STRUCTURE that compresses distances.")
    else:
        print(f"  NOTE: Random Qwen deviates by {deviation:.1f}% from theoretical")


if __name__ == "__main__":
    main()

