#!/usr/bin/env python3
"""
Experiment: Collect full distance distribution from PoC.

Runs generation with r_target=2.0 (accepts ALL) to measure actual distribution.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import requests
import numpy as np

SERVER_PORT = 8765
GENERATION_TIME = 20  # seconds per seed
SERVER_STARTUP_TIMEOUT = 120

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "llama": "unsloth/Llama-3.2-1B-Instruct",
}

SEEDS = ["seed_A", "seed_B", "seed_C"]


def start_server(model: str, model_key: str) -> subprocess.Popen:
    """Start vLLM server with PoC enabled."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--enable-poc",
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", "0.4",
            "--max-model-len", "256",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    
    print(f"  Waiting for server...")
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"  Server ready in {i+1}s")
                return proc
        except:
            pass
        time.sleep(1)
    
    proc.kill()
    raise RuntimeError("Server failed to start")


def stop_server(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()


def collect_distances(seed: str, duration: int) -> list:
    """Run generation and collect ALL distances."""
    # Start generation with r_target=2.0 (accepts all)
    config = {
        "block_hash": seed,
        "block_height": 100,
        "public_key": "experiment_node",
        "r_target": 2.0,  # Accept ALL nonces
        "node_id": 0,
        "node_count": 1,
        "batch_size": 32,
        "seq_len": 256,
    }
    
    r = requests.post(f"http://localhost:{SERVER_PORT}/api/v1/pow/init/generate", json=config)
    r.raise_for_status()
    
    time.sleep(duration)
    
    # Get status (all distances since r_target=2.0)
    r = requests.get(f"http://localhost:{SERVER_PORT}/api/v1/pow/status")
    status = r.json()
    
    # Stop
    requests.post(f"http://localhost:{SERVER_PORT}/api/v1/pow/stop")
    
    return status["valid_distances"]


def analyze_distribution(distances: list, model_name: str, theoretical_10pct: float):
    """Analyze and print distribution statistics."""
    d = np.array(distances)
    
    print(f"\n  Samples: {len(d)}")
    print(f"  Mean: {np.mean(d):.4f}")
    print(f"  Std:  {np.std(d):.4f}")
    print(f"  Min:  {np.min(d):.4f}")
    print(f"  Max:  {np.max(d):.4f}")
    
    percentiles = [10, 20, 30, 50, 70, 90]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(d, p)
        print(f"    {p:2d}th: {val:.4f}")
    
    p10 = np.percentile(d, 10)
    print(f"\n  Empirical r_target for 10% valid: {p10:.4f}")
    print(f"  Theoretical r_target for 10% valid: {theoretical_10pct:.4f}")
    print(f"  Deviation: {(p10 - theoretical_10pct) / theoretical_10pct * 100:.1f}%")
    
    return {
        "count": len(d),
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "p10": float(p10),
        "p50": float(np.percentile(d, 50)),
        "theoretical_p10": theoretical_10pct,
    }


def run_experiment(model_key: str, theoretical_10pct: float):
    """Run full experiment for one model."""
    model_name = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Experiment B: Empirical Distribution - {model_name}")
    print(f"{'='*60}")
    
    proc = start_server(model_name, model_key)
    
    try:
        all_distances = []
        seed_results = {}
        
        for seed in SEEDS:
            print(f"\n  Seed: {seed}")
            distances = collect_distances(seed, GENERATION_TIME)
            all_distances.extend(distances)
            
            print(f"  Collected {len(distances)} distances")
            if len(distances) > 100:
                d = np.array(distances)
                print(f"  Quick stats: mean={np.mean(d):.4f}, p10={np.percentile(d, 10):.4f}")
                seed_results[seed] = {
                    "count": len(distances),
                    "mean": float(np.mean(d)),
                    "p10": float(np.percentile(d, 10)),
                }
        
        print(f"\n  Combined analysis ({len(all_distances)} total samples):")
        combined = analyze_distribution(all_distances, model_name, theoretical_10pct)
        
        return {
            "model": model_name,
            "seeds": seed_results,
            "combined": combined,
        }
        
    finally:
        stop_server(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=["qwen"], help="Models to test")
    args = parser.parse_args()
    
    # Theoretical values (from Experiment A)
    theoretical = {
        "qwen": 1.4119,
        "llama": 1.4116,
    }
    
    results = {}
    for model_key in args.models:
        results[model_key] = run_experiment(model_key, theoretical[model_key])
    
    # Save results
    with open("logs/distribution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results saved to logs/distribution_results.json")


if __name__ == "__main__":
    main()

