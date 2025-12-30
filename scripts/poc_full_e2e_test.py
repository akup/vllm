#!/usr/bin/env python3
"""
Full E2E Distribution Test for PoC with vLLM Server.

Runs 18 isolated tests (3 blocks x 3 public_keys x 2 models).
Each test runs with completely fresh server + callback instances.
All logs saved to separate files per experiment.

Usage:
    python scripts/poc_full_e2e_test.py
    python scripts/poc_full_e2e_test.py --models qwen      # Only Qwen
    python scripts/poc_full_e2e_test.py --models llama     # Only Llama
    python scripts/poc_full_e2e_test.py --duration 60      # 60s per test
    python scripts/poc_full_e2e_test.py --no-layer-hooks   # Disable layer hooks
    python scripts/poc_full_e2e_test.py --sign-flips       # Enable sign flips transform

Monitor progress:
    tail -f logs/e2e_results.jsonl
    ls -lt logs/*.log | head -4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    ("Qwen/Qwen3-0.6B", "qwen"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama"),
]

BLOCK_HASHES = ["block_alpha", "block_beta", "block_gamma"]
PUBLIC_KEYS = ["node_A", "node_B", "node_C"]

R_TARGET = 1.4160  # Above max p10 across block_hashes - expects 5-35% valid rate depending on block
TEST_DURATION = 80  # seconds per test

CALLBACK_PORT = 8081
SERVER_PORT = 8765
SERVER_STARTUP_TIMEOUT = 120  # seconds

# ============================================================================
# Process Management
# ============================================================================

def start_callback_receiver(log_file: Path) -> subprocess.Popen:
    """Start callback receiver in background with log file."""
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable, "-u",
            "scripts/poc_callback_receiver.py",
            "--port", str(CALLBACK_PORT),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=Path.cwd(),
    )
    proc._log_file = f
    
    # Wait for it to start
    for _ in range(20):
        try:
            r = requests.get(f"http://localhost:{CALLBACK_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(0.5)
    
    proc.kill()
    f.close()
    raise RuntimeError("Callback receiver failed to start")


def start_vllm_server(model: str, log_file: Path) -> subprocess.Popen:
    """Start vLLM server with PoC enabled."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--enable-poc",
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", "0.4",
            "--max-model-len", "512",
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
    )
    proc._log_file = f
    
    # Wait for server to start
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
        if i > 0 and i % 10 == 0:
            print(f"    Still waiting for server... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError(f"vLLM server failed to start within {SERVER_STARTUP_TIMEOUT}s")


def stop_process(proc: subprocess.Popen):
    """Stop a subprocess gracefully."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.flush()
        proc._log_file.close()


# ============================================================================
# API Helpers
# ============================================================================

def api_call(method: str, endpoint: str, json_data: dict = None) -> dict:
    """Make API call to vLLM server."""
    url = f"http://localhost:{SERVER_PORT}{endpoint}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=json_data, timeout=30)
    r.raise_for_status()
    return r.json()


def get_callback_stats() -> Dict[str, Any]:
    """Get stats from callback receiver."""
    try:
        r = requests.get(f"http://localhost:{CALLBACK_PORT}/stats", timeout=5)
        return r.json()
    except:
        return {"total_batches": 0, "total_nonces": 0, "total_valid": 0}


# ============================================================================
# Single Test Execution
# ============================================================================

def run_single_test(
    test_num: int,
    model_name: str,
    model_short: str,
    block_hash: str,
    public_key: str,
    test_duration: int,
    logs_dir: Path,
    use_layer_hooks: bool = True,
    use_sign_flips: bool = False,
) -> Dict[str, Any]:
    """Run a single isolated test. Returns result dict."""
    
    prefix = f"test_{test_num:02d}_{model_short}_{block_hash}_{public_key}"
    server_log = logs_dir / f"{prefix}_server.log"
    callback_log = logs_dir / f"{prefix}_callback.log"
    
    result = {
        "test": test_num,
        "model": model_short,
        "model_full": model_name,
        "block_hash": block_hash,
        "public_key": public_key,
        "r_target": R_TARGET,
        "duration": test_duration,
        "timestamp": datetime.now().isoformat(),
        "total_checked": 0,
        "total_valid": 0,
        "valid_rate": 0.0,
        "callback_batches": 0,
        "callback_nonces": 0,
        "rate_per_second": 0.0,
        "error": None,
        "server_log": str(server_log),
        "callback_log": str(callback_log),
        "use_layer_hooks": use_layer_hooks,
        "use_sign_flips": use_sign_flips,
    }
    
    callback_proc = None
    server_proc = None
    
    try:
        # 1. Start callback receiver
        print(f"    Starting callback receiver...")
        callback_proc = start_callback_receiver(callback_log)
        
        # 2. Start vLLM server
        print(f"    Starting vLLM server ({model_name})...")
        server_proc = start_vllm_server(model_name, server_log)
        print(f"    Server ready")
        
        # 3. Init PoC generation
        config = {
            "block_hash": block_hash,
            "block_height": 100,
            "public_key": public_key,
            "r_target": R_TARGET,
            "node_id": 0,
            "node_count": 1,
            "callback_url": f"http://localhost:{CALLBACK_PORT}",
            "seq_len": 256,
            "use_layer_hooks": use_layer_hooks,
            "use_sign_flips": use_sign_flips,
        }
        api_call("POST", "/api/v1/pow/init/generate", config)
        hooks_str = "hooks=ON" if use_layer_hooks else "hooks=OFF"
        signs_str = "+signs" if use_sign_flips else ""
        print(f"    PoC generation started ({hooks_str}{signs_str}), running for {test_duration}s...")
        
        # 4. Wait for test duration
        time.sleep(test_duration)
        
        # 5. Get status
        status = api_call("GET", "/api/v1/pow/status")
        
        # 6. Stop PoC
        api_call("POST", "/api/v1/pow/stop")
        time.sleep(1)  # Let callback finish
        
        # 7. Get callback stats
        callback_stats = get_callback_stats()
        
        # 8. Fill in results
        result["total_checked"] = status.get("total_checked", 0)
        result["total_valid"] = status.get("total_valid", 0)
        result["rate_per_second"] = status.get("rate_per_second", 0.0)
        
        if result["total_checked"] > 0:
            result["valid_rate"] = result["total_valid"] / result["total_checked"] * 100
        
        result["callback_batches"] = callback_stats.get("total_batches", 0)
        result["callback_nonces"] = callback_stats.get("total_nonces", 0)
        
        print(f"    Results: {result['total_checked']} checked, {result['total_valid']} valid ({result['valid_rate']:.1f}%)")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"    ERROR: {e}")
    finally:
        # 9. Stop processes
        stop_process(server_proc)
        stop_process(callback_proc)
    
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full E2E Distribution Test")
    parser.add_argument("--models", nargs="+", choices=["qwen", "llama"],
                        help="Models to test (default: all)")
    parser.add_argument("--duration", type=int, default=TEST_DURATION,
                        help=f"Test duration in seconds (default: {TEST_DURATION})")
    parser.add_argument("--no-layer-hooks", action="store_true",
                        help="Disable per-layer normalization hooks")
    parser.add_argument("--sign-flips", action="store_true",
                        help="Enable random sign flips transform")
    args = parser.parse_args()
    
    # Filter models if specified
    models = MODELS
    if args.models:
        models = [(m, s) for m, s in MODELS if s in args.models]
    
    test_duration = args.duration
    use_layer_hooks = not args.no_layer_hooks
    use_sign_flips = args.sign_flips
    
    print("=" * 70)
    print("Full E2E Distribution Test")
    print("=" * 70)
    print(f"Models: {[s for _, s in models]}")
    print(f"Block hashes: {BLOCK_HASHES}")
    print(f"Public keys: {PUBLIC_KEYS}")
    print(f"r_target: {R_TARGET}")
    print(f"Duration per test: {test_duration}s")
    print(f"Total tests: {len(models) * len(BLOCK_HASHES) * len(PUBLIC_KEYS)}")
    print(f"Layer hooks: {'ON' if use_layer_hooks else 'OFF'}")
    print(f"Sign flips: {'ON' if use_sign_flips else 'OFF'}")
    print()
    
    # Setup logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    results_file = logs_dir / "e2e_results.jsonl"
    print(f"Results file: {results_file}")
    print(f"Monitor with: tail -f {results_file}")
    print()
    
    # Run all tests
    test_num = 0
    all_results = []
    
    for model_name, model_short in models:
        for block_hash in BLOCK_HASHES:
            for public_key in PUBLIC_KEYS:
                test_num += 1
                total_tests = len(models) * len(BLOCK_HASHES) * len(PUBLIC_KEYS)
                
                print("-" * 70)
                print(f"Test {test_num}/{total_tests}: {model_short} | {block_hash} | {public_key}")
                print("-" * 70)
                
                result = run_single_test(
                    test_num=test_num,
                    model_name=model_name,
                    model_short=model_short,
                    block_hash=block_hash,
                    public_key=public_key,
                    test_duration=test_duration,
                    logs_dir=logs_dir,
                    use_layer_hooks=use_layer_hooks,
                    use_sign_flips=use_sign_flips,
                )
                
                all_results.append(result)
                
                # Append to JSONL immediately (real-time tracking)
                with open(results_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                
                print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Group by model
    for model_name, model_short in models:
        model_results = [r for r in all_results if r["model"] == model_short]
        
        print(f"{model_short.upper()}:")
        valid_rates = []
        
        for r in model_results:
            status = "OK" if r["error"] is None else "ERR"
            rate_str = f"{r['valid_rate']:.1f}%" if r["error"] is None else "N/A"
            print(f"  [{status}] {r['block_hash']:12} | {r['public_key']:8} | "
                  f"checked={r['total_checked']:5} | valid={r['total_valid']:4} | rate={rate_str}")
            if r["error"] is None:
                valid_rates.append(r["valid_rate"])
        
        if valid_rates:
            avg_rate = sum(valid_rates) / len(valid_rates)
            min_rate = min(valid_rates)
            max_rate = max(valid_rates)
            spread = max_rate - min_rate
            print(f"  ----")
            print(f"  Valid rate: avg={avg_rate:.1f}%, min={min_rate:.1f}%, max={max_rate:.1f}%, spread={spread:.1f}%")
        print()
    
    # Overall stats
    all_rates = [r["valid_rate"] for r in all_results if r["error"] is None]
    failed_tests = [r for r in all_results if r["error"] is not None]
    
    print("-" * 70)
    if all_rates:
        print(f"Overall valid rate: avg={sum(all_rates)/len(all_rates):.1f}%, "
              f"min={min(all_rates):.1f}%, max={max(all_rates):.1f}%")
    print(f"Successful tests: {len(all_rates)}/{len(all_results)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for r in failed_tests:
            print(f"  - Test {r['test']}: {r['error']}")
    
    print()
    print(f"Full results: {results_file}")
    print(f"Individual logs: logs/test_*_server.log, logs/test_*_callback.log")
    
    return 0 if len(failed_tests) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

