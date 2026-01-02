#!/usr/bin/env python3
"""
Simple E2E test for PoC with vLLM Server.

Tests the simplified pipeline: random input -> per-layer hooks -> k-dim orthogonal transform.

Usage:
    python scripts/poc_e2e_simple_test.py
    python scripts/poc_e2e_simple_test.py --models qwen
    python scripts/poc_e2e_simple_test.py --models llama
    python scripts/poc_e2e_simple_test.py --duration 60
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

R_TARGET = 1.39
BLOCK_HASH = "TEST_BLOCK_HASH_123"
PUBLIC_KEY = "ascdcdfvf"

MODELS = [
    ("Qwen/Qwen3-0.6B", "qwen"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama"),
]

SERVER_PORT = 8765
TEST_DURATION = 60


def start_vllm_server(model: str, log_file: Path) -> subprocess.Popen:
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
        start_new_session=True,
    )
    proc._log_file = f
    
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
        if i > 0 and i % 10 == 0:
            print(f"    Waiting for server... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError("vLLM server failed to start")


def stop_process(proc: subprocess.Popen):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.flush()
        proc._log_file.close()


def api_call(method: str, endpoint: str, json_data: dict = None) -> dict:
    url = f"http://localhost:{SERVER_PORT}{endpoint}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=json_data, timeout=30)
    r.raise_for_status()
    return r.json()


def run_test(model_name: str, model_short: str, duration: int, logs_dir: Path) -> dict:
    server_log = logs_dir / f"poc_simple_{model_short}_server.log"
    
    result = {
        "model": model_short,
        "block_hash": BLOCK_HASH,
        "public_key": PUBLIC_KEY,
        "r_target": R_TARGET,
        "total_checked": 0,
        "total_valid": 0,
        "valid_rate": 0.0,
        "error": None,
    }
    
    server_proc = None
    
    try:
        print(f"  Starting vLLM server ({model_name})...")
        server_proc = start_vllm_server(model_name, server_log)
        print(f"  Server ready")
        
        config = {
            "block_hash": BLOCK_HASH,
            "block_height": 100,
            "public_key": PUBLIC_KEY,
            "r_target": R_TARGET,
            "node_id": 0,
            "node_count": 1,
            "seq_len": 256,
        }
        api_call("POST", "/api/v1/pow/init/generate", config)
        print(f"  PoC generation started, running for {duration}s...")
        
        time.sleep(duration)
        
        status = api_call("GET", "/api/v1/pow/status")
        api_call("POST", "/api/v1/pow/stop")
        
        result["total_checked"] = status.get("total_checked", 0)
        result["total_valid"] = status.get("total_valid", 0)
        
        if result["total_checked"] > 0:
            result["valid_rate"] = result["total_valid"] / result["total_checked"] * 100
        
        print(f"  Results: {result['total_checked']} checked, {result['total_valid']} valid ({result['valid_rate']:.1f}%)")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"  ERROR: {e}")
    finally:
        stop_process(server_proc)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Simple E2E PoC Test")
    parser.add_argument("--models", nargs="+", choices=["qwen", "llama"],
                        help="Models to test (default: all)")
    parser.add_argument("--duration", type=int, default=TEST_DURATION,
                        help=f"Test duration in seconds (default: {TEST_DURATION})")
    args = parser.parse_args()
    
    models = MODELS
    if args.models:
        models = [(m, s) for m, s in MODELS if s in args.models]
    
    print("=" * 60)
    print("Simple E2E PoC Test")
    print("=" * 60)
    print(f"Block hash: {BLOCK_HASH}")
    print(f"Public key: {PUBLIC_KEY}")
    print(f"r_target: {R_TARGET}")
    print(f"Duration: {args.duration}s per model")
    print(f"Models: {[s for _, s in models]}")
    print()
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    results = []
    for model_name, model_short in models:
        print("-" * 60)
        print(f"Testing {model_short}")
        print("-" * 60)
        result = run_test(model_name, model_short, args.duration, logs_dir)
        results.append(result)
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "OK" if r["error"] is None else "ERR"
        rate = f"{r['valid_rate']:.1f}%" if r["error"] is None else "N/A"
        print(f"  [{status}] {r['model']:8} | checked={r['total_checked']:5} | valid={r['total_valid']:4} | rate={rate}")
    
    failed = [r for r in results if r["error"]]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

