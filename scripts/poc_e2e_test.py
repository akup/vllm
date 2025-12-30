#!/usr/bin/env python3
"""
E2E Test for PoC with vLLM Server.

This script:
1. Starts callback receiver
2. For each model:
   - Starts vLLM server with PoC enabled
   - Runs generation round, collects valid nonces
   - Stops and restarts server (simulates node restart)
   - Validates previously found nonces after restart
   - Tests seed scenarios (wrong hash/pubkey = different distances)
   - Stops server
3. Collects logs to logs/ directory
4. Reports results

Usage:
    python scripts/poc_e2e_test.py
    python scripts/poc_e2e_test.py --models qwen  # Only test Qwen
    python scripts/poc_e2e_test.py --models llama  # Only test Llama
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

# Configuration
CALLBACK_PORT = 8081
SERVER_PORT = 8765
GENERATION_TIME = 10  # seconds to run generation
SERVER_STARTUP_TIMEOUT = 120  # seconds to wait for server

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "llama": "unsloth/Llama-3.2-1B-Instruct",
}

# r_target for ~10% valid rate (consistent across models with per-layer normalization)
# Per phase 4.2: with per-layer normalization + random lm_head, both models use ~1.405
MODEL_R_TARGETS = {
    "qwen": 1.405,
    "llama": 1.405,
}

# Test configuration (r_target overridden per model)
TEST_CONFIG = {
    "block_hash": "e2e_test_block_hash_12345",
    "block_height": 100,
    "public_key": "e2e_test_node",
    "r_target": 1.405,  # Consistent across models with per-layer normalization
    "node_id": 0,
    "node_count": 1,
    "batch_size": 32,
    "seq_len": 256,  # Must match PoCConfig default
}


@dataclass
class TestResult:
    model: str
    generation_nonces: int = 0
    generation_valid: int = 0
    validation_match: bool = False
    wrong_hash_different: bool = False
    wrong_pubkey_different: bool = False
    passed: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass 
class TestSuite:
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    results: List[TestResult] = field(default_factory=list)
    all_passed: bool = False


def setup_logs_dir() -> Path:
    """Create logs directory."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def start_callback_receiver(logs_dir: Path) -> subprocess.Popen:
    """Start callback receiver in background."""
    log_file = logs_dir / "callback.log"
    f = open(log_file, "w", buffering=1)  # Line buffered
    proc = subprocess.Popen(
        [
            sys.executable, "-u",  # Unbuffered Python output
            "scripts/poc_callback_receiver.py",
            "--port", str(CALLBACK_PORT),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=Path.cwd(),
    )
    proc._log_file = f  # Keep reference to close later
    
    # Wait for it to start
    for _ in range(10):
        try:
            r = requests.get(f"http://localhost:{CALLBACK_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(0.5)
    
    proc.kill()
    raise RuntimeError("Callback receiver failed to start")


def start_vllm_server(model: str, logs_dir: Path, model_key: str) -> subprocess.Popen:
    """Start vLLM server with PoC enabled."""
    log_file = logs_dir / f"server_{model_key}.log"
    
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output
    
    f = open(log_file, "a", buffering=1)  # Append mode, line buffered
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--enable-poc",
            "--port", str(SERVER_PORT),
            "--gpu-memory-utilization", "0.4",
            "--max-model-len", "256",
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
    )
    proc._log_file = f  # Keep reference to close later
    
    # Wait for server to start
    print(f"  Waiting for server to start (up to {SERVER_STARTUP_TIMEOUT}s)...")
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"  Server started in {i+1}s")
                return proc
        except:
            pass
        time.sleep(1)
    
    proc.kill()
    raise RuntimeError(f"vLLM server failed to start within {SERVER_STARTUP_TIMEOUT}s")


def stop_process(proc: subprocess.Popen, name: str = "process"):
    """Stop a subprocess gracefully."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Close log file if attached
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.flush()
        proc._log_file.close()
    print(f"  {name} stopped")


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


def clear_callback_stats():
    """Clear callback receiver stats."""
    try:
        requests.delete(f"http://localhost:{CALLBACK_PORT}/clear", timeout=5)
    except:
        pass


def run_generation(callback_url: str, model_key: str) -> Dict[str, Any]:
    """Run generation and return status with valid nonces."""
    # Clear callback stats before starting
    clear_callback_stats()
    
    # Use model-specific r_target
    r_target = MODEL_R_TARGETS.get(model_key, TEST_CONFIG["r_target"])
    
    config = {
        **TEST_CONFIG,
        "r_target": r_target,
        "callback_url": callback_url,
    }
    
    # Start generation
    api_call("POST", "/api/v1/pow/init/generate", config)
    
    # Wait for generation
    print(f"  Generating for {GENERATION_TIME}s...")
    time.sleep(GENERATION_TIME)
    
    # Get status
    status = api_call("GET", "/api/v1/pow/status")
    
    # Stop generation
    api_call("POST", "/api/v1/pow/stop")
    
    # Give callback sender time to finish sending queued batches
    time.sleep(1)
    
    # Get callback stats
    callback_stats = get_callback_stats()
    print(f"  Callback receiver got: {callback_stats['total_batches']} batches, {callback_stats['total_nonces']} nonces")
    
    return status


def validate_nonces(nonces: List[int], distances: List[float], 
                    block_hash: str = None, public_key: str = None,
                    r_target: float = None) -> Dict[str, Any]:
    """Validate nonces and return result."""
    if block_hash is None:
        block_hash = TEST_CONFIG["block_hash"]
    if public_key is None:
        public_key = TEST_CONFIG["public_key"]
    if r_target is None:
        r_target = TEST_CONFIG["r_target"]
    
    # First init for validate mode
    init_config = {
        "block_hash": block_hash,
        "block_height": TEST_CONFIG["block_height"],
        "public_key": public_key,
        "r_target": r_target,
    }
    api_call("POST", "/api/v1/pow/init/validate", init_config)
    
    # Validate nonces
    validate_request = {
        "public_key": public_key,
        "block_hash": block_hash,
        "block_height": TEST_CONFIG["block_height"],
        "nonces": nonces,
        "dist": distances,
        "node_id": 0,
    }
    
    # For validation, we need to get the computed distances
    # The /validate endpoint is fire-and-forget, so we use status to get computed distances
    # Actually, we need to call run_batch for validation, but the API doesn't expose that
    # Let's use the validate endpoint and check fraud_detected
    result = api_call("POST", "/api/v1/pow/validate", validate_request)
    
    return result


def test_model(model_key: str, model_name: str, logs_dir: Path, 
               callback_url: str) -> TestResult:
    """Test a single model."""
    result = TestResult(model=model_name)
    start_time = time.time()
    
    server_proc = None
    
    try:
        # Phase 1: Start server and generate
        print(f"\n  Phase 1: Generation")
        server_proc = start_vllm_server(model_name, logs_dir, model_key)
        
        status = run_generation(callback_url, model_key)
        result.generation_nonces = status["total_checked"]
        result.generation_valid = status["total_valid"]
        
        valid_nonces = status["valid_nonces"]
        valid_distances = status["valid_distances"]
        
        print(f"  Generated: {result.generation_nonces} checked, {result.generation_valid} valid")
        
        if len(valid_nonces) == 0:
            result.error = "No valid nonces found during generation"
            return result
        
        # Phase 2: Restart server and validate
        print(f"\n  Phase 2: Restart and Validate")
        stop_process(server_proc, "Server")
        time.sleep(2)
        
        server_proc = start_vllm_server(model_name, logs_dir, model_key)
        
        # Validate same nonces - they should produce same distances
        # Note: Current API doesn't return computed distances, only fraud_detected
        # We'll test determinism by checking fraud_detected is False
        r_target = MODEL_R_TARGETS.get(model_key, TEST_CONFIG["r_target"])
        val_result = validate_nonces(valid_nonces[:10], valid_distances[:10], r_target=r_target)
        result.validation_match = not val_result.get("fraud_detected", True)
        
        # Debug: show distances comparison
        if not result.validation_match:
            print(f"  DEBUG: r_target={r_target}")
            print(f"  DEBUG: Original distances (first 5): {valid_distances[:5]}")
            # Get computed distances from the result if available
            computed = val_result.get("computed_distances", [])
            if computed:
                print(f"  DEBUG: Computed distances (first 5): {computed[:5]}")
                # Show which nonces have discrepancy
                for i, (orig, comp) in enumerate(zip(valid_distances[:10], computed[:10])):
                    orig_valid = orig < r_target
                    comp_valid = comp < r_target
                    if orig_valid != comp_valid:
                        print(f"  MISMATCH nonce[{i}]: orig={orig:.6f} (<r_target={orig_valid}), comp={comp:.6f} (<r_target={comp_valid})")
        
        print(f"  Validation after restart: {'PASS' if result.validation_match else 'FAIL'}")
        
        # Phase 3: Wrong seed tests
        print(f"\n  Phase 3: Wrong Seed Tests")
        
        # Wrong block_hash - should detect fraud (different distances)
        # To trigger fraud detection, we claim very low distances (valid) but with
        # wrong seed, computed distances will be ~1.2 (invalid with strict r_target)
        fake_low_distances = [0.05] * len(valid_nonces[:5])
        wrong_hash_result = validate_nonces(
            valid_nonces[:5], fake_low_distances,
            block_hash="wrong_block_hash_xyz",
            r_target=0.1  # Strict: 0.05 < 0.1 (claimed valid), computed ~1.2 >= 0.1 (invalid)
        )
        # With wrong hash and strict r_target, fraud should be detected
        result.wrong_hash_different = wrong_hash_result.get("fraud_detected", False)
        print(f"  Wrong block_hash -> fraud detected: {'PASS' if result.wrong_hash_different else 'FAIL'}")
        
        # Wrong public_key - should detect fraud
        wrong_pubkey_result = validate_nonces(
            valid_nonces[:5], fake_low_distances,
            public_key="wrong_public_key_xyz",
            r_target=0.1
        )
        result.wrong_pubkey_different = wrong_pubkey_result.get("fraud_detected", False)
        print(f"  Wrong public_key -> fraud detected: {'PASS' if result.wrong_pubkey_different else 'FAIL'}")
        
        # Determine overall pass
        result.passed = (
            result.validation_match and
            result.wrong_hash_different and
            result.wrong_pubkey_different
        )
        
    except Exception as e:
        result.error = str(e)
        print(f"  ERROR: {e}")
    finally:
        if server_proc:
            stop_process(server_proc, "Server")
        result.duration_seconds = time.time() - start_time
    
    return result


def main():
    parser = argparse.ArgumentParser(description="PoC E2E Test Suite")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help="Models to test (default: all)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PoC E2E Test Suite")
    print("=" * 60)
    
    logs_dir = setup_logs_dir()
    print(f"\nLogs directory: {logs_dir.absolute()}")
    
    suite = TestSuite()
    callback_proc = None
    
    try:
        # Start callback receiver
        print("\n[1/3] Starting callback receiver...")
        callback_proc = start_callback_receiver(logs_dir)
        callback_url = f"http://localhost:{CALLBACK_PORT}"
        print(f"  Callback receiver running on {callback_url}")
        
        # Test each model
        for i, model_key in enumerate(args.models):
            model_name = MODELS[model_key]
            print(f"\n[{i+2}/{len(args.models)+2}] Testing {model_name}")
            print("-" * 50)
            
            result = test_model(model_key, model_name, logs_dir, callback_url)
            suite.results.append(result)
            
            status = "PASS" if result.passed else "FAIL"
            print(f"\n  Result: {status} ({result.duration_seconds:.1f}s)")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        all_passed = True
        for result in suite.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.model}")
            if result.error:
                print(f"        Error: {result.error}")
            else:
                print(f"        Generation: {result.generation_valid}/{result.generation_nonces} valid")
                print(f"        Validation after restart: {'OK' if result.validation_match else 'FAIL'}")
                print(f"        Wrong hash test: {'OK' if result.wrong_hash_different else 'FAIL'}")
                print(f"        Wrong pubkey test: {'OK' if result.wrong_pubkey_different else 'FAIL'}")
            all_passed = all_passed and result.passed
        
        suite.all_passed = all_passed
        
        # Save results
        results_file = logs_dir / "test_results.json"
        with open(results_file, "w") as f:
            # Convert to serializable format
            data = {
                "start_time": suite.start_time,
                "all_passed": suite.all_passed,
                "results": [asdict(r) for r in suite.results],
            }
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {results_file}")
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL TESTS PASSED!")
            return 0
        else:
            print("SOME TESTS FAILED!")
            return 1
        
    finally:
        if callback_proc:
            stop_process(callback_proc, "Callback receiver")


if __name__ == "__main__":
    sys.exit(main())

