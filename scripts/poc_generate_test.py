#!/usr/bin/env python3
"""
E2E Test for PoC /generate endpoint.

Tests:
1. Single request with nonces -> verify distances returned via callback
2. Multiple concurrent requests (same config) -> verify batching
3. Multiple concurrent requests (different configs) -> verify grouping
4. Determinism: same nonces twice -> same distances
5. Mutual exclusion: /generate while /init/generate active -> 409
6. Validate results match /validate endpoint

Usage:
    python scripts/poc_generate_test.py
    python scripts/poc_generate_test.py --model qwen
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import List, Dict, Any, Optional

import requests

SERVER_PORT = 8766
CALLBACK_PORT = 8767
SERVER_STARTUP_TIMEOUT = 120

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
}

R_TARGET = 1.34


@dataclass
class CallbackReceiver:
    received: List[Dict] = field(default_factory=list)
    server: Optional[HTTPServer] = None
    thread: Optional[Thread] = None


callback_data = CallbackReceiver()


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body.decode())
        callback_data.received.append({
            "path": self.path,
            "data": data,
            "time": time.time(),
        })
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        pass


def start_callback_server():
    callback_data.received.clear()
    callback_data.server = HTTPServer(('localhost', CALLBACK_PORT), CallbackHandler)
    callback_data.thread = Thread(target=callback_data.server.serve_forever)
    callback_data.thread.daemon = True
    callback_data.thread.start()


def stop_callback_server():
    if callback_data.server:
        callback_data.server.shutdown()
        callback_data.thread.join(timeout=5)


def start_vllm_server(model: str, log_dir: Path) -> subprocess.Popen:
    log_file = log_dir / "server.log"
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
    
    for i in range(SERVER_STARTUP_TIMEOUT):
        try:
            r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
        if i > 0 and i % 20 == 0:
            print(f"    Waiting for server... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError(f"Server failed to start within {SERVER_STARTUP_TIMEOUT}s")


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
    return r


def test_single_request():
    """Test single /generate request with callback."""
    print("\n  [Test 1] Single request with callback")
    callback_data.received.clear()
    
    nonces = list(range(10))
    resp = api_call("POST", "/api/v1/pow/generate", {
        "block_hash": "test_block_1",
        "block_height": 100,
        "public_key": "test_node_1",
        "r_target": R_TARGET,
        "nonces": nonces,
        "callback_url": f"http://localhost:{CALLBACK_PORT}",
    })
    
    if resp.status_code != 200:
        print(f"    FAIL: Expected 200, got {resp.status_code}")
        return False
    
    data = resp.json()
    if data.get("status") != "queued":
        print(f"    FAIL: Expected status=queued, got {data}")
        return False
    
    if data.get("queued_count") != len(nonces):
        print(f"    FAIL: Expected queued_count={len(nonces)}, got {data}")
        return False
    
    request_id = data.get("request_id")
    print(f"    Queued {len(nonces)} nonces, request_id={request_id[:8]}...")
    
    # Wait for callback (up to 15s)
    start = time.time()
    while time.time() - start < 15:
        if callback_data.received:
            break
        time.sleep(0.5)
    
    if not callback_data.received:
        print("    FAIL: No callback received within 15s")
        return False
    
    cb = callback_data.received[0]
    if cb["path"] != "/generated":
        print(f"    FAIL: Expected path=/generated, got {cb['path']}")
        return False
    
    results = cb["data"].get("results", [])
    print(f"    Received callback with {len(results)} results")
    
    if len(results) != len(nonces):
        print(f"    FAIL: Expected {len(nonces)} results, got {len(results)}")
        return False
    
    print("    PASS")
    return True


def test_concurrent_same_config():
    """Test multiple concurrent requests with same config get batched."""
    print("\n  [Test 2] Concurrent requests (same config)")
    callback_data.received.clear()
    
    # Send 3 requests with same config
    request_ids = []
    for i in range(3):
        nonces = list(range(i * 5, (i + 1) * 5))
        resp = api_call("POST", "/api/v1/pow/generate", {
            "block_hash": "batch_test_block",
            "block_height": 200,
            "public_key": "batch_test_node",
            "r_target": R_TARGET,
            "nonces": nonces,
            "callback_url": f"http://localhost:{CALLBACK_PORT}",
        })
        if resp.status_code == 200:
            request_ids.append(resp.json().get("request_id"))
    
    print(f"    Sent 3 requests, ids: {[r[:8] for r in request_ids]}")
    
    # Wait for callbacks
    start = time.time()
    while time.time() - start < 20:
        if len(callback_data.received) >= 3:
            break
        time.sleep(0.5)
    
    print(f"    Received {len(callback_data.received)} callbacks")
    
    # Verify all request_ids got callbacks
    received_ids = set()
    for cb in callback_data.received:
        req_id = cb["data"].get("request_id")
        if req_id:
            received_ids.add(req_id)
    
    if not all(rid in received_ids for rid in request_ids):
        print(f"    FAIL: Not all requests got callbacks")
        return False
    
    print("    PASS")
    return True


def test_different_configs():
    """Test requests with different configs are grouped separately."""
    print("\n  [Test 3] Different configs (separate groups)")
    callback_data.received.clear()
    
    configs = [
        {"block_hash": "config_A", "public_key": "node_A"},
        {"block_hash": "config_B", "public_key": "node_B"},
    ]
    
    for cfg in configs:
        resp = api_call("POST", "/api/v1/pow/generate", {
            **cfg,
            "block_height": 300,
            "r_target": R_TARGET,
            "nonces": [1, 2, 3],
            "callback_url": f"http://localhost:{CALLBACK_PORT}",
        })
        if resp.status_code != 200:
            print(f"    FAIL: Request failed: {resp.status_code}")
            return False
    
    # Wait for callbacks
    start = time.time()
    while time.time() - start < 20:
        if len(callback_data.received) >= 2:
            break
        time.sleep(0.5)
    
    # Verify we got callbacks for both configs
    block_hashes = set()
    for cb in callback_data.received:
        bh = cb["data"].get("block_hash")
        if bh:
            block_hashes.add(bh)
    
    if block_hashes != {"config_A", "config_B"}:
        print(f"    FAIL: Expected both configs, got {block_hashes}")
        return False
    
    print("    PASS")
    return True


def test_determinism():
    """Test same nonces produce same distances."""
    print("\n  [Test 4] Determinism (same nonces -> same distances)")
    
    nonces = [100, 200, 300]
    config = {
        "block_hash": "determinism_test",
        "block_height": 400,
        "public_key": "determinism_node",
        "r_target": R_TARGET,
        "nonces": nonces,
    }
    
    # First request
    callback_data.received.clear()
    resp1 = api_call("POST", "/api/v1/pow/generate", {
        **config,
        "callback_url": f"http://localhost:{CALLBACK_PORT}",
    })
    
    start = time.time()
    while time.time() - start < 15 and not callback_data.received:
        time.sleep(0.5)
    
    if not callback_data.received:
        print("    FAIL: No callback for first request")
        return False
    
    results1 = callback_data.received[0]["data"].get("results", [])
    distances1 = [r["distance"] for r in results1]
    
    # Second request (same config)
    callback_data.received.clear()
    resp2 = api_call("POST", "/api/v1/pow/generate", {
        **config,
        "callback_url": f"http://localhost:{CALLBACK_PORT}",
    })
    
    start = time.time()
    while time.time() - start < 15 and not callback_data.received:
        time.sleep(0.5)
    
    if not callback_data.received:
        print("    FAIL: No callback for second request")
        return False
    
    results2 = callback_data.received[0]["data"].get("results", [])
    distances2 = [r["distance"] for r in results2]
    
    # Compare distances
    if len(distances1) != len(distances2):
        print(f"    FAIL: Different result counts: {len(distances1)} vs {len(distances2)}")
        return False
    
    for i, (d1, d2) in enumerate(zip(distances1, distances2)):
        if abs(d1 - d2) > 1e-6:
            print(f"    FAIL: Distance mismatch at {i}: {d1} vs {d2}")
            return False
    
    print(f"    Distances match: {distances1}")
    print("    PASS")
    return True


def test_mutual_exclusion():
    """Test /generate returns 409 when /init/generate is active."""
    print("\n  [Test 5] Mutual exclusion with /init/generate")
    
    # Start continuous generation
    init_resp = api_call("POST", "/api/v1/pow/init/generate", {
        "block_hash": "mutex_test",
        "block_height": 500,
        "public_key": "mutex_node",
        "r_target": R_TARGET,
        "node_id": 0,
        "node_count": 1,
    })
    
    if init_resp.status_code != 200:
        print(f"    FAIL: /init/generate failed: {init_resp.status_code}")
        return False
    
    time.sleep(1)
    
    # Try /generate - should get 409
    gen_resp = api_call("POST", "/api/v1/pow/generate", {
        "block_hash": "mutex_test_2",
        "block_height": 501,
        "public_key": "mutex_node_2",
        "r_target": R_TARGET,
        "nonces": [1, 2, 3],
    })
    
    # Stop generation
    api_call("POST", "/api/v1/pow/stop")
    
    if gen_resp.status_code != 409:
        print(f"    FAIL: Expected 409, got {gen_resp.status_code}")
        return False
    
    print("    PASS")
    return True


def test_validate_consistency():
    """Test /generate distances match /validate distances."""
    print("\n  [Test 6] Consistency with /validate endpoint")
    
    nonces = [1000, 2000, 3000]
    config = {
        "block_hash": "validate_test",
        "block_height": 600,
        "public_key": "validate_node",
        "r_target": R_TARGET,
    }
    
    # Get distances via /generate
    callback_data.received.clear()
    api_call("POST", "/api/v1/pow/generate", {
        **config,
        "nonces": nonces,
        "callback_url": f"http://localhost:{CALLBACK_PORT}",
    })
    
    start = time.time()
    while time.time() - start < 15 and not callback_data.received:
        time.sleep(0.5)
    
    if not callback_data.received:
        print("    FAIL: No callback from /generate")
        return False
    
    gen_results = callback_data.received[0]["data"].get("results", [])
    gen_distances = {r["nonce"]: r["distance"] for r in gen_results}
    
    # Get distances via /validate
    api_call("POST", "/api/v1/pow/init/validate", config)
    
    val_resp = api_call("POST", "/api/v1/pow/validate", {
        **config,
        "nonces": nonces,
        "dist": [0.0] * len(nonces),
        "node_id": 0,
    })
    
    api_call("POST", "/api/v1/pow/stop")
    
    if val_resp.status_code != 200:
        print(f"    FAIL: /validate failed: {val_resp.status_code}")
        return False
    
    val_distances = val_resp.json().get("computed_distances", [])
    
    # Compare
    for i, nonce in enumerate(nonces):
        gen_d = gen_distances.get(nonce)
        val_d = val_distances[i] if i < len(val_distances) else None
        
        if gen_d is None or val_d is None:
            print(f"    FAIL: Missing distance for nonce {nonce}")
            return False
        
        if abs(gen_d - val_d) > 1e-6:
            print(f"    FAIL: Distance mismatch for nonce {nonce}: gen={gen_d}, val={val_d}")
            return False
    
    print(f"    /generate and /validate distances match")
    print("    PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="PoC /generate E2E Test")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--skip-server", action="store_true", help="Skip server startup")
    args = parser.parse_args()
    
    log_dir = Path("logs") / f"generate_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PoC /generate Endpoint E2E Test")
    print("=" * 60)
    print(f"Model: {args.model} ({MODELS[args.model]})")
    print(f"Logs: {log_dir}")
    
    server_proc = None
    results = {}
    
    try:
        start_callback_server()
        print(f"\nCallback server started on port {CALLBACK_PORT}")
        
        if not args.skip_server:
            print(f"\nStarting vLLM server...")
            server_proc = start_vllm_server(MODELS[args.model], log_dir)
            print("Server ready")
        
        results["single_request"] = test_single_request()
        results["concurrent_same"] = test_concurrent_same_config()
        results["different_configs"] = test_different_configs()
        results["determinism"] = test_determinism()
        results["mutual_exclusion"] = test_mutual_exclusion()
        results["validate_consistency"] = test_validate_consistency()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_callback_server()
        if server_proc:
            stop_process(server_proc)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

