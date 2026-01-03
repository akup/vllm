#!/usr/bin/env python3
"""
Performance comparison: /generate vs /init/generate

Usage:
    python scripts/poc_perf_compare.py
    python scripts/poc_perf_compare.py --duration 60
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

import requests

SERVER_PORT = 8768
CALLBACK_PORT = 8769
SERVER_STARTUP_TIMEOUT = 120

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
}

R_TARGET = 1.34
SEQ_LEN = 256
NONCES_PER_REQUEST = 1024  # Large batch


class CallbackCollector:
    received = []
    total_nonces = 0


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body.decode())
        CallbackCollector.received.append(data)
        results = data.get("results", data.get("nonces", []))
        CallbackCollector.total_nonces += len(results) if isinstance(results, list) else 0
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        pass


def start_callback_server():
    CallbackCollector.received.clear()
    CallbackCollector.total_nonces = 0
    server = HTTPServer(('localhost', CALLBACK_PORT), CallbackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


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
            print(f"    Waiting... ({i}s)")
    
    proc.kill()
    f.close()
    raise RuntimeError("Server startup failed")


def stop_process(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except:
                pass
            proc.wait()
    if hasattr(proc, '_log_file') and proc._log_file:
        proc._log_file.close()


def api(method: str, endpoint: str, data: dict = None):
    url = f"http://localhost:{SERVER_PORT}{endpoint}"
    if method == "GET":
        return requests.get(url, timeout=60)
    return requests.post(url, json=data, timeout=60)


def run_init_generate(duration: int) -> dict:
    """Test /init/generate (continuous)."""
    print(f"\n  [1/2] /init/generate for {duration}s...")
    
    CallbackCollector.received.clear()
    CallbackCollector.total_nonces = 0
    start = time.time()
    
    api("POST", "/api/v1/pow/init/generate", {
        "block_hash": "perf_init",
        "block_height": 100,
        "public_key": "perf_node",
        "r_target": R_TARGET,
        "node_id": 0,
        "node_count": 1,
        "batch_size": 32,
        "seq_len": SEQ_LEN,
        "callback_url": f"http://localhost:{CALLBACK_PORT}",
    })
    
    time.sleep(duration)
    
    status = api("GET", "/api/v1/pow/status").json()
    api("POST", "/api/v1/pow/stop")
    
    elapsed = time.time() - start
    total = status.get("total_checked", 0)
    valid = status.get("total_valid", 0)
    
    print(f"        Processed: {total} nonces")
    print(f"        Throughput: {total/elapsed:.1f} nonces/sec")
    
    return {
        "method": "/init/generate",
        "total": total,
        "valid": valid,
        "elapsed": elapsed,
        "throughput": total / elapsed,
    }


def run_generate(duration: int) -> dict:
    """Test /generate (batch)."""
    print(f"\n  [2/2] /generate for {duration}s (batch={NONCES_PER_REQUEST})...")
    
    CallbackCollector.received.clear()
    CallbackCollector.total_nonces = 0
    start = time.time()
    nonce_counter = 0
    requests_sent = 0
    
    while time.time() - start < duration:
        nonces = list(range(nonce_counter, nonce_counter + NONCES_PER_REQUEST))
        nonce_counter += NONCES_PER_REQUEST
        
        resp = api("POST", "/api/v1/pow/generate", {
            "block_hash": "perf_gen",
            "block_height": 100,
            "public_key": "perf_node",
            "r_target": R_TARGET,
            "nonces": nonces,
            "seq_len": SEQ_LEN,
            "callback_url": f"http://localhost:{CALLBACK_PORT}",
        })
        
        if resp.status_code == 200:
            requests_sent += 1
    
    # Wait for remaining callbacks
    time.sleep(2)
    
    elapsed = time.time() - start
    total = CallbackCollector.total_nonces
    
    # Count valid from callbacks
    valid = 0
    for cb in CallbackCollector.received:
        for r in cb.get("results", []):
            if r.get("distance") is not None and r["distance"] < R_TARGET:
                valid += 1
    
    print(f"        Processed: {total} nonces ({requests_sent} requests)")
    print(f"        Throughput: {total/elapsed:.1f} nonces/sec")
    
    return {
        "method": "/generate",
        "total": total,
        "valid": valid,
        "elapsed": elapsed,
        "throughput": total / elapsed,
        "requests": requests_sent,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen")
    parser.add_argument("--duration", type=int, default=30)
    args = parser.parse_args()
    
    log_dir = Path("logs") / f"perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PoC Performance: /generate vs /init/generate")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Duration: {args.duration}s per test")
    print(f"Batch:    {NONCES_PER_REQUEST} nonces per /generate request")
    
    server_proc = None
    
    try:
        cb_server = start_callback_server()
        
        print(f"\nStarting server...")
        server_proc = start_vllm_server(MODELS[args.model], log_dir)
        print("Server ready")
        
        r1 = run_init_generate(args.duration)
        time.sleep(2)
        r2 = run_generate(args.duration)
        
        cb_server.shutdown()
        
        # Results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        ratio = r2["throughput"] / r1["throughput"] * 100 if r1["throughput"] > 0 else 0
        
        print(f"\n  /init/generate: {r1['throughput']:.1f} nonces/sec (baseline)")
        print(f"  /generate:      {r2['throughput']:.1f} nonces/sec ({ratio:.0f}%)")
        
        print(f"\n  Performance ratio: {ratio:.1f}%")
        
        if ratio >= 95:
            print("  Status: PASS (>= 95%)")
            status = 0
        else:
            print(f"  Status: FAIL (< 95%)")
            status = 1
        
        # Save
        with open(log_dir / "results.json", "w") as f:
            json.dump({"init_generate": r1, "generate": r2, "ratio": ratio}, f, indent=2)
        
        return status
        
    finally:
        if server_proc:
            stop_process(server_proc)


if __name__ == "__main__":
    sys.exit(main())
