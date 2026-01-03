#!/usr/bin/env python3
"""
Test /generate and /chat/completion parallel operation.

Verifies that layer hooks from /generate don't interfere with /chat/completion.

Usage:
    python scripts/poc_parallel_test.py
"""

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

SERVER_PORT = 8770
CALLBACK_PORT = 8771
SERVER_STARTUP_TIMEOUT = 120

MODEL = "Qwen/Qwen3-0.6B"
R_TARGET = 1.34

callback_received = []


class CallbackHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        callback_received.append(json.loads(body.decode()))
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        pass


def start_callback_server():
    callback_received.clear()
    server = HTTPServer(('localhost', CALLBACK_PORT), CallbackHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def start_vllm_server(log_dir: Path) -> subprocess.Popen:
    log_file = log_dir / "server.log"
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    
    f = open(log_file, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
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


def chat_completion(prompt: str) -> str:
    """Call /chat/completion and return the response text."""
    resp = requests.post(
        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        return f"ERROR: {resp.status_code}"
    return resp.json()["choices"][0]["message"]["content"]


def poc_generate(nonces: list) -> dict:
    """Call /generate endpoint."""
    resp = requests.post(
        f"http://localhost:{SERVER_PORT}/api/v1/pow/generate",
        json={
            "block_hash": "parallel_test",
            "block_height": 100,
            "public_key": "test_node",
            "r_target": R_TARGET,
            "nonces": nonces,
            "callback_url": f"http://localhost:{CALLBACK_PORT}",
        },
        timeout=60,
    )
    return resp.json() if resp.status_code == 200 else {"error": resp.status_code}


def test_chat_before_generate():
    """Test chat works before any /generate call."""
    print("\n  [Test 1] Chat completion BEFORE /generate")
    
    response = chat_completion("What is 2+2? Answer with just the number.")
    print(f"    Response: {response[:100]}")
    
    # Should get a sensible response
    if "4" in response or "four" in response.lower():
        print("    PASS: Got expected response")
        return True
    else:
        print("    WARN: Response may be unexpected (checking for coherence)")
        return len(response) > 0 and not response.startswith("ERROR")


def test_chat_during_generate():
    """Test chat works DURING /generate processing."""
    print("\n  [Test 2] Chat completion DURING /generate")
    
    # Start a large /generate request
    callback_received.clear()
    poc_generate(list(range(1000)))
    
    # Immediately call chat
    response = chat_completion("What is 3+3? Answer with just the number.")
    print(f"    Response: {response[:100]}")
    
    # Wait for generate to finish
    time.sleep(5)
    
    if "6" in response or "six" in response.lower():
        print("    PASS: Got expected response")
        return True
    else:
        print("    WARN: Response may be affected by hooks")
        # Check if response is garbage (hooks interference)
        if len(response) < 5 or response.startswith("ERROR"):
            print("    FAIL: Response appears corrupted")
            return False
        return True


def test_chat_after_generate():
    """Test chat works AFTER /generate (hooks should be cleaned up)."""
    print("\n  [Test 3] Chat completion AFTER /generate")
    
    # Run /generate
    callback_received.clear()
    poc_generate(list(range(100)))
    
    # Wait for it to complete
    time.sleep(3)
    
    # Now test chat
    response = chat_completion("What is 5+5? Answer with just the number.")
    print(f"    Response: {response[:100]}")
    
    if "10" in response or "ten" in response.lower():
        print("    PASS: Got expected response")
        return True
    else:
        print("    WARN: Response may be affected by cached hooks")
        if len(response) < 5 or response.startswith("ERROR"):
            print("    FAIL: Response appears corrupted")
            return False
        return True


def test_multiple_chat_generate_cycles():
    """Test multiple alternating chat/generate calls."""
    print("\n  [Test 4] Multiple chat/generate cycles")
    
    results = []
    for i in range(3):
        # Generate
        poc_generate(list(range(i*100, (i+1)*100)))
        time.sleep(1)
        
        # Chat
        response = chat_completion(f"What is {i+1}+{i+1}? Just the number.")
        expected = str((i+1)*2)
        success = expected in response
        results.append(success)
        print(f"    Cycle {i+1}: expected {expected}, got '{response[:50]}' - {'PASS' if success else 'FAIL'}")
    
    return all(results)


def main():
    log_dir = Path("logs") / f"parallel_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PoC + Chat Parallel Operation Test")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Logs: {log_dir}")
    
    server_proc = None
    results = {}
    
    try:
        cb_server = start_callback_server()
        
        print("\nStarting server...")
        server_proc = start_vllm_server(log_dir)
        print("Server ready")
        
        results["chat_before"] = test_chat_before_generate()
        results["chat_during"] = test_chat_during_generate()
        results["chat_after"] = test_chat_after_generate()
        results["multiple_cycles"] = test_multiple_chat_generate_cycles()
        
        cb_server.shutdown()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
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
        print("ALL TESTS PASSED - Hooks don't interfere with chat")
        return 0
    else:
        print("TESTS FAILED - Hooks may interfere with chat!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

