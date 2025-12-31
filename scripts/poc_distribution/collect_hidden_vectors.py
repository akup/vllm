#!/usr/bin/env python3
"""
Collect raw full-d hidden vectors per model via PoC HTTP API and save artifacts only.

This script:
- Starts a local vLLM OpenAI server with PoC enabled
- Initializes a PoC round
- Switches to GENERATING (manual)
- Calls `/api/v1/pow/batch` until N vectors are collected
- Saves artifacts under:
    scripts/poc_distribution/data/{experiment}/{model_short}/raw/
      - hidden_unit_vectors.npy   (float16, shape [N, d_full])
      - meta.json                 (includes nonces, paths, timings)
      - server.log

No offline transforms, projections, or analyses are performed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import requests

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3-0.6B", "qwen"),
    ("unsloth/Llama-3.2-1B-Instruct", "llama"),
]


def _safe_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:80] if len(s) > 80 else s


def _wait_health(
    base_url: str,
    *,
    proc: subprocess.Popen | None = None,
    log_path: Path | None = None,
    timeout_s: int = 240,
) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            tail = ""
            try:
                if log_path and log_path.exists():
                    txt = log_path.read_text(errors="ignore")
                    tail = "\n".join(txt.splitlines()[-80:])
            except Exception:
                pass
            raise RuntimeError(
                f"Server process exited early (exit={proc.returncode}). Log tail:\n{tail}"
            )
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Server did not become healthy within {timeout_s}s")


def _get_poc_status(base_url: str, *, timeout_s: int = 10) -> dict:
    r = requests.get(f"{base_url}/api/v1/pow/status", timeout=timeout_s)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected /api/v1/pow/status payload")
    return payload


def _wait_poc_state(base_url: str, *, want_state: str, timeout_s: int = 60) -> None:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            st = _get_poc_status(base_url, timeout_s=10)
            last = st.get("state")
            if last == want_state:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"PoC did not reach state={want_state!r} within {timeout_s}s (last={last!r})")


def _torch_load_b64(payload: Dict[str, Any]) -> Any:
    if payload.get("_format") != "torch_save_b64":
        raise ValueError("Not a torch_save_b64 payload")
    import torch  # lazy import

    data = base64.b64decode(payload["b64"])
    return torch.load(io.BytesIO(data), map_location="cpu")


def _extract_tensor(obj: Any) -> Any:
    if isinstance(obj, dict) and obj.get("_format") == "torch_save_b64":
        return _torch_load_b64(obj)
    if isinstance(obj, dict):
        return {k: _extract_tensor(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_tensor(v) for v in obj]
    return obj


def _start_server(
    *,
    model: str,
    port: int,
    log_path: Path,
    gpu_mem_util: float,
    max_model_len: int,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    log_f = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--enable-poc",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_mem_util),
            "--max-model-len",
            str(max_model_len),
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
        start_new_session=True,
    )
    proc._log_file = log_f  # type: ignore[attr-defined]
    return proc


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            # Kill process group to avoid leaving engine workers around.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=15)
    finally:
        lf = getattr(proc, "_log_file", None)
        if lf:
            lf.flush()
            lf.close()


def collect_hidden_vectors(
    *,
    model_name: str,
    model_short: str,
    out_dir: Path,
    port: int,
    n: int,
    seq_len: int,
    batch_size: int,
    gpu_mem_util: float,
    max_model_len: int,
    block_hash: str,
    public_key: str,
    timeout_s: int,
    poc_use_sign_flips: bool,
    poc_use_nonce_householder: bool,
    poc_use_nonce_orthogonal: bool,
    poc_pick_k_dims: int,
) -> Path:
    """Collect and write raw artifact dir for a single model."""
    raw_dir = out_dir / model_short / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    server_log = raw_dir / "server.log"
    base_url = f"http://localhost:{port}"

    server = None
    t0 = time.time()
    try:
        server = _start_server(
            model=model_name,
            port=port,
            log_path=server_log,
            gpu_mem_util=gpu_mem_util,
            max_model_len=max_model_len,
        )
        _wait_health(base_url, proc=server, log_path=server_log)

        init_body = {
            "block_hash": block_hash,
            "block_height": 1,
            "public_key": public_key,
            "r_target": 10.0,
            "node_id": 0,
            "node_count": 1,
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "callback_url": None,
            # Flags are irrelevant for `last_hidden_unit_pre` but keep stable/explicit.
            "use_layer_hooks": False,
            "use_sign_flips": bool(poc_use_sign_flips),
            "use_nonce_householder": bool(poc_use_nonce_householder),
            "use_nonce_orthogonal": bool(poc_use_nonce_orthogonal),
            "pick_k_dims": int(poc_pick_k_dims),
        }
        r = requests.post(f"{base_url}/api/v1/pow/init", json=init_body, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"/pow/init failed: status={r.status_code} body={r.text[:800]}")
        r = requests.post(f"{base_url}/api/v1/pow/phase/generate_manual", json={}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(
                f"/pow/phase/generate_manual failed: status={r.status_code} body={r.text[:800]}"
            )
        # Avoid a race where /batch is called before the engine reports GENERATING.
        _wait_poc_state(base_url, want_state="GENERATING", timeout_s=60)

        vecs: List[np.ndarray] = []
        nonces_all: List[int] = []
        batches = 0
        while sum(v.shape[0] for v in vecs) < n:
            rr = None
            # Robust retry: we have observed transient 400s claiming not GENERATING
            # even when /status reports GENERATING (likely RPC ordering/latency).
            for attempt in range(6):
                rr = requests.post(
                    f"{base_url}/api/v1/pow/batch",
                    json={"return_inputs": False, "return_outputs": True},
                    timeout=timeout_s,
                )
                if rr.status_code == 200:
                    break
                # Fetch status for debugging / recovery.
                try:
                    st = _get_poc_status(base_url, timeout_s=10)
                except Exception:
                    st = {}
                body_snip = (rr.text or "")[:500]
                # If PoC isn't generating, try to re-enter generating mode.
                if rr.status_code == 400 and st.get("state") != "GENERATING":
                    try:
                        requests.post(f"{base_url}/api/v1/pow/phase/generate_manual", json={}, timeout=30)
                        _wait_poc_state(base_url, want_state="GENERATING", timeout_s=60)
                    except Exception:
                        pass
                    time.sleep(0.3)
                    continue
                # If it's the known "must be GENERATING" message, just backoff + retry.
                if rr.status_code == 400 and "must be in GENERATING state" in body_snip:
                    time.sleep(0.3 + 0.2 * attempt)
                    continue
                raise RuntimeError(
                    f"/pow/batch failed: status={rr.status_code} body={body_snip} poc_state={st.get('state')}"
                )
            if rr is None or rr.status_code != 200:
                raise RuntimeError("Failed to obtain /pow/batch success after retries")

            payload = _extract_tensor(rr.json())
            x = (payload.get("artifacts") or {}).get("last_hidden_unit_pre")
            if x is None:
                raise RuntimeError("Missing artifacts.last_hidden_unit_pre")
            vecs.append(np.asarray(x.cpu().numpy(), dtype=np.float32))
            nonces_all.extend(list(payload.get("nonces", [])))
            batches += 1

        x_all = np.concatenate(vecs, axis=0)[:n]
        nonces_all = nonces_all[:n]

        np.save(raw_dir / "hidden_unit_vectors.npy", x_all.astype(np.float16))
        meta = {
            "model": model_name,
            "model_short": model_short,
            "vector_space": "full",
            "block_hash": block_hash,
            "public_key": public_key,
            "seq_len": int(seq_len),
            "batch_size": int(batch_size),
            "num_samples_requested": int(n),
            "num_samples_written": int(x_all.shape[0]),
            "hidden_size": int(x_all.shape[1]),
            "storage_dtype": "float16",
            "storage_format": "float16",
            "vectors_path": str((raw_dir / "hidden_unit_vectors.npy").resolve()),
            "nonces": nonces_all,
            "server_log": str(server_log.resolve()),
            "elapsed_sec_collect": float(time.time() - t0),
            "num_batches": int(batches),
            "poc_init": init_body,
        }
        (raw_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return raw_dir
    finally:
        try:
            requests.post(f"{base_url}/api/v1/pow/stop", json={}, timeout=10)
        except Exception:
            pass
        if server is not None:
            _stop_proc(server)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Folder name under scripts/poc_distribution/data/ to write into.",
    )
    p.add_argument("--n", type=int, required=True, help="Number of vectors to collect per model.")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--timeout", type=int, default=360)
    p.add_argument(
        "--data-root",
        type=str,
        default="scripts/poc_distribution/data",
        help="Base folder for experiments (default: scripts/poc_distribution/data).",
    )
    p.add_argument("--block-hash", type=str, default="exp_block")
    p.add_argument("--public-key", type=str, default="exp_pubkey")
    p.add_argument(
        "--models",
        nargs="*",
        default=[m for _, m in DEFAULT_MODELS],
        help="Which default model short names to run (default: all).",
    )
    # Keep explicit/overridable for reproducibility.
    p.add_argument("--poc-use-sign-flips", action="store_true", help="Forwarded to /pow/init.")
    p.add_argument(
        "--poc-no-nonce-householder",
        action="store_true",
        help="Disable nonce Householder in /pow/init (default: enabled).",
    )
    p.add_argument(
        "--poc-use-nonce-orthogonal",
        action="store_true",
        help="Enable nonce orthogonal transform in /pow/init (default: disabled).",
    )
    p.add_argument(
        "--poc-pick-k-dims",
        type=int,
        default=10,
        help="Forwarded to /pow/init (default: 10).",
    )
    args = p.parse_args()

    out_root = Path(args.data_root) / _safe_name(str(args.experiment))
    out_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.models)
    models = [(name, short) for name, short in DEFAULT_MODELS if short in selected]
    if not models:
        raise SystemExit("No models selected.")

    for model_name, model_short in models:
        model_short = _safe_name(model_short)
        print("=" * 80)
        print(f"[collect_hidden_vectors] collecting raw: experiment={args.experiment} model={model_short} n={args.n}")
        print("=" * 80)
        raw_dir = collect_hidden_vectors(
            model_name=model_name,
            model_short=model_short,
            out_dir=out_root,
            port=int(args.port),
            n=int(args.n),
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            gpu_mem_util=float(args.gpu_memory_utilization),
            max_model_len=int(args.max_model_len),
            block_hash=str(args.block_hash),
            public_key=str(args.public_key),
            timeout_s=int(args.timeout),
            poc_use_sign_flips=bool(args.poc_use_sign_flips),
            poc_use_nonce_householder=not bool(args.poc_no_nonce_householder),
            poc_use_nonce_orthogonal=bool(args.poc_use_nonce_orthogonal),
            poc_pick_k_dims=int(args.poc_pick_k_dims),
        )
        print(f"[collect_hidden_vectors] wrote: {raw_dir.resolve()}")

    print(f"[collect_hidden_vectors] done. outputs: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


