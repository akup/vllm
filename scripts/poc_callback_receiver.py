#!/usr/bin/env python3
"""Simple callback receiver for testing PoC artifact batches.

This server receives callbacks from vLLM PoC and logs them.
Run: python scripts/poc_callback_receiver.py --port 8081
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
import uvicorn

# Force unbuffered output for logging
def log(msg: str):
    print(msg, flush=True)

app = FastAPI(title="PoC Callback Receiver")

# In-memory storage for received batches
received_batches: List[dict] = []
validated_batches: List[dict] = []
stats = {
    "total_generated_callbacks": 0,
    "total_artifacts": 0,
    "total_validated_callbacks": 0,
    "total_mismatches": 0,
}


@app.post("/generated")
async def receive_generated(request: Request) -> dict:
    """Receive a generated artifact batch from vLLM PoC.
    
    Payload format:
    {
        "public_key": "...",
        "block_hash": "...",
        "block_height": 100,
        "node_id": 0,
        "artifacts": [{"nonce": 0, "vector_b64": "..."}, ...],
        "encoding": {"dtype": "f16", "k_dim": 12, "endian": "le"}
    }
    """
    body = await request.json()
    
    timestamp = datetime.now().isoformat()
    batch = {
        "timestamp": timestamp,
        "data": body,
    }
    
    received_batches.append(batch)
    stats["total_generated_callbacks"] += 1
    
    artifacts = body.get("artifacts", [])
    stats["total_artifacts"] += len(artifacts)
    
    # Log the batch
    encoding = body.get("encoding", {})
    log(f"[{timestamp}] Received {len(artifacts)} artifacts:")
    log(f"  Block: {body.get('block_hash', 'N/A')[:16]}...")
    log(f"  Public key: {body.get('public_key', 'N/A')}")
    log(f"  Node ID: {body.get('node_id', 'N/A')}")
    log(f"  Encoding: dtype={encoding.get('dtype')}, k_dim={encoding.get('k_dim')}, endian={encoding.get('endian')}")
    if artifacts:
        log(f"  First nonce: {artifacts[0].get('nonce')}, Last nonce: {artifacts[-1].get('nonce')}")
    log(f"  Total artifacts so far: {stats['total_artifacts']}")
    log("")
    
    return {"status": "OK", "received": len(artifacts)}


@app.post("/validated")
async def receive_validated(request: Request) -> dict:
    """Receive a validation result from vLLM PoC.
    
    Payload format:
    {
        "request_id": "...",
        "block_hash": "...",
        "block_height": 100,
        "public_key": "...",
        "node_id": 0,
        "n_total": 100,
        "n_mismatch": 2,
        "mismatch_nonces": [5, 42],
        "p_value": 0.001,
        "fraud_detected": true
    }
    """
    body = await request.json()
    
    timestamp = datetime.now().isoformat()
    validated_batches.append({
        "timestamp": timestamp,
        "data": body,
    })
    
    stats["total_validated_callbacks"] += 1
    stats["total_mismatches"] += body.get("n_mismatch", 0)
    
    log(f"[{timestamp}] Received VALIDATION result:")
    log(f"  Request ID: {body.get('request_id', 'N/A')}")
    log(f"  Block: {body.get('block_hash', 'N/A')[:16]}...")
    log(f"  Public key: {body.get('public_key', 'N/A')}")
    log(f"  n_total: {body.get('n_total', 'N/A')}")
    log(f"  n_mismatch: {body.get('n_mismatch', 'N/A')}")
    log(f"  p_value: {body.get('p_value', 'N/A')}")
    log(f"  fraud_detected: {body.get('fraud_detected', 'N/A')}")
    if body.get("mismatch_nonces"):
        log(f"  mismatch_nonces: {body.get('mismatch_nonces')}")
    log("")
    
    return {"status": "OK"}


@app.get("/batches")
async def get_batches() -> dict:
    """Get all received batches."""
    return {
        "generated": received_batches,
        "validated": validated_batches,
        "stats": stats,
    }


@app.get("/stats")
async def get_stats() -> dict:
    """Get batch statistics."""
    return stats


@app.delete("/clear")
async def clear_batches() -> dict:
    """Clear all received batches."""
    global received_batches, validated_batches, stats
    received_batches = []
    validated_batches = []
    stats = {
        "total_generated_callbacks": 0,
        "total_artifacts": 0,
        "total_validated_callbacks": 0,
        "total_mismatches": 0,
    }
    return {"status": "cleared"}


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "healthy"}


@app.post("/save")
async def save_batches(request: Request) -> dict:
    """Save received batches to a file in logs/v2/callbacks/."""
    body = await request.json()
    filename = body.get("filename", f"batches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    output_dir = Path("logs/v2/callbacks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump({
            "generated": received_batches,
            "validated": validated_batches,
            "stats": stats,
        }, f, indent=2)
    
    log(f"Saved batches to {filepath}")
    return {"status": "saved", "path": str(filepath)}


def main():
    parser = argparse.ArgumentParser(description="PoC Callback Receiver")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081,
                        help="Port to listen on")
    args = parser.parse_args()
    
    log(f"Starting PoC Callback Receiver on {args.host}:{args.port}")
    log(f"Callback URL: http://localhost:{args.port}")
    log(f"  POST /generated - receive artifact batches")
    log(f"  POST /validated - receive validation results")
    log(f"  GET  /batches   - get all received batches")
    log(f"  GET  /stats     - get statistics")
    log(f"  POST /save      - save batches to logs/v2/callbacks/")
    log(f"  DELETE /clear   - clear all batches")
    log("")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
