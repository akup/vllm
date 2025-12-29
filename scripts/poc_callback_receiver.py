#!/usr/bin/env python3
"""Simple callback receiver for testing PoC batches.

This server receives callbacks from vLLM PoC sender and logs them.
Run: python scripts/poc_callback_receiver.py --port 8081
"""
import argparse
import json
import sys
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request
import uvicorn

# Force unbuffered output for logging
def log(msg: str):
    print(msg, flush=True)

app = FastAPI(title="PoC Callback Receiver")

# In-memory storage for received batches
received_batches: List[dict] = []
stats = {
    "total_batches": 0,
    "total_nonces": 0,
    "total_valid": 0,
}


@app.post("/generated")
async def receive_generated(request: Request) -> dict:
    """Receive a generated batch from vLLM PoC.
    
    Note: This batch already contains only VALID nonces (filtered by sub_batch).
    We receive nonces that passed r_target threshold on the server.
    """
    body = await request.json()
    
    timestamp = datetime.now().isoformat()
    batch = {
        "timestamp": timestamp,
        "data": body,
    }
    
    received_batches.append(batch)
    stats["total_batches"] += 1
    
    nonces = body.get("nonces", [])
    distances = body.get("dist", [])
    
    # All nonces in this batch are valid (already filtered by server)
    stats["total_nonces"] += len(nonces)
    stats["total_valid"] += len(nonces)
    
    # Log the batch
    r_target = body.get('r_target', 'N/A')
    log(f"[{timestamp}] Received {len(nonces)} valid nonces:")
    log(f"  Block: {body.get('block_hash', 'N/A')[:16]}...")
    log(f"  Public key: {body.get('public_key', 'N/A')}")
    log(f"  r_target: {r_target}")
    if distances:
        log(f"  Distance range: [{min(distances):.4f}, {max(distances):.4f}]")
    log(f"  Total valid so far: {stats['total_valid']}")
    log("")
    
    return {"status": "OK", "received": len(nonces)}


@app.post("/validated")
async def receive_validated(request: Request) -> dict:
    """Receive a validated batch from vLLM PoC (matching original API)."""
    body = await request.json()
    
    timestamp = datetime.now().isoformat()
    
    log(f"[{timestamp}] Received VALIDATED batch:")
    log(f"  Block: {body.get('block_hash', 'N/A')[:16]}...")
    log(f"  Public key: {body.get('public_key', 'N/A')}")
    log(f"  Nonces: {len(body.get('nonces', []))}")
    log(f"  Fraud detected: {body.get('fraud_detected', 'N/A')}")
    log("")
    
    return {"status": "OK"}


@app.get("/batches")
async def get_batches() -> dict:
    """Get all received batches."""
    return {
        "batches": received_batches,
        "stats": stats,
    }


@app.get("/stats")
async def get_stats() -> dict:
    """Get batch statistics."""
    return stats


@app.delete("/clear")
async def clear_batches() -> dict:
    """Clear all received batches."""
    global received_batches, stats
    received_batches = []
    stats = {
        "total_batches": 0,
        "total_nonces": 0,
        "total_valid": 0,
    }
    return {"status": "cleared"}


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "healthy"}


def main():
    parser = argparse.ArgumentParser(description="PoC Callback Receiver")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081,
                        help="Port to listen on")
    args = parser.parse_args()
    
    log(f"Starting PoC Callback Receiver on {args.host}:{args.port}")
    log(f"Callback URL: http://localhost:{args.port}")
    log(f"  POST /generated - receive generated batches")
    log(f"  POST /validated - receive validated batches")
    log("")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

