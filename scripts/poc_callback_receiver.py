#!/usr/bin/env python3
"""Simple callback receiver for testing PoC batches.

This server receives callbacks from vLLM PoC sender and logs them.
Run: python scripts/poc_callback_receiver.py --port 8081
"""
import argparse
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(title="PoC Callback Receiver")

# In-memory storage for received batches
received_batches: List[dict] = []
stats = {
    "total_batches": 0,
    "total_nonces": 0,
    "total_valid": 0,
}


@app.post("/callback/batch")
async def receive_batch(request: Request) -> dict:
    """Receive a generated/validated batch from vLLM PoC."""
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
    r_target = body.get("r_target", 1.0)
    
    stats["total_nonces"] += len(nonces)
    
    valid_count = sum(1 for d in distances if d < r_target)
    stats["total_valid"] += valid_count
    
    # Log the batch
    print(f"[{timestamp}] Received batch:")
    print(f"  Block: {body.get('block_hash', 'N/A')[:16]}...")
    print(f"  Nonces: {len(nonces)} | Valid: {valid_count}")
    if distances:
        print(f"  Distance range: [{min(distances):.4f}, {max(distances):.4f}]")
    print(f"  Stats: {stats['total_valid']}/{stats['total_nonces']} valid so far")
    print()
    
    return {"status": "OK", "received": len(nonces)}


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
    
    print(f"Starting PoC Callback Receiver on {args.host}:{args.port}")
    print(f"Callback URL: http://localhost:{args.port}/callback/batch")
    print()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

