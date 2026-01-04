#!/usr/bin/env python3
"""
Data collection script for PoW experiments.

Collects nonces, distances, and vectors from multiple servers.

Usage:
    python scripts/collect_data.py --name my_experiment --config configs/servers_4.json
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import requests


def api_call(url: str, endpoint: str, method: str = "POST", json_data: dict = None) -> dict:
    """Make API call to server."""
    full_url = f"{url}{endpoint}"
    if method == "GET":
        r = requests.get(full_url, timeout=30)
    else:
        r = requests.post(full_url, json=json_data, timeout=300)
    r.raise_for_status()
    return r.json()


def collect_from_server(name: str, url: str, config: dict) -> dict:
    """Collect data from a single server."""
    # Stop any running generation
    try:
        api_call(url, "/api/v1/pow/stop")
    except Exception:
        pass  # Ignore if nothing running

    # Build generation config
    gen_config = {
        "block_hash": config["block_hash"],
        "block_height": config.get("block_height", 100),
        "public_key": config["public_key"],
        "r_target": config["r_target"],
        "node_id": 0,
        "node_count": 1,
        "seq_len": config.get("seq_len", 128),
        "batch_size": config.get("batch_size", 128),
        "nonces": list(range(config.get("nonce_count", 500))),
        "wait": True,
        "return_vectors": True,
    }

    # Generate
    result = api_call(url, "/api/v1/pow/generate", json_data=gen_config)

    return {
        "server_name": name,
        "server_url": url,
        "nonces": result.get("valid_nonces", []),
        "distances": result.get("valid_distances", []),
        "vectors": result.get("vectors", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect PoW data from multiple servers")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("logs") / f"{args.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy
    shutil.copy(config_path, out_dir / "config.json")

    print(f"Output: {out_dir}")
    print(f"Servers: {list(config['servers'].keys())}")
    print()

    # Collect from each server
    for name, url in config["servers"].items():
        print(f"Collecting from {name} ({url})...", end=" ", flush=True)
        try:
            result = collect_from_server(name, url, config)
            
            # Save result
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"OK ({len(result['nonces'])} nonces, {len(result['vectors'])} vectors)")
        except Exception as e:
            print(f"FAILED: {e}")
            # Save error
            with open(out_dir / f"{name}.json", "w") as f:
                json.dump({"server_name": name, "server_url": url, "error": str(e)}, f, indent=2)

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()

