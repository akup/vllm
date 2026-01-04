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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


def api_call(url: str, endpoint: str, method: str = "POST", json_data: dict = None) -> dict:
    """Make API call to server."""
    full_url = f"{url}{endpoint}"
    if method == "GET":
        r = requests.get(full_url, timeout=30)
    else:
        r = requests.post(full_url, json=json_data, timeout=600)
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


def find_latest_run(name: str) -> Path | None:
    """Find the most recent output directory for given experiment name."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # Find directories matching pattern: {name}_{timestamp}
    matching = sorted(
        [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{name}_")],
        key=lambda d: d.name,
        reverse=True
    )
    return matching[0] if matching else None


def get_completed_servers(out_dir: Path) -> set[str]:
    """Get set of server names that have successful data (not errors)."""
    completed = set()
    for json_file in out_dir.glob("*.json"):
        if json_file.name == "config.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Check if it has actual data (not an error)
            if "error" not in data and data.get("nonces"):
                completed.add(json_file.stem)
        except Exception:
            pass
    return completed


def main():
    parser = argparse.ArgumentParser(description="Collect PoW data from multiple servers")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Continue from last run, skipping completed servers")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # Determine output directory
    if args.continue_run:
        out_dir = find_latest_run(args.name)
        if out_dir is None:
            print(f"No previous run found for '{args.name}', starting fresh")
            args.continue_run = False
    
    if not args.continue_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("logs") / f"{args.name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save config copy
        shutil.copy(config_path, out_dir / "config.json")
    
    # Get already completed servers if continuing
    completed = get_completed_servers(out_dir) if args.continue_run else set()

    # Group by URL (dedupe same address:port)
    url_to_names = {}
    for name, url in config["servers"].items():
        if name not in completed:
            url_to_names.setdefault(url, []).append(name)

    print(f"Output: {out_dir}")
    print(f"Servers: {list(config['servers'].keys())}")
    if completed:
        print(f"Skipping (already done): {sorted(completed)}")
    print(f"Unique URLs to query: {len(url_to_names)}")
    print()

    def collect_and_save(url, names):
        """Collect from one URL and save for all names sharing it."""
        try:
            result = collect_from_server(names[0], url, config)
            # Save for all names that share this URL
            for name in names:
                result_copy = {**result, "server_name": name}
                with open(out_dir / f"{name}.json", "w") as f:
                    json.dump(result_copy, f, indent=2)
            return url, names, len(result["nonces"]), len(result["vectors"]), None
        except Exception as e:
            for name in names:
                with open(out_dir / f"{name}.json", "w") as f:
                    json.dump({"server_name": name, "server_url": url, "error": str(e)}, f, indent=2)
            return url, names, 0, 0, str(e)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=len(url_to_names) or 1) as executor:
        futures = {
            executor.submit(collect_and_save, url, names): url
            for url, names in url_to_names.items()
        }
        for future in as_completed(futures):
            url, names, nonce_count, vector_count, error = future.result()
            if error:
                print(f"{names} ({url}): FAILED - {error}")
            else:
                print(f"{names} ({url}): OK ({nonce_count} nonces, {vector_count} vectors)")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()

