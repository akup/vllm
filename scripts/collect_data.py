#!/usr/bin/env python3
"""
Data collection script for PoW experiments.

Collects nonces, distances, and vectors from multiple servers.

Supports:
- Single seed: block_hash + public_key
- Multi-seed: block_hashes (list) + public_keys (list) -> iterates over all combinations

Usage:
    python scripts/collect_data.py --name my_experiment --config configs/servers_4.json
"""

import argparse
import itertools
import json
import shutil
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# Global shutdown event for Ctrl-C handling
shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Handle Ctrl-C by setting shutdown event."""
    print("\n\nInterrupt received, shutting down...")
    shutdown_event.set()


def api_call(url: str, endpoint: str, method: str = "POST", json_data: dict = None) -> dict:
    """Make API call to server."""
    full_url = f"{url}{endpoint}"
    if method == "GET":
        r = requests.get(full_url, timeout=30)
    else:
        r = requests.post(full_url, json=json_data, timeout=600)
    r.raise_for_status()
    return r.json()


def collect_from_server(name: str, url: str, config: dict, block_hash: str, public_key: str) -> dict:
    """Collect data from a single server for a specific seed."""
    # Stop any running generation
    try:
        api_call(url, "/api/v1/pow/stop")
    except Exception:
        pass  # Ignore if nothing running

    # Build generation config
    gen_config = {
        "block_hash": block_hash,
        "block_height": config.get("block_height", 100),
        "public_key": public_key,
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
        "block_hash": block_hash,
        "public_key": public_key,
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


def get_completed_tasks(out_dir: Path) -> set[str]:
    """Get set of completed task keys (filename stems) that have successful data."""
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


def get_output_filename(server_name: str, block_hash: str, public_key: str, multi_seed: bool) -> str:
    """Generate output filename based on seed mode."""
    if multi_seed:
        return f"{server_name}_{block_hash}_{public_key}.json"
    else:
        return f"{server_name}.json"


def get_task_key(server_name: str, block_hash: str, public_key: str, multi_seed: bool) -> str:
    """Generate task key for tracking completion."""
    if multi_seed:
        return f"{server_name}_{block_hash}_{public_key}"
    else:
        return server_name


def main():
    # Register signal handler for Ctrl-C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Collect PoW data from multiple servers")
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--continue", dest="continue_run", action="store_true",
                        help="Continue from last run, skipping completed tasks")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path) as f:
        config = json.load(f)

    # Resolve seeds - support both single and multi-seed configs
    if "block_hashes" in config:
        block_hashes = config["block_hashes"]
    else:
        block_hashes = [config["block_hash"]]
    
    if "public_keys" in config:
        public_keys = config["public_keys"]
    else:
        public_keys = [config["public_key"]]
    
    seeds = list(itertools.product(block_hashes, public_keys))
    multi_seed = len(seeds) > 1

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
    
    # Get already completed tasks if continuing
    completed = get_completed_tasks(out_dir) if args.continue_run else set()

    # Build task list grouped by URL
    # Each URL gets all its (server_name, block_hash, public_key) tasks
    url_to_tasks = {}
    for name, url in config["servers"].items():
        for block_hash, public_key in seeds:
            task_key = get_task_key(name, block_hash, public_key, multi_seed)
            if task_key not in completed:
                url_to_tasks.setdefault(url, []).append((name, block_hash, public_key))

    total_tasks = sum(len(tasks) for tasks in url_to_tasks.values())
    
    print(f"Output: {out_dir}")
    print(f"Servers: {list(config['servers'].keys())}")
    print(f"Seeds: {len(seeds)} combinations")
    if multi_seed:
        print(f"  block_hashes: {block_hashes}")
        print(f"  public_keys: {public_keys}")
    print(f"Total tasks: {total_tasks}")
    print(f"Workers: {len(url_to_tasks)} (one per URL)")
    if completed:
        print(f"Skipping (already done): {len(completed)} tasks")
    print()

    def collect_all_seeds_for_url(url, task_list):
        """Process all seeds for one URL sequentially. Returns list of results."""
        results = []
        for name, block_hash, public_key in task_list:
            # Check for shutdown before each task
            if shutdown_event.is_set():
                break
            
            task_key = get_task_key(name, block_hash, public_key, multi_seed)
            filename = get_output_filename(name, block_hash, public_key, multi_seed)
            
            try:
                result = collect_from_server(name, url, config, block_hash, public_key)
                with open(out_dir / filename, "w") as f:
                    json.dump(result, f, indent=2)
                results.append((name, block_hash, public_key, len(result["nonces"]), len(result["vectors"]), None))
            except Exception as e:
                if shutdown_event.is_set():
                    break
                error_result = {
                    "server_name": name,
                    "server_url": url,
                    "block_hash": block_hash,
                    "public_key": public_key,
                    "error": str(e),
                }
                with open(out_dir / filename, "w") as f:
                    json.dump(error_result, f, indent=2)
                results.append((name, block_hash, public_key, 0, 0, str(e)))
        
        return url, results

    if not url_to_tasks:
        print("No tasks to run.")
        return

    # Parallel execution - one worker per URL
    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=len(url_to_tasks)) as executor:
            futures = {
                executor.submit(collect_all_seeds_for_url, url, task_list): url
                for url, task_list in url_to_tasks.items()
            }
            
            for future in as_completed(futures):
                if shutdown_event.is_set():
                    interrupted = True
                    break
                    
                url, results = future.result()
                for name, block_hash, public_key, nonce_count, vector_count, error in results:
                    seed_str = f" [{block_hash}+{public_key}]" if multi_seed else ""
                    if error:
                        print(f"{name}{seed_str}: FAILED - {error}")
                    else:
                        print(f"{name}{seed_str}: OK ({nonce_count} nonces, {vector_count} vectors)")
    except KeyboardInterrupt:
        interrupted = True
        shutdown_event.set()
        print("\n\nInterrupt received, cancelling pending tasks...")

    if interrupted:
        print(f"\nInterrupted. Partial results in {out_dir}")
        print("Use --continue to resume from where you left off.")
        sys.exit(1)
    else:
        print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
