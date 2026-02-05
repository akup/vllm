#!/usr/bin/env python3
"""
Warm Linux page cache for a directory tree by reading files in parallel.
Use before starting vLLM so "Loading weights" reads from RAM instead of disk.

Usage:
  python3 warm-page-cache.py [--root /data/huggingface] [--workers 8] [--chunk-size 1048576]
  HF_HOME=/data/huggingface python3 warm-page-cache.py

Reads all files under --root in parallel (default 8 workers), in chunks (default 1 MiB),
so the kernel keeps them in page cache. Does not load files into Python memory.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# Extensions that are typically model weights (optional filter)
WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".msgpack"}


def get_files(root: Path, only_weights: bool) -> list[Path]:
    """List all regular files under root, optionally only weight-like extensions."""
    root = root.resolve()
    if not root.is_dir():
        return []
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            if only_weights and path.suffix.lower() not in WEIGHT_EXTENSIONS:
                continue
            files.append(path)
    return sorted(files)


def read_file_into_page_cache(
    path: Path,
    chunk_size: int,
    progress_ref: dict | None = None,
) -> tuple[Path, int]:
    """Read file in chunks to populate page cache; return (path, bytes_read).
    If progress_ref is set, update progress_ref['bytes_read'] after each chunk (with progress_ref['lock']).
    """
    total = 0
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if progress_ref is not None:
                    with progress_ref["lock"]:
                        progress_ref["bytes_read"] += len(chunk)
                # Touch the memory so the kernel keeps the pages
                _ = chunk[0]
    except (OSError, IOError):
        return (path, -1)
    return (path, total)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Warm Linux page cache by reading files under a directory in parallel."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data/huggingface")),
        help="Root directory to read (default: HF_HOME or /data/huggingface)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel readers (default: 8)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Read chunk size in bytes (default: 1 MiB)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Read all files; default is only weight-like extensions (.safetensors, .bin, .msgpack)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    # Unbuffer stdout so progress is visible when run from start.sh (non-TTY)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    t0 = time.perf_counter()
    print(f"Discovering files under {root}...", flush=True)
    files = get_files(root, only_weights=not args.all_files)
    t_after_rglob = time.perf_counter()
    if not files:
        print(f"No files found under {root}", file=sys.stderr)
        return 1

    total_size = sum(f.stat().st_size for f in files)
    t_after_stat = time.perf_counter()
    total_gib = total_size / (1024**3)
    discovery_s = t_after_rglob - t0
    stat_s = t_after_stat - t_after_rglob
    print(f"Discovery: {len(files)} files in {discovery_s:.1f}s, stat(sizes) in {stat_s:.1f}s", flush=True)
    print(f"Warming page cache: {len(files)} files, {total_gib:.1f} GiB, {args.workers} workers", flush=True)
    start = time.perf_counter()
    last_log_time = start
    last_log_gib = 0.0
    log_interval_gib = 1.0  # log every 1 GiB
    log_interval_sec = 5.0  # log at least every 5 seconds (timer thread)

    # Shared state: bytes_read (updated per chunk by workers), start, total_size, done, lock
    progress = {
        "bytes_read": 0,
        "start": start,
        "total_size": total_size,
        "done": False,
        "lock": threading.Lock(),
    }

    def log_progress_every_5s() -> None:
        while not progress["done"]:
            time.sleep(log_interval_sec)
            if progress["done"]:
                return
            b = progress["bytes_read"]
            elapsed = time.perf_counter() - progress["start"]
            if elapsed > 0:
                gib = b / (1024**3)
                pct = 100.0 * b / progress["total_size"] if progress["total_size"] else 0
                mibs = (b / (1024**2)) / elapsed
                print(f"  Progress: {gib:.1f} GiB read and cached ({pct:.0f}%), {mibs:.0f} MiB/s, elapsed {elapsed:.0f}s", flush=True)

    timer = threading.Thread(target=log_progress_every_5s, daemon=True)
    timer.start()

    read_count = 0
    error_count = 0
    bytes_read = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(read_file_into_page_cache, p, args.chunk_size, progress): p
            for p in files
        }
        for future in as_completed(futures):
            path, n = future.result()
            if n >= 0:
                read_count += 1
                bytes_read += n
                now = time.perf_counter()
                gib_read = bytes_read / (1024**3)
                elapsed = now - start
                if gib_read - last_log_gib >= log_interval_gib:
                    pct = 100.0 * bytes_read / total_size if total_size else 0
                    mibs = (bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {gib_read:.1f} GiB read and cached ({pct:.0f}%), {mibs:.0f} MiB/s, elapsed {elapsed:.0f}s", flush=True)
                    last_log_gib = gib_read
                    last_log_time = now
            else:
                error_count += 1
                print(f"  Warning: failed to read {path}", file=sys.stderr)

    progress["done"] = True
    elapsed = time.perf_counter() - start
    gib_read = bytes_read / (1024**3)
    throughput_mibs = (bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0
    print(f"Done: {gib_read:.1f} GiB read and cached in {elapsed:.1f}s total ({throughput_mibs:.0f} MiB/s), {read_count} files read, {error_count} errors", flush=True)
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
