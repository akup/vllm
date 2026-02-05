#!/usr/bin/env bash
# Warm Linux page cache for Hugging Face model files under HF_HOME (default /data/huggingface)
# by reading them in parallel. Run before starting vLLM to make "Loading weights" fast (from RAM).
#
# Usage:
#   ./warm-page-cache.sh
#   HF_HOME=/data/huggingface ./warm-page-cache.sh
#   PARALLEL_READERS=16 ./warm-page-cache.sh
#
# Requires: Python 3, path to warm-page-cache.py (same directory or in PATH)
set -e

HF_HOME="${HF_HOME:-/data/huggingface}"
PARALLEL_READERS="${PARALLEL_READERS:-8}"
SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$0")}"

if [ ! -d "$HF_HOME" ]; then
    echo "ERROR: HF_HOME not a directory: $HF_HOME" >&2
    exit 1
fi

echo "Warming page cache for $HF_HOME (parallel readers: $PARALLEL_READERS)"
if [ -f "$SCRIPT_DIR/warm-page-cache.py" ]; then
    python3 "$SCRIPT_DIR/warm-page-cache.py" --root "$HF_HOME" --workers "$PARALLEL_READERS"
else
    python3 warm-page-cache.py --root "$HF_HOME" --workers "$PARALLEL_READERS"
fi
