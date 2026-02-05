#!/usr/bin/env bash
# Download vLLM cache tarball from S3 and extract to /data/vllm-cache.
#
# Usage:
#   download-vllm-cache-from-s3.sh MODEL_NAME [S3_URI]
#   download-vllm-cache-from-s3.sh s3://bucket/path/vllm-cache.tar.gz
#
# If S3_URI is omitted, it is built as s3://VLLM_CACHE_S3_BUCKET/vllm-cache/MODEL_SLUG/TP{N}/vllm-cache.tar.gz
# (same bucket as upload-vllm-cache-to-s3.sh; default bucket gonka-vllm-cache).
#
# Instance IAM profile needs s3:GetObject (and s3:ListBucket) on the bucket; see upload-vllm-cache-to-s3.sh header.
#
# Requires: AWS CLI. Extracts to DEST_DIR (default /data) so final path is /data/vllm-cache.
set -e

# Same hardcoded default as upload-vllm-cache-to-s3.sh
VLLM_CACHE_S3_BUCKET="${VLLM_CACHE_S3_BUCKET:-gonka-vllm-cache}"

DEST_DIR="${DEST_DIR:-/data}"
# Final cache path after extract (tarball must contain top-level "vllm-cache" dir)
VLLM_CACHE_PATH="${DEST_DIR}/vllm-cache"

if [ $# -eq 0 ]; then
    echo "Usage: $0 MODEL_NAME [S3_URI]" >&2
    echo "   or: $0 s3://bucket/path/vllm-cache.tar.gz" >&2
    echo "" >&2
    echo "Extracts to $VLLM_CACHE_PATH. Set DEST_DIR to change (e.g. DEST_DIR=/data)." >&2
    exit 1
fi

# If first arg looks like s3://..., use it as S3_URI and no MODEL_NAME
if [[ "$1" == s3://* ]]; then
    S3_URI="$1"
else
    MODEL_NAME="$1"
    S3_URI="${2:-}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    MODEL_SLUG="${MODEL_NAME//\//-}"
    if [ -z "$S3_URI" ]; then
        S3_URI="s3://${VLLM_CACHE_S3_BUCKET}/vllm-cache/${MODEL_SLUG}/TP${TENSOR_PARALLEL_SIZE}/vllm-cache.tar.gz"
    fi
fi

echo "[$(date +%H:%M:%S)] Downloading $S3_URI to $VLLM_CACHE_PATH"
mkdir -p "$DEST_DIR"

# Stream from S3 and extract; tarball has top-level "vllm-cache" so -C /data gives /data/vllm-cache
aws s3 cp "$S3_URI" - | tar -xzf - -C "$DEST_DIR"

if [ ! -d "$VLLM_CACHE_PATH" ]; then
    echo "ERROR: After extract, $VLLM_CACHE_PATH not found (tarball should contain top-level 'vllm-cache' dir)" >&2
    exit 1
fi

echo "[$(date +%H:%M:%S)] Cache extracted to $VLLM_CACHE_PATH"
echo "Set in your env: export VLLM_CACHE_ROOT=$VLLM_CACHE_PATH"
