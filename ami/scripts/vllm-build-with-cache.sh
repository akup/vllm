#!/usr/bin/env bash
# Build vLLM (including flash-attn) or install from S3 cache.
# Set VLLM_BUILD_CACHE_BUCKET to reuse a pre-built wheel from a previous Packer run.
# Requires: venv active, torch installed, /tmp/vllm-src with vllm source.
# Optional: VLLM_BUILD_VERSION (default 0.9.1), PYVER (default 3.11), VLLM_BUILD_VERBOSE (default 1; set 0 to quiet).

set -e
VERSION="${VLLM_BUILD_VERSION:-0.9.1}"
PYVER="${PYVER:-3.11}"
CACHE_KEY="vllm-${VERSION}-cu128-py${PYVER}.whl"
WHEELS_DIR="/tmp/vllm-wheels"
MAX_RETRIES="${VLLM_BUILD_MAX_RETRIES:-10}"
mkdir -p "$WHEELS_DIR"

# Try cache download if bucket set (reuse wheel from a previous Packer run)
if [ -n "${VLLM_BUILD_CACHE_BUCKET}" ]; then
  if ! command -v aws &>/dev/null; then
    echo "WARNING: aws CLI not found; install it (e.g. sudo dnf install -y aws-cli) and ensure instance has S3 access (IAM role). Skipping cache."
  else
    PREFIX="${VLLM_BUILD_CACHE_PREFIX:-vllm-wheels}"
    S3_URI="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${CACHE_KEY}"
    if aws s3 cp "$S3_URI" "$WHEELS_DIR/$CACHE_KEY" 2>/dev/null; then
      echo "vLLM cache hit: $S3_URI (reusing artifact from previous run)"
      pip install "$WHEELS_DIR/$CACHE_KEY"
      echo "vLLM installed from cache."
      exit 0
    fi
    echo "vLLM cache miss; building (final wheel will be uploaded to S3 on success)..."
  fi
fi

# Build wheel then install (resumable: retry without cleaning so Ninja continues)
cd /tmp/vllm-src
export SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}"
# Build verbosity: VERBOSE=1 => CMAKE_VERBOSE_MAKEFILE=ON (vLLM setup.py)
export VERBOSE="${VLLM_BUILD_VERBOSE:-1}"
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
  export PYTHONUNBUFFERED=1
  if pip wheel -v --no-build-isolation -w "$WHEELS_DIR" . 2>&1 | tee /tmp/vllm-wheel.log; then
    WHEEL=$(ls "$WHEELS_DIR"/vllm-*.whl 2>/dev/null | head -1)
    if [ -z "$WHEEL" ]; then
      echo "ERROR: No wheel produced."
      tail -40 /tmp/vllm-wheel.log
      exit 1
    fi
    pip install "$WHEEL"
    echo "vLLM installed from local build."

    # Upload final wheel so next run can use it
    if [ -n "${VLLM_BUILD_CACHE_BUCKET}" ] && command -v aws &>/dev/null; then
      PREFIX="${VLLM_BUILD_CACHE_PREFIX:-vllm-wheels}"
      S3_URI="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${CACHE_KEY}"
      echo "Uploading wheel to $S3_URI for future runs..."
      aws s3 cp "$WHEEL" "$S3_URI"
      echo "Cache updated."
    fi
    exit 0
  fi

  RETRY=$((RETRY + 1))
  echo "Last 60 lines of /tmp/vllm-wheel.log:"
  tail -60 /tmp/vllm-wheel.log
  if [ $RETRY -lt $MAX_RETRIES ]; then
    echo "Retry $RETRY/$MAX_RETRIES: re-running build without cleaning (Ninja will resume)..."
    sleep 10
  else
    echo "ERROR: vLLM build failed after $MAX_RETRIES retries."
    exit 1
  fi
done
exit 1
