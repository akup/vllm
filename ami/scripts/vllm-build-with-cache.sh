#!/usr/bin/env bash
# Build vLLM (including flash-attn) or install from S3 cache.
# Set VLLM_BUILD_CACHE_BUCKET to reuse a pre-built wheel or intermediate build state.
# - Final wheel cache: successful builds; later runs install from wheel.
# - Intermediate cache: uploaded to S3 every 10 min during build; on connection loss, next run restores and resumes.
# Requires: venv active, torch installed, /tmp/vllm-src with vllm source.
# Optional: VLLM_BUILD_VERSION (default 0.9.1), PYVER (default 3.11),
#   VLLM_INTERMEDIATE_UPLOAD_INTERVAL (default 600 = 10 min), VLLM_BUILD_VERBOSE (default 1).

set -e
VERSION="${VLLM_BUILD_VERSION:-0.9.1}"
PYVER="${PYVER:-3.11}"
CACHE_KEY="vllm-${VERSION}-cu128-py${PYVER}.whl"
INTERMEDIATE_KEY="vllm-intermediate-${VERSION}-cu128-py${PYVER}.tar.gz"
WHEELS_DIR="/tmp/vllm-wheels"
MAX_RETRIES="${VLLM_BUILD_MAX_RETRIES:-10}"
UPLOAD_INTERVAL="${VLLM_INTERMEDIATE_UPLOAD_INTERVAL:-600}"
UPLOAD_PID=""
mkdir -p "$WHEELS_DIR"

cleanup_background_upload() {
  if [ -n "$UPLOAD_PID" ] && kill -0 "$UPLOAD_PID" 2>/dev/null; then
    kill "$UPLOAD_PID" 2>/dev/null || true
    wait "$UPLOAD_PID" 2>/dev/null || true
  fi
}
trap cleanup_background_upload EXIT

upload_intermediate_to_s3() {
  [ -z "${VLLM_BUILD_CACHE_BUCKET}" ] || ! command -v aws &>/dev/null && return 0
  [ ! -d /tmp/vllm-src ] && return 0
  PREFIX="${VLLM_BUILD_CACHE_PREFIX:-vllm-wheels}"
  S3_URI="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${INTERMEDIATE_KEY}"
  TARFILE="/tmp/intermediate-upload.tar.gz"
  rm -f "$TARFILE"
  ( nice tar -C / -czf "$TARFILE" tmp/vllm-src 2>/dev/null || true
    if [ -s "$TARFILE" ]; then
      SIZE=$(du -h "$TARFILE" | cut -f1)
      if nice aws s3 cp "$TARFILE" "$S3_URI"; then
        echo "[$(date -Iseconds)] Cache upload (intermediate, size $SIZE): $S3_URI"
      fi
    fi
    rm -f "$TARFILE"
  )
}

# Try final wheel from S3 (reuse from a previous successful run)
if [ -n "${VLLM_BUILD_CACHE_BUCKET}" ]; then
  if ! command -v aws &>/dev/null; then
    echo "WARNING: aws CLI not found; S3 cache disabled."
  else
    PREFIX="${VLLM_BUILD_CACHE_PREFIX:-vllm-wheels}"
    S3_URI="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${CACHE_KEY}"
    echo "Trying to load final wheel cache: $S3_URI"
    if aws s3 cp "$S3_URI" "$WHEELS_DIR/$CACHE_KEY" 2>/dev/null; then
      echo "Cache hit (final wheel): $S3_URI"
      pip install "$WHEELS_DIR/$CACHE_KEY"
      echo "vLLM installed from cache."
      exit 0
    fi
    # No final wheel; try intermediate cache (resume after connection loss)
    S3_INTERMEDIATE="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${INTERMEDIATE_KEY}"
    echo "Trying to load intermediate cache: $S3_INTERMEDIATE"
    if aws s3 cp "$S3_INTERMEDIATE" /tmp/intermediate.tar.gz 2>/dev/null; then
      SIZE=$(du -h /tmp/intermediate.tar.gz | cut -f1)
      echo "Cache hit (intermediate, size $SIZE): $S3_INTERMEDIATE"
      rm -rf /tmp/vllm-src
      tar -xzf /tmp/intermediate.tar.gz -C /
      rm -f /tmp/intermediate.tar.gz
      echo "Restored /tmp/vllm-src; build will resume."
    fi
    echo "vLLM cache miss; building (final wheel + intermediate state every ${UPLOAD_INTERVAL}s will be uploaded to S3)..."
  fi
fi

# Build wheel then install (resumable: retry without cleaning so Ninja continues)
cd /tmp/vllm-src
export SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}"
export VERBOSE="${VLLM_BUILD_VERBOSE:-1}"
RETRY=0
export PYTHONUNBUFFERED=1
while [ $RETRY -lt $MAX_RETRIES ]; do
  # Background: upload intermediate state to S3 every 10 min (echo to console)
  if [ -n "${VLLM_BUILD_CACHE_BUCKET}" ] && command -v aws &>/dev/null; then
    (
      while true; do
        sleep "$UPLOAD_INTERVAL"
        echo "[$(date -Iseconds)] Uploading intermediate build state..."
        upload_intermediate_to_s3
      done
    ) &
    UPLOAD_PID=$!
    echo "Background intermediate upload every ${UPLOAD_INTERVAL}s (PID $UPLOAD_PID)."
  fi

  PIP_VERBOSE_FLAG=''; [ "${VLLM_BUILD_VERBOSE:-1}" = '2' ] && PIP_VERBOSE_FLAG='-v'
  pip wheel $PIP_VERBOSE_FLAG --no-build-isolation -w "$WHEELS_DIR" . 2>&1 | tee /tmp/vllm-wheel.log | grep -v "Skipping link" || true
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    cleanup_background_upload
    WHEEL=$(ls "$WHEELS_DIR"/vllm-*.whl 2>/dev/null | head -1)
    if [ -z "$WHEEL" ]; then
      echo "ERROR: No wheel produced."
      tail -40 /tmp/vllm-wheel.log
      exit 1
    fi
    pip install "$WHEEL"
    echo "vLLM installed from local build."

    if [ -n "${VLLM_BUILD_CACHE_BUCKET}" ] && command -v aws &>/dev/null; then
      PREFIX="${VLLM_BUILD_CACHE_PREFIX:-vllm-wheels}"
      S3_URI="s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${CACHE_KEY}"
      echo "Uploading wheel to $S3_URI for future runs..."
      aws s3 cp "$WHEEL" "$S3_URI"
      aws s3 rm "s3://${VLLM_BUILD_CACHE_BUCKET}/${PREFIX}/${INTERMEDIATE_KEY}" 2>/dev/null || true
      echo "Cache updated."
    fi
    exit 0
  fi

  cleanup_background_upload
  RETRY=$((RETRY + 1))
  upload_intermediate_to_s3
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
