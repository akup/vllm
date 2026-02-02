#!/usr/bin/env bash
# Build vLLM (no S3). Requires: /tmp/vllm-src, torch in venv.
# Optional: VLLM_BUILD_VERSION (default 0.9.1), VLLM_BUILD_VERBOSE (default 1).

set -e
VERSION="${VLLM_BUILD_VERSION:-0.9.1}"
WHEELS_DIR="/tmp/vllm-wheels"
MAX_RETRIES="${VLLM_BUILD_MAX_RETRIES:-10}"
VENV="${VLLM_VENV:-/app/vllm-poc/.venv}"
mkdir -p "$WHEELS_DIR"

# CUDA
export CUDA_HOME=""
for d in /usr/local/cuda-12.8 /usr/local/cuda-12; do [ -x "$d/bin/nvcc" ] 2>/dev/null && CUDA_HOME=$d && break; done
[ -z "$CUDA_HOME" ] && for d in /usr/local/cuda-12*; do [ -x "$d/bin/nvcc" ] 2>/dev/null && CUDA_HOME=$d && break; done
[ -z "$CUDA_HOME" ] && { echo "ERROR: No CUDA toolkit found."; exit 1; }
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Build env
export TMPDIR="${TMPDIR:-/var/tmp/pip-build}"
export TMP="$TMPDIR"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
source "$VENV/bin/activate"

cd /tmp/vllm-src
export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION"
export VERBOSE="${VLLM_BUILD_VERBOSE:-1}"
RETRY=0
export PYTHONUNBUFFERED=1
while [ $RETRY -lt $MAX_RETRIES ]; do
  if pip wheel -v --no-build-isolation -w "$WHEELS_DIR" . 2>&1 | tee /tmp/vllm-wheel.log; then
    WHEEL=$(ls "$WHEELS_DIR"/vllm-*.whl 2>/dev/null | head -1)
    [ -z "$WHEEL" ] && { echo "ERROR: No wheel produced."; tail -40 /tmp/vllm-wheel.log; exit 1; }
    pip install "$WHEEL"
    echo "vLLM installed from local build."
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
