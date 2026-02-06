#!/bin/bash
set -e

# Start script for vLLM PoC AMI: runs vLLM server natively (no Docker) with PoC backend,
# compatible with MLNode pow_v2_routes (GET/POST /api/v1/pow/*). Self-contained, fast startup.

# Source environment file if present (e.g. from user-data or /etc/gonka-container.env)
if [ -f /etc/gonka-container.env ]; then
    echo "Sourcing /etc/gonka-container.env"
    set -a
    source /etc/gonka-container.env
    set +a
fi
source /etc/profile.d/gonka-api.sh 2>/dev/null || true

# Required for MLNode registration and FRP
if [ -z "$API_NODES" ]; then
    echo "API_NODES is required (comma-separated ip:port)." >&2
    exit 1
fi
if [ -z "$CLIENT_ID" ]; then
    echo "CLIENT_ID is required (four-digit, e.g. 0001)." >&2
    exit 1
fi
if [[ ! "$CLIENT_ID" =~ ^[0-9]{4}$ ]]; then
    echo "CLIENT_ID must be a four-digit number (0001-9999)." >&2
    exit 1
fi
if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME is required for vLLM (e.g. Qwen/Qwen3-8B)." >&2
    exit 1
fi

# Optional
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
HF_HOME=${HF_HOME:-/home/ec2-user/.cache/huggingface}
NODE_ID=${NODE_ID:-$CLIENT_ID}
ID_PREFIX=${ID_PREFIX:-}
GPU_TYPE=${GPU_TYPE:-nvidia}
NUM_GPUS=${NUM_GPUS:-1}
POC_NODE=${POC_NODE:-true}
REGISTRATION_ENDPOINT=${REGISTRATION_ENDPOINT:-/admin/v1/nodes}

IFS=',' read -ra API_NODES_ARRAY <<< "${API_NODES// /}"

# Ensure venv exists
if [ ! -f /app/vllm-poc/.venv/bin/python ]; then
    echo "ERROR: /app/vllm-poc/.venv not found. Run from AMI built with native vLLM." >&2
    exit 1
fi

# FRP (optional): if FRP_SERVERS and SECRET_FRP_TOKEN are set, start frpc
if [ -n "$FRP_SERVERS" ] && [ -n "$SECRET_FRP_TOKEN" ]; then
    FRP_CONFIG_DIR="${FRP_CONFIG_DIR:-/etc/frp}"
    mkdir -p "$FRP_CONFIG_DIR"
    IFS=',' read -ra FRP_SERVERS_ARRAY <<< "${FRP_SERVERS// /}"
    for i in "${!FRP_SERVERS_ARRAY[@]}"; do
        server="${FRP_SERVERS_ARRAY[$i]}"
        FRP_SERVER_IP="${server%%:*}"
        FRP_SERVER_PORT="${server##*:}"
        cat > "${FRP_CONFIG_DIR}/frpc${i}.ini" <<EOF
[common]
server_addr = ${FRP_SERVER_IP}
server_port = ${FRP_SERVER_PORT}
token = ${SECRET_FRP_TOKEN}

[client-mlnode-poc-${CLIENT_ID}]
type = tcp
local_ip = 127.0.0.1
local_port = 8080
remote_port = 2${CLIENT_ID}
EOF
        (command -v frpc &>/dev/null && frpc -c "${FRP_CONFIG_DIR}/frpc${i}.ini" &) || true
    done
fi

# Build vLLM run args (native process, no Docker)
export VLLM_USE_V1=0
export HF_HOME="/data/huggingface"
# Set VLLM_LOGGING_LEVEL=DEBUG before start.sh to see detailed weight-load/disk activity during startup
# export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_SKIP_P2P_CHECK=1

HF_HOME="/data/huggingface"
TENSOR_PARALLEL_SIZE=4
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/vllm-cache}"
export VLLM_CACHE_ROOT
#export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset NCCL_CUMEM_ENABLE 2>/dev/null || true

echo "Starting uvicorn application..."
UVICORN_START_TIME=$(date +%s)
echo "[$(date +%H:%M:%S)] Uvicorn startup initiated"

# Create log directory for uvicorn if it doesn't exist
UVICORN_LOG_DIR="${LOG_DIR:-/tmp/logs}"
mkdir -p "$UVICORN_LOG_DIR"

source /app/packages/api/.venv/bin/activate

echo "Activated and want to read huggingface"

# Warm Linux page cache for model weights in background (so uvicorn starts immediately and logs are not blocked)
# PYTHONUNBUFFERED=1 so progress lines appear in the log immediately
if [ -x /data/compressa-tests/warm-page-cache.sh ]; then
    echo "[$(date +%H:%M:%S)] Starting page-cache warm in background (PARALLEL_READERS=16); log: $UVICORN_LOG_DIR/warm-page-cache.log"
    PARALLEL_READERS=16 PYTHONUNBUFFERED=1 /data/compressa-tests/warm-page-cache.sh >> "${UVICORN_LOG_DIR}/warm-page-cache.log" 2>&1 &
fi

# Start uvicorn and capture both stdout and stderr
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 2>&1 | tee "$UVICORN_LOG_DIR/uvicorn.log" &
UVICORN_PID=$!
UVICORN_LAUNCH_TIME=$(date +%s)
LAUNCH_DURATION=$((UVICORN_LAUNCH_TIME - UVICORN_START_TIME))
echo "[$(date +%H:%M:%S)] Uvicorn process launched (PID: $UVICORN_PID, launch took ${LAUNCH_DURATION}s)"
echo "Uvicorn logs: $UVICORN_LOG_DIR/uvicorn.log"

# Start vLLM natively in background (PoC routes are in the overlay)
# Normally the API starts two vLLM backends (ports 5001, 5002) via inference-up; each uses 4 GPUs.
# To start two vLLMs directly here with the same args, uncomment and use a venv that has vllm:

VLLM_PYTHON="${VLLM_PYTHON_PATH:-/app/vllm-poc/.venv/bin/python}"
VLLM_BASE_ARGS="--host 0.0.0.0 --model $MODEL_NAME --dtype float16 --quantization fp8 --enforce-eager --load-format fastsafetensors --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
echo "[$(date +%H:%M:%S)] Starting vLLM on port 5001 (GPUs 0-3)..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 $VLLM_PYTHON -m vllm.entrypoints.openai.api_server $VLLM_BASE_ARGS --port 5001 >> "$UVICORN_LOG_DIR/vllm-5001.log" 2>&1 &
VLLM_PID_1=$!
echo "[$(date +%H:%M:%S)] Starting vLLM on port 5002 (GPUs 4-7)..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5,6,7 $VLLM_PYTHON -m vllm.entrypoints.openai.api_server $VLLM_BASE_ARGS --port 5002 >> "$UVICORN_LOG_DIR/vllm-5002.log" 2>&1 &
VLLM_PID_2=$!
echo "vLLM started (PIDs $VLLM_PID_1, $VLLM_PID_2). Logs: $UVICORN_LOG_DIR/vllm-5001.log, vllm-5002.log"
# (Then call the API to setup proxy with ports 5001,5002 so routes use these backends.)


