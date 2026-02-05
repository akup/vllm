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
# echo "Starting vLLM PoC server natively (port 8080)..."
# /app/vllm-poc/.venv/bin/python -m vllm.entrypoints.openai.api_server $VLLM_ARGS &
# VLLM_PID=$!
# echo "vLLM started (PID $VLLM_PID). Waiting for /api/v1/state..."

# Give uvicorn a moment to start the process
sleep 1

# Check if process is actually running
if ! kill -0 $UVICORN_PID 2>/dev/null; then
    echo "ERROR: Uvicorn process died immediately after start!" >&2
    if [ -f "$UVICORN_LOG_DIR/uvicorn.log" ]; then
        echo "Last 50 lines of uvicorn log:" >&2
        tail -50 "$UVICORN_LOG_DIR/uvicorn.log" >&2
    fi
    exit 1
else
    echo "[$(date +%H:%M:%S)] Uvicorn process is running (PID: $UVICORN_PID)"
    # Show initial log output to see startup progress
    if [ -f "$UVICORN_LOG_DIR/uvicorn.log" ]; then
        echo "Initial uvicorn log output:"
        cat "$UVICORN_LOG_DIR/uvicorn.log" 2>/dev/null || echo "Log file empty or not readable"
    fi
fi

# Wait for uvicorn to be ready (check if pow/init/generate endpoint is responding)
echo "Waiting for uvicorn to be ready..."
READINESS_START_TIME=$(date +%s)
max_attempts=120
attempt=0
LAST_LOG_CHECK=0

while [ $attempt -lt $max_attempts ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - READINESS_START_TIME))
    
    # Show progress every 5 seconds
    if [ $((attempt % 5)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Attempt $attempt/$max_attempts (elapsed: ${ELAPSED}s)"
        
        # Check uvicorn log for startup messages
        if [ -f "$UVICORN_LOG_DIR/uvicorn.log" ]; then
            CURRENT_LOG_LINES=$(wc -l < "$UVICORN_LOG_DIR/uvicorn.log" 2>/dev/null || echo "0")
            if [ "$CURRENT_LOG_LINES" -gt "$LAST_LOG_CHECK" ]; then
                NEW_LINES=$((CURRENT_LOG_LINES - LAST_LOG_CHECK))
                echo "  New log lines since last check ($NEW_LINES):"
                tail -n "$NEW_LINES" "$UVICORN_LOG_DIR/uvicorn.log" | sed 's/^/    /' | tail -5
                LAST_LOG_CHECK=$CURRENT_LOG_LINES
            fi
        fi
        
        # Check if uvicorn process is still running
        if ! kill -0 $UVICORN_PID 2>/dev/null; then
            echo "ERROR: Uvicorn process (PID: $UVICORN_PID) has died!" >&2
            if [ -f "$UVICORN_LOG_DIR/uvicorn.log" ]; then
                echo "Last 50 lines of uvicorn log:" >&2
                tail -50 "$UVICORN_LOG_DIR/uvicorn.log" >&2
            fi
            exit 1
        fi
    fi
    
    # Test the /api/v1/state endpoint
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 2 \
        --connect-timeout 1 \
        "http://127.0.0.1:8080/api/v1/state" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" = "200" ]; then
        READY_TIME=$(date +%s)
        TOTAL_STARTUP_TIME=$((READY_TIME - UVICORN_START_TIME))
        READINESS_WAIT_TIME=$((READY_TIME - READINESS_START_TIME))
        echo "[$(date +%H:%M:%S)] ✅ Uvicorn is ready! (HTTP status: $HTTP_CODE)"
        echo "  Total startup time: ${TOTAL_STARTUP_TIME}s"
        echo "  Readiness check wait: ${READINESS_WAIT_TIME}s"
        break
    fi
    
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    if [ "${INFERENCE_NODE}" = "true" ]; then
        echo "WARNING: Uvicorn may not be ready, but proceeding with inference-up call..." >&2
    else
        echo "WARNING: Uvicorn may not be ready..." >&2
    fi
fi

echo "Calling inference-up.py to load model..."
if [ -n "$TENSOR_PARALLEL_SIZE" ] && [ "$TENSOR_PARALLEL_SIZE" -gt 1 ]; then
    python3 /data/compressa-tests/inference-up.py \
        --model "$MODEL_NAME" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --base-url "http://localhost:8080" || {
        echo "WARNING: inference-up.py failed, but continuing..." >&2
    }
else
    python3 /data/compressa-tests/inference-up.py \
        --model "$MODEL_NAME" \
        --base-url "http://localhost:8080" || {
        echo "WARNING: inference-up.py failed, but continuing..." >&2
    }
fi
echo "Inference engine is up!"

# Register with MLNode API (so pow_v2_routes can discover this backend)
REGISTRATION_JSON=${REGISTRATION_JSON:-}
if [ -z "$REGISTRATION_JSON" ]; then
  REGISTRATION_JSON='{
   "id": "'${ID_PREFIX}${NODE_ID}'",
   "host": "frps",
   "inference_port": 1'${CLIENT_ID}',
   "poc_port": 2'${CLIENT_ID}',
   "max_concurrent": 500,
   "models": {
     "'$MODEL_NAME'": {
       "args": ["--tensor-parallel-size","'$TENSOR_PARALLEL_SIZE'", "--load-format","fastsafetensors", "--quantization","fp8"]
     }
   },
   "poc_hw": {
     "type": "'${GPU_TYPE}'",
     "num": '${NUM_GPUS}'
   },
   "access": true
 }'
fi
for API_NODE in "${API_NODES_ARRAY[@]}"; do
  echo "Registering with API at ${API_NODE}"
  echo "$REGISTRATION_JSON" | curl -s -X POST "http://${API_NODE}${REGISTRATION_ENDPOINT}" \
    -H "Content-Type: application/json" -d @- || true
done

# Optionally report GPU devices to API
GPU_DEVICES=$(curl -s "http://127.0.0.1:8080/api/v1/gpu/devices" 2>/dev/null || echo "{}")
if [ -n "$GPU_DEVICES" ] && [ "$GPU_DEVICES" != "{}" ]; then
  if command -v jq &>/dev/null; then
    GPU_DEVICES_WITH_ACCESS=$(echo "$GPU_DEVICES" | jq '. + {"access": true}')
  else
    GPU_DEVICES_WITH_ACCESS=$(echo "$GPU_DEVICES" | python3 -c "import sys,json; d=json.load(sys.stdin); d['access']=True; print(json.dumps(d))" 2>/dev/null) || echo "$GPU_DEVICES"
  fi
  for API_NODE in "${API_NODES_ARRAY[@]}"; do
    echo "$GPU_DEVICES_WITH_ACCESS" | curl -s -X POST "http://${API_NODE}/admin/v1/nodes/${NODE_ID}/hardware" \
      -H "Content-Type: application/json" -d @- || true
  done
fi


FIRST_API_NODE="${API_NODES_ARRAY[0]}"
echo "Requesting init_generate from API: ${FIRST_API_NODE}"
INIT_RESPONSE=$(curl -s --max-time 30 "http://${FIRST_API_NODE}/admin/v1/poc_init/${CLIENT_ID}?access=true" 2>/dev/null || echo "{}")
INIT_GENERATE_JSON=$(echo "$INIT_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('init_generate') or {}))" 2>/dev/null || echo "")
if [ -n "$INIT_GENERATE_JSON" ] && [ "$INIT_GENERATE_JSON" != "null" ] && [ "$INIT_GENERATE_JSON" != "{}" ]; then
  # Use inference endpoint for version v2, else pow endpoint
  INIT_VERSION=$(echo "$INIT_GENERATE_JSON" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('version', ''))" 2>/dev/null || echo "")
  if [ "$INIT_VERSION" = "v2" ]; then
    POW_INIT_ENDPOINT="http://127.0.0.1:8080/api/v1/inference/pow/init/generate"
  else
    POW_INIT_ENDPOINT="http://127.0.0.1:8080/api/v1/pow/init/generate"
  fi
  echo "POW version: $INIT_VERSION"
  echo "Posting init_generate to local $POW_INIT_ENDPOINT"
  HTTP_CODE=$(curl -s -o /tmp/pow_init_response.json -w "%{http_code}" --max-time 30 -X POST "$POW_INIT_ENDPOINT" \
    -H "Content-Type: application/json" -d "$INIT_GENERATE_JSON" 2>/dev/null || echo "000")
  echo "Local pow/init/generate HTTP $HTTP_CODE"
  rm -f /tmp/pow_init_response.json
fi

echo "vLLM PoC backend is running (native, no Docker). MLNode can proxy to this instance (e.g. poc_port 2${CLIENT_ID} via FRP)."
echo "Waiting for vLLM process..."
wait $VLLM_PID 2>/dev/null || true
