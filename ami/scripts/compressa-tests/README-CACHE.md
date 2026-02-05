# Warmed-cache AMI (p6-b200.48xlarge)

To get an AMI with vLLM cache and warmed state for fast startup, launch a **p6-b200.48xlarge** instance from the API AMI, start vLLM with the required parameters (using inference-up.py), then create an AMI and snapshot from that running instance.

## 1. Launch a p6-b200.48xlarge instance

Launch an instance from the API AMI (built with `packer build ami/packer-api.json`). Use instance type **p6-b200.48xlarge**.

## 2. (Optional) Warm Linux page cache for model weights

To reduce “Loading weights” time (e.g. from ~74 s to much less), read the model files into the Linux page cache **before** starting uvicorn. Run this once (uses parallel reads over `/data/huggingface`):

```bash
export HF_HOME=/data/huggingface
PARALLEL_READERS=16 /data/compressa-tests/warm-page-cache.sh
```

Or with Python only (default 8 workers, only weight-like files):

```bash
HF_HOME=/data/huggingface python3 /data/compressa-tests/warm-page-cache.py --workers 16
```

Then continue with step 3.

## 3. On the instance: set env and start uvicorn

Set the same environment as in `ami/start.sh` (lines 77–98), then start uvicorn:

```bash
export VLLM_USE_V1=0
export HF_HOME="/data/huggingface"
TENSOR_PARALLEL_SIZE=4
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/vllm-cache}"
export VLLM_CACHE_ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset NCCL_CUMEM_ENABLE 2>/dev/null || true

UVICORN_LOG_DIR="${LOG_DIR:-/tmp/logs}"
mkdir -p "$UVICORN_LOG_DIR"

source /app/packages/api/.venv/bin/activate
export PYTHONPATH=/app:/app/packages/api/src

# Start uvicorn and capture both stdout and stderr
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 2>&1 | tee "$UVICORN_LOG_DIR/uvicorn.log" &
```

## 4. On the instance: run inference-up to load the model and warm cache

After the API is up, run inference-up with the same `MODEL_NAME` and `TENSOR_PARALLEL_SIZE`:

```bash
python3 /data/compressa-tests/inference-up.py \
  --model "$MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --base-url "http://localhost:8080" || {
  echo "WARNING: inference-up.py failed, but continuing..." >&2
}
```

Wait until vLLM has loaded the model and is ready.

## 5. Create the AMI and snapshot (from your local machine)

From your **local machine** (or any host with AWS CLI and permissions), create an AMI from the running instance. This creates EBS snapshots of the instance’s volumes.

```bash
INSTANCE_ID="i-0c45aa822cdd4a0df"   # replace with your instance id
AMI_NAME="gonka-mlnode-with-cache-$(date +%Y%m%d-%H%M)"

aws ec2 create-image \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "AMI from running instance (includes vLLM cache, warmed state)" \
  --no-reboot
```

Use `--no-reboot` to avoid rebooting the instance (crash-consistent snapshot). Omit it for a reboot and more consistent state.

Check AMI status until `State` is `available`:

```bash
aws ec2 describe-images --image-ids <ImageId> --query 'Images[0].State' --output text
```

New instances launched from this AMI will have the warmed cache and start faster.
