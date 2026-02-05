# vLLM PoC AMI (self-contained, no Docker)

AMI that runs **vLLM with the PoC backend** as a **native install** (no Docker). Same stack as `Dockerfile.quick` (vLLM + `vllm/poc` overlay), but installed directly on the instance for **faster startup** (saves ~20–30 seconds vs Docker). MLNode uses it as a PoC backend for `/inference/pow` (pow_v2 routes).

## What this AMI provides

- **Base**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)
- **Runtime**: **No Docker** – vLLM runs as a normal process from a Python venv
- **Install**: vLLM (including `vllm/poc`) installed from the local repo with `pip install .` in `/app/vllm-poc/.venv`
- **API**: vLLM server on port **8080** exposing:
  - `/api/v1/state`
  - `/api/v1/pow/init/generate`
  - `/api/v1/pow/generate`
  - `/api/v1/pow/status`
  - `/api/v1/pow/stop`
  - `/api/v1/gpu/devices`

These are the same paths that **MLNode pow_v2 routes** call when proxying to vLLM backends.

## Why native (no Docker)

- **Faster startup**: No container pull/start overhead (~20–30 seconds saved)
- **Self-contained**: Everything in one AMI; no image registry or Docker daemon
- **Simpler**: Single process, same pattern as the standalone pow AMI

## How MLNode references this backend

In MLNode, `packages/api` uses:

- **`api/inference/pow_v2_routes.py`** – defines `/inference/pow` endpoints that proxy to vLLM backends
- **`api/proxy.py`** – `get_healthy_backends()`, `call_backend(port, method, path, ...)`

Each instance from this AMI runs vLLM on port **8080**. With FRP, the API reaches it via the FRP server (e.g. `poc_port` = `2<CLIENT_ID>`). Once the node is registered, the API proxies PoC requests to this instance’s `/api/v1/pow/*`.

## Prerequisites

- Packer installed
- AWS credentials and permissions (EC2, AMI)
- vLLM repo

## Build

Run Packer **from the vLLM repo root** (Packer uploads the necessary source files directly):

```bash
cd /path/to/vllm
packer build ami/packer.json
```

With custom variables:

```bash
cd /path/to/vllm
packer build \
  -var 'aws_region=us-west-2' \
  -var 'instance_type=g5.xlarge' \
  -var 'ami_name=gonka-vllm-poc-{{timestamp}}' \
  ami/packer.json
```

| Variable | Default | Description |
|----------|---------|-------------|
| `python_version` | `3.11` | Python version for the venv |

Build uploads the repo (from the current directory) to the instance and runs `pip install .` so vLLM and PoC match the project version.

## API AMI (packer-api.json) and warmed-cache AMI

The **API AMI** adds the MLNode API (uvicorn + packages) on top of the PoC base. After building it, you can create a **warmed-cache AMI** by launching a large GPU instance, starting vLLM with the target model (so cache and compile are populated), then creating an AMI from that running instance.

### 1. Build the API AMI

```bash
cd /path/to/vllm
packer build ami/packer-api.json
```

### 2. Launch a p6-b200.48xlarge instance

Launch an instance from the API AMI you just built, using instance type **p6-b200.48xlarge** (or the same type you use in production). Configure env (e.g. `API_NODES`, `CLIENT_ID`, `MODEL_NAME`) as needed; the start script uses defaults like `MODEL_NAME=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` and `TENSOR_PARALLEL_SIZE=4` (see `ami/start.sh`).

### 3. Start vLLM and warm the cache

On the instance, start the API (uvicorn) so vLLM backends can be brought up. Either run the full start script:

```bash
/usr/local/bin/start-vllm-poc.sh   # or ami/start.sh if you are in the repo
```

or start uvicorn and then run inference-up with the same model and tensor-parallel size as in `start.sh`:

```bash
# Start uvicorn (API) first
source /app/packages/api/.venv/bin/activate
export PYTHONPATH=/app:/app/packages/api/src
python -m uvicorn api.app:app --host 0.0.0.0 --port 8080 &

# Wait for API to be up, then load the model and warm cache (same MODEL_NAME / TENSOR_PARALLEL_SIZE as start.sh)
MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
TENSOR_PARALLEL_SIZE=4
python3 /data/compressa-tests/inference-up.py \
  --model "$MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --base-url "http://localhost:8080" || { echo "WARNING: inference-up.py failed, but continuing..." >&2; }
```

Wait until vLLM has loaded the model and (if applicable) compilation has finished so the cache under `VLLM_CACHE_ROOT` (and any compile cache) is populated.

### 4. Create AMI and snapshot from the running instance

From your **local machine** (or any host with AWS CLI and permissions), create an AMI from the running instance. This automatically creates EBS snapshots of the instance’s volumes.

```bash
INSTANCE_ID="i-0c45aa822cdd4a0df"   # replace with your instance id
AMI_NAME="gonka-mlnode-with-cache-$(date +%Y%m%d-%H%M)"

aws ec2 create-image \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "AMI from running instance (includes vLLM cache, warmed state)" \
  --no-reboot
```

Use `--no-reboot` to avoid rebooting the instance (crash-consistent snapshot). Omit it for a reboot and more consistent state. Check AMI status with `aws ec2 describe-images --image-ids <ImageId>` until `State` is `available`. New instances launched from this AMI will have the warmed cache and start faster.

## Start on instance

Set environment (e.g. in `/etc/gonka-container.env` or user-data), then:

```bash
/usr/local/bin/start-vllm-poc.sh
```

### Required env (example)

| Variable      | Example           | Description                          |
|---------------|-------------------|--------------------------------------|
| `API_NODES`   | `10.0.1.1:5000`   | MLNode API host:port (comma-separated) |
| `CLIENT_ID`   | `0001`            | Four-digit node id                   |
| `MODEL_NAME`  | `Qwen/Qwen3-8B`   | Model for vLLM                       |

### Optional env

| Variable               | Default   | Description                    |
|------------------------|-----------|--------------------------------|
| `TENSOR_PARALLEL_SIZE` | `1`       | vLLM tensor parallel size      |
| `HF_HOME`              | `~/.cache/huggingface` | Hugging Face cache   |
| `POC_NODE`             | `true`    | If true, run poc init after start |
| `FRP_SERVERS`          | -         | Comma-separated `host:port` for FRP |
| `SECRET_FRP_TOKEN`     | -         | FRP token                      |
| `NODE_ID`              | `$CLIENT_ID` | Node id for registration   |
| `REGISTRATION_ENDPOINT`| `/admin/v1/nodes` | API registration path  |

## Fast disk (EBS IOPS)

**Disk speed is not fixed by the image.** It is determined when you **launch** the instance: **instance type** and **EBS volume type, IOPS, and throughput**.

- **API AMI (`packer-api.json`)**: The root volume is built and registered as **gp3 with 16,000 IOPS and 1,000 MiB/s**. Instances launched from this AMI use that mapping by default, so you get a fast root (and HF_HOME) without extra steps.
- **Other AMIs**: If the Packer config does not set `iops`/`throughput`, gp3 defaults to 3,000 IOPS and 125 MiB/s. To get fast disk when launching:
  - **Launch template / EC2**: Set the root (or data) volume to gp3 with e.g. **16,000 IOPS** and **1,000 MiB/s** (or io2 for higher guarantees).
  - **Instance store**: Use an instance type with **NVMe instance store** (e.g. some `g5`, `p4d`) and put the model there; fastest, but ephemeral.

For slow “Loading safetensors checkpoint shards”, see `ami/scripts/compressa-tests/README-CACHE.md` (warm page cache, IOPS, fastsafetensors).

## Directory layout

```
vllm/
├── Dockerfile.quick   # Reference: same overlay approach for Docker
├── vllm/              # Python package (entrypoints, poc, ...)
└── ami/
    ├── packer.json    # Native vLLM + PoC install (no Docker)
    ├── start.sh       # Start vLLM process + register + poc init
    └── README.md      # This file
```

On the AMI:

- `/app/vllm-poc/.venv` – Python venv with vLLM and PoC overlay
- `/usr/local/bin/start-vllm-poc.sh` – Start script
- `/etc/profile.d/gonka-vllm-poc.sh` – Env (VLLM_USE_V1=0, HF_HOME, PATH)
- `/etc/systemd/system/vllm-poc.service` – Optional systemd unit

## Relation to pow AMI

- **gonka-pow-ami** (e.g. `mlnode/packages/pow/ami`): standalone PoW service (PyTorch, `pow.service.app`), no vLLM
- **gonka-vllm-poc-ami** (this): vLLM with PoC backend, **native install**, no Docker. MLNode pow_v2 routes call `/api/v1/pow/*` on these instances.
