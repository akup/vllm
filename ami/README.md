# vLLM PoC AMI (self-contained, no Docker)

AMI that runs **vLLM with the PoC backend** as a **native install** (no Docker). Same stack as `Dockerfile.quick` (vLLM + `vllm/poc` overlay), but installed directly on the instance for **faster startup** (saves ~20–30 seconds vs Docker). MLNode uses it as a PoC backend for `/inference/pow` (pow_v2 routes).

## Two-stage build

- **Base image** (`packer.json`): Installs vLLM (full repo) in `/app/vllm-poc/.venv`, CloudWatch agent (same logic as gonka pow AMI), FRP, CUDA. **No** `start.sh` or systemd PoC service.
- **PoC overlay image** (`packer-poc.json`): Starts from the base AMI, copies `vllm/poc` and `start.sh`, sets up profile.d and `vllm-poc.service`. Use this for MLNode PoC backends.

## What the PoC AMI provides

- **Base**: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.9 (Amazon Linux 2023)
- **Runtime**: **No Docker** – vLLM runs as a normal process from a Python venv
- **Install**: vLLM (including `vllm/poc`) in `/app/vllm-poc/.venv`; base image has vLLM, overlay adds/updates PoC and start script
- **CloudWatch**: Agent installed and enabled (RPM + systemd unit + config template), same pattern as `mlnode/packages/pow/ami`
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

Run Packer **from the vLLM repo root**.

### 1. Base image (vLLM only, no start.sh)

```bash
cd /path/to/vllm
packer build ami/packer.json
```

Produces `project-vllm-base-ami-<timestamp>`. Contains vLLM + CloudWatch agent; no PoC start script.

#### Build via GitHub Actions

A workflow at `.github/workflows/build-ami.yml` builds the base AMI on AWS (manual trigger; run can take up to ~3 hours).

**Manual run from `poc-layers`:** The Actions UI "Run workflow" button only shows workflows on the default branch. If this workflow exists only on `poc-layers`, run it with GitHub CLI:

```bash
gh workflow run "build-ami.yml" --ref poc-layers
```

Optional: `--field aws_region=us-east-1` `--field instance_type=r6i.4xlarge`

**Required repository secrets** (Settings → Secrets and variables → Actions):

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM access key for the account that will own the AMI |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key for the above |

**IAM permissions** for that user/role: EC2 (create/describe/delete instances, volumes, security groups, key pairs; create/register/copy/deregister AMIs), and optionally `sts:GetCallerIdentity` for debugging. A typical policy includes `ec2:*` and `ami:*` in the build region.

**How to run:** Actions → "Build vLLM base AMI" → "Run workflow". Optionally set `aws_region` (default `us-east-1`) and `instance_type` (default `r6i.4xlarge`). The job has a 4-hour timeout.

#### Build with S3 cache (branch `build-ami`)

A second workflow **Build vLLM base AMI (with S3 cache)** (`.github/workflows/build-ami-cache.yml`) builds the base AMI using `ami/scripts/vllm-build-with-cache.sh`:

- **Final wheel cache:** successful builds upload the wheel to S3; later runs install from it and skip the build.
- **Intermediate cache:** during the build, `/tmp/vllm-src` is uploaded to S3 **every 10 minutes**. If the run is interrupted (e.g. connection loss), the next run downloads that state and resumes.

Use the **`build-ami`** branch for cache builds. The workflow runs on push to `build-ami` (when `ami/` or the workflow file changes) or manually.

**Required secrets** (in addition to AMI build credentials):

| Secret | Description |
|--------|-------------|
| `AWS_AMI_CACHE_BUCKET` | S3 bucket for vLLM wheel + intermediate cache |
| `AWS_AMI_IAM_PROFILE` | IAM instance profile name for the build instance (S3 access) |

**One-time setup:** Create the bucket and IAM profile with `ami/scripts/setup-cache-bucket.sh`, then add the bucket name and profile name as `AWS_AMI_CACHE_BUCKET` and `AWS_AMI_IAM_PROFILE` in repo secrets.

**Manual run from branch build-ami:**

```bash
gh workflow run "build-ami-cache.yml" --ref build-ami
```

Optional: `--field branch=build-ami` `--field aws_region=us-east-1` `--field instance_type=r6i.4xlarge`

### 2. PoC overlay image (base + vllm/poc + start.sh)

Uses the base AMI (default in `packer-poc.json` is `ami-06fade14c40cef676`). To build with the default base:

```bash
cd /path/to/vllm
packer build ami/packer-poc.json
```

To use a different base AMI, pass the variable:

```bash
cd /path/to/vllm
packer build -var "base_ami_id=ami-06fade14c40cef676" ami/packer-poc.json
```

Or resolve the latest base AMI by name and pass it:

```bash
BASE_AMI=$(aws ec2 describe-images --owners self \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text \
  --filters "Name=name,Values=project-vllm-base-ami*")
packer build -var "base_ami_id=$BASE_AMI" ami/packer-poc.json
```

Produces `project-vllm-poc-ami-<timestamp>` (PoC overlay for MLNode).

**Connection stability (long builds):** The base build runs 1–2+ hours. The template sets SSH keepalive (client and server) and long read/write timeouts so the connection is less likely to drop. If it still drops, re-run `packer build`; the vLLM build step retries without cleaning, so Ninja resumes from the last completed object.

### Variables (base: packer.json)

| Variable | Default | Description |
|----------|---------|-------------|
| `python_version` | `3.11` | Python version for the venv |
| `instance_type` | `r6i.4xlarge` | Instance type for build |
| `ami_name` | `project-vllm-base-ami-{{timestamp}}` | Output AMI name |

### Variables (overlay: packer-poc.json)

| Variable | Default | Description |
|----------|---------|-------------|
| `base_ami_id` | `ami-06fade14c40cef676` | Base AMI ID (vLLM base image) |
| `instance_type` | `r6i.2xlarge` | Instance type for overlay build |
| `ami_name` | `project-vllm-poc-ami-{{timestamp}}` | Output AMI name |

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

## Directory layout

```
vllm/
├── Dockerfile.quick   # Reference: same overlay approach for Docker
├── vllm/              # Python package (entrypoints, poc, ...)
└── ami/
    ├── packer.json         # Base: vLLM + CloudWatch (no start.sh)
    ├── packer-poc.json     # Overlay: base + vllm/poc + start.sh
    ├── start.sh            # Start vLLM process + register + poc init
    ├── setup-cloudwatch-agent.sh
    ├── cloudwatch-agent.service
    ├── cloudwatch-config.json
    └── README.md           # This file
```

On the **base** AMI:

- `/app/vllm-poc/.venv` – Python venv with vLLM (and PoC from install)
- CloudWatch agent installed and enabled; config template in `/opt/aws/amazon-cloudwatch-agent/etc/`

On the **PoC overlay** AMI (in addition):

- `/usr/local/bin/start-vllm-poc.sh` – Start script
- `/etc/profile.d/gonka-vllm-poc.sh` – Env (VLLM_USE_V1=0, HF_HOME, PATH)
- `/etc/systemd/system/vllm-poc.service` – Optional systemd unit
- `vllm/poc` in site-packages overwritten with overlay copy

## Relation to pow AMI

- **gonka-pow-ami** (e.g. `mlnode/packages/pow/ami`): standalone PoW service (PyTorch, `pow.service.app`), no vLLM
- **gonka-vllm-poc-ami** (this): vLLM with PoC backend, **native install**, no Docker. MLNode pow_v2 routes call `/api/v1/pow/*` on these instances.
