# vLLM PoC AMI (self-contained, no Docker)

AMI that runs **vLLM with the PoC backend** as a **native install** (no Docker). Same stack as `Dockerfile.quick` (vLLM + `vllm/poc` overlay), but installed directly on the instance for **faster startup** (saves ~20–30 seconds vs Docker). MLNode uses it as a PoC backend for `/inference/pow` (pow_v2 routes).

## Build stages

- **Base image** (`packer.json`): Installs vLLM (full repo) in `/app/vllm-poc/.venv`, CloudWatch agent (same logic as gonka pow AMI), FRP, CUDA. **No** `start.sh` or systemd PoC service.
- **PoC overlay image** (`packer-poc.json`): Starts from the base AMI, copies `vllm/poc` and `start.sh`, sets up profile.d and `vllm-poc.service`. Use this for MLNode PoC backends.
- **API overlay image** (`packer-api.json`): Starts from the **PoC AMI** (passed by parameter), installs the MLNode API (same stack as `ami/packages/api/Dockerfile` app target) natively: `/app/packages` with api, pow, train, common; Poetry venv at `/app/packages/api/.venv`; start script `/usr/local/bin/start-api.sh` and systemd `vllm-api.service`. No Docker; equivalent to the Docker base `ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-blackwell` = PoC AMI.

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

### 0. Prereq image (CUDA + PyTorch)

The full base image (`packer.json`) **always** starts from the latest **prereq AMI** (`project-vllm-prereq-ami-*`), so CUDA and PyTorch are not reinstalled. The prereq is found automatically by name in your account.

**Recommended: build both images with one script** (builds prereq if missing, then full base):

```bash
cd /path/to/vllm
./ami/scripts/build-vllm-ami.sh
```

Options: `--region us-east-1`, `--force-prereq` (rebuild prereq), `--prereq-only` (build only prereq), `--skip-prereq` (fail if no prereq), `--verbose 0|1|2` (vLLM build verbosity), `--cache-bucket BUCKET`, `--cache-prefix PREFIX`, `--iam-profile NAME`, `--var KEY=VALUE`.

**Or build manually:** First build the prereq (once per region), then the base image:

```bash
packer build ami/packer-prereq.json
packer build ami/packer.json
```

`packer.json` uses the latest `project-vllm-prereq-ami-*` in the region (by filter). Override with `-var "source_image=ami-XXXXXXXX"` to use a specific AMI.

### 1. Base image (vLLM only, no start.sh)

```bash
cd /path/to/vllm
./ami/scripts/build-vllm-ami.sh
# or: packer build ami/packer.json  (requires prereq AMI to exist)
```

Produces `project-vllm-base-ami-<timestamp>`. Contains vLLM + CloudWatch agent; no PoC start script.

**Packer variables** (optional):

| Variable | Default | Description |
|----------|--------|-------------|
| `vllm_build_verbose` | `1` | vLLM build verbosity: `0` = quiet, `1` = normal (Ninja progress), `2` = verbose |
| `vllm_build_cache_bucket` | `""` | S3 bucket for wheel/intermediate cache; empty = no cache |
| `vllm_build_cache_prefix` | `vllm-wheels` | S3 key prefix for cache objects |
| `vllm_build_overlay_files` | `""` | Space-separated paths under repo to apply over cache (e.g. tokenizer fix); only used when cache is enabled |

**Example: build vLLM verbose** (more CMake/Ninja output):

```bash
packer build -var "vllm_build_verbose=2" ami/packer.json
```

**Example: build with S3 cache** (faster re-runs; requires bucket and IAM profile):

```bash
packer build \
  -var "vllm_build_cache_bucket=my-vllm-cache" \
  -var "iam_instance_profile=PackerVLLMCache" \
  ami/packer.json
```

**Example: build with cache and overlay tokenizer fix** (reuse cached wheel but apply current `tokenizer.py`; **required for Qwen models**):

```bash
packer build \
  -var "vllm_build_cache_bucket=my-vllm-cache" \
  -var "vllm_build_overlay_files=vllm/transformers_utils/tokenizer.py" \
  -var "iam_instance_profile=PackerVLLMCache" \
  ami/packer.json
```

**Qwen / `all_special_tokens_extended`:** If at runtime you see `Qwen2Tokenizer has no attribute all_special_tokens_extended`, the base AMI was built without the overlay (or from an older build). Either **rebuild the base AMI** with `vllm_build_overlay_files=vllm/transformers_utils/tokenizer.py` (GitHub workflow does this by default), or on the **running instance** copy the fixed file from the repo into the venv:

```bash
# From your dev machine (repo root), copy fixed tokenizer into the instance’s installed vLLM:
scp vllm/transformers_utils/tokenizer.py ec2-user@<instance-ip>:/tmp/
# On the instance:
sudo cp /tmp/tokenizer.py /app/vllm-poc/.venv/lib64/python3.11/site-packages/vllm/transformers_utils/tokenizer.py
```

Then restart the vLLM server.

#### Build via GitHub Actions

A workflow at `.github/workflows/build-ami.yml` builds the base AMI on AWS (manual trigger; run can take up to ~3 hours).

**Manual run from `poc-layers`:** The Actions UI "Run workflow" button only shows workflows on the default branch. If this workflow exists only on `poc-layers`, run it with GitHub CLI:

```bash
gh workflow run "build-ami.yml" --ref poc-layers
```

Optional: `--field aws_region=us-east-1` `--field instance_type=r6i.4xlarge` `--field clean_cache=true`

**Clean cache (full rebuild):** To force a full vLLM build without reusing S3 wheel/intermediate cache, set **clean_cache** to `true` when running the workflow (or run the AWS CLI below before triggering the workflow). The workflow will delete all objects under the cache prefix, then build from source (with overlay).

```bash
gh workflow run "build-ami.yml" --ref poc-layers -f clean_cache=true
```

Or manually delete the cache (same bucket/prefix as the workflow):

```bash
aws s3 rm s3://gonka-vllm-build-cache/vllm-wheels/ --recursive
```

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

### 3. API overlay image (base + MLNode API)

Uses the **PoC AMI** (same logical base as Docker `ghcr.io/gonka-ai/vllm:v0.9.1-poc-v2-blackwell`). Installs the API stack from `ami/packages/api` (api, pow, train, common) natively with Poetry.

```bash
# Build API overlay (pass your PoC AMI id)
packer build -var "poc_ami_id=ami-XXXXXXXX" ami/packer-api.json
```

Or resolve latest PoC AMI:

```bash
POC_AMI=$(aws ec2 describe-images --owners self \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text \
  --filters "Name=name,Values=project-vllm-poc-ami*")
packer build -var "poc_ami_id=$POC_AMI" ami/packer-api.json
```

Produces `project-vllm-api-ami-<timestamp>`. Start API: `/usr/local/bin/start-api.sh` or `systemctl start vllm-api`. PoC (vLLM) remains: `/usr/local/bin/start-vllm-poc.sh` or `systemctl start vllm-poc`.

### Variables (overlay: packer-api.json)

| Variable | Default | Description |
|----------|---------|-------------|
| `poc_ami_id` | `ami-0981d509bf50f86fa` | PoC AMI ID (project-vllm-poc-ami-*) |
| `instance_type` | `r6i.2xlarge` | Instance type for overlay build |
| `ami_name` | `project-vllm-api-ami-{{timestamp}}` | Output AMI name |

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
    ├── packer-api.json     # Overlay: PoC AMI + MLNode API (uvicorn)
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
