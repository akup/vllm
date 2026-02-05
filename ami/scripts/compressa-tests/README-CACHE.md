# Warmed-cache AMI (p6-b200.48xlarge)

To get an AMI with vLLM cache and warmed state for fast startup, launch a **p6-b200.48xlarge** instance from the API AMI, start vLLM with the required parameters (using inference-up.py), then create an AMI and snapshot from that running instance.

## Page-cache warm in start.sh

`ami/start.sh` runs the page-cache warm **in background** so uvicorn starts immediately and logs are not blocked. Warm output goes to `$LOG_DIR/warm-page-cache.log` (default `/tmp/logs/warm-page-cache.log`).

## Slow first ~200s then fast? Check EBS IOPS

If the warm script shows a long delay before the first progress (e.g. ~200s to read the first few GiB, then ~50s for the rest), the root volume may have **low IOPS/throughput** at launch (e.g. default gp3 3,000 IOPS, 125 MiB/s). Check the instance’s EBS settings:

```bash
# From your machine (replace with your instance id)
INSTANCE_ID=i-080f709ab30281449 /data/compressa-tests/check-instance-ebs-iops.sh

# Or on the instance (uses metadata)
/data/compressa-tests/check-instance-ebs-iops.sh
```

If IOPS/throughput are low, update your **launch template** so the root (or data) volume uses gp3 with e.g. **16,000 IOPS** and **1,000 MiB/s** (see `ami/README.md` → Fast disk (EBS IOPS)).

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

## Why is the first read so slow? (0 GiB for minutes)

If progress stays at **0.0 GiB for many minutes** (e.g. 200+ seconds), the disk is extremely slow for the first I/O. Common causes:

- **EBS burst credits exhausted** (gp2) or low baseline IOPS (gp3)
- **Cold EBS volume** or first I/O after boot
- **NFS/EFS** – network storage with high latency or throttling

**Diagnostics (on the instance):**

```bash
/data/compressa-tests/disk-read-diagnostics.sh /data/huggingface
```

This prints: mount and filesystem, block devices, EBS IOPS (if AWS), a quick `dd` read test on one file, and `vmstat`. Run it while warm-page-cache is stuck at 0 GiB, or right after boot.

**Fixes:** Increase EBS IOPS/throughput (see below), or pre-warm the volume (e.g. run a quick sequential read before the warm script), or reuse a warm EBS volume (see below).

## Solving cold EBS / first I/O after boot

Yes. Options:

1. **Pre-warm the volume before warm-page-cache**  
   Run a short sequential read over the device or path so the EBS backend is “woken up”, then run the real warm. Example (on the instance, before or at the start of your start script):
   ```bash
   # Optional: wake EBS (adjust device if /data is on a different volume)
   DATA_DEV=$(findmnt -n -o SOURCE --target /data 2>/dev/null)
   if [ -b "$DATA_DEV" ]; then
     dd if="$DATA_DEV" of=/dev/null bs=1M count=4096 2>/dev/null || true
   fi
   ```
   Or read the first bit of the first model file:  
   `dd if=/data/huggingface/models--*/*/snapshots/*/model*.safetensors of=/dev/null bs=1M count=1024 2>/dev/null || true`  
   Then run warm-page-cache as usual.

2. **Use gp3 with provisioned IOPS/throughput**  
   Cold and burst behavior is worse with gp2 or low-baseline gp3. Use gp3 with higher baseline IOPS (and throughput if needed) so performance is consistent from first I/O.

3. **Reuse the same EBS volume**  
   Stop/start the same instance (don’t terminate) so the root (and any data volume) stays the same; the volume is often still warm. For new instances, use a **separate data volume** that you create once, warm once, and reattach via launch template (see below).

## Reusing warm disks with a launch template

Yes. Use a **dedicated EBS volume for model data** (e.g. `/data`), warm it once, then attach that **same volume** to new instances in the launch template.

**1. Create and warm the volume once**

- Create an EBS volume (gp3 recommended, with the IOPS/throughput you need). Optionally create it from a snapshot that already contains the model data.
- Attach it to an instance, mount it at `/data`, run warm-page-cache (and your workload) so it’s warm.
- Detach the volume (you can stop/terminate the instance if nothing else needs it). Note the **volume ID** (e.g. `vol-0123abcd`).

**2. Launch template: attach the existing volume**

In the launch template (or RunInstances), add a **block device mapping** that attaches that existing volume to the new instance (e.g. as `/dev/sdb` or `/dev/xvdb`):

```json
"BlockDeviceMappings": [
  { "DeviceName": "/dev/xvda", "Ebs": { ... } },
  { "DeviceName": "/dev/sdb", "Ebs": { "VolumeId": "vol-0123abcd" } }
]
```

Or AWS CLI:

```bash
aws ec2 run-instances ... \
  --block-device-mappings 'DeviceName=/dev/sdb,Ebs={VolumeId=vol-0123abcd}'
```

**3. Mount at boot (user data or start.sh)**

The new instance gets the same volume on e.g. `/dev/nvme1n1` (or `/dev/sdb`). In user data or in `start.sh`, mount it at `/data` before running warm-page-cache:

```bash
# Example: if the warm data volume is attached as second NVMe device
sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data || true
# or: sudo mount /dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0123abcd /data
```

Then `HF_HOME=/data/huggingface` and warm-page-cache use the warm volume. Root can stay from the AMI (small, fast boot); the warm data volume is reused so first I/O is much faster.

**Spot instances:** Yes, you can reuse the same volume for **spot** instances. Attach the existing volume in the launch template (same block device mapping as above). When the spot is interrupted, the instance terminates but the **volume persists** (it’s a pre-existing volume, not created with the instance). Launch a new spot instance with the same launch template to reattach the same warm volume. Only one instance can have the volume attached at a time, so one warm volume per spot instance; use a fleet of volumes if you run many spots in parallel.

**Caveats:**

- **You cannot detach the root volume** from an instance (AWS returns "Unable to detach root volume"). The volume you warm and reuse must be a **secondary** EBS volume (e.g. created separately and attached as `/dev/sdb`), not the root `/dev/xvda`. Create a dedicated data volume, attach it, mount at `/data`, warm it, then detach that volume and reattach it to new instances.
- The same volume can only be attached to one instance at a time. To run multiple instances with warm data, use one volume per instance (each warmed once and reattached via launch template), or use a shared filesystem (NFS/EFS).

## Checking EBS IOPS (slow first reads)

If page-cache warming or model load is slow at first (e.g. ~47 MiB/s for the first 9 GiB, then ~900 MiB/s), the volume may be using burst credits or have low baseline IOPS. To inspect EBS settings for an instance:

```bash
# On the instance (uses instance-id from metadata) or from a host with AWS CLI and permissions
/data/compressa-tests/check-instance-ebs-iops.sh
# or with explicit instance ID (e.g. i-080f709ab30281449)
/data/compressa-tests/check-instance-ebs-iops.sh i-080f709ab30281449
```

This prints volume type, size, IOPS, and throughput. For `gp3`, consider increasing IOPS/throughput; for `gp2`, consider migrating to `gp3` with higher baseline IOPS.
