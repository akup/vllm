#!/usr/bin/env bash
# Launch a plain Amazon Linux instance with a new io2 data volume (350 GiB, 50000 IOPS,
# delete_on_termination=false). Download the model with HuggingFace hub layout (HF_HOME/hub)
# so vLLM finds it and does not re-download. Then terminate the instance and print the volume ID.
#
# For gated models you may need to run on the instance: huggingface-cli login (then re-run download).
#
# Usage:
#   ./launch-download-model-volume.sh
#
# Env (optional):
#   REGION           default us-east-1
#   INSTANCE_TYPE    default r6i.2xlarge (no GPU needed for download)
#   KEY_NAME         default gnk-key-pair
#   SG_ID            default sg-01400fcd44b3c9742
#   MODEL_ID         default Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
#   VOLUME_SIZE      default 350 (GiB)
#   VOLUME_IOPS      default 50000
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-r6i.2xlarge}"
KEY_NAME="${KEY_NAME:-gnk-key-pair}"
SG_ID="${SG_ID:-sg-01400fcd44b3c9742}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$SCRIPT_DIR/gnk-key-pair.pem}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
VOLUME_SIZE="${VOLUME_SIZE:-350}"
VOLUME_IOPS="${VOLUME_IOPS:-50000}"

if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "SSH key not found: $SSH_KEY_PATH (set SSH_KEY_PATH if different)." >&2
  exit 1
fi

echo "Resolving latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-*-kernel-*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
  echo "Could not find Amazon Linux 2023 AMI." >&2
  exit 1
fi
echo "AMI: $AMI_ID"

echo "Launching instance with new data volume: ${VOLUME_SIZE} GiB io2, ${VOLUME_IOPS} IOPS, delete_on_termination=false..."
RUN_OUT=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  ${SUBNET_ID:+--subnet-id "$SUBNET_ID"} \
  --block-device-mappings "[
    {\"DeviceName\": \"/dev/sdb\", \"Ebs\": {\"VolumeSize\": ${VOLUME_SIZE}, \"VolumeType\": \"io2\", \"Iops\": ${VOLUME_IOPS}, \"DeleteOnTermination\": false}}
  ]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=download-model-$(date +%Y%m%d-%H%M)}]" \
  --output json)

INSTANCE_ID=$(echo "$RUN_OUT" | jq -r '.Instances[0].InstanceId')
echo "Instance: $INSTANCE_ID"

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Waiting for SSH..."
for i in $(seq 1 30); do
  if [ -n "$PUBLIC_IP" ] && ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes -i "$SSH_KEY_PATH" "ec2-user@$PUBLIC_IP" "echo ok" 2>/dev/null; then
    break
  fi
  [ $i -eq 30 ] && { echo "SSH timeout."; exit 1; }
  sleep 10
done

echo "Installing Hugging Face CLI and downloading model to attached volume (HF_HOME=/data/huggingface)..."
ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY_PATH" "ec2-user@$PUBLIC_IP" "bash -s" "$MODEL_ID" <<'REMOTE'
set -e
MODEL_ID="${1:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
# Second block device (data volume): nvme1n1 on Nitro, sdb on older
DEV=""
[ -b /dev/nvme1n1 ] && DEV=/dev/nvme1n1
[ -b /dev/sdb ] && [ -z "$DEV" ] && DEV=/dev/sdb
if [ -z "$DEV" ]; then
  echo "No second block device found." >&2
  exit 1
fi
sudo mkfs -t xfs -f "$DEV" 2>/dev/null || true
sudo mkdir -p /data
sudo mount "$DEV" /data
sudo chown ec2-user:ec2-user /data
mkdir -p /data/huggingface
export HF_HOME="/data/huggingface"
# pip may not be installed on minimal AL2023
sudo dnf install -y python3-pip 2>/dev/null || true
# Use hub cache layout (HF_HOME/hub) so vLLM finds the model; pass cache_dir to avoid HF_HOME/huggingface.
python3 -m pip install -q huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', cache_dir='/data/huggingface/hub')
"
sudo umount /data
echo "Download and unmount done."
REMOTE

echo "Fetching data volume ID..."
DATA_VOLUME_ID=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --output json | jq -r '.Reservations[0].Instances[0].BlockDeviceMappings[] | select(.DeviceName=="/dev/sdb") | .Ebs.VolumeId')
if [ -z "$DATA_VOLUME_ID" ] || [ "$DATA_VOLUME_ID" = "null" ]; then
  echo "Could not get data volume ID." >&2
  exit 1
fi

echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output text >/dev/null

echo ""
echo "Done. Model volume ID (attach as /dev/sdb and mount for HF_HOME or /data):"
echo "  $DATA_VOLUME_ID"
