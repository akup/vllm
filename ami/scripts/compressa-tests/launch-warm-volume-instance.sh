#!/usr/bin/env bash
# Launch an instance from the API AMI with a large io2 root volume (same spec as packer-api.json),
# run warm-page-cache on the instance to warm the disk, then terminate the instance and leave
# the root volume behind (delete_on_termination=false). Output the volume ID for attaching to
# other instances (e.g. in a launch template).
#
# Prerequisites:
#   - Model data under /data/huggingface on the instance (AMI or copy before warming).
#   - AWS CLI, SSH access (KEY_NAME + security group allowing SSH).
#
# Usage:
#   ./launch-warm-volume-instance.sh
#   KEY_NAME=my-key SG_ID=sg-xxx SUBNET_ID=subnet-xxx ./launch-warm-volume-instance.sh
#
# Env (optional):
#   AMI_ID          default ami-07cfbd29488f65527
#   REGION          default us-east-1
#   INSTANCE_TYPE   default r6i.2xlarge
#   KEY_NAME        default gnk-key-pair
#   SG_ID           default sg-01400fcd44b3c9742
#   SSH_KEY_PATH    default same dir as script: gnk-key-pair.pem
#   SUBNET_ID       optional (default VPC if omitted)
#   PARALLEL_READERS  default 16 for warm-page-cache
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AMI_ID="${AMI_ID:-ami-07cfbd29488f65527}"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-r6i.2xlarge}"
PARALLEL_READERS="${PARALLEL_READERS:-16}"
KEY_NAME="${KEY_NAME:-gnk-key-pair}"
SG_ID="${SG_ID:-sg-01400fcd44b3c9742}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$SCRIPT_DIR/gnk-key-pair.pem}"

if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "SSH key not found: $SSH_KEY_PATH (set SSH_KEY_PATH if different)." >&2
  exit 1
fi

echo "Launching instance from $AMI_ID in $REGION (instance type: $INSTANCE_TYPE)..."
echo "Root volume: 350 GiB io2, 64000 IOPS, delete_on_termination=false"

RUN_OUT=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  ${SUBNET_ID:+--subnet-id "$SUBNET_ID"} \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 350,
        "VolumeType": "io2",
        "Iops": 64000,
        "DeleteOnTermination": false
      }
    }
  ]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=warm-volume-temp-$(date +%Y%m%d-%H%M)}]" \
  --output json)

INSTANCE_ID=$(echo "$RUN_OUT" | jq -r '.Instances[0].InstanceId')
if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "null" ]; then
  echo "Failed to get instance ID from run-instances." >&2
  echo "$RUN_OUT" >&2
  exit 1
fi
echo "Instance ID: $INSTANCE_ID"

echo "Waiting for instance to be running..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP for SSH (use public IP if in public subnet, else describe for private IP and suggest SSM)
PUBLIC_IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null || true)
# Wait for SSH to be reachable
echo "Waiting for SSH (ec2-user@${PUBLIC_IP:-$INSTANCE_ID})..."
for i in $(seq 1 30); do
  if [ -n "$PUBLIC_IP" ] && ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes -i "$SSH_KEY_PATH" "ec2-user@$PUBLIC_IP" "echo ok" 2>/dev/null; then
    break
  fi
  [ $i -eq 30 ] && { echo "SSH timeout."; exit 1; }
  sleep 10
done

echo "Running warm-page-cache on instance (HF_HOME=/data/huggingface, PARALLEL_READERS=$PARALLEL_READERS)..."
ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY_PATH" "ec2-user@$PUBLIC_IP" \
  "sudo mkdir -p /data/huggingface; export HF_HOME=/data/huggingface; PARALLEL_READERS=$PARALLEL_READERS PYTHONUNBUFFERED=1 /data/compressa-tests/warm-page-cache.sh" || {
  echo "WARNING: warm-page-cache failed or no files under /data/huggingface; continuing to detach and terminate." >&2
}

echo "Fetching root volume ID..."
VOLUME_ID=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].BlockDeviceMappings[?DeviceName==`/dev/xvda`].Ebs.VolumeId' --output text)
if [ -z "$VOLUME_ID" ] || [ "$VOLUME_ID" = "None" ]; then
  echo "Failed to get volume ID." >&2
  exit 1
fi
echo "Root volume ID: $VOLUME_ID"

echo "Terminating instance $INSTANCE_ID (volume $VOLUME_ID will be preserved)..."
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" --output text

echo ""
echo "Done. Warm volume ID (attach to new instances as /dev/sdb or use in launch template):"
echo "  $VOLUME_ID"
