#!/bin/bash
# Gonka EC2 user-data main script (run from bootstrap; do not embed in user-data).
# Expects: /etc/gonka-container.env already created, SSH_PUBLIC_KEY_VALUE in environment.
#
# Preload via AMI: copy this script to the image at e.g. /usr/local/bin/gonka-user-data-main.sh
# and make it executable (chmod +x). Then launch with:
#   ./launch-instance.sh ... --user-data-from-ami
# or with a custom path:
#   ./launch-instance.sh ... --user-data-from-ami /path/on/ami/gonka-user-data-main.sh

# Don't use set -e so script can continue after non-fatal failures
exec > >(tee -a /var/log/user-data.log) 2>&1
echo "=== User Data Main Script Started at $(date) ==="

retry_with_backoff() {
  local max_attempts=5
  local delay=2
  local attempt=1
  local cmd="$*"
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt/$max_attempts: $cmd"
    if eval "$cmd"; then
      return 0
    fi
    if [ $attempt -lt $max_attempts ]; then
      echo "Failed, waiting ${delay}s before retry..."
      sleep $delay
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done
  echo "WARNING: Command failed after $max_attempts attempts: $cmd"
  return 1
}

# Inject SSH public key if provided
if [ -n "${SSH_PUBLIC_KEY_VALUE:-}" ]; then
  echo "Adding SSH public key to authorized_keys (permanent)..."
  SSH_DIR="/home/ec2-user/.ssh"
  AUTH_KEYS_FILE="$SSH_DIR/authorized_keys"
  retry_with_backoff "mkdir -p $SSH_DIR" || true
  retry_with_backoff "chmod 700 $SSH_DIR" || true
  KEY_FINGERPRINT=$(echo "$SSH_PUBLIC_KEY_VALUE" | awk '{print $2}' | head -c 50)
  if [ -f "$AUTH_KEYS_FILE" ]; then
    if grep -qF "$KEY_FINGERPRINT" "$AUTH_KEYS_FILE" 2>/dev/null; then
      echo "SSH key already present in authorized_keys, skipping..."
    else
      echo "$SSH_PUBLIC_KEY_VALUE" >> "$AUTH_KEYS_FILE" || true
      [ ! -f "$AUTH_KEYS_FILE" ] || ! grep -qF "$KEY_FINGERPRINT" "$AUTH_KEYS_FILE" 2>/dev/null && echo "$SSH_PUBLIC_KEY_VALUE" >> "$AUTH_KEYS_FILE" || true
    fi
  else
    echo "$SSH_PUBLIC_KEY_VALUE" > "$AUTH_KEYS_FILE" || true
    [ ! -f "$AUTH_KEYS_FILE" ] && echo "$SSH_PUBLIC_KEY_VALUE" > "$AUTH_KEYS_FILE" || true
  fi
  chmod 600 "$AUTH_KEYS_FILE" 2>/dev/null || true
  chown -R ec2-user:ec2-user "$SSH_DIR" 2>/dev/null || true
  if [ -f "$AUTH_KEYS_FILE" ] && grep -qF "$KEY_FINGERPRINT" "$AUTH_KEYS_FILE" 2>/dev/null; then
    echo "SUCCESS: SSH key permanently added to authorized_keys"
  else
    echo "WARNING: Could not verify SSH key was added"
  fi
else
  echo "WARNING: No SSH public key provided (SSH_PUBLIC_KEY_VALUE not set)"
fi

# Load env (created by bootstrap)
if [ -f /etc/gonka-container.env ]; then
  echo "Sourcing /etc/gonka-container.env"
  set -a
  source /etc/gonka-container.env
  set +a
else
  echo "WARNING: /etc/gonka-container.env not found"
fi

# Mount EBS volume at /data/huggingface (attached at launch as /dev/sdf -> /dev/nvme1n1 on Nitro)
# echo "Waiting for EBS volume block device (attached at /dev/sdf, appears as /dev/nvme1n1 on Nitro)..."
# DATA_DEV=""
# for i in $(seq 1 60); do
#   if [ -b /dev/nvme1n1 ]; then
#     DATA_DEV="/dev/nvme1n1"
#     echo "Found block device $DATA_DEV"
#     break
#   fi
#   echo "  Attempt $i/60: device not ready, waiting 2s..."
#   sleep 2
# done
# if [ -n "$DATA_DEV" ]; then
#   mkdir -p /data/huggingface
#   if mountpoint -q /data/huggingface; then
#     echo "Already mounted: /data/huggingface"
#   else
#     MOUNTED=false
#     # Try whole device first (volume formatted as single fs)
#     for retry in 1 2 3 4 5; do
#       if mount "$DATA_DEV" /data/huggingface 2>/dev/null; then
#         echo "Mounted $DATA_DEV at /data/huggingface"
#         MOUNTED=true
#         break
#       fi
#       echo "  Mount attempt $retry/5 failed, waiting 5s..."
#       sleep 5
#     done
#     # If whole device failed, try first partition (volume may have been formatted with a partition table)
#     if [ "$MOUNTED" = false ] && [ -b "${DATA_DEV}p1" ]; then
#       echo "Trying partition ${DATA_DEV}p1..."
#       for retry in 1 2 3; do
#         if mount "${DATA_DEV}p1" /data/huggingface 2>/dev/null; then
#           echo "Mounted ${DATA_DEV}p1 at /data/huggingface"
#           chown ec2-user -R "/data/huggingface"
#           MOUNTED=true
#           break
#         fi
#         sleep 2
#       done
#     fi
#     if [ "$MOUNTED" = false ]; then
#       echo "WARNING: Could not mount $DATA_DEV (or partition) at /data/huggingface. Data on volume is NOT touched. To mount manually: sudo mount <device-or-partition> /data/huggingface"
#     fi
#   fi
# else
#   echo "WARNING: EBS volume block device did not appear within 120s; /data/huggingface may be unavailable"
# fi

# Fallback: try /dev/sdf (and sdf1 if partitioned)
SDF_DEV="/dev/sdf"
for i in $(seq 1 60); do
  if [ -b "$SDF_DEV" ]; then
    echo "Detected block device $SDF_DEV."
    break
  fi
  echo "Waiting for $SDF_DEV to become available... ($i/60)"
  sleep 1
done
if [ -b "$SDF_DEV" ] && ! mountpoint -q /data/huggingface; then
  echo "Attempting to mount $SDF_DEV to /data/huggingface..."
  mkdir -p /data/huggingface
  if mount "$SDF_DEV" /data/huggingface 2>/dev/null; then
    echo "Mounted $SDF_DEV to /data/huggingface successfully."
  elif [ -b "${SDF_DEV}1" ] && mount "${SDF_DEV}1" /data/huggingface 2>/dev/null; then
    echo "Mounted ${SDF_DEV}1 to /data/huggingface successfully."
  else
    echo "WARNING: Failed to mount $SDF_DEV (or partition). Data on volume is NOT touched."
  fi
fi

# Docker / container startup
echo "SKIP_DOCKER: ${SKIP_DOCKER:-true}"
if [ "${SKIP_DOCKER}" = "true" ]; then
  echo "SKIP_DOCKER is set to true - skipping Docker container startup"
  echo "Starting without container"

  /usr/local/bin/start-vllm.sh || {
    echo "ERROR: start.sh failed, but continuing..."
  }
else
  echo "Waiting for Docker to be ready..."
  for i in $(seq 1 30); do
    if systemctl is-active --quiet docker 2>/dev/null; then
      echo "Docker is ready"
      break
    fi
    echo "Waiting for Docker... ($i/30)"
    sleep 2
  done
  echo "Starting Docker container..."
  if [ -f /usr/local/bin/docker-run.sh ]; then
    /usr/local/bin/docker-run.sh || echo "ERROR: docker-run.sh failed, continuing..."
  else
    echo "WARNING: /usr/local/bin/docker-run.sh not found"
  fi
fi

# Final SSH verification
if [ -n "${SSH_PUBLIC_KEY_VALUE:-}" ]; then
  KEY_FINGERPRINT=$(echo "$SSH_PUBLIC_KEY_VALUE" | awk '{print $2}' | head -c 50)
  AUTH_KEYS_FILE="/home/ec2-user/.ssh/authorized_keys"
  if [ -f "$AUTH_KEYS_FILE" ] && grep -qF "$KEY_FINGERPRINT" "$AUTH_KEYS_FILE" 2>/dev/null; then
    echo "SUCCESS: Final verification: SSH key is present in $AUTH_KEYS_FILE"
  else
    echo "ERROR: Final verification FAILED: SSH key not found; attempting emergency fix..."
    SSH_DIR="/home/ec2-user/.ssh"
    mkdir -p "$SSH_DIR" 2>/dev/null || true
    echo "$SSH_PUBLIC_KEY_VALUE" >> "$AUTH_KEYS_FILE" 2>/dev/null || true
    chmod 600 "$AUTH_KEYS_FILE" 2>/dev/null || true
    chown ec2-user:ec2-user "$AUTH_KEYS_FILE" 2>/dev/null || true
  fi
fi

# CloudWatch agent status
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
if command -v amazon-cloudwatch-agent &>/dev/null; then
  if /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -m ec2 -a status 2>/dev/null | grep -q "running"; then
    echo "CloudWatch agent is running (logs: /aws/ec2/gonka/user-data, stream: $INSTANCE_ID)"
  else
    echo "CloudWatch agent not running; attempting to start..."
    [ -f /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json ] && \
      /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 \
        -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s || true
  fi
else
  echo "WARNING: CloudWatch agent not installed"
fi

echo "SUCCESS: User data main script completed at $(date)"
echo "Log: /var/log/user-data.log"
