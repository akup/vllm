#!/usr/bin/env bash
# Print diagnostics to find why reads from a path (e.g. /data/huggingface) are slow.
# Run on the instance when warm-page-cache shows 0 GiB for many minutes.
#
# Usage:
#   ./disk-read-diagnostics.sh
#   ./disk-read-diagnostics.sh /data/huggingface
#
set -e

PATH_TO_CHECK="${1:-/data/huggingface}"

echo "=== Disk read diagnostics for ${PATH_TO_CHECK} ==="
echo ""

echo "--- Mount and filesystem ---"
if [ -d "$PATH_TO_CHECK" ]; then
  df -h "$PATH_TO_CHECK" 2>/dev/null || true
  echo ""
  mount | grep -E "$(df "$PATH_TO_CHECK" 2>/dev/null | tail -1 | awk '{print $1}')" || mount | grep -E "data|huggingface" || true
else
  echo "Path does not exist: $PATH_TO_CHECK"
  exit 1
fi

echo ""
echo "--- Block devices (lsblk) ---"
lsblk 2>/dev/null || true

echo ""
echo "--- Device for path ---"
DEV=$(findmnt -n -o SOURCE --target "$PATH_TO_CHECK" 2>/dev/null || true)
if [ -n "$DEV" ]; then
  echo "Device: $DEV"
  if [ -b "$DEV" ]; then
    echo "Read scheduler: $(cat /sys/block/$(basename "$DEV")/queue/scheduler 2>/dev/null || echo 'N/A')"
  fi
fi

echo ""
echo "--- EBS volume IOPS (if AWS instance) ---"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if command -v aws &>/dev/null; then
  INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null) || true
  if [ -n "$INSTANCE_ID" ]; then
    echo "Instance: $INSTANCE_ID"
    if [ -x "$SCRIPT_DIR/check-instance-ebs-iops.sh" ]; then
      "$SCRIPT_DIR/check-instance-ebs-iops.sh" 2>/dev/null || true
    else
      echo "Run: $SCRIPT_DIR/check-instance-ebs-iops.sh (or /data/compressa-tests/check-instance-ebs-iops.sh on instance)"
    fi
  fi
else
  echo "AWS CLI not found; run check-instance-ebs-iops.sh from a host with AWS CLI"
fi

echo ""
echo "--- Quick read test (first 1 GiB of first file under path) ---"
FIRST_FILE=$(find "$PATH_TO_CHECK" -type f -name "*.safetensors" 2>/dev/null | head -1)
if [ -n "$FIRST_FILE" ] && [ -r "$FIRST_FILE" ]; then
  echo "File: $FIRST_FILE"
  echo "Size: $(ls -lh "$FIRST_FILE" | awk '{print $5}')"
  echo "Reading first 1 GiB with dd (measure speed)..."
  dd if="$FIRST_FILE" of=/dev/null bs=1M count=1024 2>&1 || true
else
  echo "No readable .safetensors file found under $PATH_TO_CHECK"
fi

echo ""
echo "--- I/O wait (run during warm-page-cache: vmstat 5 or iostat -x 5) ---"
echo "If iowait is high, disk is the bottleneck. If not, check NFS/network storage."
vmstat 1 3 2>/dev/null || true

echo ""
echo "Done. Slow first reads are often: EBS burst credits exhausted, gp2/gp3 baseline IOPS low, or NFS/EFS cold."
