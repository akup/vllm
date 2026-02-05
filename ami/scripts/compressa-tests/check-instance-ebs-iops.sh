#!/usr/bin/env bash
# Print EBS volume and IOPS/throughput settings for an EC2 instance.
# Run on the instance (uses instance metadata) or from your machine with INSTANCE_ID.
#
# Usage:
#   ./check-instance-ebs-iops.sh              # current instance (from metadata)
#   INSTANCE_ID=i-080f709ab30281449 ./check-instance-ebs-iops.sh
#   ./check-instance-ebs-iops.sh i-080f709ab30281449
#
# Requires: AWS CLI, jq (optional, for pretty output)
set -e

INSTANCE_ID="${1:-${INSTANCE_ID:-}}"
if [ -z "$INSTANCE_ID" ]; then
    INSTANCE_ID=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null) || true
fi
if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 [INSTANCE_ID] or set INSTANCE_ID" >&2
    echo "  Example: INSTANCE_ID=i-080f709ab30281449 $0" >&2
    exit 1
fi

echo "Instance: $INSTANCE_ID"
echo "Block devices and volumes:"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].BlockDeviceMappings[*].[DeviceName,Ebs.VolumeId]' \
    --output text | while read -r dev vol; do
    [ -z "$vol" ] && continue
    echo "  $dev -> $vol"
    aws ec2 describe-volumes --volume-ids "$vol" \
        --query 'Volumes[0].[VolumeType,Iops,Throughput,Size]' \
        --output text 2>/dev/null | while read -r vtype iops throughput size; do
        [ "$iops" = "None" ] && iops="N/A"
        [ "$throughput" = "None" ] && throughput="N/A"
        echo "    Type=$vtype Size=${size}GiB IOPS=${iops:-N/A} Throughput=${throughput:-N/A} MiB/s"
    done
done
