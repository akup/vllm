#!/bin/bash
# Script to configure and start CloudWatch agent early in boot process
# This runs before user-data/cloud-init to capture all logs from the start

set -e

CONFIG_FILE="/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"
CONFIG_TEMPLATE="/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json.template"

# Get instance metadata
# Try IMDS (Instance Metadata Service) first - this is the standard AWS way
# 169.254.169.254 is a special link-local address that AWS provides to EC2 instances
INSTANCE_ID=""
REGION=""

# Method 1: Try IMDS (Instance Metadata Service v1)
if [ -z "$INSTANCE_ID" ]; then
  INSTANCE_ID=$(curl -s --max-time 5 --connect-timeout 2 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
fi

# Method 2: Try IMDSv2 (requires token)
if [ -z "$INSTANCE_ID" ]; then
  TOKEN=$(curl -s --max-time 5 --connect-timeout 2 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
  if [ -n "$TOKEN" ]; then
    INSTANCE_ID=$(curl -s --max-time 5 --connect-timeout 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "")
  fi
fi

# Method 3: Try AWS CLI (if available and IAM role is attached)
if [ -z "$INSTANCE_ID" ] && command -v aws >/dev/null 2>&1; then
  # Get instance ID from AWS CLI using the instance's own identity
  INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=network-interface.private-ip-address,Values=$(hostname -I | awk '{print $1}')" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text 2>/dev/null || echo "")
  
  # Alternative: Get from instance identity document
  if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" == "None" ]; then
    INSTANCE_ID=$(aws ec2 describe-instances \
      --filters "Name=instance-state-name,Values=running" \
      --query 'Reservations[0].Instances[0].InstanceId' \
      --output text 2>/dev/null || echo "")
  fi
fi

# Get region - try IMDS first
if [ -z "$REGION" ]; then
  REGION=$(curl -s --max-time 5 --connect-timeout 2 http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "")
fi

# Try IMDSv2 for region
if [ -z "$REGION" ] && [ -n "$TOKEN" ]; then
  REGION=$(curl -s --max-time 5 --connect-timeout 2 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "")
fi

# Try AWS CLI for region
if [ -z "$REGION" ] && command -v aws >/dev/null 2>&1; then
  REGION=$(aws configure get region 2>/dev/null || echo "")
  if [ -z "$REGION" ]; then
    REGION=$(aws ec2 describe-availability-zones --query 'AvailabilityZones[0].RegionName' --output text 2>/dev/null || echo "")
  fi
fi

# Fallback defaults
INSTANCE_ID="${INSTANCE_ID:-unknown}"
REGION="${REGION:-us-east-1}"

# Log what we got
echo "Instance ID: ${INSTANCE_ID:-unknown} (Region: ${REGION:-us-east-1})"
if [ "$INSTANCE_ID" = "unknown" ] || [ -z "$INSTANCE_ID" ]; then
  echo "WARNING: Could not determine instance ID, using 'unknown'"
  echo "CloudWatch logs will use 'unknown' as stream name"
fi

# Check if CloudWatch agent is installed
# Check both PATH and the standard installation location
CLOUDWATCH_AGENT_BIN="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent"
CLOUDWATCH_AGENT_CTL_BIN="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"

if ! command -v amazon-cloudwatch-agent &> /dev/null && [ ! -f "$CLOUDWATCH_AGENT_BIN" ]; then
  echo "CloudWatch agent not found, skipping configuration"
  exit 0
fi

# Determine which ctl command to use
if [ -f "$CLOUDWATCH_AGENT_CTL_BIN" ]; then
  CLOUDWATCH_AGENT_CTL="$CLOUDWATCH_AGENT_CTL_BIN"
elif command -v amazon-cloudwatch-agent-ctl &> /dev/null; then
  CLOUDWATCH_AGENT_CTL="amazon-cloudwatch-agent-ctl"
else
  echo "ERROR: CloudWatch agent ctl command not found"
  exit 1
fi

# Verify the ctl command exists and is executable
if [ ! -x "$CLOUDWATCH_AGENT_CTL" ] && ! command -v "$CLOUDWATCH_AGENT_CTL" &> /dev/null; then
  echo "ERROR: CloudWatch agent ctl command is not executable: $CLOUDWATCH_AGENT_CTL"
  exit 1
fi

# Create config directory if it doesn't exist
mkdir -p /opt/aws/amazon-cloudwatch-agent/etc

# If template exists, use it; otherwise create default config
if [ -f "$CONFIG_TEMPLATE" ]; then
  # Copy template and replace instance_id placeholder
  cp "$CONFIG_TEMPLATE" "$CONFIG_FILE"
  sed -i "s/{instance_id}/$INSTANCE_ID/g" "$CONFIG_FILE"
  echo "CloudWatch agent config created from template (instance: $INSTANCE_ID)"
else
  # Create default config if template doesn't exist
  cat > "$CONFIG_FILE" <<EOF
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "root"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/user-data.log",
            "log_group_name": "/aws/ec2/gonka/user-data",
            "log_stream_name": "$INSTANCE_ID",
            "retention_in_days": 7
          },
          {
            "file_path": "/var/log/cloud-init-output.log",
            "log_group_name": "/aws/ec2/gonka/cloud-init",
            "log_stream_name": "$INSTANCE_ID",
            "retention_in_days": 7
          },
          {
            "file_path": "/var/log/docker-container.log",
            "log_group_name": "/aws/ec2/gonka/docker-container",
            "log_stream_name": "$INSTANCE_ID",
            "retention_in_days": 7
          }
        ]
      }
    }
  }
}
EOF
  echo "CloudWatch agent config created (instance: $INSTANCE_ID)"
fi

# Start CloudWatch agent
echo "Starting CloudWatch agent..."
$CLOUDWATCH_AGENT_CTL \
  -a fetch-config \
  -m ec2 \
  -c file:"$CONFIG_FILE" \
  -s || {
  echo "WARNING: Failed to start CloudWatch agent (check IAM permissions)"
  exit 0  # Don't fail boot if CloudWatch fails
}

echo "CloudWatch agent started successfully"
echo "Logs will be sent to:"
echo "  - /aws/ec2/gonka/user-data (stream: $INSTANCE_ID)"
echo "  - /aws/ec2/gonka/cloud-init (stream: $INSTANCE_ID)"
echo "  - /aws/ec2/gonka/docker-container (stream: $INSTANCE_ID)"
echo "  - /aws/ec2/gonka/vllm-poc (stream: $INSTANCE_ID)"

