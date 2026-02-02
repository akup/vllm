#!/usr/bin/env bash
# Build vLLM AMIs sequentially: prereq (CUDA + PyTorch) if missing or forced, then full base image.
# By default uses latest project-vllm-prereq-ami-* as source for the full build; build prereq first if none exists.
#
# Usage (from vLLM repo root):
#   ./ami/scripts/build-vllm-ami.sh [OPTIONS]
#
# Options:
#   --region REGION         AWS region (default: us-east-1)
#   --force-prereq          Rebuild prereq AMI even if one exists
#   --prereq-only           Build only prereq AMI, then exit
#   --skip-prereq           Skip prereq build; fail if no prereq AMI found (packer will use filter)
#   --instance-type TYPE    Instance type (default: r6i.4xlarge)
#   --verbose 0|1|2         vLLM build verbosity: 0=quiet, 1=normal, 2=verbose pip (default: 1)
#   --cache-bucket BUCKET   S3 bucket for vLLM build cache (optional)
#   --cache-prefix PREFIX   S3 key prefix for cache (default: vllm-wheels)
#   --iam-profile NAME      IAM instance profile for cache (optional)
#   --var KEY=VALUE         Pass through to packer (can repeat)
#
# Examples:
#   ./ami/scripts/build-vllm-ami.sh
#   ./ami/scripts/build-vllm-ami.sh --force-prereq --region us-east-2
#   ./ami/scripts/build-vllm-ami.sh --prereq-only
#   ./ami/scripts/build-vllm-ami.sh --verbose 2 --cache-bucket my-cache --cache-prefix vllm-wheels --iam-profile PackerVLLMCache

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AMI_DIR="$REPO_ROOT/ami"
REGION="${AWS_REGION:-us-east-1}"
FORCE_PREREQ=false
PREREQ_ONLY=false
SKIP_PREREQ=false
INSTANCE_TYPE="r6i.4xlarge"
VLLM_VERBOSE="1"
CACHE_BUCKET=""
CACHE_PREFIX=""
IAM_PROFILE=""
PACKER_VARS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --region) REGION="$2"; shift 2 ;;
    --force-prereq) FORCE_PREREQ=true; shift ;;
    --prereq-only) PREREQ_ONLY=true; shift ;;
    --skip-prereq) SKIP_PREREQ=true; shift ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    --verbose) VLLM_VERBOSE="$2"; shift 2 ;;
    --cache-bucket) CACHE_BUCKET="$2"; shift 2 ;;
    --cache-prefix) CACHE_PREFIX="$2"; shift 2 ;;
    --iam-profile) IAM_PROFILE="$2"; shift 2 ;;
    --var) PACKER_VARS+=("-var" "$2"); shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

cd "$REPO_ROOT"

# Find latest prereq AMI in the region (owned by self)
get_latest_prereq_ami() {
  aws ec2 describe-images \
    --region "$REGION" \
    --owners self \
    --filters "Name=name,Values=project-vllm-prereq-ami-*" "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
    --output text 2>/dev/null || true
}

# Build prereq AMI (CUDA + PyTorch)
build_prereq() {
  echo "Building prereq AMI (project-vllm-prereq-ami-*)..."
  packer build \
    -var "aws_region=$REGION" \
    -var "instance_type=$INSTANCE_TYPE" \
    "${PACKER_VARS[@]}" \
    "$AMI_DIR/packer-prereq.json"
}

# Build full vLLM base AMI (uses latest prereq when source_image not set)
build_full() {
  local extra_vars=()
  extra_vars+=(-var "vllm_build_verbose=$VLLM_VERBOSE")
  [ -n "$CACHE_BUCKET" ] && extra_vars+=(-var "vllm_build_cache_bucket=$CACHE_BUCKET")
  [ -n "$CACHE_PREFIX" ] && extra_vars+=(-var "vllm_build_cache_prefix=$CACHE_PREFIX")
  [ -n "$IAM_PROFILE" ] && extra_vars+=(-var "iam_instance_profile=$IAM_PROFILE")
  echo "Building full vLLM base AMI (source: latest project-vllm-prereq-ami-*; verbose=$VLLM_VERBOSE)..."
  packer build \
    -var "aws_region=$REGION" \
    -var "instance_type=$INSTANCE_TYPE" \
    "${extra_vars[@]}" \
    "${PACKER_VARS[@]}" \
    "$AMI_DIR/packer.json"
}

PREREQ_AMI=$(get_latest_prereq_ami)

if [ "$PREREQ_ONLY" = true ]; then
  build_prereq
  echo "Prereq-only build done. Run again without --prereq-only to build full base AMI."
  exit 0
fi

if [ -z "$PREREQ_AMI" ] || [ "$PREREQ_AMI" = "None" ]; then
  if [ "$SKIP_PREREQ" = true ]; then
    echo "ERROR: No prereq AMI (project-vllm-prereq-ami-*) found in region $REGION. Build prereq first or omit --skip-prereq."
    exit 1
  fi
  echo "No prereq AMI found in $REGION; building prereq first..."
  build_prereq
elif [ "$FORCE_PREREQ" = true ]; then
  echo "Force rebuild of prereq..."
  build_prereq
else
  echo "Using existing prereq AMI: $PREREQ_AMI"
fi

build_full
echo "Done. Full base AMI built from latest project-vllm-prereq-ami-*."
