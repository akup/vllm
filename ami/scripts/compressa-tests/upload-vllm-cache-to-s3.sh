#!/usr/bin/env bash
# Tar VLLM_CACHE_ROOT and upload to S3. Does not prepare the cache; run warmup first if needed.
#
# Usage:
#   upload-vllm-cache-to-s3.sh MODEL_NAME
#   MODEL_NAME is used only for the S3 key path (e.g. vllm-cache/Qwen-Qwen2.5-7B-Instruct/TP1/).
#
# Bucket: VLLM_CACHE_S3_BUCKET (hardcoded default below). Created if it does not exist.
#
# Instance IAM profile (for upload and for download-vllm-cache-from-s3.sh) must allow:
#   - s3:CreateBucket, s3:ListBucket (for create-if-missing and list)
#   - s3:PutObject, s3:GetObject on s3://BUCKET/vllm-cache/*
# Example policy (attach to instance profile or role):
#   {
#     "Version": "2012-10-17",
#     "Statement": [
#       {
#         "Effect": "Allow",
#         "Action": ["s3:CreateBucket", "s3:ListBucket"],
#         "Resource": "arn:aws:s3:::VLLM_CACHE_S3_BUCKET"
#       },
#       {
#         "Effect": "Allow",
#         "Action": ["s3:PutObject", "s3:GetObject"],
#         "Resource": "arn:aws:s3:::VLLM_CACHE_S3_BUCKET/vllm-cache/*"
#       }
#     ]
#   }
#
# Requires: AWS CLI.
set -e

# Hardcoded default bucket; override with VLLM_CACHE_S3_BUCKET if set
VLLM_CACHE_S3_BUCKET="${VLLM_CACHE_S3_BUCKET:-gonka-vllm-cache}"
AWS_REGION="${AWS_DEFAULT_REGION:-$AWS_REGION}"
AWS_REGION="${AWS_REGION:-us-east-1}"

MODEL_NAME="${1:?Usage: $0 MODEL_NAME}"

VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/data/vllm-cache}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# S3 key: vllm-cache/MODEL_SLUG/TP{N}/vllm-cache.tar.gz
MODEL_SLUG="${MODEL_NAME//\//-}"
S3_URI="s3://${VLLM_CACHE_S3_BUCKET}/vllm-cache/${MODEL_SLUG}/TP${TENSOR_PARALLEL_SIZE}/vllm-cache.tar.gz"

if [ ! -d "$VLLM_CACHE_ROOT" ]; then
    echo "ERROR: VLLM_CACHE_ROOT not a directory: $VLLM_CACHE_ROOT (run warmup first, or set VLLM_CACHE_ROOT)" >&2
    exit 1
fi

# Create bucket if it does not exist
if ! aws s3api head-bucket --bucket "$VLLM_CACHE_S3_BUCKET" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] Creating bucket s3://$VLLM_CACHE_S3_BUCKET (region=$AWS_REGION)"
    aws s3 mb "s3://${VLLM_CACHE_S3_BUCKET}" --region "$AWS_REGION"
fi

# Tar cache with one top-level dir so extract -C /data gives /data/vllm-cache
PARENT=$(dirname "$VLLM_CACHE_ROOT")
CHILD=$(basename "$VLLM_CACHE_ROOT")
TARBALL="/tmp/vllm-cache-$$.tar.gz"
trap "rm -f $TARBALL" EXIT

echo "[$(date +%H:%M:%S)] Creating $TARBALL from $VLLM_CACHE_ROOT"
tar -czf "$TARBALL" -C "$PARENT" "$CHILD"

echo "[$(date +%H:%M:%S)] Uploading to $S3_URI"
aws s3 cp "$TARBALL" "$S3_URI"

echo "[$(date +%H:%M:%S)] Done. Download on instances with:"
echo "  download-vllm-cache-from-s3.sh $MODEL_NAME"
