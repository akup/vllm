#!/usr/bin/env bash
# Create S3 bucket and IAM instance profile for vLLM build cache.
# Run once (e.g. from your laptop with AWS CLI and credentials).
# Then run: packer build -var "vllm_build_cache_bucket=BUCKET" -var "iam_instance_profile=PROFILE" ami/packer.json

set -e
BUCKET="${1:-gonka-vllm-build-cache}"
REGION="${2:-us-east-1}"
PROFILE_NAME="${3:-PackerVLLMCache}"
ROLE_NAME="${PROFILE_NAME}Role"
POLICY_NAME="${PROFILE_NAME}Policy"

echo "Bucket: $BUCKET"
echo "Region: $REGION"
echo "Instance profile: $PROFILE_NAME"
echo ""

# 1. Create S3 bucket
echo "Creating S3 bucket s3://$BUCKET ..."
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  echo "Bucket $BUCKET already exists."
else
  aws s3 mb "s3://$BUCKET" --region "$REGION"
  echo "Bucket created."
fi

# 2. IAM policy: allow GetObject, PutObject, ListBucket on bucket/prefix
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"
POLICY_DOC=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/vllm-wheels/*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::${BUCKET}",
      "Condition": { "StringLike": { "s3:prefix": ["vllm-wheels/*"] } }
    }
  ]
}
EOF
)

echo "Creating IAM policy $POLICY_NAME ..."
if aws iam get-policy --policy-arn "$POLICY_ARN" 2>/dev/null; then
  echo "Policy $POLICY_NAME already exists. Creating new version..."
  aws iam create-policy-version --policy-arn "$POLICY_ARN" --policy-document "$POLICY_DOC" --set-as-default
else
  aws iam create-policy --policy-name "$POLICY_NAME" --policy-document "$POLICY_DOC"
fi
echo "Policy OK."

# 3. IAM role for EC2
TRUST_DOC='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
echo "Creating IAM role $ROLE_NAME ..."
if aws iam get-role --role-name "$ROLE_NAME" 2>/dev/null; then
  echo "Role $ROLE_NAME already exists."
else
  aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document "$TRUST_DOC"
fi
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn "$POLICY_ARN"
echo "Role OK."

# 4. Instance profile
echo "Creating instance profile $PROFILE_NAME ..."
if aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null; then
  echo "Instance profile $PROFILE_NAME already exists."
  if ! aws iam get-instance-profile --instance-profile-name "$PROFILE_NAME" --query "InstanceProfile.Roles[?RoleName=='$ROLE_NAME']" --output text 2>/dev/null | grep -q .; then
    aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
  fi
else
  aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
  aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
fi
echo "Instance profile OK."

echo ""
echo "Done. Wait a few seconds for IAM to propagate, then run:"
echo ""
echo "  cd /path/to/vllm"
echo "  packer build \\"
echo "    -var \"vllm_build_cache_bucket=$BUCKET\" \\"
echo "    -var \"iam_instance_profile=$PROFILE_NAME\" \\"
echo "    -var \"aws_region=$REGION\" \\"
echo "    ami/packer.json"
echo ""
