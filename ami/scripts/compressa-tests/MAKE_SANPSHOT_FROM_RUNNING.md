```bash
# Replace with your instance ID and desired AMI name
INSTANCE_ID="i-0123456789abcdef0"
AMI_NAME="my-ami-$(date +%Y%m%d-%H%M)"

aws ec2 create-image \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "AMI from running instance" \
  --no-reboot
```