#!/usr/bin/env bash
# One-time AWS infrastructure setup for LOB-RL training.
#
# Creates (idempotent):
#   - S3 bucket for cache + results
#   - ECR repository for Docker image
#   - IAM role + instance profile with least-privilege policy
#   - Security group (outbound all, inbound SSH for debugging)
#
# Usage:
#   ./aws/setup.sh
#   ./aws/setup.sh --region us-east-1   # override region
#
# After running, export the printed env vars (or add to .envrc / shell profile).

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
BUCKET_NAME="${AWS_S3_BUCKET:-lob-rl-training}"
ECR_REPO_NAME="lob-rl"
ROLE_NAME="lob-rl-training"
SG_NAME="lob-rl-training"
PROJECT_TAG="lob-rl"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO_NAME}"

echo "=== LOB-RL AWS Setup ==="
echo "Region:   $REGION"
echo "Account:  $ACCOUNT_ID"
echo "Bucket:   $BUCKET_NAME"
echo "ECR:      $ECR_URI"
echo ""

# ─── S3 Bucket ───────────────────────────────────────────────────────
echo "--- S3 Bucket ---"
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket $BUCKET_NAME already exists."
else
    if [[ "$REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    else
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "Created bucket: $BUCKET_NAME"
fi

# Enable versioning for safety
aws s3api put-bucket-versioning --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled
echo "Versioning enabled."

# Tag
aws s3api put-bucket-tagging --bucket "$BUCKET_NAME" \
    --tagging "TagSet=[{Key=Project,Value=$PROJECT_TAG}]"

# ─── ECR Repository ──────────────────────────────────────────────────
echo ""
echo "--- ECR Repository ---"
if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "ECR repo $ECR_REPO_NAME already exists."
else
    aws ecr create-repository --repository-name "$ECR_REPO_NAME" --region "$REGION" \
        --image-scanning-configuration scanOnPush=true
    echo "Created ECR repo: $ECR_REPO_NAME"
fi

# ─── IAM Role + Instance Profile ─────────────────────────────────────
echo ""
echo "--- IAM Role ---"

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "ec2.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}'

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    echo "IAM role $ROLE_NAME already exists."
else
    aws iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --tags "Key=Project,Value=$PROJECT_TAG"
    echo "Created IAM role: $ROLE_NAME"
fi

# Inline policy: least privilege
POLICY_DOC=$(cat <<POLICY_EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3ReadWrite",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
      "Resource": [
        "arn:aws:s3:::${BUCKET_NAME}",
        "arn:aws:s3:::${BUCKET_NAME}/*"
      ]
    },
    {
      "Sid": "ECRPull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetAuthorizationToken"
      ],
      "Resource": "*"
    },
    {
      "Sid": "SelfTerminate",
      "Effect": "Allow",
      "Action": "ec2:TerminateInstances",
      "Resource": "arn:aws:ec2:*:${ACCOUNT_ID}:instance/*",
      "Condition": {
        "StringEquals": {"aws:ResourceTag/Project": "${PROJECT_TAG}"}
      }
    }
  ]
}
POLICY_EOF
)

aws iam put-role-policy --role-name "$ROLE_NAME" \
    --policy-name "${ROLE_NAME}-policy" \
    --policy-document "$POLICY_DOC"
echo "Policy attached."

# Instance profile
if aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" >/dev/null 2>&1; then
    echo "Instance profile $ROLE_NAME already exists."
else
    aws iam create-instance-profile --instance-profile-name "$ROLE_NAME"
    aws iam add-role-to-instance-profile \
        --instance-profile-name "$ROLE_NAME" \
        --role-name "$ROLE_NAME"
    echo "Created instance profile: $ROLE_NAME"
    echo "Waiting 10s for IAM propagation..."
    sleep 10
fi

# ─── Security Group ──────────────────────────────────────────────────
echo ""
echo "--- Security Group ---"

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" --output text)

if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then
    echo "WARNING: No default VPC found in $REGION. Create a VPC or set AWS_SECURITY_GROUP manually."
    SG_ID="NONE"
else
    SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
        --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || echo "None")

    if [[ "$SG_ID" == "None" || -z "$SG_ID" ]]; then
        SG_ID=$(aws ec2 create-security-group --region "$REGION" \
            --group-name "$SG_NAME" \
            --description "LOB-RL training instances" \
            --vpc-id "$VPC_ID" \
            --query "GroupId" --output text)

        # Inbound SSH (for debugging failed instances)
        aws ec2 authorize-security-group-ingress --region "$REGION" \
            --group-id "$SG_ID" \
            --protocol tcp --port 22 --cidr 0.0.0.0/0

        # Tag
        aws ec2 create-tags --region "$REGION" \
            --resources "$SG_ID" \
            --tags "Key=Project,Value=$PROJECT_TAG"

        echo "Created security group: $SG_ID"
    else
        echo "Security group $SG_NAME already exists: $SG_ID"
    fi
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  Setup complete. Export these env vars:"
echo "=========================================="
echo ""
echo "export AWS_S3_BUCKET=\"$BUCKET_NAME\""
echo "export AWS_ECR_REPO=\"$ECR_URI\""
echo "export AWS_REGION=\"$REGION\""
echo "export AWS_IAM_PROFILE=\"$ROLE_NAME\""
echo "export AWS_SECURITY_GROUP=\"$SG_ID\""
echo ""
echo "Optional (for SSH debugging of failed instances):"
echo "export AWS_KEY_NAME=\"your-ec2-key-pair-name\""
echo ""
echo "Push Docker image to ECR:"
echo "  aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI"
echo "  docker buildx build --platform linux/amd64 -t ${ECR_URI}:latest --push ."
echo ""
echo "Upload cache:"
echo "  AWS_S3_BUCKET=$BUCKET_NAME ./aws/upload-cache.sh"
