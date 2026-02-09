#!/usr/bin/env bash
# Launch an EC2 Spot instance for training.
#
# Usage:
#   EXP_NAME=<label> ./aws/launch.sh <train.py args...>
#
# Examples:
#   # LSTM training (auto-detects GPU instance)
#   ./aws/launch.sh \
#     --recurrent --bar-size 1000 --execution-cost \
#     --ent-coef 0.05 --learning-rate 0.001 \
#     --shuffle-split --seed 42 \
#     --total-timesteps 5000000 --checkpoint-freq 500000
#
#   # MLP training (auto-detects CPU instance)
#   ./aws/launch.sh \
#     --bar-size 1000 --execution-cost \
#     --total-timesteps 5000000
#
#   # Override instance type
#   AWS_INSTANCE_TYPE=g5.2xlarge ./aws/launch.sh --recurrent ...
#
# Required env vars:
#   AWS_S3_BUCKET     S3 bucket for cache/results
#   AWS_ECR_REPO      Full ECR URI (account.dkr.ecr.region.amazonaws.com/repo)
#
# Optional env vars:
#   AWS_INSTANCE_TYPE  Override auto-detect (default: g5.xlarge for GPU, c7a.4xlarge for CPU)
#   AWS_REGION         Default: us-east-1
#   AWS_SECURITY_GROUP Security group ID (from setup.sh)
#   AWS_IAM_PROFILE    IAM instance profile name (from setup.sh)
#   AWS_KEY_NAME       EC2 key pair for SSH debugging
#   EXP_NAME           Experiment label (auto-detected from args if unset)

set -euo pipefail

S3_BUCKET="${AWS_S3_BUCKET:?Set AWS_S3_BUCKET (e.g. lob-rl-training)}"
ECR_REPO="${AWS_ECR_REPO:?Set AWS_ECR_REPO (e.g. 123456.dkr.ecr.us-east-1.amazonaws.com/lob-rl)}"
REGION="${AWS_REGION:-us-east-1}"
IAM_PROFILE="${AWS_IAM_PROFILE:-lob-rl-training}"
SECURITY_GROUP="${AWS_SECURITY_GROUP:-}"
KEY_NAME="${AWS_KEY_NAME:-}"

# ─── Auto-detect experiment name from training args ──────────────────
if [[ -z "${EXP_NAME:-}" ]]; then
    EXP_NAME="mlp"
    for arg in "$@"; do
        case "$arg" in
            --recurrent) EXP_NAME="lstm" ;;
            --frame-stack) EXP_NAME="framestack" ;;
        esac
    done
fi

# ─── Auto-detect instance type ───────────────────────────────────────
# GPU (g5.xlarge, A10G 24GB) for --recurrent, CPU (c7a.4xlarge) otherwise
NEEDS_GPU=false
for arg in "$@"; do
    case "$arg" in
        --recurrent) NEEDS_GPU=true ;;
    esac
done

if [[ -z "${AWS_INSTANCE_TYPE:-}" ]]; then
    if $NEEDS_GPU; then
        INSTANCE_TYPE="g5.xlarge"
    else
        INSTANCE_TYPE="c7a.4xlarge"
    fi
else
    INSTANCE_TYPE="$AWS_INSTANCE_TYPE"
fi

# ─── Detect if GPU instance (for --gpus all flag) ────────────────────
IS_GPU_INSTANCE=false
case "$INSTANCE_TYPE" in
    g[0-9]*|p[0-9]*|trn*|inf*|dl*) IS_GPU_INSTANCE=true ;;
esac

# ─── Resolve AMI (Deep Learning AMI — Ubuntu, has Docker + NVIDIA runtime) ─
# Try PyTorch DL AMI first (has Docker + NVIDIA runtime pre-installed)
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*(Ubuntu 22.04)*" \
        "Name=architecture,Values=x86_64" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text 2>/dev/null)

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    # Fallback: Base DL AMI with NVIDIA driver (has Docker, no PyTorch)
    AMI_ID=$(aws ec2 describe-images --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=architecture,Values=x86_64" \
            "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)
fi

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    # Last resort: any Deep Learning AMI with CUDA
    AMI_ID=$(aws ec2 describe-images --region "$REGION" \
        --owners amazon \
        --filters \
            "Name=name,Values=Deep Learning Base AMI with Single CUDA (Ubuntu 22.04)*" \
            "Name=architecture,Values=x86_64" \
            "Name=state,Values=available" \
        --query "sort_by(Images, &CreationDate)[-1].ImageId" \
        --output text 2>/dev/null)
fi

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "ERROR: Could not find a suitable Deep Learning AMI in $REGION."
    echo "Try: aws ec2 describe-images --region $REGION --owners amazon --filters 'Name=name,Values=Deep Learning*Ubuntu*22.04*' --query 'Images[*].{Name:Name,Id:ImageId}' --output table"
    exit 1
fi

# ─── Build user-data script ──────────────────────────────────────────
TRAIN_ARGS="$*"

DOCKER_GPU_FLAG=""
if $IS_GPU_INSTANCE; then
    DOCKER_GPU_FLAG="--gpus all"
fi

USER_DATA=$(cat <<'USERDATA_OUTER'
#!/bin/bash
set -euo pipefail
exec > /var/log/bootstrap.log 2>&1
echo "=== Bootstrap started at $(date) ==="

REGION="__REGION__"
S3_BUCKET="__S3_BUCKET__"
ECR_REPO="__ECR_REPO__"
EXP_NAME="__EXP_NAME__"
TRAIN_ARGS="__TRAIN_ARGS__"
DOCKER_GPU_FLAG="__DOCKER_GPU_FLAG__"

# ─── Get instance ID (IMDSv2) ────────────────────────────────────────
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-id)
echo "Instance: $INSTANCE_ID"

RUN_ID="${EXP_NAME}_${INSTANCE_ID}"
WORK_DIR="/tmp/lob-rl"
CACHE_DIR="$WORK_DIR/cache/mes"
OUTPUT_DIR="$WORK_DIR/runs/$RUN_ID"
mkdir -p "$CACHE_DIR" "$OUTPUT_DIR"

# Upload bootstrap log periodically
upload_logs() {
    cp /var/log/bootstrap.log "$OUTPUT_DIR/bootstrap.log" 2>/dev/null || true
    aws s3 cp "$OUTPUT_DIR/bootstrap.log" "s3://$S3_BUCKET/runs/$RUN_ID/bootstrap.log" \
        --region "$REGION" --quiet 2>/dev/null || true
}
trap upload_logs EXIT

# ─── Spot interruption monitor (background) ──────────────────────────
(
    while true; do
        sleep 5
        TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 30" 2>/dev/null || true)
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            -H "X-aws-ec2-metadata-token: $TOKEN" \
            "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null || echo "000")
        if [[ "$HTTP_CODE" == "200" ]]; then
            echo "=== SPOT INTERRUPTION DETECTED at $(date) ==="
            echo "Uploading partial results..."
            aws s3 sync "$OUTPUT_DIR/" "s3://$S3_BUCKET/runs/$RUN_ID/" \
                --region "$REGION" --quiet 2>/dev/null || true
            upload_logs
            echo "Partial upload complete. Instance will be terminated by AWS."
            exit 0
        fi
    done
) &
SPOT_MONITOR_PID=$!
echo "Spot monitor started (PID: $SPOT_MONITOR_PID)"

# ─── Download cache from S3 ──────────────────────────────────────────
echo "=== Downloading cache from S3 ($(date)) ==="
aws s3 sync "s3://$S3_BUCKET/cache/mes/" "$CACHE_DIR/" --region "$REGION"
CACHE_COUNT=$(find "$CACHE_DIR" -name "*.npz" 2>/dev/null | wc -l)
echo "Downloaded $CACHE_COUNT .npz files."
upload_logs

# ─── Pull Docker image from ECR ──────────────────────────────────────
echo "=== Pulling Docker image ($(date)) ==="
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$(echo "$ECR_REPO" | cut -d/ -f1)"
docker pull "${ECR_REPO}:latest"
echo "Image pulled."
upload_logs

# ─── Run training ────────────────────────────────────────────────────
echo "=== Starting training ($(date)) ==="
echo "Args: $TRAIN_ARGS"
echo "GPU flag: $DOCKER_GPU_FLAG"

DOCKER_EXIT=0
docker run --rm \
    $DOCKER_GPU_FLAG \
    -v "$CACHE_DIR":/workspace/cache/mes:ro \
    -v "$OUTPUT_DIR":/workspace/runs/"$RUN_ID" \
    -e EXP_NAME="$EXP_NAME" \
    -e RUN_ID="$INSTANCE_ID" \
    "${ECR_REPO}:latest" \
    --cache-dir /workspace/cache/mes/ $TRAIN_ARGS \
    || DOCKER_EXIT=$?

echo "=== Training exited with code $DOCKER_EXIT ($(date)) ==="
upload_logs

# ─── Upload results to S3 ────────────────────────────────────────────
echo "=== Uploading results to S3 ($(date)) ==="
aws s3 sync "$OUTPUT_DIR/" "s3://$S3_BUCKET/runs/$RUN_ID/" --region "$REGION"
echo "Results uploaded."
upload_logs

# ─── Stop spot monitor ───────────────────────────────────────────────
kill $SPOT_MONITOR_PID 2>/dev/null || true

# ─── Self-terminate or stay alive ────────────────────────────────────
if [[ $DOCKER_EXIT -eq 0 ]]; then
    echo "Training succeeded. Self-terminating..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
else
    echo "Training failed (exit $DOCKER_EXIT). Instance staying alive for SSH debugging."
    echo "  SSH: ssh -i <key> ubuntu@<public-ip>"
    echo "  Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
    # Keep instance alive (cloud-init has already finished; instance stays running)
fi
USERDATA_OUTER
)

# Substitute variables into user-data
USER_DATA="${USER_DATA//__REGION__/$REGION}"
USER_DATA="${USER_DATA//__S3_BUCKET__/$S3_BUCKET}"
USER_DATA="${USER_DATA//__ECR_REPO__/$ECR_REPO}"
USER_DATA="${USER_DATA//__EXP_NAME__/$EXP_NAME}"
USER_DATA="${USER_DATA//__TRAIN_ARGS__/$TRAIN_ARGS}"
USER_DATA="${USER_DATA//__DOCKER_GPU_FLAG__/$DOCKER_GPU_FLAG}"

# Base64 encode for EC2 API
USER_DATA_B64=$(echo "$USER_DATA" | base64)

# ─── Build launch spec ───────────────────────────────────────────────
echo "=== Launching EC2 Spot Instance ==="
echo "Instance type: $INSTANCE_TYPE"
echo "GPU instance:  $IS_GPU_INSTANCE"
echo "AMI:           $AMI_ID"
echo "Region:        $REGION"
echo "Exp:           $EXP_NAME"
echo "Args:          $TRAIN_ARGS"
echo ""

# Build optional args
EXTRA_ARGS=""
if [[ -n "$SECURITY_GROUP" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --security-group-ids $SECURITY_GROUP"
fi
if [[ -n "$KEY_NAME" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --key-name $KEY_NAME"
fi

INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --user-data "$USER_DATA_B64" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=lob-rl-${EXP_NAME}},{Key=Project,Value=lob-rl},{Key=ExpName,Value=${EXP_NAME}}]" \
    $EXTRA_ARGS \
    --query "Instances[0].InstanceId" --output text)

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
    echo "ERROR: Failed to launch instance."
    echo "Check: instance type availability, IAM profile, AMI, security group."
    exit 1
fi

RUN_DIR="${EXP_NAME}_${INSTANCE_ID}"

echo "Instance launched: $INSTANCE_ID"
echo "Run dir:           $RUN_DIR"
echo "S3 results:        s3://$S3_BUCKET/runs/$RUN_DIR/"
echo ""
echo "=== Next steps ==="
echo "  Status:    aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name' --output text"
echo "  Console:   aws ec2 get-console-output --instance-id $INSTANCE_ID --region $REGION --output text"
echo "  Bootstrap: aws s3 cp s3://$S3_BUCKET/runs/$RUN_DIR/bootstrap.log - --region $REGION"
echo "  Fetch:     ./aws/fetch-results.sh $RUN_DIR"
echo "  Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
