# aws/ — AWS EC2 Spot Training Infrastructure

Replaces RunPod for remote training. Uses EC2 Spot instances with S3 for storage and ECR for Docker images.

## File Table

| File | Role |
|------|------|
| `setup.sh` | One-time infrastructure setup (S3 bucket, ECR repo, IAM role, security group) |
| `launch.sh` | Launch EC2 Spot instance for training — drop-in replacement for `runpod/launch.sh` |
| `fetch-results.sh` | Download results from S3 to `results/<run_dir>/` |
| `upload-cache.sh` | One-time upload of `cache/mes/` to S3 |
| `monitor.sh` | Poll instances, auto-fetch results when done |

## Quick Start

```bash
# 1. One-time setup
./aws/setup.sh
# Export the printed env vars

# 2. Push Docker image to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin $AWS_ECR_REPO
docker buildx build --platform linux/amd64 -t ${AWS_ECR_REPO}:latest --push .

# 3. Upload cache
AWS_S3_BUCKET=lob-rl-training ./aws/upload-cache.sh

# 4. Launch training
./aws/launch.sh --bar-size 1000 --execution-cost --total-timesteps 5000000

# 5. Fetch results
./aws/fetch-results.sh mlp_i-0abc123def456
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_S3_BUCKET` | Yes | — | S3 bucket for cache and results |
| `AWS_ECR_REPO` | Yes | — | Full ECR URI (account.dkr.ecr.region.amazonaws.com/lob-rl) |
| `AWS_REGION` | No | `us-east-1` | AWS region |
| `AWS_INSTANCE_TYPE` | No | Auto-detect | Override instance type (g5.xlarge for GPU, c7a.4xlarge for CPU) |
| `AWS_IAM_PROFILE` | No | `lob-rl-training` | IAM instance profile name |
| `AWS_SECURITY_GROUP` | No | — | Security group ID for SSH access |
| `AWS_KEY_NAME` | No | — | EC2 key pair for SSH debugging |
| `EXP_NAME` | No | Auto-detect | Experiment label (mlp/lstm/framestack) |

## Instance Types

| Instance | GPU | vCPU | RAM | Spot $/hr | Use for |
|----------|-----|------|-----|-----------|---------|
| `g5.xlarge` | A10G 24GB | 4 | 16GB | ~$0.24 | LSTM (RecurrentPPO) |
| `c7a.4xlarge` | — | 16 | 32GB | ~$0.39 | MLP, frame-stack (CPU-only) |

Auto-detection: `--recurrent` flag → `g5.xlarge`, otherwise → `c7a.4xlarge`.

## S3 Layout

```
s3://lob-rl-training/
├── cache/mes/                          # 249 .npz files (18GB, uploaded once)
└── runs/
    └── {exp_name}_{instance_id}/       # Per-run output
        ├── train.log
        ├── ppo_lob.zip
        ├── vec_normalize.pkl
        ├── bootstrap.log               # User-data execution log
        ├── tb_logs/
        └── checkpoints/
```

## Cost Comparison

| Provider | Instance | GPU | Spot $/hr | Notes |
|----------|----------|-----|-----------|-------|
| **AWS** | g5.xlarge | A10G 24GB | ~$0.24 | 60% cheaper than RunPod |
| **AWS** | c7a.4xlarge | — (16 vCPU) | ~$0.39 | No GPU idle cost for MLP |
| RunPod | RTX 4090 | RTX 4090 24GB | $0.59 | Previous provider (deprecated) |

## Spot Interruption Handling

1. User-data starts a background monitor polling `http://169.254.169.254/latest/meta-data/spot/instance-action` every 5s (IMDSv2)
2. On HTTP 200 (interruption notice): uploads partial results to S3, then AWS terminates
3. ~2 minutes to save — enough for S3 sync of checkpoints
4. Recovery: fetch partial results, find latest checkpoint, relaunch with `--resume`
5. Recommendation: `--checkpoint-freq 500000` for spot runs

## Debugging Failed Instances

```bash
# Check bootstrap log (uploaded periodically to S3)
aws s3 cp s3://$AWS_S3_BUCKET/runs/<run_dir>/bootstrap.log -

# SSH into a failed instance (requires AWS_KEY_NAME in launch)
aws ec2 describe-instances --instance-ids <id> \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
ssh -i <key.pem> ubuntu@<ip>

# Manually terminate
aws ec2 terminate-instances --instance-ids <id>
```

## Cross-File Dependencies

- **Depends on:** `Dockerfile` (container image), `scripts/start.sh` (container entrypoint), `scripts/train.py` (training script)
- **Depended on by:** `experiment.sh` (orchestrator), `.claude/prompts/run.md` (RUN agent), `.claude/prompts/frame.md` (FRAME agent)

## Modification Hints

- **Add new instance type:** Update auto-detect logic in `launch.sh` (the `NEEDS_GPU` check and `IS_GPU_INSTANCE` case match). Add to the table in this README.
- **Change S3 layout:** Update paths in `launch.sh` user-data, `fetch-results.sh`, and `upload-cache.sh`.
- **Change AMI:** Update the `describe-images` filter in `launch.sh`.
