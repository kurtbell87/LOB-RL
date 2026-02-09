# RunPod GPU Training

GPU training infrastructure for LOB-RL. Uses persistent network volumes for data and ephemeral pods for compute.

## Files

| File | Role |
|------|------|
| `README.md` | This guide |
| `upload-cache.sh` | Legacy cache upload via rsync (prefer `aws s3 sync` — see setup below) |
| `launch.sh` | Launch a training pod with given args. Auto-detects `EXP_NAME` from args, passes `--cache-dir /workspace/cache/mes/`. Output dir is `/workspace/runs/{exp_name}_{pod_id}/`. |
| `fetch-results.sh` | Download trained model + logs from network volume via S3. Takes run dir name (e.g., `lstm_abc123`). No running pod required. |

## Prerequisites

1. **RunPod account** with API key and S3 API credentials
2. **`runpodctl`** CLI installed: `brew install runpod/runpodctl/runpodctl`
3. **Docker Hub** account (for pushing the training image)
4. **AWS CLI** (for S3-compatible access to RunPod volumes): `brew install awscli`

## Secrets

**RunPod API key** — configured via `runpodctl config --apiKey`.

**RunPod S3 credentials** — stored as an AWS CLI named profile:
```bash
aws configure --profile runpod
# Access Key: user_...  (RunPod S3 access key)
# Secret Key: rps_...   (RunPod S3 secret key)
# Region: us-nc-1
# Output format: (blank)
```

All `aws s3` commands use `--profile runpod --endpoint-url https://s3api-us-nc-1.runpod.io`.

## One-Time Setup

### 1. Configure RunPod API key

```bash
runpodctl config --apiKey YOUR_API_KEY
```

### 2. Create a network volume

Go to RunPod console > Storage > Create Network Volume:
- **Name:** `lob-rl-data`
- **Size:** 50 GB (18GB cache + room for models/checkpoints)
- **Region:** **US-NC-1** (required — must match GPU region)

Note the volume ID (e.g., `4w2m8hek66`).

### 3. Upload cache data via S3

```bash
aws s3 sync cache/mes/ s3://$RUNPOD_VOLUME_ID/cache/mes/ \
  --profile runpod --endpoint-url https://s3api-us-nc-1.runpod.io
```

This uploads `cache/mes/` (~13GB, 249 `.npz` files). ~37 MiB/s, takes ~6 min. Resume-capable (uses `aws s3 sync`).

Verify:
```bash
aws s3 ls s3://$RUNPOD_VOLUME_ID/cache/mes/ \
  --profile runpod --endpoint-url https://s3api-us-nc-1.runpod.io | wc -l
# Should be 249
```

### 4. Build and push Docker image

**IMPORTANT:** Must build for `linux/amd64` (RunPod GPUs are x86_64, not ARM).

```bash
export DOCKERHUB_USER=yourusername

# Build for amd64 and push in one step
docker buildx build --platform linux/amd64 -t $DOCKERHUB_USER/lob-rl:latest --push .
```

Image is ~8GB (PyTorch 2.5.1 + CUDA 12.4 + sshd + rsync + SB3 deps). No data baked in.

## Docker Image

The `Dockerfile` uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` as base and adds:
- `openssh-server` + `rsync` (kept for debugging failed pods via SSH)
- Python deps: `gymnasium`, `stable-baselines3`, `sb3-contrib`, `tensorboard`
- `scripts/start.sh` as entrypoint (runs training, exits on success)
- `scripts/train.py` + `python/lob_rl/` (training code)

The `start.sh` entrypoint:
- Starts sshd (picks up RunPod-injected `PUBLIC_KEY` env var for auth)
- Constructs unique output dir: `/workspace/runs/{EXP_NAME}_{RUNPOD_POD_ID}/`
- Runs `train.py`, logging to `{output_dir}/train.log`
- **On success (exit 0):** exits immediately — pod stops, billing stops. Results on network volume.
- **On failure (non-zero):** keeps container alive via `sleep infinity` for SSH debugging.
- With no args, idles for manual SSH use

## Running Experiments

### Launch a training pod

```bash
export RUNPOD_VOLUME_ID=4w2m8hek66
export DOCKERHUB_USER=kurtbell87

# LSTM, 5M steps, best hyperparams
./runpod/launch.sh \
  --recurrent \
  --bar-size 1000 \
  --execution-cost \
  --ent-coef 0.05 \
  --learning-rate 0.001 \
  --shuffle-split \
  --seed 42 \
  --total-timesteps 5000000 \
  --checkpoint-freq 500000
```

Default GPU: RTX 4090 ($0.59/hr, 24GB — sufficient for PPO/LSTM on 21-dim obs). Override with `GPU_TYPE`:
```bash
GPU_TYPE="NVIDIA L40" ./runpod/launch.sh --recurrent ...
```

### Monitor training

```bash
# Check pod status (pods auto-stop on training success)
runpodctl get pod

# View logs
runpodctl logs <pod-id>

# Automated monitoring — polls until all pods finish, fetches results, analyzes
RUNPOD_VOLUME_ID=4w2m8hek66 ./research/monitor.sh
```

### Fetch results

Results are on the persistent network volume. Fetch via S3 — no running pod required:

```bash
RUNPOD_VOLUME_ID=4w2m8hek66 ./runpod/fetch-results.sh lstm_abc123
```

Downloads via S3 to `results/lstm_abc123/`. The argument is the run directory name (`{exp_name}_{pod_id}`), shown in `launch.sh` output.

### Resume from checkpoint

After a crash or spot preemption:
```bash
./runpod/launch.sh \
  --resume /workspace/runs/checkpoints/rl_model_2000000_steps.zip \
  --recurrent --total-timesteps 5000000 \
  --checkpoint-freq 500000 ...
```

## When Code Changes

```bash
docker buildx build --platform linux/amd64 -t $DOCKERHUB_USER/lob-rl:latest --push .
# Launch a new pod — it picks up the latest image
```

Data never moves again unless you add new cache files.

## GPU Options

24GB VRAM is sufficient for PPO/LSTM on a 21-dim observation space.

| GPU | $/hr | VRAM | Region | Stock | Notes |
|-----|------|------|--------|-------|-------|
| **RTX 4090** | **$0.59** | 24GB | US-NC-1 | Medium | **Default.** Fast clocks, good availability |
| L40 | $0.99 | 48GB | US-KS-2 | Medium | Fallback if 4090 unavailable (different region — needs separate volume) |

5M steps LSTM takes ~6-8hrs. At $0.59/hr (RTX 4090), that's ~$3.50-4.70.

## Volume Layout

```
/workspace/                    # Network volume mount point
├── cache/mes/                 # 249 .npz files (uploaded once via S3)
│   ├── 2022-01-03.npz
│   ├── 2022-01-04.npz
│   └── ...
└── runs/                      # Training output (created by start.sh/train.py)
    └── {exp_name}_{pod_id}/   # Per-experiment dir (e.g. lstm_abc123/)
        ├── train.log              # Combined stdout/stderr from train.py
        ├── ppo_lob.zip            # Final model
        ├── vec_normalize.pkl      # VecNormalize stats
        ├── tb_logs/               # TensorBoard logs
        └── checkpoints/           # Periodic checkpoints
            ├── rl_model_500000_steps.zip
            └── rl_model_500000_steps_vecnormalize.pkl
```

## Gotchas

- **Image must be `linux/amd64`** — use `docker buildx build --platform linux/amd64`. Building on Apple Silicon without `--platform` produces an ARM image that silently fails on RunPod.
- **Pods auto-stop on success** — `start.sh` exits 0 after training completes. The pod shows as "Exited" in `runpodctl get pod`. Results are on the network volume; fetch via S3.
- **Failed pods stay alive** — if training fails (non-zero exit), the pod stays running for SSH debugging. Remember to `runpodctl remove pod` after debugging.
- **Fetch uses S3, not SSH** — `fetch-results.sh` downloads from the network volume via `aws s3 sync`. No running pod required. Requires `RUNPOD_VOLUME_ID` env var and `--profile runpod` configured.
- **Cache upload uses S3** — `upload-cache.sh` is legacy (creates a GPU pod for rsync). Prefer `aws s3 sync --profile runpod` (see setup above).
