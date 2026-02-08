# RunPod GPU Training

GPU training infrastructure for LOB-RL. Uses persistent network volumes for data and ephemeral pods for compute.

## Files

| File | Role |
|------|------|
| `README.md` | This guide |
| `upload-cache.sh` | One-time upload of `cache/mes/` to RunPod network volume via S3 API |
| `launch.sh` | Launch a training pod with given args |
| `fetch-results.sh` | Download trained model + logs from pod |

## Prerequisites

1. **RunPod account** with API key
2. **`runpodctl`** CLI installed: `brew install runpod/runpodctl/runpodctl` (or see [runpodctl docs](https://github.com/runpod/runpodctl))
3. **Docker Hub** account (for pushing the training image)
4. **AWS CLI** (for S3-compatible upload to RunPod volumes): `brew install awscli`

## One-Time Setup

### 1. Configure RunPod API key

```bash
runpodctl config --apiKey YOUR_API_KEY
```

### 2. Create a network volume

Go to RunPod console > Storage > Create Network Volume:
- **Name:** `lob-rl-data`
- **Size:** 50 GB (18GB cache + room for models/checkpoints)
- **Region:** Pick the region with A40 availability (US-TX-3 recommended)

Note the volume ID (e.g., `vol_abc123`).

### 3. Upload cache data

```bash
export RUNPOD_VOLUME_ID=vol_abc123
./runpod/upload-cache.sh
```

This uploads `cache/mes/` (~18GB, 249 `.npz` files) to the network volume. Takes ~10 min on a good connection. Resume-capable — safe to interrupt and re-run.

### 4. Build and push Docker image

```bash
# Set your Docker Hub username
export DOCKERHUB_USER=yourusername

# Build
docker build -t $DOCKERHUB_USER/lob-rl:latest .

# Push
docker push $DOCKERHUB_USER/lob-rl:latest
```

Image is ~7GB (PyTorch base + deps). No data baked in.

## Running Experiments

### Launch a training pod

```bash
export RUNPOD_VOLUME_ID=vol_abc123
export DOCKERHUB_USER=yourusername

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

Default GPU: A40 ($0.35/hr, 48GB VRAM). Override with `--gpu` flag:
```bash
./runpod/launch.sh --gpu "NVIDIA A100 80GB PCIe" --recurrent ...
```

### Monitor training

SSH into the pod (connection info printed by `launch.sh`), then:
```bash
# Watch logs
tail -f /workspace/runs/tb_logs/*/events*

# TensorBoard
tensorboard --logdir /workspace/runs/tb_logs --port 6006 --bind_all
# Then SSH tunnel: ssh -L 6006:localhost:6006 root@pod-ip
```

### Fetch results

```bash
./runpod/fetch-results.sh <pod-id>
```

Downloads model, checkpoints, VecNormalize stats, and TensorBoard logs to `results/<pod-id>/`.

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
docker build -t $DOCKERHUB_USER/lob-rl:latest .
docker push $DOCKERHUB_USER/lob-rl:latest
# Launch a new pod — it picks up the latest image
```

Data never moves again unless you add new cache files.

## GPU Options

| GPU | $/hr | VRAM | Notes |
|-----|------|------|-------|
| **A40** | **$0.35** | 48GB | Best value for this model size |
| A100 PCIe | $1.19 | 80GB | Overkill for PPO/LSTM on 21-dim obs |
| H100 SXM | $2.69 | 80GB | Overkill |

A40 is the right choice. 5M steps LSTM ~6-8hrs = ~$2-3.

## Volume Layout

```
/workspace/                    # Network volume mount point
├── cache/mes/                 # 249 .npz files (uploaded once)
│   ├── 2022-01-03.npz
│   ├── 2022-01-04.npz
│   └── ...
└── runs/                      # Training output (created by train.py)
    ├── ppo_lob.zip            # Final model
    ├── vec_normalize.pkl      # VecNormalize stats
    ├── tb_logs/               # TensorBoard logs
    └── checkpoints/           # Periodic checkpoints
        ├── rl_model_500000_steps.zip
        └── rl_model_500000_steps_vecnormalize.pkl
```
