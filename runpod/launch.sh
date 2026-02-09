#!/usr/bin/env bash
# Launch a training pod on RunPod.
#
# Usage:
#   export RUNPOD_VOLUME_ID=vol_abc123
#   export DOCKERHUB_USER=yourusername
#   ./runpod/launch.sh [train.py args...]
#
# Examples:
#   # LSTM training, 5M steps, best hyperparams
#   ./runpod/launch.sh \
#     --recurrent --bar-size 1000 --execution-cost \
#     --ent-coef 0.05 --learning-rate 0.001 \
#     --shuffle-split --seed 42 \
#     --total-timesteps 5000000 --checkpoint-freq 500000
#
#   # Resume from checkpoint
#   ./runpod/launch.sh \
#     --resume /workspace/runs/checkpoints/rl_model_2000000_steps.zip \
#     --recurrent --total-timesteps 5000000
#
#   # Use a specific GPU
#   GPU_TYPE="NVIDIA L40" ./runpod/launch.sh --recurrent ...
#
# Default GPU: NVIDIA GeForce RTX 4090 (US-NC-1, $0.59/hr, 24GB)
#
# The pod runs train.py with:
#   --cache-dir /workspace/cache/mes/
#   --output-dir /workspace/runs/{exp_name}_{pod_id}/
#   [your additional args]
#
# EXP_NAME is auto-detected (--recurrent → lstm, --frame-stack → framestack,
# else mlp) or set via EXP_NAME env var.

set -euo pipefail

VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"
DOCKER_USER="${DOCKERHUB_USER:?Set DOCKERHUB_USER to your Docker Hub username}"
IMAGE="${DOCKER_USER}/lob-rl:latest"

GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 4090}"
CLOUD_TYPE="${CLOUD_TYPE:---secureCloud}"  # --secureCloud or --communityCloud

# Auto-detect experiment name from training args
if [ -z "${EXP_NAME:-}" ]; then
    EXP_NAME="mlp"
    for arg in "$@"; do
        case "$arg" in
            --recurrent) EXP_NAME="lstm" ;;
            --frame-stack) EXP_NAME="framestack" ;;
        esac
    done
fi

# --cache-dir is always set to volume path
# --output-dir is constructed by start.sh using EXP_NAME + RUNPOD_POD_ID
TRAIN_ARGS="--cache-dir /workspace/cache/mes/ $*"

echo "=== Launching training pod ==="
echo "Image:    $IMAGE"
echo "GPU:      $GPU_TYPE"
echo "Volume:   $VOLUME_ID"
echo "Exp:      $EXP_NAME"
echo "Args:     $TRAIN_ARGS"
echo ""

CREATE_OUTPUT=$(runpodctl create pod \
    --name "lob-rl-${EXP_NAME}" \
    --gpuType "$GPU_TYPE" \
    --imageName "$IMAGE" \
    --networkVolumeId "$VOLUME_ID" \
    --volumePath /workspace \
    --ports "6006/http" \
    --env "EXP_NAME=$EXP_NAME" \
    $CLOUD_TYPE \
    --args "$TRAIN_ARGS" \
    2>&1) || true
echo "$CREATE_OUTPUT"
POD_ID=$(echo "$CREATE_OUTPUT" | sed -n 's/^pod "\([^"]*\)".*/\1/p')

if [ -z "$POD_ID" ]; then
    echo "ERROR: Failed to create pod. Check runpodctl config, volume ID, and GPU availability."
    echo "GPU requested: $GPU_TYPE"
    exit 1
fi

RUN_DIR="${EXP_NAME}_${POD_ID}"

echo "Pod created: $POD_ID"
echo "Output dir: /workspace/runs/${RUN_DIR}/"
echo ""
echo "=== Next steps ==="
echo "  Status:   runpodctl get pod"
echo "  Logs:     runpodctl logs $POD_ID"
echo "  Stop:     runpodctl stop pod $POD_ID"
echo "  Remove:   runpodctl remove pod $POD_ID"
echo ""
echo "  Pod auto-stops on training success. Fetch results via S3:"
echo "    RUNPOD_VOLUME_ID=$VOLUME_ID ./runpod/fetch-results.sh $RUN_DIR"
echo ""
echo "  Automated monitoring (polls until done, fetches, analyzes):"
echo "    RUNPOD_VOLUME_ID=$VOLUME_ID ./research/monitor.sh"
