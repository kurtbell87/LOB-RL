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
#   # Use a different GPU
#   GPU_TYPE="NVIDIA A100 80GB PCIe" ./runpod/launch.sh --recurrent ...
#
# The pod runs train.py with:
#   --cache-dir /workspace/cache/mes/
#   --output-dir /workspace/runs/
#   [your additional args]

set -euo pipefail

VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"
DOCKER_USER="${DOCKERHUB_USER:?Set DOCKERHUB_USER to your Docker Hub username}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A40}"
IMAGE="${DOCKER_USER}/lob-rl:latest"

# Build the training command
# --cache-dir and --output-dir are always set to volume paths
TRAIN_ARGS="--cache-dir /workspace/cache/mes/ --output-dir /workspace/runs/ $*"

echo "=== Launching training pod ==="
echo "Image:    $IMAGE"
echo "GPU:      $GPU_TYPE"
echo "Volume:   $VOLUME_ID"
echo "Args:     $TRAIN_ARGS"
echo ""

POD_ID=$(runpodctl create pod \
    --name "lob-rl-train" \
    --gpuType "$GPU_TYPE" \
    --imageName "$IMAGE" \
    --volumeId "$VOLUME_ID" \
    --ports "22/tcp,6006/tcp" \
    --args "$TRAIN_ARGS" \
    2>&1 | grep -oP 'pod_[a-zA-Z0-9]+' | head -1)

if [ -z "$POD_ID" ]; then
    echo "ERROR: Failed to create pod. Check runpodctl config, volume ID, and image name."
    exit 1
fi

echo "Pod created: $POD_ID"
echo ""
echo "=== Next steps ==="
echo "  Monitor:  runpodctl logs $POD_ID"
echo "  SSH:      runpodctl ssh $POD_ID"
echo "  Stop:     runpodctl stop pod $POD_ID"
echo "  Remove:   runpodctl remove pod $POD_ID"
echo "  Results:  ./runpod/fetch-results.sh $POD_ID"
echo ""
echo "  TensorBoard (after SSH):"
echo "    tensorboard --logdir /workspace/runs/tb_logs --port 6006 --bind_all"
