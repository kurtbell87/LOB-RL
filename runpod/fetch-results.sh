#!/usr/bin/env bash
# Fetch training results from a RunPod pod.
#
# Usage:
#   ./runpod/fetch-results.sh <pod-id>
#
# Downloads from /workspace/runs/ on the pod to results/<pod-id>/ locally.
# Includes: model, VecNormalize stats, checkpoints, TensorBoard logs.

set -euo pipefail

POD_ID="${1:?Usage: ./runpod/fetch-results.sh <pod-id>}"
LOCAL_DIR="results/${POD_ID}"

mkdir -p "$LOCAL_DIR"

echo "=== Fetching results from pod $POD_ID ==="
echo "Destination: $LOCAL_DIR/"
echo ""

# Download the runs directory
runpodctl receive "$POD_ID":/workspace/runs/ "$LOCAL_DIR/"

echo ""
echo "=== Results downloaded ==="
echo "Location: $LOCAL_DIR/"
echo ""

# Show what was downloaded
if command -v tree &> /dev/null; then
    tree "$LOCAL_DIR/"
else
    find "$LOCAL_DIR/" -type f | head -20
fi

echo ""
echo "To evaluate locally:"
echo "  cd build-release"
echo "  PYTHONPATH=.:../python uv run python ../scripts/train.py \\"
echo "    --cache-dir ../cache/mes/ --resume ../$LOCAL_DIR/ppo_lob.zip ..."
