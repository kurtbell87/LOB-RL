#!/usr/bin/env bash
# Fetch training results from a RunPod pod.
#
# Usage:
#   ./runpod/fetch-results.sh <pod-id>
#
# Downloads from /workspace/runs/ on the pod to results/<pod-id>/ locally.
# Includes: model, VecNormalize stats, checkpoints, TensorBoard logs.
#
# Prerequisites:
#   - runpodctl installed and configured
#   - SSH key added to RunPod (runpodctl ssh add-key)
#   - Pod must be RUNNING

set -euo pipefail

POD_ID="${1:?Usage: ./runpod/fetch-results.sh <pod-id>}"
LOCAL_DIR="results/${POD_ID}"

mkdir -p "$LOCAL_DIR"

echo "=== Fetching results from pod $POD_ID ==="
echo "Destination: $LOCAL_DIR/"
echo ""

# Verify pod is running
STATUS=$(runpodctl get pod 2>&1 | grep "$POD_ID" | grep -o 'RUNNING' || true)
if [ "$STATUS" != "RUNNING" ]; then
    echo "ERROR: Pod $POD_ID is not running."
    echo "Current pods:"
    runpodctl get pod 2>&1
    exit 1
fi

# Get SSH connection info
echo "=== Getting SSH connection info ==="
SSH_INFO=$(runpodctl ssh connect "$POD_ID" 2>&1)
echo "$SSH_INFO"

SSH_HOST=$(echo "$SSH_INFO" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
SSH_PORT=$(echo "$SSH_INFO" | grep -oE '\-p [0-9]+' | grep -oE '[0-9]+' | head -1)
SSH_KEY=$(echo "$SSH_INFO" | grep -oE '\-i [^ ]+' | sed 's/-i //' | head -1)

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "ERROR: Could not parse SSH connection info."
    echo "Raw output: $SSH_INFO"
    echo ""
    echo "You can manually download with:"
    echo "  rsync -avz -e 'ssh -p PORT -i KEY' root@HOST:/workspace/runs/ $LOCAL_DIR/"
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

echo ""
echo "=== Downloading results via rsync ==="
echo "SSH: root@$SSH_HOST:$SSH_PORT"
echo ""

# Ensure rsync is available on the pod
ssh $SSH_OPTS root@"$SSH_HOST" "which rsync > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync)" 2>/dev/null

# Download via rsync over SSH
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    root@"$SSH_HOST":/workspace/runs/ \
    "$LOCAL_DIR/"

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
