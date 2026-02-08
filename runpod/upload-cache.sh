#!/usr/bin/env bash
# Upload cache/mes/ to a RunPod network volume via SSH/rsync.
#
# Usage:
#   export RUNPOD_VOLUME_ID=vol_abc123
#   ./runpod/upload-cache.sh
#
# Prerequisites:
#   - runpodctl installed and configured (runpodctl config --apiKey ...)
#   - SSH key added to RunPod (runpodctl ssh add-key)
#   - Network volume already created in RunPod console
#
# This script:
#   1. Starts a minimal pod with the network volume mounted
#   2. Uploads cache/mes/ to /workspace/cache/mes/ on the volume via rsync
#   3. Verifies file count
#   4. Terminates the upload pod
#
# Safe to re-run — rsync only uploads new/changed files.

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-cache/mes}"
VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"
GPU_TYPE="${UPLOAD_GPU:-NVIDIA GeForce RTX 4090}"
CLOUD_TYPE="${CLOUD_TYPE:---secureCloud}"  # --secureCloud or --communityCloud

if [ ! -d "$CACHE_DIR" ]; then
    echo "ERROR: Cache directory not found: $CACHE_DIR"
    echo "Run precompute_cache.py first, or set CACHE_DIR to the correct path."
    exit 1
fi

FILE_COUNT=$(find "$CACHE_DIR" -name "*.npz" | wc -l | tr -d ' ')
echo "Found $FILE_COUNT .npz files in $CACHE_DIR"

echo ""
echo "=== Step 1: Creating upload pod ==="
echo "Volume: $VOLUME_ID"
echo "This will create a temporary pod to receive the upload."
echo ""

# Create a pod with the volume mounted and SSH enabled
CREATE_OUTPUT=$(runpodctl create pod \
    --name "lob-rl-upload" \
    --gpuType "$GPU_TYPE" \
    --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04" \
    --networkVolumeId "$VOLUME_ID" \
    --ports "22/tcp" \
    --startSSH \
    $CLOUD_TYPE \
    2>&1) || true
echo "$CREATE_OUTPUT"

# Parse pod ID from output like: pod "abc123def" created for $0.59/hr
POD_ID=$(echo "$CREATE_OUTPUT" | sed -n 's/^pod "\([^"]*\)".*/\1/p')

if [ -z "$POD_ID" ]; then
    echo "ERROR: Failed to create pod. Check runpodctl config and volume/GPU availability."
    echo "Try a different GPU: UPLOAD_GPU='NVIDIA L40' ./runpod/upload-cache.sh"
    exit 1
fi

echo "Pod created: $POD_ID"
echo "Waiting for pod to be ready..."

# Wait for pod to be running (up to 5 minutes)
STATUS=""
for i in $(seq 1 60); do
    STATUS=$(runpodctl get pod 2>&1 | grep "$POD_ID" | grep -o 'RUNNING' || true)
    if [ "$STATUS" = "RUNNING" ]; then
        break
    fi
    sleep 5
done

if [ "$STATUS" != "RUNNING" ]; then
    echo "ERROR: Pod did not start within 5 minutes."
    echo "Check RunPod console. Pod ID: $POD_ID"
    echo "Cleaning up..."
    runpodctl remove pod "$POD_ID" 2>/dev/null || true
    exit 1
fi

echo "Pod is running."

# Get SSH connection info
echo ""
echo "=== Step 2: Getting SSH connection info ==="
SSH_INFO=$(runpodctl ssh connect "$POD_ID" 2>&1)
echo "$SSH_INFO"

# Extract SSH host and port from the connect command
# Format is typically: ssh root@<ip> -p <port> -i ~/.runpod/ssh/RunPod-Key-Go
SSH_HOST=$(echo "$SSH_INFO" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
SSH_PORT=$(echo "$SSH_INFO" | grep -oE '\-p [0-9]+' | grep -oE '[0-9]+' | head -1)
SSH_KEY=$(echo "$SSH_INFO" | grep -oE '\-i [^ ]+' | sed 's/-i //' | head -1)

if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "ERROR: Could not parse SSH connection info."
    echo "Raw output: $SSH_INFO"
    echo ""
    echo "You can manually upload with:"
    echo "  rsync -avz -e 'ssh -p PORT -i KEY' $CACHE_DIR/ root@HOST:/workspace/cache/mes/"
    echo ""
    echo "Then remove the pod: runpodctl remove pod $POD_ID"
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT"
if [ -n "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

echo ""
echo "=== Step 3: Uploading cache files ==="
echo "Uploading $FILE_COUNT files to /workspace/cache/mes/ via rsync..."
echo "SSH: root@$SSH_HOST:$SSH_PORT"
echo ""

# Create target directory and ensure rsync is available
ssh $SSH_OPTS root@"$SSH_HOST" "mkdir -p /workspace/cache/mes && (which rsync > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync))" 2>/dev/null

# Upload via rsync over SSH
rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    "$CACHE_DIR/" \
    root@"$SSH_HOST":/workspace/cache/mes/

echo ""
echo "=== Step 4: Verifying upload ==="

REMOTE_COUNT=$(ssh $SSH_OPTS root@"$SSH_HOST" \
    "find /workspace/cache/mes -name '*.npz' | wc -l" 2>/dev/null | tr -d ' ')
echo "Remote file count: $REMOTE_COUNT"
echo "Local file count:  $FILE_COUNT"

if [ "$REMOTE_COUNT" -eq "$FILE_COUNT" ]; then
    echo "Upload verified successfully."
else
    echo "WARNING: File count mismatch. Re-run this script to retry."
fi

echo ""
echo "=== Step 5: Terminating upload pod ==="

runpodctl remove pod "$POD_ID"
echo "Upload pod terminated."

echo ""
echo "=== Done ==="
echo "Cache is now on volume $VOLUME_ID at /workspace/cache/mes/"
echo "This data persists across pod restarts. No need to re-upload."
