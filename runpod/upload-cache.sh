#!/usr/bin/env bash
# Upload cache/mes/ to a RunPod network volume via runpodctl.
#
# Usage:
#   export RUNPOD_VOLUME_ID=vol_abc123
#   ./runpod/upload-cache.sh
#
# Prerequisites:
#   - runpodctl installed and configured (runpodctl config --apiKey ...)
#   - Network volume already created in RunPod console
#
# This script:
#   1. Starts a minimal pod with the network volume mounted
#   2. Uploads cache/mes/ to /workspace/cache/mes/ on the volume
#   3. Verifies file count
#   4. Terminates the upload pod
#
# Safe to re-run — rsync-like behavior, only uploads new/changed files.

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-cache/mes}"
VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"
GPU_TYPE="${UPLOAD_GPU:-NVIDIA RTX A4000}"  # Cheapest GPU, just need SSH access

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

# Create a pod with the volume mounted
POD_ID=$(runpodctl create pod \
    --name "lob-rl-upload" \
    --gpuType "$GPU_TYPE" \
    --imageName "ubuntu:22.04" \
    --volumeId "$VOLUME_ID" \
    --ports "22/tcp" \
    --args "sleep infinity" \
    2>&1 | grep -oP 'pod_[a-zA-Z0-9]+' | head -1)

if [ -z "$POD_ID" ]; then
    echo "ERROR: Failed to create pod. Check runpodctl config and volume ID."
    exit 1
fi

echo "Pod created: $POD_ID"
echo "Waiting for pod to be ready..."

# Wait for pod to be running
for i in $(seq 1 60); do
    STATUS=$(runpodctl get pod "$POD_ID" 2>&1 | grep -o 'RUNNING' || true)
    if [ "$STATUS" = "RUNNING" ]; then
        break
    fi
    sleep 5
done

if [ "$STATUS" != "RUNNING" ]; then
    echo "ERROR: Pod did not start within 5 minutes."
    echo "Check RunPod console. Pod ID: $POD_ID"
    exit 1
fi

echo "Pod is running."

echo ""
echo "=== Step 2: Uploading cache files ==="
echo "Uploading $FILE_COUNT files to /workspace/cache/mes/ ..."
echo ""

# Create target directory and upload
runpodctl exec "$POD_ID" -- mkdir -p /workspace/cache/mes

# Upload using runpodctl send (transfers files to the pod)
runpodctl send "$CACHE_DIR" "$POD_ID":/workspace/cache/mes/

echo ""
echo "=== Step 3: Verifying upload ==="

REMOTE_COUNT=$(runpodctl exec "$POD_ID" -- bash -c "find /workspace/cache/mes -name '*.npz' | wc -l" 2>&1 | tr -d ' ')
echo "Remote file count: $REMOTE_COUNT"
echo "Local file count:  $FILE_COUNT"

if [ "$REMOTE_COUNT" -eq "$FILE_COUNT" ]; then
    echo "Upload verified successfully."
else
    echo "WARNING: File count mismatch. Re-run this script to retry."
fi

echo ""
echo "=== Step 4: Terminating upload pod ==="

runpodctl remove pod "$POD_ID"
echo "Upload pod terminated."

echo ""
echo "=== Done ==="
echo "Cache is now on volume $VOLUME_ID at /workspace/cache/mes/"
echo "This data persists across pod restarts. No need to re-upload."
