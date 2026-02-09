#!/usr/bin/env bash
# Fetch training results from RunPod network volume via S3.
#
# Usage:
#   ./runpod/fetch-results.sh <run_dir>
#
# Example:
#   RUNPOD_VOLUME_ID=4w2m8hek66 ./runpod/fetch-results.sh lstm_6kwbf810ribiza
#
# Downloads from s3://$VOLUME_ID/runs/<run_dir>/ to results/<run_dir>/ locally.
# Includes: model, VecNormalize stats, checkpoints, TensorBoard logs.
#
# No running pod required — data is on the persistent network volume.
#
# Prerequisites:
#   - AWS CLI installed with RunPod S3 profile configured:
#       aws configure --profile runpod
#   - RUNPOD_VOLUME_ID env var set

set -euo pipefail

RUN_DIR="${1:?Usage: ./runpod/fetch-results.sh <run_dir>  (e.g. lstm_6kwbf810ribiza)}"
VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"
S3_ENDPOINT="https://s3api-us-nc-1.runpod.io"
S3_OPTS="--profile runpod --endpoint-url $S3_ENDPOINT"

S3_PATH="s3://${VOLUME_ID}/runs/${RUN_DIR}/"
LOCAL_DIR="results/${RUN_DIR}"

mkdir -p "$LOCAL_DIR"

echo "=== Fetching results via S3 ==="
echo "Source:      $S3_PATH"
echo "Destination: $LOCAL_DIR/"
echo ""

# Verify the remote directory exists
FILE_COUNT=$(aws s3 ls "$S3_PATH" $S3_OPTS --recursive 2>&1 | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No files found at $S3_PATH"
    echo "Check that RUNPOD_VOLUME_ID ($VOLUME_ID) and run dir ($RUN_DIR) are correct."
    echo ""
    echo "Available run dirs:"
    aws s3 ls "s3://${VOLUME_ID}/runs/" $S3_OPTS 2>&1 || true
    exit 1
fi
echo "Found $FILE_COUNT files on volume."
echo ""

# Download via S3 sync
aws s3 sync "$S3_PATH" "$LOCAL_DIR/" $S3_OPTS

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

# Verify expected files
MISSING=""
[ -f "$LOCAL_DIR/train.log" ] || MISSING="$MISSING train.log"
[ -f "$LOCAL_DIR/ppo_lob.zip" ] || MISSING="$MISSING ppo_lob.zip"
if [ -n "$MISSING" ]; then
    echo ""
    echo "WARNING: Expected files missing:$MISSING"
    echo "Training may not have completed successfully."
fi

echo ""
echo "To evaluate locally:"
echo "  cd build-release"
echo "  PYTHONPATH=.:../python uv run python ../scripts/train.py \\"
echo "    --cache-dir ../cache/mes/ --resume ../$LOCAL_DIR/ppo_lob.zip ..."
