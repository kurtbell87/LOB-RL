#!/usr/bin/env bash
# Fetch training results from S3.
#
# Usage:
#   ./aws/fetch-results.sh <run_dir>
#
# Example:
#   ./aws/fetch-results.sh lstm_i-0abc123def456
#
# Downloads from s3://$AWS_S3_BUCKET/runs/<run_dir>/ to results/<run_dir>/ locally.
#
# Required env vars:
#   AWS_S3_BUCKET   S3 bucket name
#
# Optional env vars:
#   AWS_REGION      Default: us-east-1

set -euo pipefail

RUN_DIR="${1:?Usage: ./aws/fetch-results.sh <run_dir>  (e.g. lstm_i-0abc123def456)}"
S3_BUCKET="${AWS_S3_BUCKET:?Set AWS_S3_BUCKET}"
REGION="${AWS_REGION:-us-east-1}"

S3_PATH="s3://${S3_BUCKET}/runs/${RUN_DIR}/"
LOCAL_DIR="results/${RUN_DIR}"

mkdir -p "$LOCAL_DIR"

echo "=== Fetching results from S3 ==="
echo "Source:      $S3_PATH"
echo "Destination: $LOCAL_DIR/"
echo ""

# Verify remote directory exists
FILE_COUNT=$(aws s3 ls "$S3_PATH" --region "$REGION" --recursive 2>&1 | wc -l | tr -d ' ')
if [[ "$FILE_COUNT" -eq 0 ]]; then
    echo "ERROR: No files found at $S3_PATH"
    echo ""
    echo "Available run dirs:"
    aws s3 ls "s3://${S3_BUCKET}/runs/" --region "$REGION" 2>&1 || true
    exit 1
fi
echo "Found $FILE_COUNT files in S3."
echo ""

# Download
aws s3 sync "$S3_PATH" "$LOCAL_DIR/" --region "$REGION"

echo ""
echo "=== Results downloaded ==="
echo "Location: $LOCAL_DIR/"
echo ""

# Show what was downloaded
if command -v tree &>/dev/null; then
    tree "$LOCAL_DIR/"
else
    find "$LOCAL_DIR/" -type f | head -20
fi

# Verify expected files
MISSING=""
[[ -f "$LOCAL_DIR/train.log" ]] || MISSING="$MISSING train.log"
[[ -f "$LOCAL_DIR/ppo_lob.zip" ]] || MISSING="$MISSING ppo_lob.zip"
if [[ -n "$MISSING" ]]; then
    echo ""
    echo "WARNING: Expected files missing:$MISSING"
    echo "Training may not have completed successfully."
    echo "Check bootstrap log: aws s3 cp s3://$S3_BUCKET/runs/$RUN_DIR/bootstrap.log -"
fi

echo ""
echo "To evaluate locally:"
echo "  cd build-release"
echo "  PYTHONPATH=.:../python uv run python ../scripts/train.py \\"
echo "    --cache-dir ../cache/mes/ --resume ../$LOCAL_DIR/ppo_lob.zip ..."
