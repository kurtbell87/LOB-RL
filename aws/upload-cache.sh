#!/usr/bin/env bash
# Upload cache/mes/ to S3 (one-time).
#
# Usage:
#   AWS_S3_BUCKET=lob-rl-training ./aws/upload-cache.sh
#
# Safe to re-run — only uploads new/changed files.

set -euo pipefail

S3_BUCKET="${AWS_S3_BUCKET:?Set AWS_S3_BUCKET}"
REGION="${AWS_REGION:-us-east-1}"
CACHE_DIR="${CACHE_DIR:-cache/mes}"

if [[ ! -d "$CACHE_DIR" ]]; then
    echo "ERROR: Cache directory not found: $CACHE_DIR"
    echo "Run precompute_cache.py first, or set CACHE_DIR."
    exit 1
fi

LOCAL_COUNT=$(find "$CACHE_DIR" -name "*.npz" | wc -l | tr -d ' ')
echo "=== Uploading cache to S3 ==="
echo "Source:      $CACHE_DIR/ ($LOCAL_COUNT .npz files)"
echo "Destination: s3://$S3_BUCKET/cache/mes/"
echo ""

aws s3 sync "$CACHE_DIR/" "s3://$S3_BUCKET/cache/mes/" --region "$REGION"

REMOTE_COUNT=$(aws s3 ls "s3://$S3_BUCKET/cache/mes/" --region "$REGION" --recursive 2>&1 | grep -c '\.npz$' || echo "0")
echo ""
echo "Local:  $LOCAL_COUNT files"
echo "Remote: $REMOTE_COUNT files"

if [[ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ]]; then
    echo "Upload verified."
else
    echo "WARNING: Count mismatch. Re-run to retry."
fi
