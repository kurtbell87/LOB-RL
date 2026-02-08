#!/bin/bash
# RunPod container startup script.
# Runs train.py in the background and keeps the container alive for SSH access.
#
# Output dir: /workspace/runs/{EXP_NAME}_{RUNPOD_POD_ID}/
# Training logs: {output_dir}/train.log

set -euo pipefail

# Start SSH daemon for remote access
if [ -f /usr/sbin/sshd ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    # RunPod injects public keys via PUBLIC_KEY env var
    if [ -n "${PUBLIC_KEY:-}" ]; then
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
    # Also check RUNPOD_PUBLIC_KEY
    if [ -n "${RUNPOD_PUBLIC_KEY:-}" ]; then
        echo "$RUNPOD_PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi
    /usr/sbin/sshd
    echo "SSH daemon started."
fi

# Construct unique output dir: /workspace/runs/{exp_name}_{pod_id}
RUN_NAME="${EXP_NAME:-exp}_${RUNPOD_POD_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="/workspace/runs/${RUN_NAME}"
LOG_DIR="$OUTPUT_DIR"
LOG_FILE="$LOG_DIR/train.log"
mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
    echo "No training args provided. Container is idle — SSH in to run manually."
    echo "  python /app/scripts/train.py --cache-dir /workspace/cache/mes/ --output-dir $OUTPUT_DIR ..."
    sleep infinity
fi

echo "=== Starting training ===" | tee "$LOG_FILE"
echo "Run:  $RUN_NAME" | tee -a "$LOG_FILE"
echo "Out:  $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Args: $*" | tee -a "$LOG_FILE"
echo "Log:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python /app/scripts/train.py --output-dir "$OUTPUT_DIR" "$@" >> "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID" | tee -a "$LOG_FILE"

# Wait for training to finish; keep container alive either way
wait $TRAIN_PID || EXIT_CODE=$?
EXIT_CODE=${EXIT_CODE:-0}

echo "" >> "$LOG_FILE"
echo "=== Training exited with code $EXIT_CODE ===" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed. Container staying alive for debugging via SSH." | tee -a "$LOG_FILE"
    echo "Check log: tail -f $LOG_FILE" | tee -a "$LOG_FILE"
fi

# Keep container alive for result fetching / debugging
sleep infinity
