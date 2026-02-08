#!/usr/bin/env bash
# Watch for Databento download to finish, move it to data/, log everything.
set -euo pipefail

TARGET="GLBX-20260207-L953CAPU5B.zip"
WATCH_DIR="$HOME/Downloads"
DEST_DIR="/Users/brandonbell/LOCAL_DEV/LOB-RL/data"
LOG="/tmp/download_watcher.log"
POLL=30        # seconds between checks
STABLE_WAIT=5  # seconds to confirm size is stable

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Download watcher started ==="
log "Watching: ${WATCH_DIR}/${TARGET}"
log "Destination: ${DEST_DIR}/"

while true; do
    FILE="${WATCH_DIR}/${TARGET}"

    # Check for temp/partial download artifacts
    if [[ -f "${FILE}.crdownload" || -f "${FILE}.part" || -f "${FILE}.download" ]]; then
        log "Temp file detected — download in progress, waiting..."
        sleep "$POLL"
        continue
    fi

    if [[ ! -f "$FILE" ]]; then
        log "File not found yet, polling again in ${POLL}s..."
        sleep "$POLL"
        continue
    fi

    # File exists and no temp files — confirm size is stable
    log "File found! Checking size stability..."
    SIZE1=$(stat -f%z "$FILE" 2>/dev/null || echo 0)
    sleep "$STABLE_WAIT"
    SIZE2=$(stat -f%z "$FILE" 2>/dev/null || echo 0)

    if [[ "$SIZE1" != "$SIZE2" ]]; then
        log "Size changed (${SIZE1} -> ${SIZE2}), still downloading. Waiting..."
        sleep "$POLL"
        continue
    fi

    if [[ "$SIZE1" -eq 0 ]]; then
        log "File exists but is 0 bytes — waiting..."
        sleep "$POLL"
        continue
    fi

    # Download is complete
    log "Download complete! Size: ${SIZE1} bytes ($(( SIZE1 / 1024 / 1024 )) MB)"
    log "Moving to ${DEST_DIR}/"
    mv "$FILE" "$DEST_DIR/"
    log "Moved successfully: ${DEST_DIR}/${TARGET}"
    log "=== Download watcher finished ==="
    exit 0
done
