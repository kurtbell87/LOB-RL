#!/usr/bin/env bash
# Monitor LOB-RL EC2 instances, auto-fetch results when done.
#
# Usage:
#   ./aws/monitor.sh                          # discover all lob-rl instances
#   ./aws/monitor.sh i-0abc123 i-0def456      # monitor specific instances
#
# Polls every 5 minutes. When an instance terminates/stops:
#   1. Fetches results from S3
#   2. Verifies expected files
#
# Required env vars:
#   AWS_S3_BUCKET   S3 bucket name
#
# Optional env vars:
#   AWS_REGION         Default: us-east-1
#   POLL_INTERVAL      Seconds between polls (default: 300)

set -euo pipefail

S3_BUCKET="${AWS_S3_BUCKET:?Set AWS_S3_BUCKET}"
REGION="${AWS_REGION:-us-east-1}"
POLL_INTERVAL="${POLL_INTERVAL:-300}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_instance_state() {
    aws ec2 describe-instances --region "$REGION" \
        --instance-ids "$1" \
        --query "Reservations[0].Instances[0].State.Name" \
        --output text 2>/dev/null || echo "not-found"
}

get_instance_exp_name() {
    aws ec2 describe-instances --region "$REGION" \
        --instance-ids "$1" \
        --query "Reservations[0].Instances[0].Tags[?Key=='ExpName'].Value | [0]" \
        --output text 2>/dev/null || echo "unknown"
}

# ─── Discover or use provided instance IDs ───────────────────────────
if [[ $# -gt 0 ]]; then
    INSTANCE_IDS=("$@")
else
    log "Discovering lob-rl instances..."
    mapfile -t INSTANCE_IDS < <(aws ec2 describe-instances --region "$REGION" \
        --filters "Name=tag:Project,Values=lob-rl" \
                  "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
        --query "Reservations[*].Instances[*].InstanceId" --output text)

    if [[ ${#INSTANCE_IDS[@]} -eq 0 || -z "${INSTANCE_IDS[0]:-}" ]]; then
        log "No active lob-rl instances found."
        exit 0
    fi
fi

# ─── Track completion ────────────────────────────────────────────────
declare -A DONE

log "Monitoring ${#INSTANCE_IDS[@]} instances. Poll interval: ${POLL_INTERVAL}s"
for iid in "${INSTANCE_IDS[@]}"; do
    exp_name=$(get_instance_exp_name "$iid")
    state=$(get_instance_state "$iid")
    log "  $iid ($exp_name) — $state"
    DONE[$iid]="no"
done

while true; do
    all_done=true

    for iid in "${INSTANCE_IDS[@]}"; do
        [[ "${DONE[$iid]}" == "yes" ]] && continue
        all_done=false

        state=$(get_instance_state "$iid")
        exp_name=$(get_instance_exp_name "$iid")
        run_dir="${exp_name}_${iid}"

        log "Checking $iid ($exp_name): $state"

        case "$state" in
            terminated|stopped|shutting-down|not-found)
                log "  Instance done ($state). Fetching results..."
                if AWS_S3_BUCKET="$S3_BUCKET" AWS_REGION="$REGION" \
                    ./aws/fetch-results.sh "$run_dir" 2>&1 | tail -5; then
                    log "  Results saved to results/$run_dir/"
                else
                    log "  WARNING: Fetch failed. Manual: ./aws/fetch-results.sh $run_dir"
                fi
                DONE[$iid]="yes"
                ;;
            *)
                log "  Still running."
                ;;
        esac
    done

    if $all_done; then
        log "All instances finished."
        break
    fi

    log "Sleeping ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
