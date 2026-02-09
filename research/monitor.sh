#!/usr/bin/env bash
# Monitor RunPod training pods, fetch results when done, then launch
# Claude Code to synthesize findings.
#
# Usage:
#   RUNPOD_VOLUME_ID=4w2m8hek66 ./research/monitor.sh
#
# Polls every hour. When a pod finishes (exits or disappears):
#   1. Fetches results via S3 (no SSH needed, no running pod needed)
#   2. Verifies expected files were downloaded
#   3. Removes the pod
#   4. After ALL pods finish, launches Claude Code to analyze results
#
# Pods auto-stop on training success (start.sh exits 0). This monitor
# detects the "Exited" state via `runpodctl get pod`.
#
# Edit the PODS array below for each experiment batch.

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────
# Format: "pod_id:exp_name"
# The run directory on the volume is: {exp_name}_{pod_id}
PODS=(
    "6kwbf810ribiza:lstm"
    "yvag35jcok2egk:mlp"
    "0w3gtzsu1h3nhl:framestack"
)
POLL_INTERVAL="${POLL_INTERVAL:-3600}"  # seconds between polls (default: 1 hour)
VOLUME_ID="${RUNPOD_VOLUME_ID:?Set RUNPOD_VOLUME_ID to your network volume ID}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
FETCH_SCRIPT="$REPO_ROOT/runpod/fetch-results.sh"
RESEARCH_DIR="$REPO_ROOT/research"

# ─── Helpers ─────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_pod_status() {
    # Returns the status string for a pod (RUNNING, EXITED, etc.) or empty if not found
    local pod_id="$1"
    runpodctl get pod 2>&1 | grep "$pod_id" | awk '{print $2}' || true
}

is_pod_done() {
    # A pod is "done" if it's EXITED (success) or no longer listed (removed/crashed)
    local pod_id="$1"
    local status
    status=$(get_pod_status "$pod_id")
    if [ -z "$status" ]; then
        # Pod no longer listed — treat as done
        return 0
    fi
    case "$status" in
        EXITED|TERMINATED|COMPLETED) return 0 ;;
        *) return 1 ;;
    esac
}

fetch_and_verify() {
    # Fetch results via S3 and verify expected files exist locally.
    # Returns 0 on success, 1 on failure.
    local run_dir="$1"
    local local_dir="$RESULTS_DIR/$run_dir"

    log "  Fetching results via S3: $run_dir"
    if RUNPOD_VOLUME_ID="$VOLUME_ID" "$FETCH_SCRIPT" "$run_dir" 2>&1 | tail -10; then
        # Verify expected files
        if [ -f "$local_dir/train.log" ] && [ -f "$local_dir/ppo_lob.zip" ]; then
            log "  Verified: train.log and ppo_lob.zip present."
            return 0
        else
            log "  WARNING: Download succeeded but expected files missing."
            [ ! -f "$local_dir/train.log" ] && log "    Missing: train.log"
            [ ! -f "$local_dir/ppo_lob.zip" ] && log "    Missing: ppo_lob.zip"
            return 1
        fi
    else
        log "  WARNING: fetch-results.sh failed for $run_dir."
        return 1
    fi
}

# ─── Track completion with simple parallel arrays ────────────────────
POD_IDS=()
POD_NAMES=()
POD_DONE=()

for entry in "${PODS[@]}"; do
    POD_IDS+=("${entry%%:*}")
    POD_NAMES+=("${entry##*:}")
    POD_DONE+=("no")
done

NUM_PODS=${#POD_IDS[@]}

log "Monitoring $NUM_PODS pods. Poll interval: ${POLL_INTERVAL}s"
for i in $(seq 0 $((NUM_PODS - 1))); do
    run_dir="${POD_NAMES[$i]}_${POD_IDS[$i]}"
    log "  ${POD_NAMES[$i]} -> ${POD_IDS[$i]}  (run dir: $run_dir)"
done

while true; do
    all_done=true

    for i in $(seq 0 $((NUM_PODS - 1))); do
        [ "${POD_DONE[$i]}" = "yes" ] && continue
        all_done=false

        pod_id="${POD_IDS[$i]}"
        exp_name="${POD_NAMES[$i]}"
        run_dir="${exp_name}_${pod_id}"

        log "Checking $exp_name ($pod_id)..."

        status=$(get_pod_status "$pod_id")
        if [ -n "$status" ]; then
            log "  Status: $status"
        else
            log "  Pod not found in listing."
        fi

        if is_pod_done "$pod_id"; then
            log "  Pod finished (status: ${status:-not found}). Fetching results..."

            # Fetch and verify
            if fetch_and_verify "$run_dir"; then
                log "  Results saved to $RESULTS_DIR/$run_dir/"

                # Only remove the pod if it still exists and results verified
                if [ -n "$status" ]; then
                    log "  Removing pod $pod_id..."
                    runpodctl remove pod "$pod_id" 2>&1 || log "  WARNING: Failed to remove pod."
                fi
            else
                log "  Fetch failed or incomplete. NOT removing pod $pod_id."
                log "  Manual fetch: RUNPOD_VOLUME_ID=$VOLUME_ID ./runpod/fetch-results.sh $run_dir"

                # Still mark as done to avoid infinite loop, but warn
                log "  WARNING: Marking as done despite fetch failure. Check results manually."
            fi

            POD_DONE[$i]="yes"
            log "  $exp_name complete."
        else
            log "  Still running."
        fi
    done

    if $all_done; then
        log "All pods finished."
        break
    fi

    log "Sleeping ${POLL_INTERVAL}s until next check..."
    sleep "$POLL_INTERVAL"
done

# ─── Synthesize results with Claude Code ─────────────────────────────
log "Launching Claude Code to analyze results..."

# Build a summary of what was fetched
RESULT_DIRS=""
for i in $(seq 0 $((NUM_PODS - 1))); do
    pod_id="${POD_IDS[$i]}"
    exp_name="${POD_NAMES[$i]}"
    run_dir="${exp_name}_${pod_id}"
    RESULT_DIRS="$RESULT_DIRS  - $exp_name: $RESULTS_DIR/$run_dir/"$'\n'
done

read -r -d '' ANALYSIS_PROMPT << 'PROMPT_EOF' || true
You are analyzing the results of three parallel RL training experiments for a
limit order book (LOB) trading agent. Each experiment ran for 5M timesteps on
an RTX 4090 GPU with the same hyperparameters (bar_size=1000, ent_coef=0.05,
lr=1e-3, shuffle-split, seed=42) but different model architectures:

1. **LSTM** (RecurrentPPO) — recurrent policy for temporal context
2. **MLP** (PPO) — feedforward baseline, 256x256 ReLU
3. **Frame-stack** (PPO + VecFrameStack 4) — concatenated last 4 observations

TASK:
1. Read the train.log from each experiment's results directory.
2. Extract key metrics: final val/test mean_return, sortino_ratio, positive_episodes,
   training fps, total wall time, entropy trajectory.
3. If TensorBoard event files exist, extract the reward/entropy curves.
4. Write a comprehensive analysis report to research/experiment_report.md covering:
   - Summary table comparing all three architectures
   - Which model generalizes best OOS (val and test metrics)
   - Training efficiency (fps, wall time)
   - Whether any model shows signs of learning (positive Sortino, positive episodes)
   - Comparison with prior 2M-step local results (MLP: val -51.5/test -62.5,
     Frame-stack: val -48.4/test -50.2, LSTM: killed at 15%)
   - Recommendations for next steps
5. Keep the report concise and data-driven. Use tables and bullet points.
PROMPT_EOF

# Append the result directory info
ANALYSIS_PROMPT="$ANALYSIS_PROMPT

RESULT DIRECTORIES:
$RESULT_DIRS

Read the train.log files in each result directory to find the metrics.
Write your report to: $RESEARCH_DIR/experiment_report.md"

cd "$RESEARCH_DIR"
claude --dangerously-skip-permissions -p "$ANALYSIS_PROMPT" \
    --allowedTools "Read,Write,Glob,Grep,Bash" \
    2>&1 | tee "$RESEARCH_DIR/claude_analysis.log"

log "Analysis complete. Report at: $RESEARCH_DIR/experiment_report.md"
log "Claude log at: $RESEARCH_DIR/claude_analysis.log"
