#!/usr/bin/env bash
# Monitor RunPod training pods, fetch results when done, then launch
# Claude Code to synthesize findings.
#
# Usage:
#   ./research/monitor.sh
#
# Polls every hour. When all pods finish:
#   1. Fetches results via rsync
#   2. Removes pods
#   3. Launches Claude Code to analyze results and write a report
#
# Edit the PODS array below for each experiment batch.

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────
# Format: "pod_id:exp_name"
PODS=(
    "6kwbf810ribiza:lstm"
    "yvag35jcok2egk:mlp"
    "0w3gtzsu1h3nhl:framestack"
)
POLL_INTERVAL="${POLL_INTERVAL:-3600}"  # seconds between polls (default: 1 hour)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/results"
RESEARCH_DIR="$REPO_ROOT/research"
FETCH_SCRIPT="$REPO_ROOT/runpod/fetch-results.sh"
SSH_KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -i $SSH_KEY"

# ─── Helpers ─────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

get_ssh_info() {
    local pod_id="$1"
    local ssh_line
    ssh_line=$(runpodctl ssh connect "$pod_id" 2>&1) || return 1
    local host port
    host=$(echo "$ssh_line" | grep -oE 'root@[^ ]+' | sed 's/root@//')
    port=$(echo "$ssh_line" | grep -oE '\-p [0-9]+' | grep -oE '[0-9]+')
    if [ -z "$host" ] || [ -z "$port" ]; then
        return 1
    fi
    echo "$host $port"
}

check_training_done() {
    local pod_id="$1" exp_name="$2"
    local info host port
    info=$(get_ssh_info "$pod_id") || return 1
    host=$(echo "$info" | awk '{print $1}')
    port=$(echo "$info" | awk '{print $2}')

    local log_path="/workspace/runs/${exp_name}_${pod_id}/train.log"
    local result
    result=$(ssh $SSH_OPTS root@"$host" -p "$port" \
        "grep -c 'Training exited with code' $log_path 2>/dev/null" 2>/dev/null) || return 1

    [ "$result" -ge 1 ] && return 0 || return 1
}

get_exit_code() {
    local pod_id="$1" exp_name="$2"
    local info host port
    info=$(get_ssh_info "$pod_id") || { echo "unknown"; return; }
    host=$(echo "$info" | awk '{print $1}')
    port=$(echo "$info" | awk '{print $2}')

    local log_path="/workspace/runs/${exp_name}_${pod_id}/train.log"
    ssh $SSH_OPTS root@"$host" -p "$port" \
        "grep 'Training exited with code' $log_path 2>/dev/null | grep -oE '[0-9]+$'" 2>/dev/null || echo "unknown"
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
    log "  ${POD_NAMES[$i]} -> ${POD_IDS[$i]}"
done

while true; do
    all_done=true

    for i in $(seq 0 $((NUM_PODS - 1))); do
        [ "${POD_DONE[$i]}" = "yes" ] && continue
        all_done=false

        pod_id="${POD_IDS[$i]}"
        exp_name="${POD_NAMES[$i]}"

        log "Checking $exp_name ($pod_id)..."

        # Check if pod still exists (retry once on failure)
        pod_list=$(runpodctl get pod 2>&1) || pod_list=""
        if [ -z "$pod_list" ] || ! echo "$pod_list" | grep -q "$pod_id"; then
            sleep 5
            pod_list=$(runpodctl get pod 2>&1) || pod_list=""
            if [ -z "$pod_list" ] || ! echo "$pod_list" | grep -q "$pod_id"; then
                log "  WARNING: Pod $pod_id no longer exists. Marking as done."
                POD_DONE[$i]="yes"
                continue
            fi
        fi

        # Check if training finished
        if check_training_done "$pod_id" "$exp_name"; then
            exit_code=$(get_exit_code "$pod_id" "$exp_name")
            log "  FINISHED (exit code: $exit_code). Fetching results..."

            # Fetch results
            if "$FETCH_SCRIPT" "$pod_id" 2>&1 | tail -5; then
                log "  Results saved to $RESULTS_DIR/$pod_id/"
            else
                log "  WARNING: fetch-results.sh failed. Results may be incomplete."
            fi

            # Remove pod
            log "  Removing pod $pod_id..."
            runpodctl remove pod "$pod_id" 2>&1 || log "  WARNING: Failed to remove pod."

            POD_DONE[$i]="yes"
            log "  $exp_name complete."
        else
            log "  Still training."
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
    RESULT_DIRS="$RESULT_DIRS  - $exp_name: $RESULTS_DIR/$pod_id/ (output dir: ${exp_name}_${pod_id})"$'\n'
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
