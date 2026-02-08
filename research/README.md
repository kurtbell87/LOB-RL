# Research

Experiment monitoring, analysis reports, and synthesized results.

## Files

| File | Role |
|------|------|
| `monitor.sh` | Polls RunPod pods hourly, fetches results when done, removes pods, launches Claude Code to write analysis report |
| `experiment_report.md` | Auto-generated analysis report (created by Claude Code after experiments finish) |
| `claude_analysis.log` | Full Claude Code output log from the analysis run |

## Usage

```bash
# Run in background (nohup so it survives terminal close)
nohup ./research/monitor.sh > research/monitor.log 2>&1 &

# Watch progress
tail -f research/monitor.log

# Override poll interval (default: 3600s = 1 hour)
POLL_INTERVAL=1800 nohup ./research/monitor.sh > research/monitor.log 2>&1 &
```

## Configuration

Edit the `PODS` array at the top of `monitor.sh` for each experiment batch:
```bash
PODS=(
    "pod_id_1:exp_name_1"
    "pod_id_2:exp_name_2"
)
```

## Flow

1. Polls each pod via SSH, checking train.log for "Training exited with code"
2. When a pod finishes: fetches results via `runpod/fetch-results.sh`, then removes the pod
3. When ALL pods finish: launches `claude --dangerously-skip-permissions` to read the results and write `experiment_report.md`
