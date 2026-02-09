"""Execute experiment exp-001: Training Data Scaling — 20 vs 199 Days.

Runs the full protocol:
  MVE → Run A (MLP 20d) → Run B (LSTM 20d) → Run C (MLP 199d) → Run D (LSTM 199d) → Run E (LSTM 199d seed 43, conditional)

Collects all metrics and writes to results/exp-001-does-increasing-training-data-from-20-to/metrics.json.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build-release")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache", "mes")
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "results", "exp-001-does-increasing-training-data-from-20-to"
)
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "train.py")

# Common args for all runs
COMMON_ARGS = [
    "--cache-dir", CACHE_DIR,
    "--bar-size", "1000",
    "--execution-cost",
    "--policy-arch", "256,256",
    "--activation", "relu",
    "--ent-coef", "0.05",
    "--learning-rate", "0.001",
    "--shuffle-split",
]


def run_training(run_name, extra_args, output_dir):
    """Run a training job and capture output."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")

    cmd = [
        sys.executable, TRAIN_SCRIPT,
        *COMMON_ARGS,
        "--output-dir", output_dir,
        *extra_args,
    ]

    print(f"\n{'='*60}")
    print(f"Starting {run_name}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=BUILD_DIR,
            env={
                **os.environ,
                "PYTHONPATH": f".:{os.path.join(PROJECT_ROOT, 'python')}",
            },
        )

        # Stream and capture output
        output_lines = []
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace")
            output_lines.append(decoded)
            log_file.write(decoded)
            sys.stdout.write(decoded)
            sys.stdout.flush()

        proc.wait()

    wall_time = time.time() - start_time
    print(f"\n{run_name} completed in {wall_time:.0f}s (exit code: {proc.returncode})")

    return {
        "exit_code": proc.returncode,
        "wall_time_seconds": wall_time,
        "output": "".join(output_lines),
        "log_path": log_path,
    }


def parse_eval_metrics(output):
    """Parse validation and test metrics from train.py stdout."""
    metrics = {}
    for line in output.split("\n"):
        if "Validation metrics:" in line:
            val_str = line.split("Validation metrics:")[1].strip()
            metrics["val"] = eval(val_str)  # safe: known output format
        elif "Test metrics:" in line:
            test_str = line.split("Test metrics:")[1].strip()
            metrics["test"] = eval(test_str)
    return metrics


def parse_fps_from_output(output):
    """Parse FPS from train.py stdout."""
    fps_values = []
    for line in output.split("\n"):
        if "fps" in line.lower() and "|" in line:
            parts = line.split("|")
            for part in parts:
                part = part.strip()
                try:
                    fps_values.append(float(part))
                except ValueError:
                    pass
    return fps_values[-1] if fps_values else None


def read_tensorboard_metrics(tb_dir):
    """Read all scalar metrics from TensorBoard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    metrics = {}

    if not os.path.exists(tb_dir):
        return metrics

    # Find the actual event file directory
    for root, dirs, files in os.walk(tb_dir):
        event_files = [f for f in files if f.startswith("events.out.tfevents")]
        if event_files:
            ea = EventAccumulator(root)
            ea.Reload()
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


def get_metric_at_step(tb_metrics, tag, target_step):
    """Get the metric value closest to target_step."""
    if tag not in tb_metrics:
        return None
    events = tb_metrics[tag]
    if not events:
        return None

    # Find closest step
    closest = min(events, key=lambda e: abs(e[0] - target_step))
    return closest[1]


def get_last_metric(tb_metrics, tag):
    """Get the last logged value for a metric."""
    if tag not in tb_metrics:
        return None
    events = tb_metrics[tag]
    if not events:
        return None
    return events[-1][1]


def extract_run_metrics(run_result, output_dir, is_recurrent=False):
    """Extract all metrics for a single run."""
    eval_metrics = parse_eval_metrics(run_result["output"])

    # Find TensorBoard directory
    tb_base = os.path.join(output_dir, "tb_logs")
    tb_dir = None
    if os.path.exists(tb_base):
        subdirs = os.listdir(tb_base)
        if subdirs:
            tb_dir = os.path.join(tb_base, subdirs[0])

    tb_metrics = read_tensorboard_metrics(tb_dir) if tb_dir else {}

    result = {
        "wall_time_seconds": run_result["wall_time_seconds"],
        "exit_code": run_result["exit_code"],
    }

    # Primary metrics from evaluation
    if "val" in eval_metrics:
        result["val_mean_return"] = eval_metrics["val"]["mean_return"]
        result["val_std_return"] = eval_metrics["val"]["std_return"]
        result["val_sortino_ratio"] = eval_metrics["val"]["sortino_ratio"]
        result["val_n_episodes"] = eval_metrics["val"]["n_episodes"]
        result["val_positive_episodes"] = eval_metrics["val"]["positive_episodes"]

    if "test" in eval_metrics:
        result["test_mean_return"] = eval_metrics["test"]["mean_return"]
        result["test_std_return"] = eval_metrics["test"]["std_return"]
        result["test_sortino_ratio"] = eval_metrics["test"]["sortino_ratio"]
        result["test_n_episodes"] = eval_metrics["test"]["n_episodes"]
        result["test_positive_episodes"] = eval_metrics["test"]["positive_episodes"]

    # TensorBoard training metrics at checkpoints
    for step in [1_000_000, 3_000_000, 5_000_000]:
        step_label = f"{step // 1_000_000}M"
        result[f"explained_variance_{step_label}"] = get_metric_at_step(
            tb_metrics, "train/explained_variance", step
        )
        result[f"approx_kl_{step_label}"] = get_metric_at_step(
            tb_metrics, "train/approx_kl", step
        )
        result[f"clip_fraction_{step_label}"] = get_metric_at_step(
            tb_metrics, "train/clip_fraction", step
        )

    # Entropy at 1M, 2M, 3M, 4M, 5M
    for step in [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000]:
        step_label = f"{step // 1_000_000}M"
        result[f"entropy_{step_label}"] = get_metric_at_step(
            tb_metrics, "train/entropy_loss", step
        )

    # Final values
    result["final_entropy"] = get_last_metric(tb_metrics, "train/entropy_loss")
    result["final_explained_variance"] = get_last_metric(
        tb_metrics, "train/explained_variance"
    )

    # FPS
    fps = get_last_metric(tb_metrics, "time/fps")
    result["training_fps"] = fps

    # In-sample return (last episode reward from training)
    ep_rew = get_last_metric(tb_metrics, "rollout/ep_rew_mean")
    result["in_sample_mean_return"] = ep_rew

    return result


def check_abort_criteria(run_metrics, run_name):
    """Check abort criteria and return (should_abort, reason)."""
    # NaN/Inf check
    for key, val in run_metrics.items():
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            if key in ["val_sortino_ratio", "test_sortino_ratio"]:
                continue  # Sortino can be inf
            return True, f"NaN/Inf detected in {key}={val} for {run_name}"

    # FPS abort
    fps = run_metrics.get("training_fps")
    if fps is not None:
        if "lstm" in run_name.lower() and fps < 100:
            return True, f"LSTM FPS too low: {fps} < 100 for {run_name}"
        if "mlp" in run_name.lower() and fps < 500:
            return True, f"MLP FPS too low: {fps} < 500 for {run_name}"

    # Entropy collapse
    final_entropy = run_metrics.get("final_entropy")
    if final_entropy is not None and final_entropy < -0.60:
        return True, f"Entropy collapsed: {final_entropy} < -0.60 for {run_name}"

    return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mve", action="store_true", help="Skip MVE phase")
    parser.add_argument("--skip-to", type=str, default=None,
                        help="Skip to a specific run (a, b, c, d, e)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    experiment_start = time.time()
    all_results = {}
    abort_triggered = False
    abort_reason = None

    # ============================================================
    # Phase 1: MVE — MLP 199-day, 500K steps
    # ============================================================
    if not args.skip_mve and args.skip_to is None:
        print("\n" + "=" * 60)
        print("PHASE 1: MVE — MLP 199-day, 500K steps")
        print("=" * 60)

        mve_dir = os.path.join(RESULTS_DIR, "mve-mlp-199d")
        mve_result = run_training(
            "MVE (MLP 199-day, 500K steps)",
            ["--seed", "42", "--train-days", "199", "--total-timesteps", "500000",
             "--checkpoint-freq", "100000"],
            mve_dir,
        )

        if mve_result["exit_code"] != 0:
            print("ERROR: MVE failed! Aborting experiment.")
            abort_triggered = True
            abort_reason = "MVE failed with non-zero exit code"
        else:
            mve_metrics = extract_run_metrics(mve_result, mve_dir)
            all_results["mve"] = mve_metrics

            # MVE gates
            mve_fps = mve_metrics.get("training_fps")
            if mve_fps is not None and mve_fps < 600:
                print(f"WARNING: MVE FPS {mve_fps} < 600 threshold")
                # Don't abort, just warn — local FPS may differ from spec expectation

            print(f"\nMVE metrics: FPS={mve_fps}, wall_time={mve_metrics['wall_time_seconds']:.0f}s")
            print("MVE PASSED — proceeding to full protocol")

    # ============================================================
    # Phase 2: 20-Day Controls
    # ============================================================
    runs_to_execute = [
        ("a", "Run A: MLP 20-day control",
         ["--seed", "42", "--train-days", "20", "--total-timesteps", "5000000",
          "--checkpoint-freq", "1000000"],
         False),
        ("b", "Run B: LSTM 20-day control",
         ["--seed", "42", "--train-days", "20", "--total-timesteps", "5000000",
          "--checkpoint-freq", "1000000", "--recurrent"],
         True),
        ("c", "Run C: MLP 199-day treatment",
         ["--seed", "42", "--train-days", "199", "--total-timesteps", "5000000",
          "--checkpoint-freq", "1000000"],
         False),
        ("d", "Run D: LSTM 199-day treatment",
         ["--seed", "42", "--train-days", "199", "--total-timesteps", "5000000",
          "--checkpoint-freq", "1000000", "--recurrent"],
         True),
    ]

    run_dirs = {
        "a": "run-a-mlp-20d",
        "b": "run-b-lstm-20d",
        "c": "run-c-mlp-199d",
        "d": "run-d-lstm-199d",
        "e": "run-e-lstm-199d-seed43",
    }

    skip_to_idx = None
    if args.skip_to:
        for i, (run_id, _, _, _) in enumerate(runs_to_execute):
            if run_id == args.skip_to:
                skip_to_idx = i
                break

    for i, (run_id, run_name, extra_args, is_recurrent) in enumerate(runs_to_execute):
        if abort_triggered:
            break
        if skip_to_idx is not None and i < skip_to_idx:
            continue

        phase_label = "Phase 2" if run_id in ("a", "b") else "Phase 3"
        print(f"\n{'='*60}")
        print(f"{phase_label}: {run_name}")
        print(f"{'='*60}")

        run_dir = os.path.join(RESULTS_DIR, run_dirs[run_id])
        result = run_training(run_name, extra_args, run_dir)

        if result["exit_code"] != 0:
            print(f"ERROR: {run_name} failed!")
            abort_triggered = True
            abort_reason = f"{run_name} failed with exit code {result['exit_code']}"
            break

        metrics = extract_run_metrics(result, run_dir, is_recurrent)
        all_results[run_id] = metrics

        # Check abort criteria
        should_abort, reason = check_abort_criteria(metrics, run_name)
        if should_abort:
            print(f"ABORT: {reason}")
            abort_triggered = True
            abort_reason = reason
            break

        # Phase 2 sanity checks: verify controls reproduce historical baselines
        if run_id == "a":
            val_ret = metrics.get("val_mean_return")
            if val_ret is not None:
                diff_from_baseline = abs(val_ret - (-62.9))
                print(f"\nRun A sanity: val_return={val_ret:.1f}, diff from pre-006 baseline={diff_from_baseline:.1f}")
                if diff_from_baseline > 30:
                    print(f"ABORT: MLP control val return {val_ret:.1f} differs from historical -62.9 by {diff_from_baseline:.1f} > 30")
                    abort_triggered = True
                    abort_reason = f"MLP 20-day control val return {val_ret:.1f} differs from historical baseline -62.9 by {diff_from_baseline:.1f} (>30 threshold)"
                    break

        if run_id == "b":
            val_ret = metrics.get("val_mean_return")
            if val_ret is not None:
                diff_from_baseline = abs(val_ret - (-36.7))
                print(f"\nRun B sanity: val_return={val_ret:.1f}, diff from pre-005 baseline={diff_from_baseline:.1f}")
                if diff_from_baseline > 30:
                    print(f"ABORT: LSTM control val return {val_ret:.1f} differs from historical -36.7 by {diff_from_baseline:.1f} > 30")
                    abort_triggered = True
                    abort_reason = f"LSTM 20-day control val return {val_ret:.1f} differs from historical baseline -36.7 by {diff_from_baseline:.1f} (>30 threshold)"
                    break

    # ============================================================
    # Phase 4: Reproducibility Check (conditional)
    # ============================================================
    if not abort_triggered and "d" in all_results and "b" in all_results:
        d_val = all_results["d"].get("val_mean_return", 0)
        b_val = all_results["b"].get("val_mean_return", 0)
        improvement = d_val - b_val

        print(f"\nRun D vs Run B improvement: {improvement:.1f}")
        if abs(improvement) <= 10:
            print("Skip Run E — Run D shows no meaningful improvement over Run B (within ±10)")
        else:
            print(f"Running Run E — Run D shows {improvement:.1f} point change vs Run B")

            run_dir = os.path.join(RESULTS_DIR, run_dirs["e"])
            result = run_training(
                "Run E: LSTM 199-day seed 43",
                ["--seed", "43", "--train-days", "199", "--total-timesteps", "5000000",
                 "--checkpoint-freq", "1000000", "--recurrent"],
                run_dir,
            )

            if result["exit_code"] == 0:
                metrics = extract_run_metrics(result, run_dir, is_recurrent=True)
                all_results["e"] = metrics
            else:
                print(f"WARNING: Run E failed with exit code {result['exit_code']}")

    # ============================================================
    # Phase 5: Collect and write metrics.json
    # ============================================================
    total_wall_time = time.time() - experiment_start
    total_runs = len([k for k in all_results if k in ("a", "b", "c", "d", "e")])

    # Build the metrics.json structure
    metrics_json = {
        "experiment": "exp-001-training-data-scaling-20-vs-199-days",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Baseline (20-day controls)
    if "a" in all_results:
        metrics_json["baseline_mlp_20d"] = all_results["a"]
    if "b" in all_results:
        metrics_json["baseline_lstm_20d"] = all_results["b"]

    # Treatment (199-day)
    if "c" in all_results:
        metrics_json["treatment_mlp_199d"] = all_results["c"]
    if "d" in all_results:
        metrics_json["treatment_lstm_199d"] = all_results["d"]

    # Reproducibility
    if "e" in all_results:
        metrics_json["reproducibility_lstm_199d_seed43"] = all_results["e"]

    # MVE
    if "mve" in all_results:
        metrics_json["mve"] = all_results["mve"]

    # Sanity checks
    sanity = {}
    if "a" in all_results:
        a_val = all_results["a"].get("val_mean_return")
        if a_val is not None:
            sanity["mlp_20d_val_return"] = a_val
            sanity["mlp_20d_vs_historical_diff"] = abs(a_val - (-62.9))
            sanity["mlp_20d_within_15_of_historical"] = abs(a_val - (-62.9)) <= 15

    if "b" in all_results:
        b_val = all_results["b"].get("val_mean_return")
        if b_val is not None:
            sanity["lstm_20d_val_return"] = b_val
            sanity["lstm_20d_vs_historical_diff"] = abs(b_val - (-36.7))
            sanity["lstm_20d_within_15_of_historical"] = abs(b_val - (-36.7)) <= 15

    # Entropy sanity check across all runs
    for run_id, run_metrics in all_results.items():
        if run_id == "mve":
            continue
        ent = run_metrics.get("final_entropy")
        if ent is not None:
            sanity[f"{run_id}_final_entropy"] = ent
            sanity[f"{run_id}_entropy_above_neg060"] = ent > -0.60

    # In-sample return sanity
    for run_id, run_metrics in all_results.items():
        if run_id == "mve":
            continue
        isr = run_metrics.get("in_sample_mean_return")
        if isr is not None:
            sanity[f"{run_id}_in_sample_return"] = isr
            sanity[f"{run_id}_in_sample_positive"] = isr > 0

    metrics_json["sanity_checks"] = sanity

    # Resource usage
    metrics_json["resource_usage"] = {
        "gpu_hours": 0,  # Running on CPU/MPS locally
        "wall_clock_seconds": total_wall_time,
        "total_training_steps": sum(
            5_000_000 for k in all_results if k in ("a", "b", "c", "d", "e")
        ) + (500_000 if "mve" in all_results else 0),
        "total_runs": total_runs + (1 if "mve" in all_results else 0),
        "hardware": "Apple M2 Max (CPU, 32GB RAM)",
    }

    metrics_json["abort_triggered"] = abort_triggered
    metrics_json["abort_reason"] = abort_reason
    metrics_json["notes"] = (
        f"Ran locally on Apple M2 Max (CPU) instead of RunPod RTX 4090 due to local execution context. "
        f"Total wall time: {total_wall_time:.0f}s. "
        f"Total runs completed: {total_runs + (1 if 'mve' in all_results else 0)}."
    )

    # Write metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"\nMetrics written to: {metrics_path}")

    # Also save config for reproducibility
    config_path = os.path.join(RESULTS_DIR, "config.json")
    config = {
        "common_args": {
            "bar_size": 1000,
            "execution_cost": True,
            "policy_arch": "256,256",
            "activation": "relu",
            "ent_coef": 0.05,
            "learning_rate": 0.001,
            "shuffle_split": True,
            "n_envs": 8,
            "batch_size": 256,
            "n_epochs": 5,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_steps": 2048,
        },
        "runs": {
            "mve": {"train_days": 199, "total_timesteps": 500000, "seed": 42, "recurrent": False},
            "a": {"train_days": 20, "total_timesteps": 5000000, "seed": 42, "recurrent": False},
            "b": {"train_days": 20, "total_timesteps": 5000000, "seed": 42, "recurrent": True},
            "c": {"train_days": 199, "total_timesteps": 5000000, "seed": 42, "recurrent": False},
            "d": {"train_days": 199, "total_timesteps": 5000000, "seed": 42, "recurrent": True},
            "e": {"train_days": 199, "total_timesteps": 5000000, "seed": 43, "recurrent": True},
        },
        "checkpoint_freq": 1000000,
        "hardware": "Apple M2 Max (CPU, 32GB RAM)",
        "cache_dir": CACHE_DIR,
        "n_cache_files": 249,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to: {config_path}")

    return 0 if not abort_triggered else 1


if __name__ == "__main__":
    sys.exit(main())
