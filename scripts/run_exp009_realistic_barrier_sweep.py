#!/usr/bin/env python3
"""exp-009: Realistic Barrier Sweep — B=2000 with Fixed Tick-Based Barriers R ∈ {10, 20, 30, 40}.

LR only. Chronological temporal split. Block bootstrap significance test.

Usage:
  cd build-release
  PYTHONPATH=.:../python uv run python ../scripts/run_exp009_realistic_barrier_sweep.py
"""

import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build-release"
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp-009-realistic-barrier-sweep"
DATA_DIR = PROJECT_ROOT / "data" / "mes"
ROLL_CALENDAR = DATA_DIR / "roll_calendar.json"

TICK_SIZE = 0.25
BAR_SIZE = 2000
LOOKBACK = 10
SEED = 42
N_BOOT = 1000
BLOCK_SIZE = 50
PRECOMPUTE_WORKERS = 8

# Gate thresholds (same as exp-008)
YBAR_LO = 0.20
YBAR_HI = 0.46
TIMEOUT_MAX = 0.10

# R configurations: R -> (a, b, t_max)
R_CONFIGS = {
    10: {"a": 20, "b": 10, "t_max": 40},
    20: {"a": 40, "b": 20, "t_max": 80},
    30: {"a": 60, "b": 30, "t_max": 100},
    40: {"a": 80, "b": 40, "t_max": 100},
}

# Bonferroni correction for 8 tests (4 R × 2 labels)
N_CELLS = 8
BONFERRONI_ALPHA = 0.05 / N_CELLS

# exp-008 B=2000 cross-reference
EXP008_B2000 = {
    "R12": {"bss_long": -0.0032, "bss_short": -0.0028},
    "R24": {"bss_long": -0.0059, "bss_short": -0.0019},
    "R36": {"bss_long": -0.0114, "bss_short": 0.0021},
}


def run_precompute(R, a, b, t_max, output_dir, workers=PRECOMPUTE_WORKERS):
    """Run precompute_barrier_cache.py for a given R configuration."""
    cmd = [
        "uv", "run", "python", str(SCRIPT_DIR / "precompute_barrier_cache.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(output_dir),
        "--roll-calendar", str(ROLL_CALENDAR),
        "--bar-size", str(BAR_SIZE),
        "--a", str(a),
        "--b", str(b),
        "--t-max", str(t_max),
        "--lookback", str(LOOKBACK),
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BUILD_DIR}:{PROJECT_ROOT / 'python'}"
    print(f"  Running precompute: B={BAR_SIZE}, R={R}, a={a}, b={b}, t_max={t_max}")
    print(f"  Output: {output_dir}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BUILD_DIR), env=env,
                            capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  PRECOMPUTE FAILED (exit {result.returncode}):")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        return False, elapsed
    n_files = len(list(Path(output_dir).glob("*.npz")))
    print(f"  Precompute done: {n_files} files, {elapsed:.1f}s")
    return True, elapsed


def load_and_analyze(cache_dir, R_ticks, a, b, t_max):
    """Load data, run null calibration and LR signal detection for one R config."""
    from lob_rl.barrier.first_passage_analysis import (
        brier_score,
        brier_skill_score,
        constant_brier,
        fit_logistic,
        load_binary_labels,
        null_calibration_report,
        paired_bootstrap_brier,
        temporal_split,
    )

    # Load data
    data = load_binary_labels(str(cache_dir), lookback=LOOKBACK)
    X = data["X"]
    Y_long = data["Y_long"]
    Y_short = data["Y_short"]
    timeout_long = data["timeout_long"]
    timeout_short = data["timeout_short"]
    tau_long = data["tau_long"]
    tau_short = data["tau_short"]
    boundaries = data["session_boundaries"]
    n_sessions = len(boundaries) - 1
    n_samples = len(Y_long)

    config_result = {
        "R_ticks": R_ticks,
        "a": a,
        "b": b,
        "t_max": t_max,
        "n_samples": n_samples,
        "n_sessions": n_sessions,
    }

    # Null calibration
    null_report = null_calibration_report(
        Y_long, Y_short, tau_long, tau_short,
        timeout_long, timeout_short, boundaries
    )

    config_result["ybar_long"] = null_report["ybar_long"]
    config_result["ybar_short"] = null_report["ybar_short"]
    config_result["timeout_rate_long"] = null_report["timeout_rate_long"]
    config_result["timeout_rate_short"] = null_report["timeout_rate_short"]
    config_result["mean_tau_long"] = null_report["mean_tau_long"]
    config_result["mean_tau_short"] = null_report["mean_tau_short"]

    # Apply gate: ȳ ∈ [0.20, 0.46], timeout < 10%
    ybar_ok = (YBAR_LO <= null_report["ybar_long"] <= YBAR_HI and
               YBAR_LO <= null_report["ybar_short"] <= YBAR_HI)
    timeout_ok = (null_report["timeout_rate_long"] < TIMEOUT_MAX and
                  null_report["timeout_rate_short"] < TIMEOUT_MAX)
    gate_passed = ybar_ok and timeout_ok
    config_result["gate_passed"] = gate_passed

    if not gate_passed:
        config_result["gate_failure_reason"] = (
            f"ybar_ok={ybar_ok}, timeout_ok={timeout_ok}"
        )

    # Signal detection: LR only (spec excludes GBT)
    train_sess, val_sess, test_sess = temporal_split(n_sessions)

    def sess_to_rows(sess_idx):
        rows = []
        for s in sess_idx:
            rows.append(np.arange(boundaries[s], boundaries[s + 1]))
        return np.concatenate(rows) if len(rows) > 0 else np.array([], dtype=np.int64)

    train_rows = sess_to_rows(train_sess)
    val_rows = sess_to_rows(val_sess)

    X_train = X[train_rows]
    X_val = X[val_rows]

    for label_name, Y in [("long", Y_long), ("short", Y_short)]:
        y_train = Y[train_rows].astype(int)
        y_val = Y[val_rows].astype(int)

        # Constant baseline
        ybar_train = float(y_train.mean())
        pred_const = np.full(len(y_val), ybar_train)
        bs_const = brier_score(y_val, pred_const)

        # LR
        converged = True
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model_lr = fit_logistic(X_train, y_train, max_iter=1000)
            if any("ConvergenceWarning" in str(warning.category) for warning in w):
                converged = False

        pred_lr = model_lr.predict_proba(X_val)[:, 1]
        bs_lr = brier_score(y_val, pred_lr)
        bss = brier_skill_score(y_val, pred_lr)

        # Bootstrap
        delta_result = paired_bootstrap_brier(
            y_val, pred_lr, pred_const,
            n_boot=N_BOOT, block_size=BLOCK_SIZE, seed=SEED
        )

        config_result[f"brier_constant_{label_name}"] = bs_const
        config_result[f"brier_logistic_{label_name}"] = bs_lr
        config_result[f"bss_logistic_{label_name}"] = bss
        config_result[f"delta_logistic_{label_name}"] = delta_result
        config_result[f"lr_converged_{label_name}"] = converged

    return config_result


def make_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_serializable(v) for v in obj]
    return obj


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0_experiment = time.time()

    print("=" * 60)
    print("exp-009: Realistic Barrier Sweep")
    print(f"  B={BAR_SIZE}, R ∈ {list(R_CONFIGS.keys())}")
    print("=" * 60)

    # =========================================================================
    # Phase 1: Precompute barrier caches
    # =========================================================================
    print("\n--- Phase 1: Precompute barrier caches ---")

    configs = []
    for R, params in R_CONFIGS.items():
        cache_dir = CACHE_ROOT / f"barrier-b{BAR_SIZE}-r{R}"
        configs.append({
            "R_ticks": R,
            "a": params["a"],
            "b": params["b"],
            "t_max": params["t_max"],
            "cache_dir": cache_dir,
        })

    for c in configs:
        R = c["R_ticks"]
        cache_dir = c["cache_dir"]

        # Check if cache already populated
        n_existing = len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0
        if n_existing >= 240:
            print(f"\n  R={R}: Cache already has {n_existing} files, skipping precompute")
            continue

        # Run precompute
        print(f"\n  R={R}: Precomputing...")
        success, elapsed = run_precompute(
            R, c["a"], c["b"], c["t_max"], cache_dir,
            workers=PRECOMPUTE_WORKERS
        )
        if not success:
            print(f"  WARNING: Precompute failed for R={R}")

        # Verify output
        n_files = len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0
        if n_files < 200:
            print(f"  ABORT for R={R}: only {n_files} sessions (need >= 200)")
            c["skip"] = True

        # Check wall clock (90 min abort)
        elapsed_total = time.time() - t0_experiment
        if elapsed_total > 90 * 60:
            print(f"  ABORT: Wall clock exceeded 90 min ({elapsed_total/60:.1f} min)")
            break

    # =========================================================================
    # Phase 2: MVE on R=10
    # =========================================================================
    print("\n--- Phase 2: Minimum Viable Experiment (R=10) ---")

    mve_config = None
    for c in configs:
        if c["R_ticks"] == 10 and not c.get("skip"):
            mve_config = c
            break

    if mve_config is None:
        # Fall back to first non-skipped config
        for c in configs:
            if not c.get("skip"):
                mve_config = c
                break

    if mve_config is None:
        print("  ABORT: No valid configurations to run MVE on")
        metrics = {
            "experiment": "exp-009-realistic-barrier-sweep",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": "All configurations failed precompute",
            "notes": "No valid R configurations survived Phase 1.",
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    print(f"  MVE config: R={mve_config['R_ticks']}")
    try:
        mve_result = load_and_analyze(
            mve_config["cache_dir"],
            mve_config["R_ticks"],
            mve_config["a"],
            mve_config["b"],
            mve_config["t_max"],
        )
        print(f"  MVE N_samples: {mve_result['n_samples']}")
        print(f"  MVE N_sessions: {mve_result['n_sessions']}")
        print(f"  MVE ȳ_long: {mve_result['ybar_long']:.4f}")
        print(f"  MVE ȳ_short: {mve_result['ybar_short']:.4f}")
        print(f"  MVE gate_passed: {mve_result['gate_passed']}")
        print(f"  MVE BSS_long: {mve_result['bss_logistic_long']:.6f}")
        print(f"  MVE BSS_short: {mve_result['bss_logistic_short']:.6f}")

        # Check for NaN/Inf
        for key in ["bss_logistic_long", "bss_logistic_short"]:
            v = mve_result[key]
            if not np.isfinite(v):
                print(f"  ABORT: {key} is {v}")
                metrics = {
                    "experiment": "exp-009-realistic-barrier-sweep",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "abort_triggered": True,
                    "abort_reason": f"MVE produced {key}={v}",
                }
                with open(RESULTS_DIR / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                return

        print("  MVE PASSED — pipeline validated.")

    except Exception as e:
        print(f"  MVE FAILED: {e}")
        import traceback
        traceback.print_exc()
        metrics = {
            "experiment": "exp-009-realistic-barrier-sweep",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": f"MVE failed: {e}",
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # =========================================================================
    # Phase 3: Full signal detection for all configs
    # =========================================================================
    print("\n--- Phase 3: Null calibration + signal detection ---")

    all_results = []
    gates_passed = 0
    gates_failed = 0

    for i, c in enumerate(configs):
        if c.get("skip"):
            print(f"\n  [{i+1}/{len(configs)}] R={c['R_ticks']}: SKIPPED")
            continue

        R = c["R_ticks"]
        print(f"\n  [{i+1}/{len(configs)}] R={R} (a={c['a']}, b={c['b']}, t_max={c['t_max']})")

        # If this is the MVE config, reuse results
        if c["R_ticks"] == mve_config["R_ticks"]:
            result = mve_result
        else:
            try:
                t0 = time.time()
                result = load_and_analyze(
                    c["cache_dir"], R, c["a"], c["b"], c["t_max"]
                )
                elapsed = time.time() - t0
                print(f"    Analysis done in {elapsed:.1f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                result = {
                    "R_ticks": R, "a": c["a"], "b": c["b"],
                    "t_max": c["t_max"], "error": str(e),
                }

        if result.get("gate_passed"):
            gates_passed += 1
        elif "error" not in result:
            gates_failed += 1

        print(f"    N={result.get('n_samples', '?')}, "
              f"sess={result.get('n_sessions', '?')}, "
              f"gate={'PASS' if result.get('gate_passed') else 'FAIL'}")
        if "bss_logistic_long" in result:
            print(f"    BSS_long={result['bss_logistic_long']:.6f}, "
                  f"BSS_short={result['bss_logistic_short']:.6f}")

        all_results.append(result)

        # Check wall clock (90 min abort)
        elapsed_total = time.time() - t0_experiment
        if elapsed_total > 90 * 60:
            print(f"\n  ABORT: Wall clock exceeded 90 min ({elapsed_total/60:.1f} min)")
            break

    # =========================================================================
    # Phase 4: Aggregate and save
    # =========================================================================
    print("\n--- Phase 4: Aggregate and save ---")

    # Find best cell
    best_bss = -np.inf
    best_cell = None

    for r in all_results:
        if "error" in r:
            continue
        for label in ["long", "short"]:
            bss_key = f"bss_logistic_{label}"
            delta_key = f"delta_logistic_{label}"
            if bss_key not in r:
                continue
            bss = r[bss_key]
            delta = r[delta_key]
            cell = {
                "R_ticks": r["R_ticks"],
                "label": label,
                "bss": bss,
                "p_value": delta["p_value"],
                "bonferroni_p": min(1.0, delta["p_value"] * N_CELLS),
            }
            if bss > best_bss:
                best_bss = bss
                best_cell = cell

    # Success criteria
    # C1: At least one cell has BSS > 0 AND Bonferroni-corrected p < 0.05
    C1 = False
    if best_cell:
        C1 = best_cell["bss"] > 0 and best_cell["bonferroni_p"] < 0.05

    # C2: Best BSS >= 0.005
    C2 = False
    if best_cell:
        C2 = best_cell["bss"] >= 0.005

    # C3: At least 3 of 4 R configurations pass null calibration gate
    C3 = gates_passed >= 3

    # Clean up results for JSON
    clean_results = make_serializable(all_results)

    elapsed_total = time.time() - t0_experiment

    metrics = {
        "experiment": "exp-009-realistic-barrier-sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bar_size": BAR_SIZE,
        "configs": clean_results,
        "exp008_b2000_cross_reference": EXP008_B2000,
        "best_cell": make_serializable(best_cell),
        "success_criteria": {
            "C1": C1,
            "C2": C2,
            "C3": C3,
        },
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": elapsed_total,
            "total_training_steps": 0,
            "total_runs": len([r for r in all_results if "error" not in r]),
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Phase 1-4 completed. {len(all_results)} configs analyzed. "
            f"{gates_passed} gates passed, {gates_failed} gates failed. "
            f"LR only (GBT excluded per spec). "
            f"Bootstrap: n_boot={N_BOOT}, block_size={BLOCK_SIZE}. "
            f"Bonferroni correction for {N_CELLS} tests (raw p < {BONFERRONI_ALPHA:.5f})."
        ),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write config
    config = {
        "bar_size": BAR_SIZE,
        "r_configs": {str(k): v for k, v in R_CONFIGS.items()},
        "lookback": LOOKBACK,
        "seed": SEED,
        "n_boot": N_BOOT,
        "block_size": BLOCK_SIZE,
        "tick_size": TICK_SIZE,
        "model": "LogisticRegression(C=1.0, solver=lbfgs, max_iter=1000)",
        "split": "temporal 60/20/20",
        "gate_thresholds": {
            "ybar_range": [YBAR_LO, YBAR_HI],
            "timeout_max": TIMEOUT_MAX,
        },
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "n_cells": N_CELLS,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete. Wall time: {elapsed_total/60:.1f} min")
    print(f"Results: {RESULTS_DIR / 'metrics.json'}")
    print(f"Config:  {RESULTS_DIR / 'config.json'}")
    if best_cell:
        print(f"\nBest cell: R={best_cell['R_ticks']}, "
              f"label={best_cell['label']}")
        print(f"  BSS = {best_cell['bss']:.6f}")
        print(f"  raw p = {best_cell['p_value']:.4f}")
        print(f"  Bonferroni p = {best_cell['bonferroni_p']:.4f}")
    print(f"\nC1 (signal detected): {C1}")
    print(f"C2 (BSS >= 0.005):    {C2}")
    print(f"C3 (>= 3/4 gates):   {C3}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
