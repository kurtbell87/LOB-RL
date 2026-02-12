#!/usr/bin/env python3
"""exp-008: Bar-Size Sweep — Does Any (B, R) Configuration Produce Positive BSS?

Sweeps B ∈ {200, 500, 1000, 2000} × R ∈ {1x, 2x, 3x} median bar range.
LR only. Chronological temporal split. Block bootstrap significance test.

Usage:
  cd build-release
  PYTHONPATH=.:../python uv run python ../scripts/run_exp008_bar_size_sweep.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build-release"
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp-008-bar-size-sweep"
DATA_DIR = PROJECT_ROOT / "data" / "mes"
ROLL_CALENDAR = DATA_DIR / "roll_calendar.json"

TICK_SIZE = 0.25
BAR_SIZES = [200, 500, 1000, 2000]
R_MULTIPLIERS = [1.0, 2.0, 3.0]
LOOKBACK = 10
SEED = 42
N_BOOT = 1000
BLOCK_SIZE = 50
PRECOMPUTE_WORKERS = 8

# Gate thresholds (wider per spec)
YBAR_LO = 0.20
YBAR_HI = 0.46
TIMEOUT_MAX = 0.10

# Bonferroni correction for 24 tests
N_CELLS = 24
BONFERRONI_ALPHA = 0.05 / N_CELLS


def run_precompute(bar_size, a, b, t_max, output_dir, workers=PRECOMPUTE_WORKERS):
    """Run precompute_barrier_cache.py for a given configuration."""
    cmd = [
        "uv", "run", "python", str(SCRIPT_DIR / "precompute_barrier_cache.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(output_dir),
        "--roll-calendar", str(ROLL_CALENDAR),
        "--bar-size", str(bar_size),
        "--a", str(a),
        "--b", str(b),
        "--t-max", str(t_max),
        "--lookback", str(LOOKBACK),
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BUILD_DIR}:{PROJECT_ROOT / 'python'}"
    print(f"  Running precompute: B={bar_size}, a={a}, b={b}, t_max={t_max}")
    print(f"  Output: {output_dir}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BUILD_DIR), env=env,
                            capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  PRECOMPUTE FAILED (exit {result.returncode}):")
        print(result.stderr[-2000:] if result.stderr else "no stderr")
        return False, elapsed
    # Count output files
    n_files = len(list(Path(output_dir).glob("*.npz")))
    print(f"  Precompute done: {n_files} files, {elapsed:.1f}s")
    return True, elapsed


def measure_median_bar_range(cache_dir):
    """Compute median (bar_high - bar_low) / TICK_SIZE across all sessions in cache."""
    npz_files = sorted(Path(cache_dir).glob("*.npz"))
    all_ranges = []
    for f in npz_files:
        data = np.load(f)
        bar_range = (data["bar_high"] - data["bar_low"]) / TICK_SIZE
        all_ranges.append(bar_range)
    if not all_ranges:
        return None
    combined = np.concatenate(all_ranges)
    return float(np.median(combined))


def load_and_analyze(cache_dir, bar_size, R_ticks, a, b, t_max):
    """Load data, run null calibration and LR signal detection for one (B, R) config."""
    from lob_rl.barrier.first_passage_analysis import (
        brier_score,
        constant_brier,
        brier_skill_score,
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
        "bar_size": bar_size,
        "R_ticks": R_ticks,
        "a": a,
        "b": b,
        "t_max": t_max,
        "n_samples": n_samples,
        "n_sessions": n_sessions,
    }

    # Null calibration (using wider gate per spec)
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

    # Apply wider gate from spec: ȳ ∈ [0.20, 0.46], timeout < 10%
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
        # Still try to run signal detection even if gate fails
        # (spec says to log but skip is optional; we'll note it)

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

    import warnings

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0_experiment = time.time()

    print("=" * 60)
    print("exp-008: Bar-Size Sweep")
    print("=" * 60)

    # =========================================================================
    # Phase 0: Measure median bar ranges
    # =========================================================================
    print("\n--- Phase 0: Measure median bar ranges ---")

    median_bar_ranges = {}

    # B=500: use existing cache
    existing_cache = CACHE_ROOT / "barrier"
    median_500 = measure_median_bar_range(existing_cache)
    median_bar_ranges[500] = median_500
    print(f"  B=500: median bar range = {median_500:.1f} ticks (from existing cache)")

    # B=200, 1000, 2000: run mini-precomputes on 10 sessions
    for B in [200, 1000, 2000]:
        tmp_dir = CACHE_ROOT / f"tmp-b{B}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Check if already has files (re-run case)
        existing_files = list(tmp_dir.glob("*.npz"))
        if len(existing_files) >= 10:
            median = measure_median_bar_range(tmp_dir)
            median_bar_ranges[B] = median
            print(f"  B={B}: median bar range = {median:.1f} ticks (from existing temp cache)")
            continue

        # Precompute with default barriers (just need bar_high/bar_low, barrier params don't matter)
        # Use a=20, b=10, t_max=40 as defaults
        cmd = [
            "uv", "run", "python", str(SCRIPT_DIR / "precompute_barrier_cache.py"),
            "--data-dir", str(DATA_DIR),
            "--output-dir", str(tmp_dir),
            "--roll-calendar", str(ROLL_CALENDAR),
            "--bar-size", str(B),
            "--a", "20", "--b", "10", "--t-max", "40",
            "--lookback", str(LOOKBACK),
            "--workers", str(PRECOMPUTE_WORKERS),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{BUILD_DIR}:{PROJECT_ROOT / 'python'}"

        print(f"  B={B}: running temp precompute (full dataset for bar range measurement)...")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(BUILD_DIR), env=env,
                                capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  ERROR: precompute failed for B={B}")
            print(result.stderr[-1000:] if result.stderr else "no stderr")
            median_bar_ranges[B] = None
            continue

        median = measure_median_bar_range(tmp_dir)
        median_bar_ranges[B] = median
        print(f"  B={B}: median bar range = {median:.1f} ticks ({elapsed:.1f}s)")

    # Check for degenerate bar sizes
    for B, median in median_bar_ranges.items():
        if median is not None and median < 1.0:
            print(f"  ABORT: B={B} has median bar range < 1 tick. Skipping.")
            median_bar_ranges[B] = None

    # Compute R grid
    r_grid = {}
    for B in BAR_SIZES:
        median = median_bar_ranges.get(B)
        if median is None:
            r_grid[B] = None
            continue
        r_values = []
        for mult in R_MULTIPLIERS:
            R = max(2, round(mult * median))  # minimum 2 ticks
            r_values.append(R)
        r_grid[B] = r_values

    print("\n  R Grid:")
    for B in BAR_SIZES:
        rs = r_grid.get(B)
        if rs:
            print(f"    B={B}: median={median_bar_ranges[B]:.1f}, R = {rs}")
        else:
            print(f"    B={B}: SKIPPED")

    # Build list of (B, R) configurations
    configs = []
    for B in BAR_SIZES:
        rs = r_grid.get(B)
        if rs is None:
            continue
        for R in rs:
            a = 2 * R  # 2:1 reward:risk
            t_max = min(100, max(40, round(4 * R)))
            cache_dir = CACHE_ROOT / f"barrier-b{B}-r{R}"
            configs.append({
                "bar_size": B,
                "R_ticks": R,
                "a": a,
                "b": R,
                "t_max": t_max,
                "cache_dir": cache_dir,
            })

    print(f"\n  Total configurations: {len(configs)}")
    for c in configs:
        print(f"    B={c['bar_size']}, R={c['R_ticks']}, a={c['a']}, b={c['b']}, "
              f"t_max={c['t_max']}")

    # =========================================================================
    # Phase 1: Precompute barrier caches
    # =========================================================================
    print("\n--- Phase 1: Precompute barrier caches ---")

    # Check for B=500 special case: can we reuse existing cache?
    for c in configs:
        B, R = c["bar_size"], c["R_ticks"]
        cache_dir = c["cache_dir"]

        # Check if this config matches existing cache/barrier/ (B=500, a=20, b=10, t_max=40)
        if B == 500 and c["a"] == 20 and c["b"] == 10 and c["t_max"] == 40:
            # Reuse existing cache
            print(f"\n  B={B}, R={R}: Reusing existing cache/barrier/ (a=20, b=10, t_max=40)")
            c["cache_dir"] = existing_cache
            continue

        # Check if cache already populated
        n_existing = len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0
        if n_existing >= 240:
            print(f"\n  B={B}, R={R}: Cache already has {n_existing} files, skipping precompute")
            continue

        # Run precompute
        print(f"\n  B={B}, R={R}: Precomputing...")
        success, elapsed = run_precompute(
            B, c["a"], c["b"], c["t_max"], cache_dir,
            workers=PRECOMPUTE_WORKERS
        )
        if not success:
            print(f"  WARNING: Precompute failed for B={B}, R={R}")

        # Verify output
        n_files = len(list(cache_dir.glob("*.npz"))) if cache_dir.exists() else 0
        if n_files < 200:
            print(f"  ABORT for B={B}, R={R}: only {n_files} sessions (need >= 200)")
            c["skip"] = True

        # Check wall clock
        elapsed_total = time.time() - t0_experiment
        if elapsed_total > 5 * 3600:
            print(f"  ABORT: Wall clock exceeded 5 hours ({elapsed_total/3600:.1f}h)")
            break

    # =========================================================================
    # Phase 2: MVE on first config
    # =========================================================================
    print("\n--- Phase 2: Minimum Viable Experiment ---")

    # Find first non-skipped config
    mve_config = None
    for c in configs:
        if not c.get("skip"):
            mve_config = c
            break

    if mve_config is None:
        print("  ABORT: No valid configurations to run MVE on")
        # Write abort metrics
        metrics = {
            "experiment": "exp-008-bar-size-sweep",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": "All configurations failed precompute",
            "notes": "No valid (B, R) configurations survived Phase 1.",
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    print(f"  MVE config: B={mve_config['bar_size']}, R={mve_config['R_ticks']}")
    try:
        mve_result = load_and_analyze(
            mve_config["cache_dir"],
            mve_config["bar_size"],
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
                    "experiment": "exp-008-bar-size-sweep",
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
            "experiment": "exp-008-bar-size-sweep",
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
            print(f"\n  [{i+1}/{len(configs)}] B={c['bar_size']}, R={c['R_ticks']}: SKIPPED")
            continue

        B, R = c["bar_size"], c["R_ticks"]
        print(f"\n  [{i+1}/{len(configs)}] B={B}, R={R} (a={c['a']}, b={c['b']}, t_max={c['t_max']})")

        # If this is the MVE config, reuse results
        if (c["bar_size"] == mve_config["bar_size"] and
            c["R_ticks"] == mve_config["R_ticks"]):
            result = mve_result
        else:
            try:
                t0 = time.time()
                result = load_and_analyze(
                    c["cache_dir"], B, R, c["a"], c["b"], c["t_max"]
                )
                elapsed = time.time() - t0
                print(f"    Analysis done in {elapsed:.1f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                result = {
                    "bar_size": B, "R_ticks": R, "a": c["a"], "b": c["b"],
                    "t_max": c["t_max"], "error": str(e),
                }

        if result.get("gate_passed"):
            gates_passed += 1
        else:
            gates_failed += 1

        print(f"    N={result.get('n_samples', '?')}, "
              f"sess={result.get('n_sessions', '?')}, "
              f"gate={'PASS' if result.get('gate_passed') else 'FAIL'}")
        if "bss_logistic_long" in result:
            print(f"    BSS_long={result['bss_logistic_long']:.6f}, "
                  f"BSS_short={result['bss_logistic_short']:.6f}")

        all_results.append(result)

        # Check wall clock
        elapsed_total = time.time() - t0_experiment
        if elapsed_total > 5 * 3600:
            print(f"\n  ABORT: Wall clock exceeded 5 hours ({elapsed_total/3600:.1f}h)")
            break

    # Check if all gates failed
    if gates_passed == 0 and gates_failed > 0:
        print("\n  WARNING: All null calibration gates failed.")
        # Don't abort — still report the results per spec

    # =========================================================================
    # Phase 4: Aggregate and save
    # =========================================================================
    print("\n--- Phase 4: Aggregate and save ---")

    # Find best cell
    best_bss = -np.inf
    best_cell = None
    all_cells = []

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
                "bar_size": r["bar_size"],
                "R_ticks": r["R_ticks"],
                "label": label,
                "bss": bss,
                "p_value": delta["p_value"],
                "bonferroni_p": min(1.0, delta["p_value"] * N_CELLS),
            }
            all_cells.append(cell)
            if bss > best_bss:
                best_bss = bss
                best_cell = cell

    # Success criteria
    C1 = False
    C2 = False
    C3 = gates_passed >= 10  # At least 10 of 12 pass

    if best_cell:
        C1 = best_cell["bss"] > 0 and best_cell["bonferroni_p"] < 0.05
        C2 = best_cell["bss"] >= 0.005

    # exp-006 consistency check
    exp006_consistency = None
    for r in all_results:
        if "error" in r:
            continue
        # Find the B=500 config closest to R=10
        if r["bar_size"] == 500 and r["R_ticks"] == 10:
            exp006_consistency = {
                "bss_long": r.get("bss_logistic_long"),
                "bss_short": r.get("bss_logistic_short"),
                # exp-006 reference: LR/long BSS = -0.0007, LR/short BSS = -0.0003
                "delta_vs_exp006_long": abs(r.get("bss_logistic_long", 0) - (-0.0007)),
                "delta_vs_exp006_short": abs(r.get("bss_logistic_short", 0) - (-0.0003)),
            }
            break

    # Convert calibration arrays and delta dicts for JSON serialization
    def make_serializable(obj):
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

    # Clean up results for JSON
    clean_results = []
    for r in all_results:
        clean = make_serializable(r)
        # Remove gate_failure_reason if gate passed
        clean_results.append(clean)

    elapsed_total = time.time() - t0_experiment

    metrics = {
        "experiment": "exp-008-bar-size-sweep",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "median_bar_ranges": {str(k): v for k, v in median_bar_ranges.items()},
        "r_grid": {str(k): v for k, v in r_grid.items() if v is not None},
        "configs": clean_results,
        "best_cell": make_serializable(best_cell),
        "success_criteria": {
            "C1": C1,
            "C2": C2,
            "C3": C3,
        },
        "exp006_consistency": make_serializable(exp006_consistency),
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": elapsed_total,
            "total_training_steps": 0,
            "total_runs": len([r for r in all_results if "error" not in r]),
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Phase 0-4 completed. {len(all_results)} configs analyzed. "
            f"{gates_passed} gates passed, {gates_failed} gates failed. "
            f"LR only (GBT excluded per spec). "
            f"Bootstrap: n_boot={N_BOOT}, block_size={BLOCK_SIZE}. "
            f"Bonferroni correction for {N_CELLS} tests."
        ),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Write config
    config = {
        "bar_sizes": BAR_SIZES,
        "r_multipliers": R_MULTIPLIERS,
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
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete. Wall time: {elapsed_total/60:.1f} min")
    print(f"Results: {RESULTS_DIR / 'metrics.json'}")
    print(f"Config:  {RESULTS_DIR / 'config.json'}")
    if best_cell:
        print(f"\nBest cell: B={best_cell['bar_size']}, R={best_cell['R_ticks']}, "
              f"label={best_cell['label']}")
        print(f"  BSS = {best_cell['bss']:.6f}")
        print(f"  raw p = {best_cell['p_value']:.4f}")
        print(f"  Bonferroni p = {best_cell['bonferroni_p']:.4f}")
    print(f"\nC1 (signal detected): {C1}")
    print(f"C2 (BSS >= 0.005):    {C2}")
    print(f"C3 (>= 10 gates ok):  {C3}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
