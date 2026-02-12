#!/usr/bin/env python3
"""exp-010: Permutation Test — Is T6's +5pp Accuracy Real Signal or Artifact?

Shuffles barrier labels 100 times and re-runs RF classification + LR BSS.
If real accuracy > 95th percentile of permuted distribution → signal is real.

Usage:
  cd build-release
  PYTHONPATH=.:../python uv run python ../scripts/run_exp010_permutation_test.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_DIR = PROJECT_ROOT / "build-release"
CACHE_DIR = PROJECT_ROOT / "cache" / "barrier"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp-010-permutation-test"

LOOKBACK = 10
SEED = 42
N_PERMUTATIONS = 100
N_BOOT = 1000
BLOCK_SIZE = 50


def load_data():
    """Load features and binary labels from the B=500 barrier cache."""
    from lob_rl.barrier.first_passage_analysis import load_binary_labels

    data = load_binary_labels(str(CACHE_DIR), lookback=LOOKBACK)
    return data


def make_3class_labels(Y_long, Y_short):
    """Convert binary Y_long, Y_short to 3-class bidirectional labels.

    0 = long_profit (Y_long=True)
    1 = short_profit (Y_short=True and NOT Y_long)
    2 = flat (neither)

    Note: if both Y_long and Y_short are True for a sample, long takes priority.
    """
    y = np.full(len(Y_long), 2, dtype=np.int64)  # default: flat
    y[Y_short] = 1  # short_profit
    y[Y_long] = 0   # long_profit (overwrites short if both true)
    return y


def run_rf_accuracy(X_train, y_train, X_test, y_test, seed=42):
    """Train RF and return test accuracy."""
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    return float(np.mean(preds == y_test))


def run_lr_bss(X_train, y_train, X_val, y_val, seed=42):
    """Train LR on binary labels and return BSS on val set."""
    from lob_rl.barrier.first_passage_analysis import (
        brier_score,
        brier_skill_score,
        fit_logistic,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_logistic(X_train, y_train.astype(int), max_iter=1000)
    pred = model.predict_proba(X_val)[:, 1]
    bss = brier_skill_score(y_val.astype(int), pred)
    return bss


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0_experiment = time.time()

    print("=" * 60)
    print("exp-010: Permutation Test")
    print("=" * 60)

    # =========================================================================
    # Load data
    # =========================================================================
    print("\n--- Loading data ---")
    data = load_data()
    X = data["X"]
    Y_long = data["Y_long"]
    Y_short = data["Y_short"]
    boundaries = data["session_boundaries"]
    n_sessions = len(boundaries) - 1
    n_samples = len(Y_long)

    print(f"  N_samples: {n_samples}")
    print(f"  N_sessions: {n_sessions}")
    print(f"  Feature dim: {X.shape[1]}")

    # Replace NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    if nan_mask.any():
        n_bad = nan_mask.sum()
        print(f"  Replacing {n_bad} NaN/Inf values with 0")
        X = np.where(nan_mask, 0.0, X)

    # Build 3-class labels
    y = make_3class_labels(Y_long, Y_short)
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    majority_baseline = float(counts.max()) / total
    print(f"\n  Label distribution:")
    print(f"    long_profit:  {counts[0]:>6d} ({100*counts[0]/total:.1f}%)")
    print(f"    short_profit: {counts[1]:>6d} ({100*counts[1]/total:.1f}%)")
    print(f"    flat:         {counts[2]:>6d} ({100*counts[2]/total:.1f}%)")
    print(f"    Majority baseline: {majority_baseline:.4f}")

    # =========================================================================
    # Splits
    # =========================================================================
    rng = np.random.default_rng(SEED)

    # Shuffle split (80/20) for RF accuracy test
    n_train_shuf = int(n_samples * 0.8)
    shuf_idx = rng.permutation(n_samples)
    train_shuf = shuf_idx[:n_train_shuf]
    test_shuf = shuf_idx[n_train_shuf:]

    X_train_shuf = X[train_shuf]
    X_test_shuf = X[test_shuf]
    y_train_shuf = y[train_shuf]
    y_test_shuf = y[test_shuf]

    # Chronological split (60/20/20) for BSS test (same as exp-006)
    from lob_rl.barrier.first_passage_analysis import temporal_split
    train_sess, val_sess, test_sess = temporal_split(n_sessions)

    def sess_to_rows(sess_idx):
        rows = []
        for s in sess_idx:
            rows.append(np.arange(boundaries[s], boundaries[s + 1]))
        return np.concatenate(rows) if len(rows) > 0 else np.array([], dtype=np.int64)

    train_chrono = sess_to_rows(train_sess)
    val_chrono = sess_to_rows(val_sess)

    X_train_chrono = X[train_chrono]
    X_val_chrono = X[val_chrono]

    # Also build 3-class chronological split
    y_train_chrono = y[train_chrono]
    y_val_chrono = y[val_chrono]

    # =========================================================================
    # Real-label accuracy (RF, shuffle-split)
    # =========================================================================
    print("\n--- Real-label RF accuracy (shuffle-split) ---")
    t0 = time.time()
    real_accuracy_shuf = run_rf_accuracy(X_train_shuf, y_train_shuf,
                                          X_test_shuf, y_test_shuf, seed=SEED)
    print(f"  Real accuracy (shuffle): {real_accuracy_shuf:.4f} ({time.time()-t0:.1f}s)")

    # Real-label accuracy (RF, chronological)
    print("\n--- Real-label RF accuracy (chronological) ---")
    t0 = time.time()
    real_accuracy_chrono = run_rf_accuracy(X_train_chrono, y_train_chrono,
                                            X_val_chrono, y_val_chrono, seed=SEED)
    print(f"  Real accuracy (chrono):  {real_accuracy_chrono:.4f} ({time.time()-t0:.1f}s)")

    # =========================================================================
    # Real-label BSS (LR on Y_long, chronological)
    # =========================================================================
    print("\n--- Real-label BSS (LR on Y_long, chronological) ---")
    Y_long_train = Y_long[train_chrono]
    Y_long_val = Y_long[val_chrono]
    t0 = time.time()
    real_bss = run_lr_bss(X_train_chrono, Y_long_train, X_val_chrono, Y_long_val, seed=SEED)
    print(f"  Real BSS (Y_long): {real_bss:.6f} ({time.time()-t0:.1f}s)")

    # =========================================================================
    # Permutation test: RF accuracy (shuffle-split)
    # =========================================================================
    print(f"\n--- Permutation test: RF accuracy ({N_PERMUTATIONS} permutations) ---")
    perm_accuracies = []
    t0 = time.time()
    for i in range(N_PERMUTATIONS):
        perm_rng = np.random.default_rng(i + 1)
        y_perm = perm_rng.permutation(y)
        y_perm_train = y_perm[train_shuf]
        y_perm_test = y_perm[test_shuf]
        acc = run_rf_accuracy(X_train_shuf, y_perm_train,
                              X_test_shuf, y_perm_test, seed=SEED)
        perm_accuracies.append(acc)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{N_PERMUTATIONS}] mean={np.mean(perm_accuracies):.4f}, "
                  f"elapsed={elapsed:.1f}s")

    perm_accuracies = np.array(perm_accuracies)
    acc_p_value = float(np.mean(perm_accuracies >= real_accuracy_shuf))
    print(f"\n  Permutation accuracy: mean={perm_accuracies.mean():.4f}, "
          f"std={perm_accuracies.std():.4f}")
    print(f"  Real accuracy: {real_accuracy_shuf:.4f}")
    print(f"  p-value: {acc_p_value:.4f}")

    # =========================================================================
    # Permutation test: BSS (LR on Y_long, chronological)
    # =========================================================================
    print(f"\n--- Permutation test: BSS ({N_PERMUTATIONS} permutations) ---")
    perm_bss_values = []
    t0 = time.time()
    for i in range(N_PERMUTATIONS):
        perm_rng = np.random.default_rng(i + 1)
        # Permute Y_long labels
        Y_long_perm = perm_rng.permutation(Y_long)
        Y_long_perm_train = Y_long_perm[train_chrono]
        Y_long_perm_val = Y_long_perm[val_chrono]
        bss = run_lr_bss(X_train_chrono, Y_long_perm_train,
                         X_val_chrono, Y_long_perm_val, seed=SEED)
        perm_bss_values.append(bss)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    [{i+1}/{N_PERMUTATIONS}] mean={np.mean(perm_bss_values):.6f}, "
                  f"elapsed={elapsed:.1f}s")

    perm_bss_values = np.array(perm_bss_values)
    bss_p_value = float(np.mean(perm_bss_values >= real_bss))
    real_bss_above_mean = real_bss > perm_bss_values.mean()
    print(f"\n  Permutation BSS: mean={perm_bss_values.mean():.6f}, "
          f"std={perm_bss_values.std():.6f}")
    print(f"  Real BSS: {real_bss:.6f}")
    print(f"  Real > mean permuted: {real_bss_above_mean}")
    print(f"  p-value: {bss_p_value:.4f}")

    # =========================================================================
    # Success criteria
    # =========================================================================
    # C1: Permutation p-value < 0.05 for RF accuracy
    C1 = acc_p_value < 0.05

    # C2: Real BSS > mean permuted BSS
    C2 = real_bss_above_mean

    # C3: Mean permuted accuracy within ±1pp of majority baseline
    mean_perm_acc = float(perm_accuracies.mean())
    C3 = abs(mean_perm_acc - majority_baseline) < 0.01

    # =========================================================================
    # Save metrics
    # =========================================================================
    elapsed_total = time.time() - t0_experiment

    metrics = {
        "experiment": "exp-010-permutation-test",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": {
            "cache_dir": str(CACHE_DIR),
            "n_samples": n_samples,
            "n_sessions": n_sessions,
            "feature_dim": int(X.shape[1]),
            "label_distribution": {
                "long_profit": int(counts[0]),
                "short_profit": int(counts[1]),
                "flat": int(counts[2]),
                "majority_baseline": majority_baseline,
            },
        },
        "real_labels": {
            "rf_accuracy_shuffle": real_accuracy_shuf,
            "rf_accuracy_chrono": real_accuracy_chrono,
            "bss_lr_ylong_chrono": real_bss,
        },
        "permutation_accuracy": {
            "n_permutations": N_PERMUTATIONS,
            "mean": float(perm_accuracies.mean()),
            "std": float(perm_accuracies.std()),
            "min": float(perm_accuracies.min()),
            "max": float(perm_accuracies.max()),
            "p95": float(np.percentile(perm_accuracies, 95)),
            "p_value": acc_p_value,
            "real_accuracy": real_accuracy_shuf,
            "real_minus_mean_perm": float(real_accuracy_shuf - perm_accuracies.mean()),
            "all_values": perm_accuracies.tolist(),
        },
        "permutation_bss": {
            "n_permutations": N_PERMUTATIONS,
            "mean": float(perm_bss_values.mean()),
            "std": float(perm_bss_values.std()),
            "min": float(perm_bss_values.min()),
            "max": float(perm_bss_values.max()),
            "p95": float(np.percentile(perm_bss_values, 95)),
            "p_value": bss_p_value,
            "real_bss": real_bss,
            "real_above_mean": real_bss_above_mean,
            "all_values": perm_bss_values.tolist(),
        },
        "success_criteria": {
            "C1_accuracy_signal_real": C1,
            "C2_bss_above_mean_perm": C2,
            "C3_perm_acc_near_baseline": C3,
        },
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": elapsed_total,
        },
        "abort_triggered": False,
        "abort_reason": None,
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Experiment complete. Wall time: {elapsed_total/60:.1f} min")
    print(f"Results: {RESULTS_DIR / 'metrics.json'}")
    print(f"\nReal RF accuracy (shuffle): {real_accuracy_shuf:.4f}")
    print(f"Permuted mean:             {perm_accuracies.mean():.4f} ± {perm_accuracies.std():.4f}")
    print(f"Accuracy p-value:          {acc_p_value:.4f}")
    print(f"\nReal BSS (Y_long):         {real_bss:.6f}")
    print(f"Permuted mean:             {perm_bss_values.mean():.6f} ± {perm_bss_values.std():.6f}")
    print(f"BSS p-value:               {bss_p_value:.4f}")
    print(f"\nC1 (accuracy signal real):  {C1}")
    print(f"C2 (BSS > mean permuted):   {C2}")
    print(f"C3 (perm acc ≈ baseline):   {C3}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
