#!/usr/bin/env python3
"""Run T6 Supervised Diagnostic on precomputed barrier cache.

Loads all barrier cache .npz files, constructs the full dataset,
and runs the diagnostic (overfit test, MLP, random forest) with
both shuffle-split and chronological splits.

Usage:
  cd build-release
  PYTHONPATH=.:../python uv run --with scikit-learn --with torch \
    python ../scripts/run_barrier_diagnostic.py --cache-dir ../cache/barrier/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lob_rl.barrier.supervised_diagnostic import (
    overfit_test,
    train_mlp,
    evaluate_classifier,
    train_random_forest,
)


def load_all_sessions(cache_dir):
    """Load features and labels from all cache files.

    Returns X (n_total, 130) float32 and y (n_total,) int64,
    plus a list of (date_str, n_usable) for chronological splitting.
    """
    files = sorted(Path(cache_dir).glob("*.npz"))
    if not files:
        print(f"ERROR: No .npz files in {cache_dir}")
        sys.exit(1)

    all_X = []
    all_y = []
    session_info = []
    label_map = {-1: 0, 0: 1, 1: 2}

    for f in files:
        data = np.load(f)
        features = data["features"]  # (n_usable, 130)
        labels = data["label_values"]  # (n_bars,)
        n_usable = int(data["n_usable"])
        lookback = int(data["lookback"])

        # Labels aligned: y[i] = label[i + lookback - 1]
        y_session = np.array(
            [label_map[int(labels[i + lookback - 1])] for i in range(n_usable)],
            dtype=np.int64,
        )

        all_X.append(features.astype(np.float32))
        all_y.append(y_session)

        date_str = f.stem  # e.g. "20220103"
        session_info.append((date_str, n_usable))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, session_info


def print_eval_results(name, result):
    """Pretty-print evaluation results."""
    print(f"\n  === {name} ===")
    print(f"  Accuracy:          {result['accuracy']:.4f}")
    print(f"  Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print(f"  Majority class:    {result['majority_class']} (baseline: {result['majority_baseline']:.4f})")
    print(f"  Beats baseline:    {result['beats_baseline']}")

    # Label names for readability
    label_names = {0: "stop (-1)", 1: "timeout (0)", 2: "profit (+1)"}
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for cls in range(3):
        pc = result["per_class"][cls]
        name_str = label_names.get(cls, str(cls))
        print(f"  {name_str:<16} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>16} {'pred stop':>10} {'pred t/o':>10} {'pred prof':>10}")
    for i, row_name in enumerate(["true stop", "true t/o", "true prof"]):
        row = result["confusion_matrix"][i]
        print(f"  {row_name:>16} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")


def main():
    parser = argparse.ArgumentParser(description="Run T6 supervised diagnostic")
    parser.add_argument("--cache-dir", type=str, default="cache/barrier/",
                        help="Barrier cache directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="MLP training epochs (default: 100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("T6 SUPERVISED DIAGNOSTIC — Barrier Feature Signal Test")
    print("=" * 60)

    # --- Load data ---
    print("\n--- Loading barrier cache ---")
    t0 = time.time()
    X, y, session_info = load_all_sessions(args.cache_dir)
    load_time = time.time() - t0
    print(f"  Loaded {len(session_info)} sessions in {load_time:.1f}s")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Feature dim:   {X.shape[1]}")

    # Verify dimension
    assert X.shape[1] == 130, f"Expected 130-dim features, got {X.shape[1]}. Wrong pipeline!"
    print(f"  CONFIRMED: 130-dim barrier features (13 x 10 lookback)")

    # Label distribution
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    print(f"\n  Label distribution:")
    print(f"    Class 0 (stop -1):    {counts[0]:>6d} ({100*counts[0]/total:.1f}%)")
    print(f"    Class 1 (timeout 0):  {counts[1]:>6d} ({100*counts[1]/total:.1f}%)")
    print(f"    Class 2 (profit +1):  {counts[2]:>6d} ({100*counts[2]/total:.1f}%)")
    print(f"    Majority baseline:    {max(counts)/total:.4f}")

    # Check for NaN/Inf
    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))
    print(f"\n  Data quality: NaN={has_nan}, Inf={has_inf}")
    if has_nan or has_inf:
        print("  WARNING: Bad values in features! Replacing NaN with 0.")
        X = np.where(np.isnan(X), 0.0, X)
        X = np.where(np.isinf(X), 0.0, X)

    # ===================================================================
    # STEP 1: Overfit test (must pass before proceeding)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 1: OVERFIT TEST")
    print("=" * 60)
    print("  Training MLP on 256 samples to >95% accuracy...")

    rng = np.random.default_rng(args.seed)
    overfit_idx = rng.choice(X.shape[0], size=256, replace=False)
    X_overfit = X[overfit_idx]
    y_overfit = y[overfit_idx]

    t0 = time.time()
    overfit_result = overfit_test(X_overfit, y_overfit, epochs=500, seed=args.seed)
    overfit_time = time.time() - t0
    print(f"  Train accuracy: {overfit_result['train_accuracy']:.4f}")
    print(f"  Passed (>0.95): {overfit_result['passed']}")
    print(f"  Time: {overfit_time:.1f}s")

    if not overfit_result["passed"]:
        print("\n  FATAL: Cannot overfit 256 samples. Architecture or data pipeline broken.")
        print("  DO NOT proceed to RL training. Debug first.")
        sys.exit(1)

    # ===================================================================
    # STEP 2: SHUFFLE-SPLIT DIAGNOSTIC
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 2: SHUFFLE-SPLIT (80/20)")
    print("=" * 60)

    n = X.shape[0]
    n_train = int(n * 0.8)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train_s, y_train_s = X[train_idx], y[train_idx]
    X_test_s, y_test_s = X[test_idx], y[test_idx]

    print(f"  Train: {len(X_train_s)}, Test: {len(X_test_s)}")

    # MLP
    print("\n  Training MLP [256, 256] ReLU...")
    t0 = time.time()
    mlp_model, mlp_train = train_mlp(X_train_s, y_train_s, epochs=args.epochs, seed=args.seed)
    mlp_time = time.time() - t0
    print(f"  MLP train accuracy: {mlp_train['train_accuracy']:.4f}")
    print(f"  MLP train loss:     {mlp_train['train_loss']:.4f}")
    print(f"  MLP training time:  {mlp_time:.1f}s")

    mlp_eval_s = evaluate_classifier(mlp_model, X_test_s, y_test_s)
    print_eval_results("MLP (shuffle-split)", mlp_eval_s)

    # Random Forest
    print("\n  Training Random Forest (100 trees)...")
    t0 = time.time()
    rf_eval_s = train_random_forest(X_train_s, y_train_s, X_test_s, y_test_s, seed=args.seed)
    rf_time = time.time() - t0
    print(f"  RF training time: {rf_time:.1f}s")
    print_eval_results("Random Forest (shuffle-split)", rf_eval_s)

    # ===================================================================
    # STEP 3: CHRONOLOGICAL SPLIT
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 3: CHRONOLOGICAL SPLIT (last ~50 days held out)")
    print("=" * 60)

    # Sessions are already sorted by date. Split: first ~200 train, last ~47 test.
    n_sessions = len(session_info)
    n_test_sessions = min(50, n_sessions // 4)
    n_train_sessions = n_sessions - n_test_sessions

    train_sizes = [s[1] for s in session_info[:n_train_sessions]]
    test_sizes = [s[1] for s in session_info[n_train_sessions:]]
    n_train_c = sum(train_sizes)
    n_test_c = sum(test_sizes)

    X_train_c = X[:n_train_c]
    y_train_c = y[:n_train_c]
    X_test_c = X[n_train_c:n_train_c + n_test_c]
    y_test_c = y[n_train_c:n_train_c + n_test_c]

    print(f"  Train sessions: {n_train_sessions} ({session_info[0][0]} - {session_info[n_train_sessions-1][0]})")
    print(f"  Test sessions:  {n_test_sessions} ({session_info[n_train_sessions][0]} - {session_info[-1][0]})")
    print(f"  Train samples:  {n_train_c}, Test samples: {n_test_c}")

    # MLP
    print("\n  Training MLP [256, 256] ReLU...")
    t0 = time.time()
    mlp_model_c, mlp_train_c = train_mlp(X_train_c, y_train_c, epochs=args.epochs, seed=args.seed)
    mlp_time_c = time.time() - t0
    print(f"  MLP train accuracy: {mlp_train_c['train_accuracy']:.4f}")
    print(f"  MLP training time:  {mlp_time_c:.1f}s")

    mlp_eval_c = evaluate_classifier(mlp_model_c, X_test_c, y_test_c)
    print_eval_results("MLP (chronological)", mlp_eval_c)

    # Random Forest
    print("\n  Training Random Forest (100 trees)...")
    t0 = time.time()
    rf_eval_c = train_random_forest(X_train_c, y_train_c, X_test_c, y_test_c, seed=args.seed)
    rf_time_c = time.time() - t0
    print(f"  RF training time: {rf_time_c:.1f}s")
    print_eval_results("Random Forest (chronological)", rf_eval_c)

    # ===================================================================
    # STEP 4: RF Feature Importance (for diagnostic insight)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 4: RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    rf_full = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    rf_full.fit(X_train_s, y_train_s)
    importances = rf_full.feature_importances_

    feature_names_base = [
        "trade_flow_imbal", "bbo_imbal", "depth_imbal", "bar_range",
        "bar_body", "body_range_ratio", "vwap_displace", "volume_log",
        "realized_vol", "session_time", "cancel_asym", "mean_spread",
        "session_age",
    ]

    # Group by lookback position and by feature type
    print("\n  Top 20 features by importance:")
    feat_imp = []
    for i in range(130):
        feat_idx = i % 13
        lag = i // 13
        name = f"{feature_names_base[feat_idx]}[t-{9-lag}]"
        feat_imp.append((name, importances[i], i))

    feat_imp.sort(key=lambda x: -x[1])
    for name, imp, idx in feat_imp[:20]:
        print(f"    {name:<30s} {imp:.4f}")

    # Aggregate by feature type
    print("\n  Aggregate importance by feature type:")
    for fi in range(13):
        agg = sum(importances[fi + 13 * lag] for lag in range(10))
        print(f"    {feature_names_base[fi]:<25s} {agg:.4f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    majority_baseline = max(counts) / total
    print(f"\n  Majority baseline:          {majority_baseline:.4f}")
    print(f"\n  Shuffle-split results:")
    print(f"    MLP accuracy:             {mlp_eval_s['accuracy']:.4f}  (balanced: {mlp_eval_s['balanced_accuracy']:.4f})")
    print(f"    RF accuracy:              {rf_eval_s['accuracy']:.4f}  (balanced: {rf_eval_s['balanced_accuracy']:.4f})")
    print(f"    MLP beats baseline:       {mlp_eval_s['beats_baseline']}")
    print(f"    RF beats baseline:        {rf_eval_s['beats_baseline']}")

    print(f"\n  Chronological results:")
    print(f"    MLP accuracy:             {mlp_eval_c['accuracy']:.4f}  (balanced: {mlp_eval_c['balanced_accuracy']:.4f})")
    print(f"    RF accuracy:              {rf_eval_c['accuracy']:.4f}  (balanced: {rf_eval_c['balanced_accuracy']:.4f})")
    print(f"    MLP beats baseline:       {mlp_eval_c['beats_baseline']}")
    print(f"    RF beats baseline:        {rf_eval_c['beats_baseline']}")

    print(f"\n  Overfit test passed:        {overfit_result['passed']}")

    # Interpretation
    print("\n  INTERPRETATION:")
    mlp_beats = mlp_eval_s['beats_baseline'] or mlp_eval_c['beats_baseline']
    rf_beats = rf_eval_s['beats_baseline'] or rf_eval_c['beats_baseline']

    if not overfit_result['passed']:
        print("  >> FAIL: MLP cannot overfit 256 samples. Architecture or data broken.")
    elif not mlp_beats and not rf_beats:
        print("  >> Features contain ZERO signal about barrier outcomes.")
        print("  >> Fix features before any RL. Architecture experiments are pointless.")
    elif rf_beats and not mlp_beats:
        print("  >> Signal exists but MLP architecture may not be ideal.")
        print("  >> Consider tree-based approaches or different architectures.")
    elif mlp_beats and rf_beats:
        mlp_better = (mlp_eval_s['balanced_accuracy'] > rf_eval_s['balanced_accuracy'] and
                      mlp_eval_c['balanced_accuracy'] > rf_eval_c['balanced_accuracy'])
        if mlp_better:
            print("  >> Signal EXISTS. MLP outperforms RF.")
            print("  >> Proceed to RL with confidence. Architecture comparison is P1.")
        else:
            print("  >> Signal exists. RF comparable or better than MLP.")
            print("  >> Proceed to RL, but note architecture is a variable.")
    elif mlp_beats:
        print("  >> Signal exists (MLP beats baseline).")
        print("  >> Proceed to RL. Architecture comparison is P1.")

    # Save results as JSON
    results = {
        "n_sessions": len(session_info),
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "label_distribution": {
            "stop_minus1": int(counts[0]),
            "timeout_0": int(counts[1]),
            "profit_plus1": int(counts[2]),
            "majority_baseline": float(majority_baseline),
        },
        "overfit_test": overfit_result,
        "shuffle_split": {
            "n_train": int(len(X_train_s)),
            "n_test": int(len(X_test_s)),
            "mlp": mlp_eval_s,
            "random_forest": rf_eval_s,
        },
        "chronological": {
            "n_train": int(n_train_c),
            "n_test": int(n_test_c),
            "train_dates": f"{session_info[0][0]}-{session_info[n_train_sessions-1][0]}",
            "test_dates": f"{session_info[n_train_sessions][0]}-{session_info[-1][0]}",
            "mlp": mlp_eval_c,
            "random_forest": rf_eval_c,
        },
    }

    output_path = Path(args.cache_dir).parent / "t6_diagnostic_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
