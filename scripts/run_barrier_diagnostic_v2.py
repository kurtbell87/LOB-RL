#!/usr/bin/env python3
"""T6 Supervised Diagnostic v2 — Proper bidirectional framing.

The agent decides at each bar: go LONG, go SHORT, or stay FLAT.
This script computes labels for BOTH directions and classifies the
optimal action: {long_profit, short_profit, neither}.

Key insight: long profit and short profit are MUTUALLY EXCLUSIVE
(if price rises a=20 before falling b=10, the short already stopped
out at +10). So the 3-class problem is well-defined.

Expected distribution (2:1 barriers under random walk):
  ~33% long profitable, ~33% short profitable, ~34% neither

Usage:
  PYTHONPATH=build-release:python uv run --with pandas --with scikit-learn --with torch \
    python scripts/run_barrier_diagnostic_v2.py --cache-dir cache/barrier/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add scripts dir for load_session_from_cache
sys.path.insert(0, str(Path(__file__).parent))

from precompute_barrier_cache import load_session_from_cache
from lob_rl.barrier.label_pipeline import compute_labels
from lob_rl.barrier.feature_pipeline import (
    compute_bar_features,
    normalize_features,
    assemble_lookback,
)
from lob_rl.barrier.supervised_diagnostic import (
    BarrierMLP,
    evaluate_classifier,
)


def build_bidirectional_dataset(cache_dir, lookback=10, a=20, b=10, t_max=40):
    """Load all sessions and compute bidirectional labels.

    Returns X (n_total, 13*lookback) float32, y (n_total,) int64,
    session_info list, and class names.

    Target classes:
      0 = Long profitable (long label = +1)
      1 = Short profitable (short label = +1)
      2 = Flat (neither profitable)
    """
    files = sorted(Path(cache_dir).glob("*.npz"))
    if not files:
        print(f"ERROR: No .npz files in {cache_dir}")
        sys.exit(1)

    all_X = []
    all_y = []
    session_info = []

    for f in files:
        session = load_session_from_cache(str(f))
        bars = session["bars"]
        features = session["features"]  # precomputed (n_usable, 130)

        n_usable = features.shape[0]
        if n_usable == 0:
            continue

        # Compute labels for both directions
        labels_long = compute_labels(bars, a=a, b=b, t_max=t_max, direction="long")
        labels_short = compute_labels(bars, a=a, b=b, t_max=t_max, direction="short")

        # Align: features[i] corresponds to bar[i + lookback - 1]
        y_session = np.zeros(n_usable, dtype=np.int64)
        for i in range(n_usable):
            bar_idx = i + lookback - 1
            ll = labels_long[bar_idx].label
            ls = labels_short[bar_idx].label

            if ll == 1:
                y_session[i] = 0  # Long profitable
            elif ls == 1:
                y_session[i] = 1  # Short profitable
            else:
                y_session[i] = 2  # Neither (flat)

        all_X.append(features.astype(np.float32))
        all_y.append(y_session)
        session_info.append((f.stem, n_usable))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y, session_info


def train_mlp_v2(X, y, epochs=100, batch_size=256, hidden_dim=256, lr=1e-3, seed=42):
    """Train MLP and return model + metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, d = X.shape
    effective_batch = min(batch_size, n)

    model = BarrierMLP(input_dim=d, hidden_dim=hidden_dim, n_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, effective_batch):
            end = min(start + effective_batch, n)
            idx = perm[start:end]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1)
        accuracy = float((preds == y_t).float().mean().item())
        final_loss = float(criterion(logits, y_t).item())

    return model, {"train_accuracy": accuracy, "train_loss": final_loss}


def print_eval_results(name, result):
    """Pretty-print evaluation results."""
    print(f"\n  === {name} ===")
    print(f"  Accuracy:          {result['accuracy']:.4f}")
    print(f"  Balanced accuracy: {result['balanced_accuracy']:.4f}")
    print(f"  Majority class:    {result['majority_class']} (baseline: {result['majority_baseline']:.4f})")
    print(f"  Beats baseline:    {result['beats_baseline']}")

    label_names = {0: "long", 1: "short", 2: "flat"}
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for cls in range(3):
        pc = result["per_class"][cls]
        print(f"  {label_names[cls]:<12} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>14} {'pred long':>10} {'pred short':>11} {'pred flat':>10}")
    for i, row_name in enumerate(["true long", "true short", "true flat"]):
        row = result["confusion_matrix"][i]
        print(f"  {row_name:>14} {row[0]:>10d} {row[1]:>11d} {row[2]:>10d}")


def main():
    parser = argparse.ArgumentParser(description="T6 supervised diagnostic v2 (bidirectional)")
    parser.add_argument("--cache-dir", type=str, default="cache/barrier/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("T6 SUPERVISED DIAGNOSTIC v2 — Bidirectional Framing")
    print("=" * 60)
    print("  Target: {long_profitable, short_profitable, flat}")
    print("  Long and short are mutually exclusive events.")

    # --- Load data ---
    print("\n--- Loading barrier cache + computing bidirectional labels ---")
    t0 = time.time()
    X, y, session_info = build_bidirectional_dataset(args.cache_dir)
    load_time = time.time() - t0
    print(f"  Loaded {len(session_info)} sessions in {load_time:.1f}s")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Feature dim:   {X.shape[1]}")

    assert X.shape[1] == 130, f"Expected 130-dim, got {X.shape[1]}"

    # Label distribution
    counts = np.bincount(y, minlength=3)
    total = counts.sum()
    print(f"\n  Label distribution:")
    print(f"    Class 0 (long):   {counts[0]:>6d} ({100*counts[0]/total:.1f}%)")
    print(f"    Class 1 (short):  {counts[1]:>6d} ({100*counts[1]/total:.1f}%)")
    print(f"    Class 2 (flat):   {counts[2]:>6d} ({100*counts[2]/total:.1f}%)")
    print(f"    Majority baseline: {max(counts)/total:.4f}")

    # NaN check
    has_nan = np.any(np.isnan(X))
    has_inf = np.any(np.isinf(X))
    if has_nan or has_inf:
        print(f"  Replacing NaN/Inf with 0")
        X = np.where(np.isnan(X), 0.0, X)
        X = np.where(np.isinf(X), 0.0, X)

    rng = np.random.default_rng(args.seed)

    # ===================================================================
    # STEP 1: Overfit test
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 1: OVERFIT TEST (256 samples)")
    print("=" * 60)

    overfit_idx = rng.choice(X.shape[0], size=256, replace=False)
    X_of, y_of = X[overfit_idx], y[overfit_idx]

    t0 = time.time()
    model_of, metrics_of = train_mlp_v2(X_of, y_of, epochs=500, seed=args.seed)
    print(f"  Train accuracy: {metrics_of['train_accuracy']:.4f} (time: {time.time()-t0:.1f}s)")
    overfit_passed = metrics_of["train_accuracy"] > 0.95
    print(f"  Passed (>0.95): {overfit_passed}")

    if not overfit_passed:
        print("  FATAL: Cannot overfit. Debug first.")
        sys.exit(1)

    # ===================================================================
    # STEP 2: SHUFFLE-SPLIT
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 2: SHUFFLE-SPLIT (80/20)")
    print("=" * 60)

    n = X.shape[0]
    n_train = int(n * 0.8)
    indices = rng.permutation(n)
    X_train_s, y_train_s = X[indices[:n_train]], y[indices[:n_train]]
    X_test_s, y_test_s = X[indices[n_train:]], y[indices[n_train:]]
    print(f"  Train: {len(X_train_s)}, Test: {len(X_test_s)}")

    # MLP
    print("\n  Training MLP [256, 256] ReLU...")
    t0 = time.time()
    mlp_s, mlp_train_s = train_mlp_v2(X_train_s, y_train_s, epochs=args.epochs, seed=args.seed)
    print(f"  Train accuracy: {mlp_train_s['train_accuracy']:.4f} (time: {time.time()-t0:.1f}s)")
    mlp_eval_s = evaluate_classifier(mlp_s, X_test_s, y_test_s)
    print_eval_results("MLP (shuffle-split)", mlp_eval_s)

    # Random Forest
    print("\n  Training Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf_s = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    rf_s.fit(X_train_s, y_train_s)
    rf_eval_s = evaluate_classifier(rf_s, X_test_s, y_test_s)
    print(f"  RF training time: {time.time()-t0:.1f}s")
    print_eval_results("Random Forest (shuffle-split)", rf_eval_s)

    # ===================================================================
    # STEP 3: CHRONOLOGICAL SPLIT
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 3: CHRONOLOGICAL SPLIT (last ~50 days)")
    print("=" * 60)

    n_sessions = len(session_info)
    n_test_sessions = min(50, n_sessions // 4)
    n_train_sessions = n_sessions - n_test_sessions

    train_sizes = [s[1] for s in session_info[:n_train_sessions]]
    n_train_c = sum(train_sizes)
    n_test_c = X.shape[0] - n_train_c

    X_train_c, y_train_c = X[:n_train_c], y[:n_train_c]
    X_test_c, y_test_c = X[n_train_c:], y[n_train_c:]

    print(f"  Train: {n_train_sessions} sessions ({session_info[0][0]}-{session_info[n_train_sessions-1][0]}), {n_train_c} samples")
    print(f"  Test:  {n_test_sessions} sessions ({session_info[n_train_sessions][0]}-{session_info[-1][0]}), {n_test_c} samples")

    # MLP
    print("\n  Training MLP [256, 256] ReLU...")
    t0 = time.time()
    mlp_c, mlp_train_c = train_mlp_v2(X_train_c, y_train_c, epochs=args.epochs, seed=args.seed)
    print(f"  Train accuracy: {mlp_train_c['train_accuracy']:.4f} (time: {time.time()-t0:.1f}s)")
    mlp_eval_c = evaluate_classifier(mlp_c, X_test_c, y_test_c)
    print_eval_results("MLP (chronological)", mlp_eval_c)

    # Random Forest
    print("\n  Training Random Forest...")
    t0 = time.time()
    rf_c = RandomForestClassifier(n_estimators=100, random_state=args.seed)
    rf_c.fit(X_train_c, y_train_c)
    rf_eval_c = evaluate_classifier(rf_c, X_test_c, y_test_c)
    print(f"  RF training time: {time.time()-t0:.1f}s")
    print_eval_results("Random Forest (chronological)", rf_eval_c)

    # ===================================================================
    # STEP 4: RF Feature Importance
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 4: RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 60)

    importances = rf_s.feature_importances_
    feature_names_base = [
        "trade_flow_imbal", "bbo_imbal", "depth_imbal", "bar_range",
        "bar_body", "body_range_ratio", "vwap_displace", "volume_log",
        "realized_vol", "session_time", "cancel_asym", "mean_spread",
        "session_age",
    ]

    print("\n  Top 20 features by importance:")
    feat_imp = []
    for i in range(130):
        fi = i % 13
        lag = i // 13
        name = f"{feature_names_base[fi]}[t-{9-lag}]"
        feat_imp.append((name, importances[i]))
    feat_imp.sort(key=lambda x: -x[1])
    for name, imp in feat_imp[:20]:
        print(f"    {name:<30s} {imp:.4f}")

    print("\n  Aggregate by feature type:")
    for fi in range(13):
        agg = sum(importances[fi + 13 * lag] for lag in range(10))
        print(f"    {feature_names_base[fi]:<25s} {agg:.4f}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 60)
    print("SUMMARY — BIDIRECTIONAL FRAMING")
    print("=" * 60)

    majority_baseline = max(counts) / total
    print(f"\n  Majority baseline (flat):    {majority_baseline:.4f}")

    print(f"\n  Shuffle-split:")
    print(f"    MLP accuracy:              {mlp_eval_s['accuracy']:.4f}  (balanced: {mlp_eval_s['balanced_accuracy']:.4f})")
    print(f"    RF accuracy:               {rf_eval_s['accuracy']:.4f}  (balanced: {rf_eval_s['balanced_accuracy']:.4f})")
    print(f"    MLP beats baseline:        {mlp_eval_s['beats_baseline']}")
    print(f"    RF beats baseline:         {rf_eval_s['beats_baseline']}")

    print(f"\n  Chronological:")
    print(f"    MLP accuracy:              {mlp_eval_c['accuracy']:.4f}  (balanced: {mlp_eval_c['balanced_accuracy']:.4f})")
    print(f"    RF accuracy:               {rf_eval_c['accuracy']:.4f}  (balanced: {rf_eval_c['balanced_accuracy']:.4f})")
    print(f"    MLP beats baseline:        {mlp_eval_c['beats_baseline']}")
    print(f"    RF beats baseline:         {rf_eval_c['beats_baseline']}")

    print(f"\n  INTERPRETATION:")
    mlp_beats = mlp_eval_s['beats_baseline'] or mlp_eval_c['beats_baseline']
    rf_beats = rf_eval_s['beats_baseline'] or rf_eval_c['beats_baseline']

    if not mlp_beats and not rf_beats:
        print("  >> Features contain ZERO directional signal.")
        print("  >> Fix features (activate 4 dead book features) before any RL.")
    elif rf_beats and not mlp_beats:
        print("  >> Signal exists but MLP architecture may not be ideal.")
    elif mlp_beats and rf_beats:
        mlp_bal = max(mlp_eval_s['balanced_accuracy'], mlp_eval_c['balanced_accuracy'])
        rf_bal = max(rf_eval_s['balanced_accuracy'], rf_eval_c['balanced_accuracy'])
        if mlp_bal > rf_bal:
            print("  >> Signal EXISTS. MLP outperforms RF on directional prediction.")
            print("  >> Proceed to RL with confidence.")
        else:
            print("  >> Signal exists. RF comparable or better than MLP.")
            print("  >> Proceed to RL, but architecture is a variable.")
    elif mlp_beats:
        print("  >> Signal exists (MLP beats baseline on direction).")
        print("  >> Proceed to RL.")

    # Save results
    results = {
        "framing": "bidirectional",
        "classes": ["long_profitable", "short_profitable", "flat"],
        "n_sessions": len(session_info),
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "label_distribution": {
            "long": int(counts[0]),
            "short": int(counts[1]),
            "flat": int(counts[2]),
            "majority_baseline": float(majority_baseline),
        },
        "overfit_test": {
            "train_accuracy": metrics_of["train_accuracy"],
            "passed": overfit_passed,
        },
        "shuffle_split": {
            "n_train": int(len(X_train_s)),
            "n_test": int(len(X_test_s)),
            "mlp": mlp_eval_s,
            "random_forest": rf_eval_s,
        },
        "chronological": {
            "n_train": int(n_train_c),
            "n_test": int(n_test_c),
            "mlp": mlp_eval_c,
            "random_forest": rf_eval_c,
        },
    }

    output_path = Path(args.cache_dir).parent / "t6_diagnostic_v2_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
