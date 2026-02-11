#!/usr/bin/env python3
"""exp-004 QUICK: 22-feature vs 9-feature signal detection.

Quick-tier diagnostic (~10 min): loads cached features, computes
bidirectional labels (Python — short labels not yet in C++ cache),
subsamples to 50K, runs RF on Set A (22 features) vs Set B (9 features)
with 2 seeds, shuffle split only.

Usage:
  cd build-release && PYTHONPATH=.:../python uv run python \
    ../scripts/run_exp004_quick.py --cache-dir ../cache/barrier/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from precompute_barrier_cache import load_session_from_cache
from lob_rl.barrier.label_pipeline import compute_labels

N_FEATURES = 22
LOOKBACK = 10
SUBSAMPLE = 50_000

FEATURE_NAMES = [
    "trade_flow_imbal", "bbo_imbal", "depth_imbal", "bar_range",
    "bar_body", "body_range_ratio", "vwap_displace", "volume_log",
    "realized_vol", "session_time", "cancel_asym", "mean_spread",
    "session_age", "OFI", "depth_ratio", "weighted_mid_disp",
    "spread_std", "VAMP_disp", "aggressor_imbal", "trade_arrival",
    "cancel_to_trade", "price_impact",
]

SUBSET_COLS = {
    "all":      list(range(22)),
    "original": [0, 3, 4, 5, 6, 7, 8, 9, 12],
}


def expand_cols(base_cols, lookback=LOOKBACK):
    expanded = []
    for c in base_cols:
        expanded.extend(range(c * lookback, (c + 1) * lookback))
    return expanded


def load_all_sessions(cache_dir):
    """Load features from cache, compute bidirectional labels.

    NOTE: Short-direction labels are computed in Python because the C++
    cache only stores long-direction labels. This is tech debt — a TDD
    cycle should add short labels to the C++ precompute pipeline.
    """
    files = sorted(Path(cache_dir).glob("*.npz"))
    if not files:
        print(f"ERROR: No .npz files in {cache_dir}", flush=True)
        sys.exit(1)

    all_X = []
    all_y = []
    t0 = time.time()

    for i, f in enumerate(files):
        session = load_session_from_cache(str(f))
        bars = session["bars"]
        features = session["features"]
        n_usable = features.shape[0]
        if n_usable == 0:
            continue

        # Long labels from cache, short labels from Python (tech debt)
        labels_long = session["labels"]  # already loaded from cache
        labels_short = compute_labels(bars, a=20, b=10, t_max=40, direction="short")

        y_session = np.zeros(n_usable, dtype=np.int64)
        for j in range(n_usable):
            bar_idx = j + LOOKBACK - 1
            if labels_long[bar_idx].label == 1:
                y_session[j] = 0  # long profitable
            elif labels_short[bar_idx].label == 1:
                y_session[j] = 1  # short profitable
            else:
                y_session[j] = 2  # flat

        all_X.append(features.astype(np.float32))
        all_y.append(y_session)

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(files)} sessions ({time.time()-t0:.0f}s)", flush=True)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    X = np.where(np.isnan(X), 0.0, X)
    X = np.where(np.isinf(X), 0.0, X)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="exp-004 quick diagnostic")
    parser.add_argument("--cache-dir", type=str, default="../cache/barrier/")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    t_start = time.time()
    seeds = [42, 43]

    print("=" * 60, flush=True)
    print("exp-004 QUICK: 22-feature vs 9-feature signal detection", flush=True)
    print("=" * 60, flush=True)

    # --- Load ---
    print("\n--- Loading sessions ---", flush=True)
    t0 = time.time()
    X_full, y = load_all_sessions(args.cache_dir)
    n_total = X_full.shape[0]
    print(f"  Loaded {n_total} samples, {X_full.shape[1]} features in {time.time()-t0:.1f}s", flush=True)

    counts = np.bincount(y, minlength=3)
    label_dist = {
        "long_pct": float(100 * counts[0] / n_total),
        "short_pct": float(100 * counts[1] / n_total),
        "flat_pct": float(100 * counts[2] / n_total),
    }
    print(f"  Labels: long={label_dist['long_pct']:.1f}%, "
          f"short={label_dist['short_pct']:.1f}%, "
          f"flat={label_dist['flat_pct']:.1f}%", flush=True)

    # --- Subsample ---
    rng = np.random.default_rng(42)
    if n_total > SUBSAMPLE:
        idx = rng.choice(n_total, size=SUBSAMPLE, replace=False)
        X_sub = X_full[idx]
        y_sub = y[idx]
        print(f"  Subsampled to {SUBSAMPLE}", flush=True)
    else:
        X_sub = X_full
        y_sub = y

    # --- RF: Set A vs Set B, 2 seeds, shuffle split ---
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    cols_a = expand_cols(SUBSET_COLS["all"])
    cols_b = expand_cols(SUBSET_COLS["original"])

    results_a = []
    results_b = []
    train_acc_a = []
    train_acc_b = []

    print("\n--- RF classifier (shuffle split, 2 seeds) ---", flush=True)
    for seed in seeds:
        rng_s = np.random.default_rng(seed)
        n = len(X_sub)
        n_train = int(n * 0.8)
        perm = rng_s.permutation(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]

        for label, cols, res_list, train_list in [
            ("A (all 22)", cols_a, results_a, train_acc_a),
            ("B (original 9)", cols_b, results_b, train_acc_b),
        ]:
            X_tr = X_sub[train_idx][:, cols]
            y_tr = y_sub[train_idx]
            X_te = X_sub[test_idx][:, cols]
            y_te = y_sub[test_idx]

            t0 = time.time()
            rf = RandomForestClassifier(
                n_estimators=100, max_features="sqrt",
                random_state=seed, n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            bal_acc = balanced_accuracy_score(y_te, rf.predict(X_te))
            tr_acc = balanced_accuracy_score(y_tr, rf.predict(X_tr))
            elapsed = time.time() - t0

            res_list.append(bal_acc)
            train_list.append(tr_acc)
            print(f"  Set {label} seed={seed}: bal_acc={bal_acc:.4f}, "
                  f"train={tr_acc:.4f} ({elapsed:.1f}s)", flush=True)

    mean_a = float(np.mean(results_a))
    mean_b = float(np.mean(results_b))
    delta = mean_a - mean_b
    majority = float(max(counts) / n_total)

    print(f"\n--- Results ---", flush=True)
    print(f"  Set A mean bal_acc: {mean_a:.4f}", flush=True)
    print(f"  Set B mean bal_acc: {mean_b:.4f}", flush=True)
    print(f"  Delta (A - B):      {delta*100:+.2f}pp", flush=True)
    print(f"  Majority baseline:  {majority:.4f}", flush=True)
    print(f"  T6 baseline (RF):   0.4050", flush=True)

    # --- Verdict ---
    a_beats_b = delta > 0.02
    a_above_baseline = mean_a > majority
    b_reproduces_t6 = abs(mean_b - 0.405) < 0.05  # within 5pp

    print(f"\n--- Verdicts ---", flush=True)
    print(f"  A > B by >2pp:        {a_beats_b} ({delta*100:+.2f}pp)", flush=True)
    print(f"  A > majority:         {a_above_baseline} ({mean_a:.4f} vs {majority:.4f})", flush=True)
    print(f"  B reproduces T6 ±5pp: {b_reproduces_t6} ({mean_b:.4f} vs 0.405)", flush=True)

    wall_time = time.time() - t_start
    print(f"\n  Wall time: {wall_time:.1f}s ({wall_time/60:.1f} min)", flush=True)

    # --- Write metrics ---
    metrics = {
        "experiment": "exp-004-22-feature-supervised-diagnostic",
        "tier": "quick",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "set_a": {
            "mean_balanced_acc": mean_a,
            "per_seed": {str(s): float(v) for s, v in zip(seeds, results_a)},
            "train_accs": {str(s): float(v) for s, v in zip(seeds, train_acc_a)},
        },
        "set_b": {
            "mean_balanced_acc": mean_b,
            "per_seed": {str(s): float(v) for s, v in zip(seeds, results_b)},
            "train_accs": {str(s): float(v) for s, v in zip(seeds, train_acc_b)},
        },
        "delta_a_minus_b_pp": float(delta * 100),
        "majority_baseline": majority,
        "t6_baseline": 0.405,
        "label_distribution": label_dist,
        "n_samples_total": n_total,
        "n_samples_used": len(X_sub),
        "feature_dim": int(X_full.shape[1]),
        "verdicts": {
            "a_beats_b_by_2pp": a_beats_b,
            "a_above_majority": a_above_baseline,
            "b_reproduces_t6": b_reproduces_t6,
        },
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": float(wall_time),
            "total_training_runs": 4,
            "seeds": seeds,
        },
        "notes": (
            "Quick-tier diagnostic. Short-direction labels computed in Python "
            "(tech debt: C++ cache only stores long-direction labels). "
            "Subsampled to 50K for RF speed. "
        ),
    }

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (Path(__file__).parent.parent
                      / "results" / "exp-004-22-feature-supervised-diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
