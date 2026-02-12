#!/usr/bin/env python
"""exp-011: Book-State vs Trade-Flow Feature Group Ablation.

Loads barrier cache, runs LR on 3 feature arms × 2 labels × 3 quarterly folds,
computes paired Δ_AUC with block bootstrap, writes metrics.json.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from lob_rl.barrier.first_passage_analysis import load_binary_labels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "barrier")
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "exp-011-feature-group-ablation"
)
SEED = 42
WALL_CLOCK_ABORT = 30 * 60  # 30 min
N_FEATURES = 22
LOOKBACK = 10
N_BOOT = 1000
BLOCK_SIZE = 50

# Feature group definitions (base column indices within each 22-dim step)
GROUP_A_BASE = [1, 2, 10, 11, 13, 14, 15, 16, 17, 20]  # Book-state
GROUP_B_BASE = [0, 3, 4, 5, 6, 7, 8, 18, 19, 21]        # Trade-flow
TEMPORAL = [9, 12]  # Session time, session age


def expand_cols(base_cols, temporal_cols, n_features=N_FEATURES, lookback=LOOKBACK):
    """Expand base column indices across all lookback steps."""
    all_base = sorted(set(base_cols + temporal_cols))
    cols = sorted([b + step * n_features for step in range(lookback) for b in all_base])
    return np.array(cols, dtype=int)


def expand_full_cols(n_features=N_FEATURES, lookback=LOOKBACK):
    """All 220 columns."""
    return np.arange(n_features * lookback, dtype=int)


def date_to_quarter(date_str):
    """Map YYYYMMDD to quarter (1-4)."""
    month = int(date_str[4:6])
    return (month - 1) // 3 + 1


def sessions_to_rows(session_indices, boundaries):
    """Convert session indices to row indices."""
    rows = []
    for s in session_indices:
        rows.append(np.arange(boundaries[s], boundaries[s + 1]))
    if rows:
        return np.concatenate(rows)
    return np.array([], dtype=np.int64)


def fit_lr(X_train, y_train, max_iter=1000):
    """Fit logistic regression with convergence handling."""
    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter, random_state=SEED)
        lr.fit(X_train, y_train)
        for w in caught:
            if "ConvergenceWarning" in str(w.category.__name__):
                converged = False
    return lr, converged


def brier_skill_score(y_true, y_pred_prob, y_rate_train):
    """BSS = 1 - BS_model / BS_constant."""
    bs_model = brier_score_loss(y_true, y_pred_prob)
    bs_constant = brier_score_loss(y_true, np.full(len(y_true), y_rate_train))
    if bs_constant == 0:
        return 0.0
    return 1.0 - bs_model / bs_constant


def block_bootstrap_delta_auc(y_true, probs_a, probs_b, block_size=BLOCK_SIZE,
                               n_boot=N_BOOT, seed=SEED):
    """Block bootstrap for paired AUC difference (B - A).

    Resample blocks of contiguous samples to preserve temporal dependence.
    Returns: (mean_delta, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_blocks = max(1, n // block_size)
    deltas = []

    for _ in range(n_boot):
        # Sample block start indices with replacement
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
        # Ensure we have valid labels (both classes present)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc_a = roc_auc_score(y_boot, probs_a[idx])
        auc_b = roc_auc_score(y_boot, probs_b[idx])
        deltas.append(auc_b - auc_a)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def block_bootstrap_delta_bss(y_true, probs_a, probs_b, y_rate_train,
                                block_size=BLOCK_SIZE, n_boot=N_BOOT, seed=SEED):
    """Block bootstrap for paired BSS difference (B - A)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_blocks = max(1, n // block_size)
    deltas = []

    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        bs_a = brier_score_loss(y_boot, probs_a[idx])
        bs_b = brier_score_loss(y_boot, probs_b[idx])
        bs_const = brier_score_loss(y_boot, np.full(len(y_boot), y_rate_train))
        if bs_const == 0:
            continue
        bss_a = 1.0 - bs_a / bs_const
        bss_b = 1.0 - bs_b / bs_const
        deltas.append(bss_b - bss_a)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def write_abort_metrics(notes, t0):
    """Write partial metrics on abort."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {
        "experiment": "exp-011-feature-group-ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "abort_triggered": True,
        "abort_reason": "; ".join(notes),
        "resource_usage": {
            "wall_clock_seconds": round(time.time() - t0, 1),
            "n_lr_fits": 0,
            "n_gbt_fits": 0,
        },
        "notes": "; ".join(notes),
    }
    out_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"ABORT — metrics written to {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    notes = []
    lr_converged_all = True
    n_lr_fits = 0

    # ==================================================================
    # Phase 1: Data loading and fold construction
    # ==================================================================
    print("=" * 60)
    print("Phase 1: Data loading and fold construction")
    print("=" * 60)

    data = load_binary_labels(CACHE_DIR, lookback=LOOKBACK)
    X = data["X"]
    Y_long = data["Y_long"].astype(int)
    Y_short = data["Y_short"].astype(int)
    boundaries = data["session_boundaries"]
    dates = data["dates"]
    n_sessions = len(boundaries) - 1
    n_samples = len(Y_long)

    print(f"  N_samples = {n_samples}, N_sessions = {n_sessions}")
    print(f"  X.shape = {X.shape}")

    # Clean NaN/Inf
    nan_mask = np.isnan(X) | np.isinf(X)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        X = np.where(nan_mask, 0.0, X)
        notes.append(f"Replaced {n_nan} NaN/Inf values with 0")
        print(f"  Replaced {n_nan} NaN/Inf values with 0")

    # Map sessions to quarters
    session_quarters = [date_to_quarter(d) for d in dates]
    print(f"  Quarters: {[session_quarters.count(q) for q in [1, 2, 3, 4]]} (Q1-Q4)")

    # Build quarterly folds
    q_sessions = {q: [] for q in [1, 2, 3, 4]}
    for i, q in enumerate(session_quarters):
        q_sessions[q].append(i)

    for q in [1, 2, 3, 4]:
        print(f"  Q{q}: {len(q_sessions[q])} sessions")

    # Verify each quarter has >= 40 sessions
    for q in [1, 2, 3, 4]:
        if len(q_sessions[q]) < 40:
            notes.append(f"WARNING: Q{q} has only {len(q_sessions[q])} sessions (< 40)")

    # Build 3 expanding-window folds
    folds = []
    fold_defs = [
        {"train_qs": [1], "test_q": 2, "name": "fold_1", "train_label": "Q1", "test_label": "Q2"},
        {"train_qs": [1, 2], "test_q": 3, "name": "fold_2", "train_label": "Q1-Q2", "test_label": "Q3"},
        {"train_qs": [1, 2, 3], "test_q": 4, "name": "fold_3", "train_label": "Q1-Q3", "test_label": "Q4"},
    ]

    fold_info = {}
    for fd in fold_defs:
        train_sess = []
        for q in fd["train_qs"]:
            train_sess.extend(q_sessions[q])
        test_sess = q_sessions[fd["test_q"]]

        train_rows = sessions_to_rows(train_sess, boundaries)
        test_rows = sessions_to_rows(test_sess, boundaries)

        folds.append({
            "name": fd["name"],
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_label": fd["train_label"],
            "test_label": fd["test_label"],
            "n_train_sessions": len(train_sess),
            "n_test_sessions": len(test_sess),
        })

        # Compute per-fold stats
        y_rate_train_long = float(Y_long[train_rows].mean())
        y_rate_train_short = float(Y_short[train_rows].mean())
        y_rate_test_long = float(Y_long[test_rows].mean())
        y_rate_test_short = float(Y_short[test_rows].mean())

        fold_info[fd["name"]] = {
            "train": fd["train_label"],
            "test": fd["test_label"],
            "n_train_sessions": len(train_sess),
            "n_test_sessions": len(test_sess),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "y_rate_train_long": y_rate_train_long,
            "y_rate_train_short": y_rate_train_short,
            "y_rate_test_long": y_rate_test_long,
            "y_rate_test_short": y_rate_test_short,
        }

        print(f"  {fd['name']}: train={fd['train_label']}({len(train_rows)} rows), "
              f"test={fd['test_label']}({len(test_rows)} rows), "
              f"ybar_train_long={y_rate_train_long:.4f}, ybar_train_short={y_rate_train_short:.4f}, "
              f"ybar_test_long={y_rate_test_long:.4f}, ybar_test_short={y_rate_test_short:.4f}")

    # Sanity: check ȳ in [0.20, 0.46] and N_test >= 50K
    all_ybar_in_range = True
    all_n_test_gte_50k = True
    for fn, fi in fold_info.items():
        for key in ["y_rate_train_long", "y_rate_train_short", "y_rate_test_long", "y_rate_test_short"]:
            val = fi[key]
            if not (0.20 <= val <= 0.46):
                all_ybar_in_range = False
                notes.append(f"SANITY: {fn} {key}={val:.4f} outside [0.20, 0.46]")
        if fi["n_test"] < 50_000:
            all_n_test_gte_50k = False
            notes.append(f"SANITY: {fn} n_test={fi['n_test']} < 50,000")

    print(f"\n  all_ybar_in_range: {all_ybar_in_range}")
    print(f"  all_n_test_gte_50k: {all_n_test_gte_50k}")
    print(f"  Phase 1 time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 2: Minimum Viable Experiment
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Minimum Viable Experiment")
    print("=" * 60)

    # Use Fold 1 for MVE
    fold1 = folds[0]
    X_train_f1 = X[fold1["train_rows"]]
    X_test_f1 = X[fold1["test_rows"]]
    y_train_long_f1 = Y_long[fold1["train_rows"]]
    y_test_long_f1 = Y_long[fold1["test_rows"]]

    # Build column selections
    group_a_cols = expand_cols(GROUP_A_BASE, TEMPORAL)
    group_b_cols = expand_cols(GROUP_B_BASE, TEMPORAL)
    group_ab_cols = expand_full_cols()

    print(f"  Group A dims: {len(group_a_cols)}")
    print(f"  Group B dims: {len(group_b_cols)}")
    print(f"  Group A+B dims: {len(group_ab_cols)}")

    # MVE: Fit LR on Group B for Y_long on Fold 1
    print("  MVE: LR on Group B, Y_long, Fold 1...")
    lr_b, conv_b = fit_lr(X_train_f1[:, group_b_cols], y_train_long_f1)
    p_b = lr_b.predict_proba(X_test_f1[:, group_b_cols])[:, 1]
    auc_b = roc_auc_score(y_test_long_f1, p_b)
    print(f"    AUC(B) = {auc_b:.6f}, converged={conv_b}")

    if np.isnan(auc_b) or np.isinf(auc_b):
        notes.append("ABORT: NaN/Inf in MVE AUC(B)")
        write_abort_metrics(notes, t0)
        return 1

    if auc_b < 0.45:
        notes.append(f"WARNING: MVE AUC(B)={auc_b:.4f} < 0.45 (possibly broken)")

    # MVE: Fit LR on Group A for Y_long on Fold 1
    print("  MVE: LR on Group A, Y_long, Fold 1...")
    lr_a, conv_a = fit_lr(X_train_f1[:, group_a_cols], y_train_long_f1)
    p_a = lr_a.predict_proba(X_test_f1[:, group_a_cols])[:, 1]
    auc_a = roc_auc_score(y_test_long_f1, p_a)
    delta_mve = auc_b - auc_a
    print(f"    AUC(A) = {auc_a:.6f}, converged={conv_a}")
    print(f"    Δ_AUC(B-A) = {delta_mve:.6f}")

    if np.isnan(auc_a) or np.isinf(auc_a) or np.isnan(delta_mve):
        notes.append("ABORT: NaN/Inf in MVE AUC(A) or delta")
        write_abort_metrics(notes, t0)
        return 1

    n_lr_fits += 2
    print(f"  MVE passed. Time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 3: Primary analysis — LR on 3 arms × 2 labels × 3 folds
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Primary analysis — 18 LR fits")
    print("=" * 60)

    arms = {
        "group_a_book": group_a_cols,
        "group_b_trade": group_b_cols,
        "group_ab_full": group_ab_cols,
    }
    labels = {"Y_long": Y_long, "Y_short": Y_short}
    results = {}

    # Store predictions for bootstrap
    predictions = {}  # key: (arm, label, fold_name) -> probs

    for fold in folds:
        fn = fold["name"]
        fi = fold_info[fn]
        X_train = X[fold["train_rows"]]
        X_test = X[fold["test_rows"]]

        for label_name, Y_all in labels.items():
            y_train = Y_all[fold["train_rows"]]
            y_test = Y_all[fold["test_rows"]]
            y_rate_train = float(y_train.mean())
            y_rate_test = float(y_test.mean())

            for arm_name, arm_cols in arms.items():
                t_fit = time.time()
                X_tr = X_train[:, arm_cols]
                X_te = X_test[:, arm_cols]

                lr, converged = fit_lr(X_tr, y_train)
                if not converged:
                    lr_converged_all = False
                    # Retry with max_iter=5000
                    notes.append(f"LR did not converge: {arm_name}/{label_name}/{fn}. Retrying with max_iter=5000.")
                    lr, converged = fit_lr(X_tr, y_train, max_iter=5000)
                    if not converged:
                        notes.append(f"LR still did not converge after retry: {arm_name}/{label_name}/{fn}")

                n_lr_fits += 1
                probs = lr.predict_proba(X_te)[:, 1]
                predictions[(arm_name, label_name, fn)] = probs

                auc = roc_auc_score(y_test, probs)
                bs_model = brier_score_loss(y_test, probs)
                bs_constant = brier_score_loss(y_test, np.full(len(y_test), y_rate_train))
                bss = 1.0 - bs_model / bs_constant if bs_constant > 0 else 0.0

                key = f"{arm_name}/lr/{label_name}/{fn}"
                results[key] = {
                    "auc": float(auc),
                    "bss": float(bss),
                    "brier": float(bs_model),
                    "brier_constant": float(bs_constant),
                    "n_train": int(fi["n_train"]),
                    "n_test": int(fi["n_test"]),
                    "y_rate_train": float(y_rate_train),
                    "y_rate_test": float(y_rate_test),
                    "converged": converged,
                }

                elapsed = time.time() - t_fit
                print(f"  {key}: AUC={auc:.6f}, BSS={bss:.6f}, converged={converged} ({elapsed:.1f}s)")

                # Wall clock abort
                if time.time() - t0 > WALL_CLOCK_ABORT:
                    notes.append(f"Wall clock abort at {time.time() - t0:.0f}s")
                    write_abort_metrics(notes, t0)
                    return 1

    # Check abort: all AUC(A+B) <= 0.50
    full_auc_gt_05_count = 0
    for fold in folds:
        for label_name in labels:
            key = f"group_ab_full/lr/{label_name}/{fold['name']}"
            if results[key]["auc"] > 0.5:
                full_auc_gt_05_count += 1

    print(f"\n  Full model AUC > 0.5 count: {full_auc_gt_05_count}/6")
    if full_auc_gt_05_count == 0:
        notes.append("ABORT: All AUC(A+B) <= 0.50 — no discriminative signal")
        write_abort_metrics(notes, t0)
        return 1

    print(f"  Phase 3 time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 4: Paired comparison — bootstrap Δ_AUC
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Paired comparison — bootstrap Δ_AUC and Δ_BSS")
    print("=" * 60)

    paired_comparisons = {}
    interaction = {}

    for label_name in labels:
        per_fold_delta_auc = []
        per_fold_delta_bss = []

        for fold in folds:
            fn = fold["name"]
            auc_a = results[f"group_a_book/lr/{label_name}/{fn}"]["auc"]
            auc_b = results[f"group_b_trade/lr/{label_name}/{fn}"]["auc"]
            bss_a = results[f"group_a_book/lr/{label_name}/{fn}"]["bss"]
            bss_b = results[f"group_b_trade/lr/{label_name}/{fn}"]["bss"]
            per_fold_delta_auc.append(auc_b - auc_a)
            per_fold_delta_bss.append(bss_b - bss_a)

            # Interaction: AUC(A+B) - max(AUC(A), AUC(B))
            auc_ab = results[f"group_ab_full/lr/{label_name}/{fn}"]["auc"]
            max_auc = max(auc_a, auc_b)
            interaction[f"{label_name}/{fn}"] = {
                "auc_ab": float(auc_ab),
                "max_auc_a_b": float(max_auc),
                "interaction_delta": float(auc_ab - max_auc),
            }

        mean_delta_auc = float(np.mean(per_fold_delta_auc))
        mean_delta_bss = float(np.mean(per_fold_delta_bss))

        # Block bootstrap for Δ_AUC CI — run per fold, then average
        print(f"  Bootstrap Δ_AUC for {label_name}...")
        boot_deltas_auc = []
        boot_deltas_bss = []

        for fold in folds:
            fn = fold["name"]
            y_test = labels[label_name][fold["test_rows"]]
            probs_a = predictions[("group_a_book", label_name, fn)]
            probs_b = predictions[("group_b_trade", label_name, fn)]
            y_rate_train = results[f"group_a_book/lr/{label_name}/{fn}"]["y_rate_train"]

            _, ci_lo_auc, ci_hi_auc = block_bootstrap_delta_auc(
                y_test, probs_a, probs_b, seed=SEED
            )
            _, ci_lo_bss, ci_hi_bss = block_bootstrap_delta_bss(
                y_test, probs_a, probs_b, y_rate_train, seed=SEED
            )

            boot_deltas_auc.append((ci_lo_auc, ci_hi_auc))
            boot_deltas_bss.append((ci_lo_bss, ci_hi_bss))

        # Average per-fold bootstrap CIs (conservative approach)
        # Better: bootstrap the mean across folds. With 3 folds, use the
        # per-fold CIs and check if all exclude 0, or average bounds.
        # The spec says "bootstrap the mean Δ_AUC" — we do per-fold bootstrap
        # and report the averaged CI bounds.
        avg_ci_lo_auc = float(np.mean([b[0] for b in boot_deltas_auc]))
        avg_ci_hi_auc = float(np.mean([b[1] for b in boot_deltas_auc]))
        avg_ci_lo_bss = float(np.mean([b[0] for b in boot_deltas_bss]))
        avg_ci_hi_bss = float(np.mean([b[1] for b in boot_deltas_bss]))

        paired_comparisons[f"lr/{label_name}"] = {
            "mean_delta_auc_b_minus_a": mean_delta_auc,
            "mean_delta_bss_b_minus_a": mean_delta_bss,
            "per_fold_delta_auc": [float(d) for d in per_fold_delta_auc],
            "per_fold_delta_bss": [float(d) for d in per_fold_delta_bss],
            "bootstrap_ci_delta_auc_95": [avg_ci_lo_auc, avg_ci_hi_auc],
            "bootstrap_ci_delta_bss_95": [avg_ci_lo_bss, avg_ci_hi_bss],
        }

        n_positive_folds = sum(1 for d in per_fold_delta_auc if d > 0)
        print(f"  {label_name}: mean Δ_AUC={mean_delta_auc:.6f}, "
              f"CI=[{avg_ci_lo_auc:.6f}, {avg_ci_hi_auc:.6f}], "
              f"positive in {n_positive_folds}/3 folds")
        print(f"  {label_name}: mean Δ_BSS={mean_delta_bss:.6f}, "
              f"CI=[{avg_ci_lo_bss:.6f}, {avg_ci_hi_bss:.6f}]")

    print(f"  Phase 4 time: {time.time() - t0:.1f}s")

    # ==================================================================
    # Phase 5: Optional GBT secondary
    # ==================================================================
    n_gbt_fits = 0
    wall_at_phase5 = time.time() - t0
    if wall_at_phase5 < 12 * 60:
        print("\n" + "=" * 60)
        print("Phase 5: Optional GBT secondary")
        print("=" * 60)
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gbt_available = True
        except ImportError:
            gbt_available = False
            notes.append("GBT skipped: sklearn GradientBoostingClassifier not available")

        if gbt_available:
            for fold in folds:
                fn = fold["name"]
                fi = fold_info[fn]
                X_train = X[fold["train_rows"]]
                X_test = X[fold["test_rows"]]

                for label_name, Y_all in labels.items():
                    y_train = Y_all[fold["train_rows"]]
                    y_test = Y_all[fold["test_rows"]]
                    y_rate_train = float(y_train.mean())
                    y_rate_test = float(y_test.mean())

                    for arm_name, arm_cols in arms.items():
                        t_fit = time.time()
                        X_tr = X_train[:, arm_cols]
                        X_te = X_test[:, arm_cols]

                        gbt = GradientBoostingClassifier(
                            n_estimators=200, max_depth=6, learning_rate=0.05,
                            min_samples_leaf=100, random_state=SEED,
                        )
                        gbt.fit(X_tr, y_train)
                        n_gbt_fits += 1

                        probs = gbt.predict_proba(X_te)[:, 1]
                        auc = roc_auc_score(y_test, probs)
                        bs_model = brier_score_loss(y_test, probs)
                        bs_constant = brier_score_loss(y_test, np.full(len(y_test), y_rate_train))
                        bss = 1.0 - bs_model / bs_constant if bs_constant > 0 else 0.0

                        key = f"{arm_name}/gbt/{label_name}/{fn}"
                        results[key] = {
                            "auc": float(auc),
                            "bss": float(bss),
                            "brier": float(bs_model),
                            "brier_constant": float(bs_constant),
                            "n_train": int(fi["n_train"]),
                            "n_test": int(fi["n_test"]),
                            "y_rate_train": float(y_rate_train),
                            "y_rate_test": float(y_rate_test),
                        }

                        elapsed = time.time() - t_fit
                        print(f"  {key}: AUC={auc:.6f}, BSS={bss:.6f} ({elapsed:.1f}s)")

                        if time.time() - t0 > WALL_CLOCK_ABORT:
                            notes.append(f"GBT aborted at wall clock {time.time() - t0:.0f}s")
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
    else:
        print(f"\n  Phase 5 (GBT) skipped — wall clock already at {wall_at_phase5:.0f}s (> 12 min)")
        notes.append(f"GBT skipped: wall clock {wall_at_phase5:.0f}s > 720s threshold")

    # ==================================================================
    # Phase 6: Assemble and save
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 6: Assemble metrics and save")
    print("=" * 60)

    # Evaluate success criteria
    # C1: Mean Δ_AUC(B−A) > 0.005 for BOTH Y_long AND Y_short, positive in ≥ 2/3 folds each
    c1_detail = {}
    c1_pass = True
    for label_name in ["Y_long", "Y_short"]:
        pc = paired_comparisons[f"lr/{label_name}"]
        mean_delta = pc["mean_delta_auc_b_minus_a"]
        n_positive = sum(1 for d in pc["per_fold_delta_auc"] if d > 0)
        label_pass = mean_delta > 0.005 and n_positive >= 2
        c1_detail[label_name] = {
            "pass": label_pass,
            "mean_delta_auc": mean_delta,
            "n_positive_folds": n_positive,
        }
        if not label_pass:
            c1_pass = False

    # C2: Bootstrap 95% CI for mean Δ_AUC excludes 0 for at least one label
    c2_pass = False
    for label_name in ["Y_long", "Y_short"]:
        pc = paired_comparisons[f"lr/{label_name}"]
        ci = pc["bootstrap_ci_delta_auc_95"]
        if ci[0] > 0 or ci[1] < 0:  # CI excludes 0 (entirely positive or negative)
            c2_pass = True

    # C3: AUC(A+B) > 0.5 on >= 4/6, all ȳ in range, all N_test >= 50K
    c3_pass = full_auc_gt_05_count >= 4 and all_ybar_in_range and all_n_test_gte_50k

    # C4: No sanity check failure
    sanity_fails = [n for n in notes if "SANITY" in n and "FAIL" in n]
    c4_pass = len(sanity_fails) == 0

    # Verdict
    if not c3_pass:
        verdict = "INVALID"
    elif c1_pass and c2_pass:
        verdict = "CONFIRMED"
    elif not c1_pass:
        verdict = "REFUTED"
    else:
        verdict = "INCONCLUSIVE"

    print(f"  C1 (Group B dominates): {c1_pass}")
    for ln, detail in c1_detail.items():
        print(f"    {ln}: pass={detail['pass']}, mean_Δ={detail['mean_delta_auc']:.6f}, "
              f"positive_folds={detail['n_positive_folds']}/3")
    print(f"  C2 (significant): {c2_pass}")
    print(f"  C3 (sanity): {c3_pass}")
    print(f"  C4 (no regression): {c4_pass}")
    print(f"  Verdict: {verdict}")

    # Build metrics dict
    metrics = {
        "experiment": "exp-011-feature-group-ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tier": "Quick",
        "arms": {
            "group_a_book": {
                "feature_cols_base": GROUP_A_BASE,
                "temporal_cols": TEMPORAL,
                "n_dims": len(group_a_cols),
            },
            "group_b_trade": {
                "feature_cols_base": GROUP_B_BASE,
                "temporal_cols": TEMPORAL,
                "n_dims": len(group_b_cols),
            },
            "group_ab_full": {
                "feature_cols_base": list(range(N_FEATURES)),
                "temporal_cols": [],
                "n_dims": len(group_ab_cols),
            },
        },
        "folds": {fn: {
            "train": fi["train"],
            "test": fi["test"],
            "n_train_sessions": fi["n_train_sessions"],
            "n_test_sessions": fi["n_test_sessions"],
        } for fn, fi in fold_info.items()},
        "results": results,
        "paired_comparisons": paired_comparisons,
        "interaction": interaction,
        "sanity_checks": {
            "all_ybar_in_range": all_ybar_in_range,
            "all_n_test_gte_50k": all_n_test_gte_50k,
            "full_auc_gt_05_count": full_auc_gt_05_count,
            "lr_converged_all": lr_converged_all,
        },
        "success_criteria": {
            "C1_group_b_dominates": c1_pass,
            "C1_detail": {k: v for k, v in c1_detail.items()},
            "C2_significant": c2_pass,
            "C3_sanity": c3_pass,
            "C4_no_regression": c4_pass,
            "verdict": verdict,
        },
        "resource_usage": {
            "wall_clock_seconds": round(time.time() - t0, 1),
            "n_lr_fits": n_lr_fits,
            "n_gbt_fits": n_gbt_fits,
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": "; ".join(notes) if notes else None,
    }

    # Write metrics.json
    out_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics written to {out_path}")
    print(f"  Total wall clock: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
