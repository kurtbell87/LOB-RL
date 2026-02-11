"""First-passage analysis module — Phase 1 & 2 of Asymmetric First-Passage Trading.

Provides Brier scores, temporal splits, bootstrap tests, calibration curves,
model fitting, null calibration, signal detection, and label loading.
"""

import glob
import os

import numpy as np


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score: mean((y_true - y_pred)^2)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def constant_brier(y_true: np.ndarray) -> float:
    """Brier score of the constant predictor ybar = mean(y_true)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    ybar = y_true.mean()
    return float(ybar * (1.0 - ybar))


def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """BSS = 1 - Brier(model) / Brier(constant).

    Returns 0.0 when constant Brier is 0 (all labels identical).
    """
    bs = brier_score(y_true, y_pred)
    bs_const = constant_brier(y_true)
    if bs_const == 0.0:
        return 0.0
    return float(1.0 - bs / bs_const)


# ---------------------------------------------------------------------------
# Temporal Splits
# ---------------------------------------------------------------------------

def temporal_split(
    n_sessions: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split session indices temporally (chronological order)."""
    idx = np.arange(n_sessions)
    n_train = max(1, round(n_sessions * train_frac))
    n_val = round(n_sessions * val_frac)
    if n_sessions == 1:
        return idx[:1], np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    # Ensure test gets at least 0, and we don't exceed n_sessions
    n_train = min(n_train, n_sessions)
    n_val = min(n_val, n_sessions - n_train)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def temporal_cv_folds(
    n_sessions: int, n_folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal CV folds.

    Fold k: train on [0..split_k-1], validate on [split_k..split_{k+1}-1].
    Divides into n_folds+1 segments; fold k trains on segments 0..k, validates on k+1.
    """
    # n_folds+2 boundaries → n_folds+1 segments.
    # Fold k: train = segments [0..k] = [0, splits[k+1]),
    #         val   = segment [k+1]   = [splits[k+1], splits[k+2])
    splits = np.linspace(0, n_sessions, n_folds + 2, dtype=np.int64)
    folds = []
    for k in range(n_folds):
        train_end = int(splits[k + 1])
        val_start = train_end
        val_end = int(splits[k + 2])
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Calibration Curve
# ---------------------------------------------------------------------------

def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin predictions into equal-width bins, return (mean_predicted, fraction_positive)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_preds = []
    frac_pos = []
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        # Include right edge in last bin
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        mean_preds.append(float(y_pred[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))

    return np.array(mean_preds), np.array(frac_pos)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_brier(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    n_boot: int = 10000,
    block_size: int = 50,
    seed: int = 42,
) -> dict:
    """Block bootstrap test for Brier score difference.

    delta = Brier(baseline) - Brier(model). Positive means model is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    pred_model = np.asarray(pred_model, dtype=np.float64)
    pred_baseline = np.asarray(pred_baseline, dtype=np.float64)

    n = len(y_true)
    sq_model = (y_true - pred_model) ** 2
    sq_baseline = (y_true - pred_baseline) ** 2
    diff = sq_baseline - sq_model  # positive = model better

    observed_delta = float(diff.mean())

    rng = np.random.default_rng(seed)
    n_blocks = max(1, (n + block_size - 1) // block_size)
    boot_deltas = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        # Circular block bootstrap
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [np.arange(s, s + block_size) % n for s in starts]
        )[:n]
        boot_deltas[b] = diff[indices].mean()

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_value = float(np.mean(boot_deltas <= 0))

    return {
        "delta": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Phase 1: Null Calibration
# ---------------------------------------------------------------------------

def null_calibration_report(
    Y_long: np.ndarray,
    Y_short: np.ndarray,
    tau_long: np.ndarray,
    tau_short: np.ndarray,
    timeout_long: np.ndarray,
    timeout_short: np.ndarray,
    session_boundaries: np.ndarray,
) -> dict:
    """Compute Phase 1 null calibration statistics."""
    Y_long = np.asarray(Y_long, dtype=bool)
    Y_short = np.asarray(Y_short, dtype=bool)
    tau_long = np.asarray(tau_long)
    tau_short = np.asarray(tau_short)
    timeout_long = np.asarray(timeout_long, dtype=bool)
    timeout_short = np.asarray(timeout_short, dtype=bool)
    session_boundaries = np.asarray(session_boundaries, dtype=np.int64)

    N = len(Y_long)
    n_sessions = len(session_boundaries) - 1

    ybar_long = float(Y_long.mean())
    ybar_short = float(Y_short.mean())
    sum_ybar = ybar_long + ybar_short

    timeout_rate_long = float(timeout_long.mean())
    timeout_rate_short = float(timeout_short.mean())

    mean_tau_long = float(tau_long.mean())
    mean_tau_short = float(tau_short.mean())

    # Joint distribution
    joint = {}
    for vl in [False, True]:
        for vs in [False, True]:
            mask = (Y_long == vl) & (Y_short == vs)
            joint[(int(vl), int(vs))] = int(mask.sum())

    # Rolling ybar per session
    rolling_long = np.empty(n_sessions)
    rolling_short = np.empty(n_sessions)
    for k in range(n_sessions):
        start = session_boundaries[k]
        end = session_boundaries[k + 1]
        rolling_long[k] = Y_long[start:end].mean()
        rolling_short[k] = Y_short[start:end].mean()

    # Standard error via session-level block bootstrap
    se_long = float(np.std(rolling_long, ddof=1) / np.sqrt(n_sessions)) if n_sessions > 1 else 0.0
    se_short = float(np.std(rolling_short, ddof=1) / np.sqrt(n_sessions)) if n_sessions > 1 else 0.0

    # Gate check
    ybar_ok = 0.28 <= ybar_long <= 0.38 and 0.28 <= ybar_short <= 0.38
    sum_ok = 0.58 <= sum_ybar <= 0.72
    timeout_ok = timeout_rate_long < 0.05 and timeout_rate_short < 0.05
    stability_ok = all(0.20 <= v <= 0.46 for v in rolling_long) and \
                   all(0.20 <= v <= 0.46 for v in rolling_short)
    gate_passed = ybar_ok and sum_ok and timeout_ok and stability_ok

    return {
        "ybar_long": ybar_long,
        "ybar_short": ybar_short,
        "se_long": se_long,
        "se_short": se_short,
        "sum_ybar": sum_ybar,
        "timeout_rate_long": timeout_rate_long,
        "timeout_rate_short": timeout_rate_short,
        "mean_tau_long": mean_tau_long,
        "mean_tau_short": mean_tau_short,
        "joint_distribution": joint,
        "rolling_ybar_long": rolling_long,
        "rolling_ybar_short": rolling_short,
        "gate_passed": gate_passed,
    }


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, max_iter: int = 1000):
    """Fit L2-regularized logistic regression."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def fit_gbt(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """Fit gradient-boosted tree classifier. LightGBM preferred, sklearn fallback."""
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=seed,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(
            max_depth=6,
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=50,
            random_state=seed,
        )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Label Loading
# ---------------------------------------------------------------------------

def load_binary_labels(cache_dir: str, lookback: int = 10) -> dict:
    """Load all barrier cache .npz files and extract binary labels + features."""
    npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No .npz files found in {cache_dir}")

    X_list = []
    Y_long_list = []
    Y_short_list = []
    timeout_long_list = []
    timeout_short_list = []
    tau_long_list = []
    tau_short_list = []
    boundaries = [0]
    dates = []

    for path in npz_files:
        data = np.load(path, allow_pickle=True)

        # Skip sessions without required keys
        if "features" not in data or "short_label_values" not in data:
            continue

        n_usable = int(data["n_usable"])
        n_bars = int(data["n_bars"])
        features = data["features"]  # (n_usable, feature_dim)

        # Labels are for all n_bars; take the last n_usable entries
        # (the first lookback-1 bars are consumed by the lookback window)
        offset = n_bars - n_usable
        label_values = data["label_values"][offset:]
        label_tau = data["label_tau"][offset:]

        short_label_values = data["short_label_values"][offset:]
        short_label_tau = data["short_label_tau"][offset:]

        # Binary labels
        Y_long = (label_values == 1)
        Y_short = (short_label_values == -1)

        # Timeout detection: label == 0 (pre-bias caches)
        timeout_long = (label_values == 0)
        timeout_short = (short_label_values == 0)

        X_list.append(features)
        Y_long_list.append(Y_long.astype(bool))
        Y_short_list.append(Y_short.astype(bool))
        timeout_long_list.append(timeout_long.astype(bool))
        timeout_short_list.append(timeout_short.astype(bool))
        tau_long_list.append(label_tau.astype(np.int32))
        tau_short_list.append(short_label_tau.astype(np.int32))
        boundaries.append(boundaries[-1] + n_usable)

        # Extract date from filename
        basename = os.path.basename(path)
        # Try to extract YYYYMMDD from filename
        date_str = basename.replace(".npz", "").split("-")[-1]
        dates.append(date_str)

    if not X_list:
        raise ValueError(f"No valid sessions found in {cache_dir}")

    return {
        "X": np.concatenate(X_list, axis=0),
        "Y_long": np.concatenate(Y_long_list),
        "Y_short": np.concatenate(Y_short_list),
        "timeout_long": np.concatenate(timeout_long_list),
        "timeout_short": np.concatenate(timeout_short_list),
        "tau_long": np.concatenate(tau_long_list),
        "tau_short": np.concatenate(tau_short_list),
        "session_boundaries": np.array(boundaries, dtype=np.int64),
        "dates": dates,
    }


# ---------------------------------------------------------------------------
# Lattice Verification
# ---------------------------------------------------------------------------

def verify_lattice(tick_size: float = 0.25, a: int = 20, b: int = 10) -> dict:
    """Verify barrier distances are on the tick lattice."""
    # a and b are in ticks. The price distances are a * tick_size and b * tick_size.
    # Lattice check: these price distances must be exact multiples of tick_size,
    # which is trivially true when a and b are integers.
    a_price = a * tick_size
    b_price = b * tick_size

    # Check if a_price / tick_size and b_price / tick_size are integers
    a_ticks = a_price / tick_size
    b_ticks = b_price / tick_size
    lattice_ok = (abs(a_ticks - round(a_ticks)) < 1e-9 and
                  abs(b_ticks - round(b_ticks)) < 1e-9)

    return {
        "lattice_ok": lattice_ok,
        "R_ticks": b,  # Risk barrier in ticks for long
    }


# ---------------------------------------------------------------------------
# Phase 2: Signal Detection
# ---------------------------------------------------------------------------

def signal_detection_report(
    X: np.ndarray,
    Y_long: np.ndarray,
    Y_short: np.ndarray,
    session_boundaries: np.ndarray,
    seed: int = 42,
) -> dict:
    """Run Phase 2 signal detection analysis."""
    session_boundaries = np.asarray(session_boundaries, dtype=np.int64)
    n_sessions = len(session_boundaries) - 1
    Y_long = np.asarray(Y_long, dtype=bool)
    Y_short = np.asarray(Y_short, dtype=bool)

    # 1. Temporal split by session
    train_sess, val_sess, test_sess = temporal_split(n_sessions)

    # Build row-level indices from session boundaries
    def sess_to_rows(sess_idx):
        rows = []
        for s in sess_idx:
            rows.append(np.arange(session_boundaries[s], session_boundaries[s + 1]))
        return np.concatenate(rows) if rows else np.array([], dtype=np.int64)

    train_rows = sess_to_rows(train_sess)
    val_rows = sess_to_rows(val_sess)

    X_train = X[train_rows]
    X_val = X[val_rows]

    # 2. 5-fold temporal CV within training sessions
    cv_folds = temporal_cv_folds(len(train_sess), n_folds=5)

    report = {}

    for label_name, Y in [("long", Y_long), ("short", Y_short)]:
        y_train = Y[train_rows].astype(int)
        y_val = Y[val_rows].astype(int)

        # Constant baseline on val
        ybar = y_train.mean()
        pred_const = np.full(len(y_val), ybar)
        brier_const = brier_score(y_val, pred_const)
        report[f"brier_constant_{label_name}"] = brier_const

        # Fit logistic on full train, predict val
        model_lr = fit_logistic(X_train, y_train)
        pred_lr = model_lr.predict_proba(X_val)[:, 1]
        brier_lr = brier_score(y_val, pred_lr)
        report[f"brier_logistic_{label_name}"] = brier_lr
        report[f"bss_logistic_{label_name}"] = brier_skill_score(y_val, pred_lr)
        report[f"max_pred_logistic_{label_name}"] = float(pred_lr.max())
        report[f"calibration_logistic_{label_name}"] = calibration_curve(y_val, pred_lr)

        # Fit GBT on full train, predict val
        model_gbt = fit_gbt(X_train, y_train, seed=seed)
        pred_gbt = model_gbt.predict_proba(X_val)[:, 1]
        brier_gbt = brier_score(y_val, pred_gbt)
        report[f"brier_gbt_{label_name}"] = brier_gbt
        report[f"bss_gbt_{label_name}"] = brier_skill_score(y_val, pred_gbt)
        report[f"max_pred_gbt_{label_name}"] = float(pred_gbt.max())
        report[f"calibration_gbt_{label_name}"] = calibration_curve(y_val, pred_gbt)

        # Bootstrap tests
        report[f"delta_logistic_{label_name}"] = paired_bootstrap_brier(
            y_val, pred_lr, pred_const, n_boot=1000, seed=seed
        )
        report[f"delta_gbt_{label_name}"] = paired_bootstrap_brier(
            y_val, pred_gbt, pred_const, n_boot=1000, seed=seed
        )

        # CV Brier scores
        cv_brier_lr = []
        cv_brier_gbt = []
        for cv_train_sess, cv_val_sess in cv_folds:
            # Map CV session indices to actual training session indices
            cv_tr_rows = []
            for s in cv_train_sess:
                actual_s = train_sess[s]
                cv_tr_rows.append(np.arange(
                    session_boundaries[actual_s],
                    session_boundaries[actual_s + 1],
                ))
            cv_vl_rows = []
            for s in cv_val_sess:
                actual_s = train_sess[s]
                cv_vl_rows.append(np.arange(
                    session_boundaries[actual_s],
                    session_boundaries[actual_s + 1],
                ))
            if not cv_tr_rows or not cv_vl_rows:
                continue
            cv_tr = np.concatenate(cv_tr_rows)
            cv_vl = np.concatenate(cv_vl_rows)

            X_cv_tr, y_cv_tr = X[cv_tr], Y[cv_tr].astype(int)
            X_cv_vl, y_cv_vl = X[cv_vl], Y[cv_vl].astype(int)

            m_lr = fit_logistic(X_cv_tr, y_cv_tr)
            p_lr = m_lr.predict_proba(X_cv_vl)[:, 1]
            cv_brier_lr.append(float(brier_score(y_cv_vl, p_lr)))

            m_gbt = fit_gbt(X_cv_tr, y_cv_tr, seed=seed)
            p_gbt = m_gbt.predict_proba(X_cv_vl)[:, 1]
            cv_brier_gbt.append(float(brier_score(y_cv_vl, p_gbt)))

        report[f"cv_brier_logistic_{label_name}"] = cv_brier_lr
        report[f"cv_brier_gbt_{label_name}"] = cv_brier_gbt

    # Profitability bound
    report["profitability_bound"] = {
        "threshold": 0.40,
        "note": "p > 0.40 needed for C=2, R=10",
    }

    # Signal found: any (model, label) has delta > 0 with CI excluding 0
    signal_found = False
    for label_name in ["long", "short"]:
        for model_name in ["logistic", "gbt"]:
            d = report[f"delta_{model_name}_{label_name}"]
            if d["delta"] > 0 and d["ci_lower"] > 0:
                signal_found = True
    report["signal_found"] = signal_found

    return report
