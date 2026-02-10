"""Regime-switch validation module.

Validates the label + normalization pipeline against a synthetic volatility
regime switch. Ensures the pipeline preserves regime information and that
trailing normalization does not smooth over regime transitions pathologically.
"""

import numpy as np
from scipy import stats

from lob_rl.barrier import TICK_SIZE, RTH_OPEN_NS, RTH_DURATION_NS, build_synthetic_trades
from lob_rl.barrier.bar_pipeline import build_bars_from_trades
from lob_rl.barrier.label_pipeline import compute_labels
from lob_rl.barrier.feature_pipeline import compute_bar_features, normalize_features


def generate_regime_switch_trades(n_bars_low=5000, n_bars_high=5000, bar_size=500,
                                   p=0.5, start_price=4000.0, tick_size=0.25,
                                   seed=None):
    """Generate synthetic trade data with a known volatility regime switch.

    Low-vol regime: tick increments drawn from {-1, +1} with probability p for uptick.
    High-vol regime: tick increments drawn from {-3, -2, -1, +1, +2, +3} uniform.

    Parameters
    ----------
    n_bars_low : int
        Number of bars in the low-vol regime. Default 5000.
    n_bars_high : int
        Number of bars in the high-vol regime. Default 5000.
    bar_size : int
        Trades per bar. Default 500.
    p : float
        Uptick probability for low-vol regime. Default 0.5.
    start_price : float
        Starting price. Default 4000.0.
    tick_size : float
        Tick size. Default 0.25.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, list[TradeBar]]
        Structured numpy array of trades with fields (price, size, side, ts_event),
        and list of TradeBar objects.
    """
    rng = np.random.default_rng(seed)

    n_low_trades = n_bars_low * bar_size
    n_high_trades = n_bars_high * bar_size
    n_total = n_low_trades + n_high_trades

    # Generate tick increments
    # Low-vol: +1 or -1 with probability p for uptick
    low_moves = np.where(rng.random(n_low_trades - 1) < p, 1, -1)

    # High-vol: uniform from {-3, -2, -1, +1, +2, +3}
    high_choices = np.array([-3, -2, -1, 1, 2, 3])
    high_moves = rng.choice(high_choices, size=n_high_trades)

    # Build price series via cumsum (vectorized)
    all_moves = np.concatenate([low_moves, high_moves])
    prices = np.empty(n_total, dtype=np.float64)
    prices[0] = start_price
    prices[1:] = start_price + np.cumsum(all_moves) * tick_size

    # Build timestamps: evenly spaced within a single RTH session
    ts = np.linspace(RTH_OPEN_NS, RTH_OPEN_NS + RTH_DURATION_NS - 1,
                     n_total, dtype=np.int64)

    trades = build_synthetic_trades(prices, ts)
    bars = build_bars_from_trades(trades, n=bar_size, session_date="2022-06-15")

    return trades, bars


def compute_segment_stats(labels, start, end):
    """Compute label statistics for a contiguous segment of labels.

    Parameters
    ----------
    labels : list[BarrierLabel]
        Full list of barrier labels.
    start : int
        Start index (inclusive).
    end : int
        End index (exclusive).

    Returns
    -------
    dict
        Keys: p_plus, p_minus, p_zero, mean_tau, median_tau.
    """
    segment = labels[start:end]
    n = len(segment)

    if n == 0:
        return {
            "p_plus": 0.0,
            "p_minus": 0.0,
            "p_zero": 0.0,
            "mean_tau": 0.0,
            "median_tau": 0.0,
        }

    # Extract into arrays for vectorized counting
    label_arr = np.array([lbl.label for lbl in segment])
    tau_arr = np.array([lbl.tau for lbl in segment])

    return {
        "p_plus": float(np.sum(label_arr == 1) / n),
        "p_minus": float(np.sum(label_arr == -1) / n),
        "p_zero": float(np.sum(label_arr == 0) / n),
        "mean_tau": float(np.mean(tau_arr)),
        "median_tau": float(np.median(tau_arr)),
    }


def ks_test_features(features, boundary, window=500):
    """Run Kolmogorov-Smirnov tests on feature distributions before vs after boundary.

    Parameters
    ----------
    features : np.ndarray
        Raw (unnormalized) feature array, shape (N, 13).
    boundary : int
        Bar index of regime boundary.
    window : int
        Number of bars before/after boundary to compare. Default 500.

    Returns
    -------
    dict
        KS test p-values for each feature column. Keys: ks_p_col_0 through ks_p_col_12.
    """
    pre_start = max(0, boundary - window)
    pre = features[pre_start:boundary]
    post_end = min(len(features), boundary + window)
    post = features[boundary:post_end]

    result = {}
    for col in range(features.shape[1]):
        pre_col = pre[:, col]
        post_col = post[:, col]

        # Remove NaN values
        pre_col = pre_col[~np.isnan(pre_col)]
        post_col = post_col[~np.isnan(post_col)]

        if len(pre_col) > 0 and len(post_col) > 0:
            _, p_value = stats.ks_2samp(pre_col, post_col)
        else:
            p_value = 1.0

        result[f"ks_p_col_{col}"] = float(p_value)

    return result


def measure_normalization_adaptation(normed_features, boundary, col=8,
                                      threshold_sigma=1.0):
    """Measure how many bars after the regime boundary for normalized features to adapt.

    Parameters
    ----------
    normed_features : np.ndarray
        Normalized feature array, shape (N, 13).
    boundary : int
        Bar index of regime boundary.
    col : int
        Feature column to check. Default 8 (realized vol).
    threshold_sigma : float
        How close to the post-boundary mean (in post-boundary stddevs).

    Returns
    -------
    int
        Number of bars after boundary until adapted. -1 if never adapts.
    """
    n = len(normed_features)

    # Compute post-boundary steady-state from bars boundary+500 onward
    steady_start = boundary + 500
    if steady_start >= n:
        return -1

    steady_state = normed_features[steady_start:, col]
    steady_mean = np.mean(steady_state)
    steady_std = np.std(steady_state)

    if steady_std == 0:
        steady_std = 1.0  # avoid division by zero

    # Scan from boundary forward, find first bar within threshold_sigma of steady_mean
    max_check = min(boundary + 500, n)
    for i in range(boundary, max_check):
        val = normed_features[i, col]
        if abs(val - steady_mean) <= threshold_sigma * steady_std:
            return i - boundary

    return -1


def validate_regime_switch(n_bars_low=5000, n_bars_high=5000, bar_size=500,
                            a=100, b=100, t_max=40, seed=42):
    """Run the full regime-switch validation.

    Parameters
    ----------
    n_bars_low, n_bars_high, bar_size : int
        Forwarded to generate_regime_switch_trades.
    a, b, t_max : int
        Forwarded to compute_labels.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Full validation results.
    """
    trades, bars = generate_regime_switch_trades(
        n_bars_low=n_bars_low, n_bars_high=n_bars_high,
        bar_size=bar_size, seed=seed,
    )

    # Compute labels
    labels = compute_labels(bars, a=a, b=b, t_max=t_max)

    # Compute segment stats
    boundary = n_bars_low
    low_stats = compute_segment_stats(labels, start=0, end=boundary)
    high_stats = compute_segment_stats(labels, start=boundary, end=len(labels))

    # Timeout ratio
    high_p_zero = high_stats["p_zero"]
    timeout_ratio = low_stats["p_zero"] / max(high_p_zero, 1e-10)

    # Chi-squared test on label distributions
    low_counts = np.array([
        low_stats["p_plus"] * boundary,
        low_stats["p_minus"] * boundary,
        low_stats["p_zero"] * boundary,
    ])
    n_high = len(labels) - boundary
    high_counts = np.array([
        high_stats["p_plus"] * n_high,
        high_stats["p_minus"] * n_high,
        high_stats["p_zero"] * n_high,
    ])
    # Build contingency table
    contingency = np.array([low_counts, high_counts])
    # Filter out columns where both rows are zero
    nonzero_cols = contingency.sum(axis=0) > 0
    contingency_filtered = contingency[:, nonzero_cols]
    if contingency_filtered.shape[1] >= 2:
        chi2, chi2_p, _, _ = stats.chi2_contingency(contingency_filtered)
    else:
        chi2_p = 1.0

    # Compute features
    features = compute_bar_features(bars)

    # KS tests on features at boundary
    ks_results = ks_test_features(features, boundary, window=500)
    ks_bar_range_p = ks_results.get("ks_p_col_3", 1.0)
    ks_realized_vol_p = ks_results.get("ks_p_col_8", 1.0)

    # Normalization adaptation
    normed = normalize_features(features)
    norm_adaptation_bars = measure_normalization_adaptation(
        normed, boundary, col=8, threshold_sigma=1.0,
    )

    # Determine overall pass
    passed = (
        low_stats["p_zero"] > high_stats["p_zero"]      # Test 8
        and timeout_ratio > 2.0                           # Test 9
        and low_stats["mean_tau"] > high_stats["mean_tau"]  # Test 10
        and chi2_p < 0.01                                 # Test 11
        and ks_bar_range_p < 0.01                         # Test 14
        and ks_realized_vol_p < 0.01                      # Test 15
        and 0 <= norm_adaptation_bars < 500               # Test 18
    )

    return {
        "n_bars_total": n_bars_low + n_bars_high,
        "n_bars_low": n_bars_low,
        "n_bars_high": n_bars_high,
        "boundary_bar": boundary,
        "low_vol": low_stats,
        "high_vol": high_stats,
        "timeout_ratio": float(timeout_ratio),
        "chi2_p_value": float(chi2_p),
        "ks_bar_range_p": float(ks_bar_range_p),
        "ks_realized_vol_p": float(ks_realized_vol_p),
        "norm_adaptation_bars": int(norm_adaptation_bars),
        "pass": passed,
    }
