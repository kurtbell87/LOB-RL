"""Gambler's ruin validation module.

Validates the label construction pipeline (T2) against analytic Gambler's ruin
results by generating synthetic random walk trade data, running it through the
bar and label pipelines, and comparing empirical barrier-hit frequencies to
the closed-form solution.
"""

import math

import numpy as np

from lob_rl.barrier import TICK_SIZE, RTH_OPEN_NS, RTH_DURATION_NS, build_synthetic_trades
from lob_rl.barrier.bar_pipeline import build_bars_from_trades
from lob_rl.barrier.label_pipeline import compute_labels


def gamblers_ruin_analytic(a, b, p):
    """Compute analytic P(hit upper barrier first) in a discrete random walk.

    Parameters
    ----------
    a : int
        Upper barrier distance in ticks.
    b : int
        Lower barrier distance in ticks.
    p : float
        Probability of an uptick (0 < p < 1).

    Returns
    -------
    float
        P(hit upper first).
    """
    if p == 0.5:
        return float(b) / float(a + b)
    q = 1.0 - p
    r = q / p
    numerator = 1.0 - r ** b
    denominator = 1.0 - r ** (a + b)
    return float(numerator / denominator)


def generate_random_walk(n_trades, p=0.5, start_price=4000.0, tick_size=0.25,
                         seed=None):
    """Generate synthetic trade data from a discrete random walk.

    Parameters
    ----------
    n_trades : int
        Number of trades to generate.
    p : float
        Probability of an uptick. Default 0.5.
    start_price : float
        Starting price. Default 4000.0.
    tick_size : float
        Tick size. Default 0.25.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Structured array with fields: price, size, side, ts_event.
    """
    rng = np.random.default_rng(seed)

    # Generate uptick/downtick decisions
    directions = np.where(rng.random(n_trades - 1) < p, 1, -1)

    # Build price series via cumsum (vectorized)
    prices = np.empty(n_trades, dtype=np.float64)
    prices[0] = start_price
    prices[1:] = start_price + np.cumsum(directions) * tick_size

    # Build timestamps and structured trades array
    ts = np.linspace(RTH_OPEN_NS, RTH_OPEN_NS + RTH_DURATION_NS - 1,
                     n_trades, dtype=np.int64)

    return build_synthetic_trades(prices, ts)


def validate_drift_level(p, a=20, b=10, n_bars=10000, bar_size=500, t_max=40,
                         seed=None):
    """Run full validation for a single drift level.

    Generates a random walk, builds bars, computes labels, and compares
    empirical barrier-hit frequencies to the analytic formula.

    Parameters
    ----------
    p : float
        Uptick probability.
    a : int
        Upper barrier in ticks. Default 20.
    b : int
        Lower barrier in ticks. Default 10.
    n_bars : int
        Minimum number of bars to produce. Default 10000.
    bar_size : int
        Trades per bar. Default 500.
    t_max : int
        Maximum holding period in bars. Default 40.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Keys: p, analytic, empirical, n_labeled, n_upper, n_lower, n_timeout,
        se, z_score, pass.
    """
    # Generate enough trades to produce at least n_bars bars
    # Need extra bars for forward-looking labels
    n_trades = (n_bars + t_max + 10) * bar_size
    trades = generate_random_walk(n_trades, p=p, seed=seed)

    # Build bars (trades array has a 'timestamp' field from build_synthetic_trades)
    bars = build_bars_from_trades(trades, n=bar_size)

    # Compute labels
    labels = compute_labels(bars, a=a, b=b, t_max=t_max, direction="long")

    # Only count labels that had enough forward bars for resolution
    # (exclude the last t_max bars which may have forced timeouts)
    usable = labels[:n_bars] if len(labels) >= n_bars else labels

    n_labeled = len(usable)
    n_upper = sum(1 for lbl in usable if lbl.label == 1)
    n_lower = sum(1 for lbl in usable if lbl.label == -1)
    n_timeout = sum(1 for lbl in usable if lbl.label == 0)

    # Compute empirical proportion
    # P(upper) among barrier hits (exclude timeouts for the comparison)
    barrier_hits = n_upper + n_lower
    if barrier_hits > 0:
        empirical = n_upper / barrier_hits
    else:
        empirical = 0.0

    # Analytic value
    analytic = gamblers_ruin_analytic(a, b, p)

    # Standard error of proportion
    if barrier_hits > 0:
        se = math.sqrt(empirical * (1.0 - empirical) / barrier_hits)
    else:
        se = 1.0  # avoid division by zero

    # Z-score
    if se > 0:
        z_score = (empirical - analytic) / se
    else:
        z_score = 0.0

    # Pass/fail
    passed = abs(z_score) <= 2.0

    return {
        "p": float(p),
        "analytic": float(analytic),
        "empirical": float(empirical),
        "n_labeled": int(n_labeled),
        "n_upper": int(n_upper),
        "n_lower": int(n_lower),
        "n_timeout": int(n_timeout),
        "se": float(se),
        "z_score": float(z_score),
        "pass": bool(passed),
    }


def run_validation(drift_levels=None, a=20, b=10, n_bars=10000, bar_size=500,
                   t_max=40, seed=42):
    """Run validation across multiple drift levels.

    Parameters
    ----------
    drift_levels : list[float], optional
        Uptick probabilities. Default [0.500, 0.505, 0.510, 0.490, 0.485].
    a : int
        Upper barrier in ticks. Default 20.
    b : int
        Lower barrier in ticks. Default 10.
    n_bars : int
        Minimum bars per drift level. Default 10000.
    bar_size : int
        Trades per bar. Default 500.
    t_max : int
        Maximum holding period. Default 40.
    seed : int
        Base random seed. Default 42.

    Returns
    -------
    list[dict]
        One result dict per drift level.
    """
    if drift_levels is None:
        drift_levels = [0.500, 0.505, 0.510, 0.490, 0.485]

    results = []
    for i, p in enumerate(drift_levels):
        # Use different seed per drift level for independence,
        # but deterministic from the base seed
        level_seed = seed + i if seed is not None else None
        result = validate_drift_level(
            p, a=a, b=b, n_bars=n_bars, bar_size=bar_size, t_max=t_max,
            seed=level_seed,
        )
        results.append(result)

    return results
