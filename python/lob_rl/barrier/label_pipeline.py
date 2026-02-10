"""Barrier label construction pipeline.

Computes barrier-hit labels for trade bars with three outcomes:
+1 (upper barrier hit), -1 (lower barrier hit), 0 (timeout).
Supports intrabar tiebreaking, short direction mirroring,
T_max calibration, and diagnostics.
"""

import math
from dataclasses import dataclass

import numpy as np


TICK_SIZE = 0.25  # /MES tick size


@dataclass
class BarrierLabel:
    """Label for a single bar entry."""
    bar_index: int
    label: int              # +1, -1, or 0
    tau: int                # bars from entry to resolution
    resolution_type: str    # "upper", "lower", "timeout", "tiebreak_upper", "tiebreak_lower"
    entry_price: float      # C_k
    resolution_bar: int     # bar_index + tau


def _label_single_bar(k, bars, upper_barrier, lower_barrier, t_max):
    """Compute label for a single entry bar k.

    Scans bars j > k up to min(k + t_max, len(bars) - 1).
    Returns (label, tau, resolution_type, resolution_bar).
    """
    n_bars = len(bars)
    max_j = min(k + t_max, n_bars - 1)

    for j in range(k + 1, max_j + 1):
        bar_j = bars[j]
        upper_hit = bar_j.high >= upper_barrier
        lower_hit = bar_j.low <= lower_barrier

        if upper_hit and lower_hit:
            # Dual breach — tiebreak using trade sequence
            label, res_type = _tiebreak(
                bar_j, upper_barrier, lower_barrier, bars[j - 1].close
            )
            tau = j - k
            return label, tau, res_type, j

        if upper_hit:
            tau = j - k
            return 1, tau, "upper", j

        if lower_hit:
            tau = j - k
            return -1, tau, "lower", j

    # Timeout — tau is bounded by the number of bars actually scanned
    remaining = n_bars - 1 - k
    tau = min(t_max, remaining) if remaining > 0 else 1
    # Ensure tau >= 1 for timeout
    tau = max(tau, 1)
    return 0, tau, "timeout", k + tau


def _tiebreak(bar_j, upper_barrier, lower_barrier, prev_close):
    """Resolve dual barrier breach on a single bar using trade sequence.

    Returns (label, resolution_type).
    """
    trade_prices = bar_j.trade_prices
    if len(trade_prices) == 0:
        # Fallback: gap direction
        return _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close)

    first_trade = trade_prices[0]

    # Check if first trade already exceeds a barrier
    first_upper = first_trade >= upper_barrier
    first_lower = first_trade <= lower_barrier

    if first_upper and first_lower:
        # Impossible for a single price, but handle gracefully
        return _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close)

    if first_upper:
        # First trade already past upper — resolve by gap direction
        # since we can't tell when lower was crossed
        return _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close)

    if first_lower:
        # First trade already past lower — resolve by gap direction
        return _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close)

    # Scan trade sequence to find which barrier is crossed first
    for price in trade_prices:
        if price >= upper_barrier:
            return 1, "tiebreak_upper"
        if price <= lower_barrier:
            return -1, "tiebreak_lower"

    # Should not reach here if H >= U and L <= D, but fallback
    return _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close)


def _gap_direction(bar_j, upper_barrier, lower_barrier, prev_close):
    """Resolve by gap direction from previous close."""
    first_trade = bar_j.trade_prices[0] if len(bar_j.trade_prices) > 0 else bar_j.open
    if first_trade >= prev_close:
        return 1, "tiebreak_upper"
    else:
        return -1, "tiebreak_lower"


def compute_labels(bars, a=20, b=10, t_max=40, direction="long"):
    """Compute barrier-hit labels for all bars.

    Parameters
    ----------
    bars : list[TradeBar]
        Bars from a single session.
    a : int
        Upper barrier distance in ticks (profit for long, stop for short).
    b : int
        Lower barrier distance in ticks (stop for long, profit for short).
    t_max : int
        Maximum holding period in bars.
    direction : str
        "long" or "short".

    Returns
    -------
    list[BarrierLabel]
    """
    if not bars:
        return []

    labels = []

    for k in range(len(bars)):
        entry_price = bars[k].close

        if direction == "long":
            upper_barrier = entry_price + a * TICK_SIZE
            lower_barrier = entry_price - b * TICK_SIZE
        else:
            # Short: profit barrier below, stop barrier above
            # For short: "upper" hit (price going up) = stop = label -1
            #            "lower" hit (price going down) = profit = label +1
            upper_barrier = entry_price + b * TICK_SIZE  # stop
            lower_barrier = entry_price - a * TICK_SIZE  # profit

        label_val, tau, res_type, res_bar = _label_single_bar(
            k, bars, upper_barrier, lower_barrier, t_max
        )

        if direction == "short" and label_val != 0:
            # For short: upper hit means price went up = stop = -1
            #            lower hit means price went down = profit = +1
            # The _label_single_bar returns +1 for upper, -1 for lower
            # For short, we need to flip: upper=stop=-1, lower=profit=+1
            label_val = -label_val
            # Also flip resolution type names
            if res_type == "upper":
                res_type = "lower"
            elif res_type == "lower":
                res_type = "upper"
            elif res_type == "tiebreak_upper":
                res_type = "tiebreak_lower"
            elif res_type == "tiebreak_lower":
                res_type = "tiebreak_upper"

        labels.append(BarrierLabel(
            bar_index=k,
            label=label_val,
            tau=tau,
            resolution_type=res_type,
            entry_price=entry_price,
            resolution_bar=res_bar,
        ))

    return labels


def calibrate_t_max(bars, a=20, b=10):
    """Calibrate T_max from data: ceil(P95 of tau for upper-hit labels).

    Parameters
    ----------
    bars : list[TradeBar]
    a : int
    b : int

    Returns
    -------
    int
    """
    # Run with very large t_max (effectively infinity)
    large_t_max = len(bars)
    labels = compute_labels(bars, a=a, b=b, t_max=large_t_max, direction="long")

    # Collect tau for upper hit (label == +1) labels
    winner_taus = [lbl.tau for lbl in labels if lbl.label == 1]

    if not winner_taus:
        raise ValueError("No upper barrier hits found; cannot calibrate T_max")

    p95 = float(np.percentile(winner_taus, 95))
    return int(math.ceil(p95))


def compute_tiebreak_frequency(labels):
    """Fraction of labels that required tiebreaking.

    Parameters
    ----------
    labels : list[BarrierLabel]

    Returns
    -------
    float
    """
    if not labels:
        return 0.0
    tiebreak_count = sum(
        1 for lbl in labels if "tiebreak" in lbl.resolution_type
    )
    return tiebreak_count / len(labels)


def compute_label_distribution(labels):
    """Compute fraction of each label type.

    Parameters
    ----------
    labels : list[BarrierLabel]

    Returns
    -------
    dict with keys p_plus, p_minus, p_zero
    """
    if not labels:
        return {"p_plus": 0.0, "p_minus": 0.0, "p_zero": 0.0}

    n = len(labels)
    p_plus = sum(1 for lbl in labels if lbl.label == 1) / n
    p_minus = sum(1 for lbl in labels if lbl.label == -1) / n
    p_zero = sum(1 for lbl in labels if lbl.label == 0) / n
    return {"p_plus": p_plus, "p_minus": p_minus, "p_zero": p_zero}
