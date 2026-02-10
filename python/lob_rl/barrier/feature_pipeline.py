"""Barrier feature extraction pipeline.

Computes 13 bar-level features, applies z-score normalization with a trailing
window, assembles lookback matrices, and provides an end-to-end builder.
"""

import math

import numpy as np

from lob_rl.barrier import TICK_SIZE

# RTH session boundaries for /MES (used for normalized session time)
# Open: 8:30 CT, Close: 15:00 CT
# RTH duration = 6.5 hours = 23400 seconds = 23400e9 nanoseconds
# These are used as relative offsets; the actual UTC values depend on CDT/CST.
# For session time, we use t_end relative to the session's first bar.


def compute_bar_features(bars, mbo_data=None):
    """Compute 13 features for each bar.

    Column layout:
        0  Trade flow imbalance       [-1, +1]
        1  BBO imbalance              [0, 1]    (0.5 if mbo_data=None)
        2  Depth imbalance            [0, 1]    (0.5 if mbo_data=None)
        3  Bar range (ticks)          >= 0
        4  Bar body (ticks)           signed
        5  Body/range ratio           [-1, +1]
        6  VWAP displacement          [-1, +1]
        7  Volume (log)               finite
        8  Trailing realized vol      NaN for first 19 bars
        9  Normalized session time    [0, 1]
       10  Cancel rate asymmetry      [-1, +1]  (0.0 if mbo_data=None)
       11  Mean spread                > 0       (1.0 if mbo_data=None)
       12  Session age                [0, 1]    min(bar_index/20, 1.0)

    Parameters
    ----------
    bars : list[TradeBar]
    mbo_data : optional
        MBO-level data for LOB features. If None, neutral defaults used.

    Returns
    -------
    np.ndarray of shape (N, 13), dtype float64
    """
    n = len(bars)
    features = np.zeros((n, 13), dtype=np.float64)

    # Pre-extract close prices for realized vol computation
    closes = np.array([b.close for b in bars], dtype=np.float64)

    # Determine session time normalization from bar timestamps
    # Use first bar's session_date to infer RTH boundaries.
    # For simplicity, we normalize using the min/max t_end across all bars
    # as a proxy for RTH open/close. However, the test expects us to use
    # the actual RTH open/close. We infer RTH open from the session_date.
    #
    # The tests construct bars with known _RTH_OPEN_NS and _RTH_CLOSE_NS:
    # _RTH_OPEN_NS = 1655296200_000_000_000   # 2022-06-15 13:30:00 UTC
    # _RTH_CLOSE_NS = 1655319600_000_000_000  # 2022-06-15 20:00:00 UTC
    # session_time = (t_end - RTH_open) / (RTH_close - RTH_open)
    #
    # We compute RTH boundaries from the session_date using America/Chicago TZ.
    rth_open_ns, rth_close_ns = _compute_rth_bounds(bars)

    for i, bar in enumerate(bars):
        # Col 0: Trade flow imbalance
        features[i, 0] = _trade_flow_imbalance(bar)

        # Col 1: BBO imbalance (neutral default; TODO: compute from MBO data)
        features[i, 1] = 0.5

        # Col 2: Depth imbalance (neutral default; TODO: compute from MBO data)
        features[i, 2] = 0.5

        # Col 3: Bar range in ticks
        bar_range = bar.high - bar.low
        features[i, 3] = bar_range / TICK_SIZE

        # Col 4: Bar body in ticks
        features[i, 4] = (bar.close - bar.open) / TICK_SIZE

        # Col 5: Body/range ratio
        if bar_range > 0:
            features[i, 5] = (bar.close - bar.open) / bar_range
        else:
            features[i, 5] = 0.0

        # Col 6: VWAP displacement
        if bar_range > 0:
            features[i, 6] = (bar.close - bar.vwap) / bar_range
        else:
            features[i, 6] = 0.0

        # Col 7: Volume (log)
        features[i, 7] = math.log(max(bar.volume, 1))

        # Col 8: Trailing realized vol (NaN for first 19 bars)
        if i >= 19:
            # Use closes[i-19:i+1] (20 close prices → 19 log returns)
            window_closes = closes[i - 19:i + 1]
            log_returns = np.log(window_closes[1:] / window_closes[:-1])
            features[i, 8] = np.std(log_returns, ddof=0)
        else:
            features[i, 8] = np.nan

        # Col 9: Normalized session time
        if rth_close_ns > rth_open_ns:
            t = (bar.t_end - rth_open_ns) / (rth_close_ns - rth_open_ns)
            features[i, 9] = max(0.0, min(1.0, t))
        else:
            features[i, 9] = 0.0

        # Col 10: Cancel rate asymmetry (zero default; TODO: compute from MBO data)
        features[i, 10] = 0.0

        # Col 11: Mean spread (unit default; TODO: compute from MBO data)
        features[i, 11] = 1.0

        # Col 12: Session age
        features[i, 12] = min(bar.bar_index / 20.0, 1.0)

    return features


def _compute_rth_bounds(bars):
    """Compute RTH open/close in nanoseconds from the session date.

    RTH = 8:30 AM - 3:00 PM Eastern Time.
    Falls back to inferring from bar timestamps if session_date unavailable.
    """
    if not bars:
        return 0, 1

    session_date = bars[0].session_date if bars else ""

    if session_date:
        try:
            from datetime import datetime
            from zoneinfo import ZoneInfo

            eastern = ZoneInfo("America/New_York")
            parts = session_date.split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])

            open_et = datetime(year, month, day, 8, 30, 0, tzinfo=eastern)
            close_et = datetime(year, month, day, 15, 0, 0, tzinfo=eastern)

            open_ns = int(open_et.timestamp() * 1e9)
            close_ns = int(close_et.timestamp() * 1e9)
            return open_ns, close_ns
        except (ValueError, IndexError):
            pass

    # Fallback: use first and last bar timestamps
    t_min = bars[0].t_start
    t_max = bars[-1].t_end
    if t_max <= t_min:
        t_max = t_min + 1
    return t_min, t_max


def _trade_flow_imbalance(bar):
    """Compute trade flow imbalance using tick rule on trade prices.

    Applies the tick rule: if price > prev_price → buy (+1),
    if price < prev_price → sell (-1), if equal → same as previous.
    Imbalance = (buy_vol - sell_vol) / total_vol, in [-1, +1].
    """
    prices = bar.trade_prices
    sizes = bar.trade_sizes

    if len(prices) <= 1:
        return 0.0

    buy_vol = 0.0
    sell_vol = 0.0
    prev_side = 0  # 0 = unknown

    for j in range(len(prices)):
        if j == 0:
            # First trade: classify as neutral or use open direction
            prev_side = 0
        else:
            if prices[j] > prices[j - 1]:
                prev_side = 1
            elif prices[j] < prices[j - 1]:
                prev_side = -1
            # if equal, keep prev_side

        if prev_side > 0:
            buy_vol += sizes[j]
        elif prev_side < 0:
            sell_vol += sizes[j]
        # neutral trades not counted

    total = buy_vol + sell_vol
    if total == 0:
        return 0.0

    return (buy_vol - sell_vol) / total


def normalize_features(raw, window=2000):
    """Z-score normalize features with a trailing window.

    For each row i, computes z-score using the trailing window of
    min(window, i+1) rows. Handles NaN by filling with 0 before
    computing statistics. Clips output to [-5, +5].

    Parameters
    ----------
    raw : np.ndarray of shape (N, 13)
    window : int
        Trailing window size (default 2000).

    Returns
    -------
    np.ndarray of shape (N, 13), dtype float64
    """
    n, ncols = raw.shape
    # Replace NaN with 0 for computation
    filled = np.where(np.isnan(raw), 0.0, raw)
    result = np.zeros_like(filled)

    for i in range(n):
        start = max(0, i - window + 1)
        window_data = filled[start:i + 1]

        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0)

        # Avoid division by zero: where std == 0, result is 0
        safe_std = np.where(std > 0, std, 1.0)
        result[i] = (filled[i] - mean) / safe_std
        # Where std was 0, set to 0
        result[i] = np.where(std > 0, result[i], 0.0)

    # Clip to [-5, +5]
    np.clip(result, -5.0, 5.0, out=result)

    return result


def assemble_lookback(normed, h=10):
    """Assemble lookback windows by concatenating h consecutive feature rows.

    Row i of output = normed[i:i+h].flatten()

    Parameters
    ----------
    normed : np.ndarray of shape (N, F)
    h : int
        Lookback horizon (default 10).

    Returns
    -------
    np.ndarray of shape (max(N-h+1, 0), F*h)
    """
    n, f = normed.shape
    out_rows = n - h + 1

    if out_rows <= 0:
        return np.empty((0, f * h), dtype=normed.dtype)

    result = np.zeros((out_rows, f * h), dtype=normed.dtype)
    for i in range(out_rows):
        result[i] = normed[i:i + h].flatten()

    return result


def build_feature_matrix(bars, h=10, window=2000, mbo_data=None):
    """End-to-end: compute features, normalize, assemble lookback.

    Drops rows where realized vol is NaN (first 19 bars) before
    normalization to ensure no NaN in output.

    Parameters
    ----------
    bars : list[TradeBar]
    h : int
        Lookback horizon (default 10).
    window : int
        Normalization window (default 2000).
    mbo_data : optional

    Returns
    -------
    np.ndarray of shape (M, 13*h) where M depends on bar count and warmup.
    """
    raw = compute_bar_features(bars, mbo_data=mbo_data)

    # Drop rows where realized vol is NaN (first 19 bars)
    # Find the first row where col 8 is not NaN
    valid_start = 0
    for i in range(raw.shape[0]):
        if not np.isnan(raw[i, 8]):
            valid_start = i
            break
    else:
        # All NaN in col 8 — keep all rows, normalize will handle NaN
        valid_start = 0

    # Only drop warmup rows if we have enough bars
    if valid_start > 0 and raw.shape[0] > valid_start:
        raw = raw[valid_start:]

    normed = normalize_features(raw, window=window)
    result = assemble_lookback(normed, h=h)

    return result
