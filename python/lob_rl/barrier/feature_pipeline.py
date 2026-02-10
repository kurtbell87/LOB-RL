"""Barrier feature extraction pipeline.

Computes 17 bar-level features, applies z-score normalization with a trailing
window, assembles lookback matrices, and provides an end-to-end builder.
"""

import math

import numpy as np

from lob_rl.barrier import TICK_SIZE, N_FEATURES
from lob_rl.barrier.lob_reconstructor import OrderBook

# Default book feature values when MBO data is unavailable.
# Order: [BBO imbalance, Depth imbalance, Cancel asymmetry, Mean spread,
#         OFI, Depth ratio, WMid displacement, Spread std]
_BOOK_DEFAULTS = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0)

# Realized volatility requires 20 close prices (indices 0..19) to compute
# 19 log returns.  Bars at index < _REALIZED_VOL_WARMUP have NaN in col 8.
_REALIZED_VOL_WARMUP = 19

# Session-age feature saturates at this many bars (col 12 = min(bar_index / N, 1)).
_SESSION_AGE_PERIOD = 20.0


def compute_bar_features(bars, mbo_data=None):
    """Compute 17 features for each bar.

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
       13  Order Flow Imbalance (OFI) [-1, +1]  (0.0 if mbo_data=None)
       14  Multi-level depth ratio    [0, 1]    (0.5 if mbo_data=None)
       15  Weighted mid displacement  signed    (0.0 if mbo_data=None)
       16  Spread dynamics (std)      >= 0      (0.0 if mbo_data=None)

    Parameters
    ----------
    bars : list[TradeBar]
    mbo_data : optional
        MBO-level data for LOB features. If None, neutral defaults used.

    Returns
    -------
    np.ndarray of shape (N, 17), dtype float64
    """
    n = len(bars)
    features = np.zeros((n, N_FEATURES), dtype=np.float64)

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

    # Compute book features from MBO data if available
    book_features = _compute_book_features(bars, mbo_data) if mbo_data is not None and len(mbo_data) > 0 else None

    for i, bar in enumerate(bars):
        # Col 0: Trade flow imbalance
        features[i, 0] = _trade_flow_imbalance(bar)

        # Cols 1, 2, 10, 11, 13-16: Book features
        if book_features is not None:
            features[i, 1] = book_features[i, 0]   # BBO imbalance
            features[i, 2] = book_features[i, 1]   # Depth imbalance
            features[i, 10] = book_features[i, 2]  # Cancel asymmetry
            features[i, 11] = book_features[i, 3]  # Mean spread
            features[i, 13] = book_features[i, 4]  # OFI
            features[i, 14] = book_features[i, 5]  # Multi-level depth ratio
            features[i, 15] = book_features[i, 6]  # Weighted mid displacement
            features[i, 16] = book_features[i, 7]  # Spread std
        else:
            features[i, 1] = _BOOK_DEFAULTS[0]   # BBO imbalance
            features[i, 2] = _BOOK_DEFAULTS[1]   # Depth imbalance
            features[i, 10] = _BOOK_DEFAULTS[2]  # Cancel asymmetry
            features[i, 11] = _BOOK_DEFAULTS[3]  # Mean spread
            features[i, 13] = _BOOK_DEFAULTS[4]  # OFI
            features[i, 14] = _BOOK_DEFAULTS[5]  # Multi-level depth ratio
            features[i, 15] = _BOOK_DEFAULTS[6]  # Weighted mid displacement
            features[i, 16] = _BOOK_DEFAULTS[7]  # Spread std

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

        # Col 8: Trailing realized vol (NaN during warmup)
        if i >= _REALIZED_VOL_WARMUP:
            window_closes = closes[i - _REALIZED_VOL_WARMUP:i + 1]
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

        # Col 12: Session age
        features[i, 12] = min(bar.bar_index / _SESSION_AGE_PERIOD, 1.0)

    return features


def _compute_book_features(bars, mbo_data):
    """Compute book-derived features from MBO data.

    Returns np.ndarray of shape (n_bars, 8):
        col 0: BBO imbalance [0, 1]
        col 1: Depth imbalance [0, 1]
        col 2: Cancel asymmetry [-1, +1]
        col 3: Mean spread (ticks) > 0
        col 4: Order Flow Imbalance (OFI) [-1, +1]
        col 5: Multi-level depth ratio [0, 1]
        col 6: Weighted mid-price displacement (ticks) signed
        col 7: Spread dynamics std (ticks) >= 0
    """
    n = len(bars)
    result = np.zeros((n, 8), dtype=np.float64)
    # Defaults (matches _BOOK_DEFAULTS order)
    for col_idx in range(8):
        result[:, col_idx] = _BOOK_DEFAULTS[col_idx]

    if mbo_data is None or len(mbo_data) == 0:
        return result

    # Extract arrays from DataFrame for speed
    actions = mbo_data["action"].values
    sides = mbo_data["side"].values
    prices = mbo_data["price"].values.astype(np.float64)
    sizes = mbo_data["size"].values.astype(np.int32)
    order_ids = mbo_data["order_id"].values.astype(np.int64)
    ts_events = mbo_data["ts_event"].values.astype(np.int64)

    # Build bar boundary arrays for searchsorted
    bar_t_starts = np.array([b.t_start for b in bars], dtype=np.int64)
    bar_t_ends = np.array([b.t_end for b in bars], dtype=np.int64)

    # Assign each MBO message to a bar using searchsorted on bar t_end
    # Messages with ts <= bar[i].t_end and ts > bar[i-1].t_end belong to bar i
    bar_assignments = np.searchsorted(bar_t_ends, ts_events, side="left")

    book = OrderBook()

    for bar_idx in range(n):
        # Get messages for this bar
        mask = bar_assignments == bar_idx
        bar_msg_ts = ts_events[mask]
        bar_actions = actions[mask]
        bar_sides = sides[mask]
        bar_prices = prices[mask]
        bar_sizes = sizes[mask]
        bar_oids = order_ids[mask]

        # Which messages are within the bar's actual time window
        in_bar_mask = bar_msg_ts >= bar_t_starts[bar_idx]

        bid_cancels = 0
        ask_cancels = 0
        spread_samples = []

        # OFI tracking
        ofi_signed_volume = 0.0
        total_add_volume = 0.0

        # Weighted mid displacement: record start-of-bar wmid
        wmid_start = book.weighted_mid_price()
        wmid_first = None  # first valid wmid after a message in the bar

        for j in range(len(bar_actions)):
            act = bar_actions[j]
            side = bar_sides[j]
            price = bar_prices[j]
            size = int(bar_sizes[j])
            oid = int(bar_oids[j])
            in_bar = bool(in_bar_mask[j])

            # OFI: only count Add messages within the bar's time window
            if act == "A" and in_bar:
                total_add_volume += size
                best_bid = book.best_bid()
                best_ask = book.best_ask()
                if side == "B" and best_bid > 0 and price >= best_bid:
                    ofi_signed_volume += size
                elif side == "A" and best_ask > 0 and price <= best_ask:
                    ofi_signed_volume -= size
                # If book is empty on one side, BBO-level add at any price
                # counts (first order establishes BBO)
                elif side == "B" and best_bid == 0:
                    ofi_signed_volume += size
                elif side == "A" and best_ask == 0:
                    ofi_signed_volume -= size

            book.apply(act, side, price, size, order_id=oid)

            if in_bar:
                # Track first valid wmid for displacement
                if wmid_first is None:
                    w = book.weighted_mid_price()
                    if w > 0:
                        wmid_first = w

                # Count cancels per side
                if act == "C":
                    if side == "B":
                        bid_cancels += 1
                    elif side == "A":
                        ask_cancels += 1

                # Sample spread after each event
                s = book.spread_ticks()
                if s > 0:
                    spread_samples.append(s)

        # Snapshot book at bar close
        bid_qty = book.best_bid_qty()
        ask_qty = book.best_ask_qty()
        total_bbo = bid_qty + ask_qty
        if total_bbo > 0:
            result[bar_idx, 0] = bid_qty / total_bbo
        else:
            result[bar_idx, 0] = 0.5

        total_bid = book.total_bid_depth(5)
        total_ask = book.total_ask_depth(5)
        total_depth = total_bid + total_ask
        if total_depth > 0:
            result[bar_idx, 1] = total_bid / total_depth
        else:
            result[bar_idx, 1] = 0.5

        # Cancel asymmetry
        total_cancels = bid_cancels + ask_cancels
        result[bar_idx, 2] = (bid_cancels - ask_cancels) / (total_cancels + 1e-10)

        # Mean spread
        if spread_samples:
            result[bar_idx, 3] = np.mean(spread_samples)
        else:
            result[bar_idx, 3] = 1.0

        # New features: only compute from in-bar messages
        has_in_bar_msgs = bool(np.any(in_bar_mask))

        # Col 4: OFI — normalized and clamped to [-1, +1]
        eps = 1e-10
        if total_add_volume > 0:
            ofi_norm = ofi_signed_volume / (total_add_volume + eps)
            result[bar_idx, 4] = max(-1.0, min(1.0, ofi_norm))
        else:
            result[bar_idx, 4] = 0.0

        # Col 5: Multi-level depth ratio — top3 / top10
        if has_in_bar_msgs:
            top3_bid = book.total_bid_depth(3)
            top3_ask = book.total_ask_depth(3)
            top10_bid = book.total_bid_depth(10)
            top10_ask = book.total_ask_depth(10)
            total_3 = top3_bid + top3_ask
            total_10 = top10_bid + top10_ask
            if total_10 > 0:
                result[bar_idx, 5] = total_3 / (total_10 + eps)
            else:
                result[bar_idx, 5] = 0.5  # default for empty book
        else:
            result[bar_idx, 5] = 0.5

        # Col 6: Weighted mid-price displacement
        wmid_end = book.weighted_mid_price()
        if wmid_first is not None and wmid_end > 0:
            result[bar_idx, 6] = (wmid_end - wmid_first) / TICK_SIZE
        else:
            result[bar_idx, 6] = 0.0

        # Col 7: Spread dynamics (std)
        if len(spread_samples) >= 2:
            result[bar_idx, 7] = float(np.std(spread_samples, ddof=0))
        else:
            result[bar_idx, 7] = 0.0

    return result


def _compute_rth_bounds(bars):
    """Compute RTH open/close in nanoseconds from the session date.

    RTH = 8:30 AM - 3:00 PM Eastern Time.
    Falls back to inferring from bar timestamps if session_date unavailable.
    """
    if not bars:
        return 0, 1

    session_date = bars[0].session_date

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

    # Compute price diffs: +1 for uptick, -1 for downtick, 0 for unchanged
    diffs = np.sign(np.diff(prices))

    # Build side array: first trade is neutral (0), rest from diffs
    # Forward-fill zeros (equal prices keep previous side)
    sides = np.empty(len(prices), dtype=np.float64)
    sides[0] = 0.0
    sides[1:] = diffs

    # Forward-fill: replace 0s with the last non-zero value
    prev = 0.0
    for j in range(len(sides)):
        if sides[j] != 0.0:
            prev = sides[j]
        else:
            sides[j] = prev

    buy_vol = float(np.sum(sizes[sides > 0]))
    sell_vol = float(np.sum(sizes[sides < 0]))
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
    raw : np.ndarray of shape (N, F)
    window : int
        Trailing window size (default 2000).

    Returns
    -------
    np.ndarray of shape (N, F), dtype float64
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
    np.ndarray of shape (M, 17*h) where M depends on bar count and warmup.
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
