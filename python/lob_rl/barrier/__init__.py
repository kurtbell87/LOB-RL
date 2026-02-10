"""Barrier pipeline package for LOB-RL label construction."""

import numpy as np

# Shared constants
TICK_SIZE = 0.25  # /MES tick size (quarter-point)

# Reference RTH window for synthetic data generation
# 2022-06-15 RTH: 13:30 UTC to 20:00 UTC (6.5 hours)
RTH_OPEN_NS = 1655296200_000_000_000
RTH_DURATION_NS = 23400_000_000_000  # 6.5 hours in nanoseconds


def build_synthetic_trades(prices, timestamps):
    """Build a structured trades array from price and timestamp sequences.

    Infers side from price diffs using the tick rule (uptick -> 'B', else -> 'A').
    All sizes are 1. Includes both ``ts_event`` and ``timestamp`` fields so the
    array is compatible with both test assertions and ``build_bars_from_trades``.

    Parameters
    ----------
    prices : np.ndarray
        Price series, shape (N,).
    timestamps : np.ndarray
        Timestamp series in nanoseconds, shape (N,).

    Returns
    -------
    np.ndarray
        Structured array with fields: price, size, side, ts_event, timestamp.
    """
    n = len(prices)
    sides = np.empty(n, dtype="U1")
    sides[0] = "B"
    if n > 1:
        diffs = np.diff(prices)
        sides[1:] = np.where(diffs > 0, "B", "A")

    dt = np.dtype([
        ("price", np.float64),
        ("size", np.int32),
        ("side", "U1"),
        ("ts_event", np.int64),
        ("timestamp", np.int64),
    ])
    trades = np.empty(n, dtype=dt)
    trades["price"] = prices
    trades["size"] = 1
    trades["side"] = sides
    trades["ts_event"] = timestamps
    trades["timestamp"] = timestamps

    return trades
