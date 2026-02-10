"""Shared test helpers for barrier pipeline tests.

Provides synthetic bar construction utilities used across
test_bar_pipeline.py, test_label_pipeline.py, and test_feature_pipeline.py.
"""

import numpy as np

from lob_rl.barrier import TICK_SIZE
from lob_rl.barrier.bar_pipeline import TradeBar

# RTH hours for /MES (Central Time) — used for session time tests
# Open: 8:30 CT, Close: 15:00 CT
# In CDT (summer): Open = 13:30 UTC, Close = 20:00 UTC
# RTH duration = 6.5 hours = 23400 seconds
_RTH_OPEN_NS = 1655296200_000_000_000   # 2022-06-15 13:30:00 UTC (8:30 CT CDT)
_RTH_CLOSE_NS = 1655319600_000_000_000  # 2022-06-15 20:00:00 UTC (15:00 CT CDT)
_RTH_DURATION_NS = _RTH_CLOSE_NS - _RTH_OPEN_NS


# ---------------------------------------------------------------------------
# Synthetic bar helpers (test-only, no implementation logic)
# ---------------------------------------------------------------------------

def make_bar(bar_index, open_price, high, low, close, volume=100,
             vwap=None, t_start=0, t_end=1, session_date="2022-06-15",
             trade_prices=None, trade_sizes=None):
    """Create a TradeBar with explicit OHLCV and optional trade sequences."""
    if vwap is None:
        vwap = (high + low) / 2.0
    if trade_prices is None:
        trade_prices = np.array([close], dtype=np.float64)
    else:
        trade_prices = np.asarray(trade_prices, dtype=np.float64)
    if trade_sizes is None:
        trade_sizes = np.ones(len(trade_prices), dtype=np.int32)
    else:
        trade_sizes = np.asarray(trade_sizes, dtype=np.int32)
    return TradeBar(
        bar_index=bar_index,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
        t_start=t_start,
        t_end=t_end,
        session_date=session_date,
        trade_prices=trade_prices,
        trade_sizes=trade_sizes,
    )


def make_flat_bars(n, base_price=4000.0, spread=1.0):
    """Create n bars that stay within a narrow range around base_price.

    These bars will NOT trigger any barriers with default a=20, b=10.
    """
    bars = []
    for k in range(n):
        bars.append(make_bar(
            bar_index=k,
            open_price=base_price,
            high=base_price + spread * TICK_SIZE,
            low=base_price - spread * TICK_SIZE,
            close=base_price,
            trade_prices=np.array([base_price], dtype=np.float64),
        ))
    return bars


def make_session_bars(n, base_price=4000.0, spread=2.0, volume=100):
    """Create n bars spanning an RTH session with increasing timestamps.

    Timestamps are evenly spaced across the 6.5-hour RTH window.
    Bars have some price variation but stay within a narrow range.
    Trade sides alternate buy/sell for deterministic flow imbalance.
    """
    bars = []
    step_ns = _RTH_DURATION_NS // max(n, 1)
    for k in range(n):
        # Slight price variation
        offset = (k % 5 - 2) * TICK_SIZE
        o = base_price + offset
        h = base_price + spread * TICK_SIZE
        l = base_price - spread * TICK_SIZE
        c = base_price + offset * 0.5

        t_start = _RTH_OPEN_NS + k * step_ns
        t_end = _RTH_OPEN_NS + (k + 1) * step_ns - 1

        # Deterministic trade sequences — 10 trades per bar
        n_trades = 10
        tp = np.linspace(l, h, n_trades)
        ts = np.ones(n_trades, dtype=np.int32) * (volume // n_trades)

        bars.append(TradeBar(
            bar_index=k,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=volume,
            vwap=(h + l) / 2.0,
            t_start=t_start,
            t_end=t_end,
            session_date="2022-06-15",
            trade_prices=tp,
            trade_sizes=ts,
        ))
    return bars
