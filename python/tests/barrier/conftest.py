"""Shared test helpers for barrier pipeline tests.

Provides synthetic bar construction utilities used across
test_bar_pipeline.py, test_label_pipeline.py, test_feature_pipeline.py,
test_reward_accounting.py, test_barrier_env.py, test_multi_session_env.py,
and test_barrier_vec_env.py.
"""

import numpy as np
import pytest

from lob_rl.barrier import TICK_SIZE
from lob_rl.barrier.bar_pipeline import TradeBar
from lob_rl.barrier.feature_pipeline import build_feature_matrix
from lob_rl.barrier.label_pipeline import compute_labels

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


# ---------------------------------------------------------------------------
# Session data helpers (shared across env tests)
# ---------------------------------------------------------------------------

# Default lookback and derived dimensions
DEFAULT_H = 10
DEFAULT_FEATURE_DIM = 13 * DEFAULT_H  # 130
DEFAULT_OBS_DIM = DEFAULT_FEATURE_DIM + 2  # 132


def make_session_data(n_bars=40, base_price=4000.0, h=DEFAULT_H):
    """Build a single session's data dict: {bars, labels, features}.

    Used by test_barrier_env, test_multi_session_env, test_barrier_vec_env,
    and test_training_diagnostics.
    """
    bars = make_session_bars(n_bars, base_price=base_price)
    labels = compute_labels(bars, a=20, b=10, t_max=40)
    features = build_feature_matrix(bars, h=h)
    return {"bars": bars, "labels": labels, "features": features}


def make_session_data_list(n_sessions=5, n_bars=40, h=DEFAULT_H):
    """Build a list of session data dicts with slightly different prices."""
    return [
        make_session_data(n_bars=n_bars, base_price=4000.0 + i * 10.0, h=h)
        for i in range(n_sessions)
    ]


def run_episode(env, rng=None):
    """Run one full episode with random valid actions. Return (total_reward, steps)."""
    if rng is None:
        rng = np.random.default_rng(42)
    obs, info = env.reset()
    total_reward = 0.0
    terminated = False
    steps = 0
    while not terminated:
        mask = env.action_masks()
        valid_actions = np.where(mask)[0]
        action = rng.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1
        if steps > 2000:
            pytest.fail("Episode did not terminate within 2000 steps")
    return total_reward, steps
