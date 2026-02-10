"""Tests for the barrier bar construction pipeline.

Spec: docs/t1-bar-construction.md

Tests the offline batch processor that reads raw MBO data, extracts matched
trades, and produces fixed-count trade bars with full trade sequence retention.
"""

import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic trade data helpers (test-only, no implementation logic)
# ---------------------------------------------------------------------------

def _make_trades(n, price_start=4775.00, price_step=0.25, size=1,
                 ts_start_ns=None, ts_step_ns=1_000_000):
    """Create a structured array of n trades for testing.

    Returns numpy structured array with fields:
      price (float64), size (int32), timestamp (int64), side (int8)
    """
    prices = np.array(
        [price_start + (i % 5) * price_step for i in range(n)],
        dtype=np.float64,
    )
    sizes = np.full(n, size, dtype=np.int32)
    if ts_start_ns is None:
        # Default: 2022-06-15 14:00:00 UTC (9:00 AM CT during CDT)
        ts_start_ns = int(
            datetime(2022, 6, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp()
            * 1e9
        )
    timestamps = np.array(
        [ts_start_ns + i * ts_step_ns for i in range(n)], dtype=np.int64
    )
    sides = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)

    trades = np.zeros(n, dtype=[
        ("price", np.float64),
        ("size", np.int32),
        ("timestamp", np.int64),
        ("side", np.int8),
    ])
    trades["price"] = prices
    trades["size"] = sizes
    trades["timestamp"] = timestamps
    trades["side"] = sides
    return trades


def _make_trades_with_prices(prices, sizes=None, ts_start_ns=None,
                              ts_step_ns=1_000_000):
    """Create trades with explicit price sequence for hand-computed verification."""
    n = len(prices)
    prices = np.asarray(prices, dtype=np.float64)
    if sizes is None:
        sizes = np.ones(n, dtype=np.int32)
    else:
        sizes = np.asarray(sizes, dtype=np.int32)
    if ts_start_ns is None:
        ts_start_ns = int(
            datetime(2022, 6, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp()
            * 1e9
        )
    timestamps = np.array(
        [ts_start_ns + i * ts_step_ns for i in range(n)], dtype=np.int64
    )
    sides = np.array([1 if i % 2 == 0 else -1 for i in range(n)], dtype=np.int8)

    trades = np.zeros(n, dtype=[
        ("price", np.float64),
        ("size", np.int32),
        ("timestamp", np.int64),
        ("side", np.int8),
    ])
    trades["price"] = prices
    trades["size"] = sizes
    trades["timestamp"] = timestamps
    trades["side"] = sides
    return trades


def _utc_ns(year, month, day, hour, minute, second=0, microsecond=0):
    """Convert a UTC datetime to nanoseconds since epoch."""
    dt = datetime(year, month, day, hour, minute, second, microsecond,
                  tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


# ===========================================================================
# 1. TradeBar dataclass
# ===========================================================================


class TestTradeBarDataclass:
    """TradeBar should be importable and have all required fields."""

    def test_import(self):
        """TradeBar should be importable from lob_rl.barrier.bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import TradeBar
        assert TradeBar is not None

    def test_fields_present(self):
        """TradeBar should have all spec-defined fields."""
        from lob_rl.barrier.bar_pipeline import TradeBar

        bar = TradeBar(
            bar_index=0,
            open=4775.00,
            high=4776.00,
            low=4774.00,
            close=4775.50,
            volume=500,
            vwap=4775.25,
            t_start=1000000,
            t_end=2000000,
            session_date="2022-06-15",
            trade_prices=np.array([4775.00]),
            trade_sizes=np.array([1]),
        )

        assert bar.bar_index == 0
        assert bar.open == 4775.00
        assert bar.high == 4776.00
        assert bar.low == 4774.00
        assert bar.close == 4775.50
        assert bar.volume == 500
        assert bar.vwap == 4775.25
        assert bar.t_start == 1000000
        assert bar.t_end == 2000000
        assert bar.session_date == "2022-06-15"
        assert isinstance(bar.trade_prices, np.ndarray)
        assert isinstance(bar.trade_sizes, np.ndarray)


# ===========================================================================
# 2. build_bars_from_trades — core bar construction
# ===========================================================================


class TestBuildBarsImport:
    """build_bars_from_trades should be importable and callable."""

    def test_import(self):
        """build_bars_from_trades should be importable."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades
        assert callable(build_bars_from_trades)


class TestBuildBarsOHLCV:
    """Bar OHLCV fields must match hand-computed values from known trades."""

    def test_exact_n_trades_correct_ohlcv(self):
        """Bar with exactly N trades produces correct OHLCV.

        Spec test #1: Create a known sequence of 500 trades with
        hand-computed O, H, L, C, V, VWAP. Verify all fields match.
        """
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        # 500 trades: prices cycle [100.00, 100.25, 100.50, 100.75, 101.00]
        # sizes all 2
        n = 500
        prices = [100.00 + (i % 5) * 0.25 for i in range(n)]
        sizes = [2] * n
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")

        assert len(bars) == 1
        bar = bars[0]
        assert bar.open == prices[0]       # 100.00
        assert bar.close == prices[-1]     # 100.75 (499 % 5 = 4 → 101.00... let me recalc)
        # 499 % 5 = 4, so prices[-1] = 100.00 + 4*0.25 = 101.00
        assert bar.close == 100.00 + (499 % 5) * 0.25
        assert bar.high == 101.00          # max of cycle
        assert bar.low == 100.00           # min of cycle

    def test_volume_equals_sum_of_sizes(self):
        """Spec test #10: bar.volume == sum(bar.trade_sizes)."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        sizes = [1, 2, 3, 4, 5] * 100  # 500 trades
        prices = [100.0] * 500
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 1
        bar = bars[0]
        assert bar.volume == sum(sizes)
        assert bar.volume == int(np.sum(bar.trade_sizes))

    def test_vwap_hand_computed(self):
        """VWAP must be sum(price_i * size_i) / sum(size_i)."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        # 5 trades with known prices and sizes
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        sizes = [10, 20, 30, 20, 10]
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=5, session_date="2022-06-15")
        assert len(bars) == 1

        expected_vwap = (
            100.0 * 10 + 101.0 * 20 + 102.0 * 30 + 103.0 * 20 + 104.0 * 10
        ) / (10 + 20 + 30 + 20 + 10)
        assert bars[0].vwap == pytest.approx(expected_vwap, rel=1e-10)


class TestBuildBarsVWAPBounds:
    """Spec test #2: VWAP must be bounded by [L_k, H_k] for all bars."""

    def test_vwap_between_low_and_high(self):
        """VWAP = sum(price_i * size_i) / sum(size_i) must be within [low, high]."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        # Random-ish prices and sizes — 1000 trades → 2 bars of 500
        rng = np.random.default_rng(42)
        n = 1000
        prices = 4775.0 + rng.standard_normal(n) * 2.0
        sizes = rng.integers(1, 50, size=n)
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 2

        for bar in bars:
            assert bar.vwap >= bar.low, (
                f"VWAP {bar.vwap} < low {bar.low} in bar {bar.bar_index}"
            )
            assert bar.vwap <= bar.high, (
                f"VWAP {bar.vwap} > high {bar.high} in bar {bar.bar_index}"
            )

    def test_vwap_bounds_many_bars(self):
        """VWAP in [low, high] for 20 bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        rng = np.random.default_rng(123)
        n = 20 * 200
        prices = 4000.0 + rng.uniform(-10, 10, size=n)
        sizes = rng.integers(1, 100, size=n)
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=200, session_date="2022-06-15")
        assert len(bars) == 20

        for bar in bars:
            assert bar.low <= bar.vwap <= bar.high


class TestBuildBarsTimestamps:
    """Spec tests #3 and #4: timestamp ordering within and between bars."""

    def test_t_start_less_than_t_end(self):
        """Spec test #3: t_start_k < t_end_k for all bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(1000)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 2

        for bar in bars:
            assert bar.t_start < bar.t_end, (
                f"Bar {bar.bar_index}: t_start {bar.t_start} >= t_end {bar.t_end}"
            )

    def test_no_temporal_overlap_between_consecutive_bars(self):
        """Spec test #4: t_end_{k} <= t_start_{k+1} for consecutive bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(2500, ts_step_ns=1_000_000)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 5

        for i in range(len(bars) - 1):
            assert bars[i].t_end <= bars[i + 1].t_start, (
                f"Temporal overlap: bar {i} t_end {bars[i].t_end} > "
                f"bar {i+1} t_start {bars[i+1].t_start}"
            )


class TestBuildBarsTradeSequenceRetention:
    """Spec test #9: trade sequence arrays match bar OHLC."""

    def test_trade_prices_length_equals_n(self):
        """bar.trade_prices has exactly N elements."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(500)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 1
        assert len(bars[0].trade_prices) == 500

    def test_trade_sizes_length_equals_n(self):
        """bar.trade_sizes has exactly N elements."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(500)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 1
        assert len(bars[0].trade_sizes) == 500

    def test_first_trade_is_open(self):
        """bar.trade_prices[0] == bar.open."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [100.0 + i * 0.01 for i in range(500)]
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert bars[0].trade_prices[0] == bars[0].open

    def test_last_trade_is_close(self):
        """bar.trade_prices[-1] == bar.close."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [100.0 + i * 0.01 for i in range(500)]
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert bars[0].trade_prices[-1] == bars[0].close

    def test_max_trade_price_is_high(self):
        """max(bar.trade_prices) == bar.high."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [100.0 + i * 0.01 for i in range(500)]
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert np.max(bars[0].trade_prices) == bars[0].high

    def test_min_trade_price_is_low(self):
        """min(bar.trade_prices) == bar.low."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [100.0 + i * 0.01 for i in range(500)]
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert np.min(bars[0].trade_prices) == bars[0].low

    def test_trade_prices_is_numpy_array(self):
        """trade_prices must be a numpy array."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(500)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert isinstance(bars[0].trade_prices, np.ndarray)

    def test_trade_sizes_is_numpy_array(self):
        """trade_sizes must be a numpy array."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(500)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert isinstance(bars[0].trade_sizes, np.ndarray)


class TestBuildBarsBarIndex:
    """Bar indices must be sequential starting at 0."""

    def test_bar_indices_sequential(self):
        """Bars should have sequential bar_index starting at 0."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(2500)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        for i, bar in enumerate(bars):
            assert bar.bar_index == i

    def test_session_date_propagated(self):
        """All bars should carry the session_date from the argument."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(1000)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-03-14")
        for bar in bars:
            assert bar.session_date == "2022-03-14"


class TestBuildBarsDefaultN:
    """Default n should be 500."""

    def test_default_n_is_500(self):
        """build_bars_from_trades with no n argument uses n=500."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(1000)
        # Call without explicit n — should default to 500
        bars = build_bars_from_trades(trades, session_date="2022-06-15")
        assert len(bars) == 2  # 1000 / 500 = 2 bars


# ===========================================================================
# 3. Incomplete bar discard
# ===========================================================================


class TestIncompleteBarsDiscarded:
    """Spec test #7: Incomplete bars at session end are discarded."""

    def test_750_trades_produces_1_bar(self):
        """750 trades with N=500 should produce 1 bar (not 2)."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(750)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 1

    def test_499_trades_produces_no_bars(self):
        """Fewer than N trades → no bars produced."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(499)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 0

    def test_exact_multiple_produces_exact_bars(self):
        """Spec edge case: session with exactly k*N trades → k bars, no remainder."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(2000)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 4

    def test_1_trade_produces_no_bars(self):
        """A single trade is insufficient for any bar."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(1)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 0

    def test_total_volume_accounting(self):
        """Spec test #8: sum(V_k) = total matched trades - remainder (where remainder < N)."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        total_trades = 1723
        n = 500
        trades = _make_trades(total_trades, size=1)
        bars = build_bars_from_trades(trades, n=n, session_date="2022-06-15")

        expected_bars = total_trades // n  # 3
        expected_remainder = total_trades % n  # 223
        assert len(bars) == expected_bars
        assert expected_remainder < n

        total_volume = sum(bar.volume for bar in bars)
        # Each trade has size=1, so volume counted in trades
        total_trade_count = sum(len(bar.trade_prices) for bar in bars)
        assert total_trade_count == expected_bars * n
        assert total_volume == expected_bars * n  # size=1 each


# ===========================================================================
# 4. Empty session
# ===========================================================================


class TestEmptySession:
    """Spec edge case: Empty session → empty bar list."""

    def test_no_trades(self):
        """Zero trades should produce an empty bar list."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(0)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 0
        assert isinstance(bars, list)


# ===========================================================================
# 5. filter_rth_trades — RTH boundary filtering
# ===========================================================================


class TestFilterRthTradesImport:
    """filter_rth_trades should be importable."""

    def test_import(self):
        """filter_rth_trades should be importable from lob_rl.barrier.bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades
        assert callable(filter_rth_trades)


class TestFilterRthTradesCDT:
    """RTH filtering during CDT (UTC-5): RTH = 13:30-20:00 UTC."""

    def test_trades_within_rth_pass_through(self):
        """Trades within 8:30 AM - 3:00 PM CT (CDT) pass through."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # June 15 2022 is CDT. RTH = 13:30-20:00 UTC
        ts_14_00 = _utc_ns(2022, 6, 15, 14, 0)  # 9:00 AM CT
        trades = _make_trades(100, ts_start_ns=ts_14_00)
        result = filter_rth_trades(trades)
        assert len(result) == 100

    def test_trades_before_rth_excluded(self):
        """Trades before 8:30 AM CT are excluded."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # 13:00 UTC = 8:00 AM CDT — before RTH
        ts_before = _utc_ns(2022, 6, 15, 13, 0)
        trades = _make_trades(10, ts_start_ns=ts_before, ts_step_ns=60_000_000_000)
        # First 3 trades are before 13:30 UTC (at 13:00, 13:01, 13:02...
        # each step is 1 minute. 13:00 to 13:29 = 30 trades)
        # With step 1 min, 30 trades span 30 minutes: 13:00 to 13:29
        trades_30min = _make_trades(
            50, ts_start_ns=_utc_ns(2022, 6, 15, 13, 0),
            ts_step_ns=60_000_000_000  # 1 minute steps
        )
        result = filter_rth_trades(trades_30min)
        # 13:00-13:29 are excluded (30 trades), 13:30-13:49 are included (20 trades)
        assert len(result) == 20

    def test_trades_after_rth_excluded(self):
        """Trades at or after 3:00 PM CT (20:00 UTC) are excluded."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # 20:00 UTC = 3:00 PM CDT — at close
        ts_at_close = _utc_ns(2022, 6, 15, 20, 0)
        trades = _make_trades(10, ts_start_ns=ts_at_close)
        result = filter_rth_trades(trades)
        assert len(result) == 0

    def test_trades_at_exact_open(self):
        """Trades exactly at 8:30 AM CT (13:30 UTC) should be included."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        ts_open = _utc_ns(2022, 6, 15, 13, 30)
        trades = _make_trades(5, ts_start_ns=ts_open, ts_step_ns=1000)
        result = filter_rth_trades(trades)
        assert len(result) == 5

    def test_trade_one_nanosecond_before_open_excluded(self):
        """A trade 1ns before 8:30 AM CT should be excluded."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        ts_just_before_open = _utc_ns(2022, 6, 15, 13, 30) - 1
        trades = _make_trades(1, ts_start_ns=ts_just_before_open)
        result = filter_rth_trades(trades)
        assert len(result) == 0

    def test_trade_one_nanosecond_before_close_included(self):
        """A trade 1ns before 3:00 PM CT (20:00 UTC) should be included."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        ts_just_before_close = _utc_ns(2022, 6, 15, 20, 0) - 1
        trades = _make_trades(1, ts_start_ns=ts_just_before_close)
        result = filter_rth_trades(trades)
        assert len(result) == 1


class TestFilterRthTradesCST:
    """RTH filtering during CST (UTC-6): RTH = 14:30-21:00 UTC."""

    def test_cst_rth_boundaries(self):
        """Jan 15 2022 is CST. RTH = 14:30-21:00 UTC."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # 14:00 UTC = 8:00 AM CST — before RTH
        trades_before = _make_trades(1, ts_start_ns=_utc_ns(2022, 1, 15, 14, 0))
        result = filter_rth_trades(trades_before)
        assert len(result) == 0

        # 14:30 UTC = 8:30 AM CST — at open
        trades_open = _make_trades(1, ts_start_ns=_utc_ns(2022, 1, 15, 14, 30))
        result = filter_rth_trades(trades_open)
        assert len(result) == 1

        # 21:00 UTC = 3:00 PM CST — at close (excluded)
        trades_close = _make_trades(1, ts_start_ns=_utc_ns(2022, 1, 15, 21, 0))
        result = filter_rth_trades(trades_close)
        assert len(result) == 0

        # 20:59 UTC = 2:59 PM CST — before close (included)
        trades_before_close = _make_trades(
            1, ts_start_ns=_utc_ns(2022, 1, 15, 20, 59)
        )
        result = filter_rth_trades(trades_before_close)
        assert len(result) == 1


class TestFilterRthTradesDST:
    """Spec edge case: DST transition days must use correct RTH boundaries."""

    def test_spring_forward_march_2022(self):
        """March 13, 2022 is spring-forward day. Before: CST (UTC-6), after: CDT (UTC-5).

        The RTH session on March 14 (Monday after spring-forward) uses CDT:
        RTH = 13:30-20:00 UTC.
        """
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # March 14, 2022 — first trading day after spring-forward (CDT)
        # 13:30 UTC = 8:30 AM CDT (open)
        trades_open = _make_trades(1, ts_start_ns=_utc_ns(2022, 3, 14, 13, 30))
        result = filter_rth_trades(trades_open)
        assert len(result) == 1

        # 14:30 UTC = 9:30 AM CDT (NOT 8:30 AM — that would be CST)
        # If code incorrectly uses CST offset, 14:30 UTC would be RTH open.
        # With correct CDT, 14:30 is well within RTH.
        trades_mid = _make_trades(1, ts_start_ns=_utc_ns(2022, 3, 14, 14, 30))
        result = filter_rth_trades(trades_mid)
        assert len(result) == 1

    def test_fall_back_november_2022(self):
        """November 6, 2022 is fall-back day. Before: CDT (UTC-5), after: CST (UTC-6).

        November 7 (Monday after fall-back) uses CST:
        RTH = 14:30-21:00 UTC.
        """
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # Nov 7, 2022 — first trading day after fall-back (CST)
        # 14:00 UTC = 8:00 AM CST — before RTH open
        trades_before = _make_trades(1, ts_start_ns=_utc_ns(2022, 11, 7, 14, 0))
        result = filter_rth_trades(trades_before)
        assert len(result) == 0

        # 14:30 UTC = 8:30 AM CST — at RTH open
        trades_open = _make_trades(1, ts_start_ns=_utc_ns(2022, 11, 7, 14, 30))
        result = filter_rth_trades(trades_open)
        assert len(result) == 1

        # 21:00 UTC = 3:00 PM CST — at close (excluded)
        trades_close = _make_trades(1, ts_start_ns=_utc_ns(2022, 11, 7, 21, 0))
        result = filter_rth_trades(trades_close)
        assert len(result) == 0


class TestFilterRthTradesGlobexMaintenance:
    """Spec test #6: Globex maintenance window trades are excluded."""

    def test_globex_maintenance_excluded_cdt(self):
        """Trades during 4:00-5:00 PM CT (21:00-22:00 UTC CDT) are excluded.

        This is outside RTH hours anyway for CDT (RTH ends at 20:00 UTC),
        but verify the filter handles it if such trades appear.
        """
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # 21:30 UTC = 4:30 PM CDT — during Globex maintenance
        ts_maint = _utc_ns(2022, 6, 15, 21, 30)
        trades = _make_trades(10, ts_start_ns=ts_maint)
        result = filter_rth_trades(trades)
        assert len(result) == 0

    def test_globex_maintenance_excluded_cst(self):
        """Trades during 4:00-5:00 PM CT (22:00-23:00 UTC CST) are excluded."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        ts_maint = _utc_ns(2022, 1, 15, 22, 30)
        trades = _make_trades(10, ts_start_ns=ts_maint)
        result = filter_rth_trades(trades)
        assert len(result) == 0


class TestFilterRthTradesPreservesOrder:
    """Filtered output must maintain trade order."""

    def test_output_preserves_timestamp_order(self):
        """Filtered trades should maintain their original timestamp order."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # Mix of RTH and non-RTH trades (June 15 CDT)
        ts_list = [
            _utc_ns(2022, 6, 15, 12, 0),   # pre-RTH
            _utc_ns(2022, 6, 15, 14, 0),   # RTH
            _utc_ns(2022, 6, 15, 16, 0),   # RTH
            _utc_ns(2022, 6, 15, 18, 0),   # RTH
            _utc_ns(2022, 6, 15, 21, 0),   # post-RTH
        ]
        trades = np.zeros(5, dtype=[
            ("price", np.float64), ("size", np.int32),
            ("timestamp", np.int64), ("side", np.int8),
        ])
        for i, ts in enumerate(ts_list):
            trades[i]["price"] = 100.0 + i
            trades[i]["size"] = 1
            trades[i]["timestamp"] = ts
            trades[i]["side"] = 1

        result = filter_rth_trades(trades)
        assert len(result) == 3
        # Verify order preserved
        assert result[0]["price"] == 101.0  # 14:00 UTC
        assert result[1]["price"] == 102.0  # 16:00 UTC
        assert result[2]["price"] == 103.0  # 18:00 UTC

    def test_returns_structured_array(self):
        """filter_rth_trades should return a structured numpy array."""
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        trades = _make_trades(100)
        result = filter_rth_trades(trades)
        assert isinstance(result, np.ndarray)
        assert "price" in result.dtype.names
        assert "size" in result.dtype.names
        assert "timestamp" in result.dtype.names


# ===========================================================================
# 6. Session boundary rules for bar construction
# ===========================================================================


class TestSessionBoundaryRules:
    """Spec test #5: Bars must not straddle RTH session boundaries."""

    def test_trades_spanning_close_discards_bar(self):
        """Create trades from 2:59 PM to 3:01 PM CT. Bar should be discarded.

        If we have N=5 and 5 trades that span the close, the bar is incomplete
        because some trades are outside RTH. After RTH filtering, the bar has
        fewer than N trades and is discarded.
        """
        from lob_rl.barrier.bar_pipeline import (
            build_bars_from_trades, filter_rth_trades
        )

        # June 15 CDT: close at 20:00 UTC
        close_utc = _utc_ns(2022, 6, 15, 20, 0)
        # 5 trades: 2 before close, 3 after close. Step = 30 seconds.
        ts_start = close_utc - 2 * 30_000_000_000  # 19:59:00 UTC
        trades = _make_trades(5, ts_start_ns=ts_start, ts_step_ns=30_000_000_000)

        filtered = filter_rth_trades(trades)
        # Only trades before 20:00 should remain (2 out of 5)
        assert len(filtered) < 5
        bars = build_bars_from_trades(filtered, n=5, session_date="2022-06-15")
        assert len(bars) == 0  # Incomplete bar discarded

    def test_trades_spanning_open_uses_only_post_open(self):
        """Create trades from 8:29 AM to 8:31 AM CT. Only 8:30+ used.

        After filtering, trades before open are excluded. Remaining trades
        may not fill a complete bar.
        """
        from lob_rl.barrier.bar_pipeline import filter_rth_trades

        # June 15 CDT: open at 13:30 UTC
        open_utc = _utc_ns(2022, 6, 15, 13, 30)
        # 10 trades: 5 before open, 5 after. Step = 30 seconds.
        ts_start = open_utc - 5 * 30_000_000_000  # 13:27:30 UTC
        trades = _make_trades(10, ts_start_ns=ts_start, ts_step_ns=30_000_000_000)

        filtered = filter_rth_trades(trades)
        # Only trades at 13:30 or later should remain
        for t in filtered:
            assert t["timestamp"] >= open_utc

    def test_bar_indices_reset_per_session(self):
        """Spec rule #5: Bar indices reset to 0 at the start of each session."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        # Two separate sessions
        trades1 = _make_trades(1000)
        trades2 = _make_trades(500)

        bars1 = build_bars_from_trades(trades1, n=500, session_date="2022-06-15")
        bars2 = build_bars_from_trades(trades2, n=500, session_date="2022-06-16")

        assert bars1[0].bar_index == 0
        assert bars1[1].bar_index == 1
        assert bars2[0].bar_index == 0  # Reset for new session


# ===========================================================================
# 7. extract_trades_from_mbo
# ===========================================================================


class TestExtractTradesFromMboImport:
    """extract_trades_from_mbo should be importable."""

    def test_import(self):
        """extract_trades_from_mbo should be importable from lob_rl.barrier.bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import extract_trades_from_mbo
        assert callable(extract_trades_from_mbo)


class TestExtractTradesFromMboReturnType:
    """extract_trades_from_mbo should return structured array with correct fields."""

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_return_dtype_fields(self):
        """Returned array must have fields: price, size, timestamp, side."""
        from lob_rl.barrier.bar_pipeline import extract_trades_from_mbo

        # Find first available .dbn.zst file
        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))
        if not files:
            pytest.skip("No .dbn.zst files found")

        trades = extract_trades_from_mbo(str(files[0]))
        assert "price" in trades.dtype.names
        assert "size" in trades.dtype.names
        assert "timestamp" in trades.dtype.names
        assert "side" in trades.dtype.names

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_return_dtype_types(self):
        """Field types must be: price=float64, size=int32, timestamp=int64, side=int8."""
        from lob_rl.barrier.bar_pipeline import extract_trades_from_mbo

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))
        if not files:
            pytest.skip("No .dbn.zst files found")

        trades = extract_trades_from_mbo(str(files[0]))
        assert trades.dtype["price"] == np.float64
        assert trades.dtype["size"] == np.int32
        assert trades.dtype["timestamp"] == np.int64
        assert trades.dtype["side"] == np.int8


# ===========================================================================
# 8. build_session_bars (end-to-end single file)
# ===========================================================================


class TestBuildSessionBarsImport:
    """build_session_bars should be importable."""

    def test_import(self):
        """build_session_bars should be importable from lob_rl.barrier.bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import build_session_bars
        assert callable(build_session_bars)


class TestBuildSessionBarsEndToEnd:
    """End-to-end test: reads MBO file, extracts trades, filters RTH, builds bars."""

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_produces_valid_bars_from_real_data(self):
        """build_session_bars on a real .dbn.zst file produces valid bars."""
        from lob_rl.barrier.bar_pipeline import build_session_bars

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))
        if not files:
            pytest.skip("No .dbn.zst files found")

        bars = build_session_bars(str(files[0]), n=500)
        # A typical trading day has ~200k trades → ~400 bars at N=500
        assert len(bars) > 0

        for bar in bars:
            # Basic sanity checks
            assert bar.t_start < bar.t_end
            assert bar.low <= bar.vwap <= bar.high
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.volume == int(np.sum(bar.trade_sizes))
            assert len(bar.trade_prices) == 500

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_default_n_is_500(self):
        """build_session_bars default n should be 500."""
        from lob_rl.barrier.bar_pipeline import build_session_bars

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))
        if not files:
            pytest.skip("No .dbn.zst files found")

        bars = build_session_bars(str(files[0]))
        if len(bars) > 0:
            assert len(bars[0].trade_prices) == 500


# ===========================================================================
# 9. build_dataset (batch processing)
# ===========================================================================


class TestBuildDatasetImport:
    """build_dataset should be importable."""

    def test_import(self):
        """build_dataset should be importable from lob_rl.barrier.bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import build_dataset
        assert callable(build_dataset)


class TestBuildDatasetReturnType:
    """build_dataset should return a DataFrame."""

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_returns_dataframe(self):
        """build_dataset returns a pandas DataFrame."""
        from lob_rl.barrier.bar_pipeline import build_dataset

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))[:2]
        if len(files) < 2:
            pytest.skip("Need at least 2 .dbn.zst files")

        df = build_dataset([str(f) for f in files], n=500)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_dataframe_columns(self):
        """DataFrame should have columns matching TradeBar fields (minus trade arrays)."""
        from lob_rl.barrier.bar_pipeline import build_dataset

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))[:2]
        if len(files) < 2:
            pytest.skip("Need at least 2 .dbn.zst files")

        df = build_dataset([str(f) for f in files], n=500)
        for col in ["bar_index", "open", "high", "low", "close", "volume",
                     "vwap", "t_start", "t_end", "session_date"]:
            assert col in df.columns, f"Missing column: {col}"


class TestBuildDatasetOutputArtifacts:
    """build_dataset should write parquet and trade sequence files."""

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_writes_parquet_file(self):
        """build_dataset with output_path creates bars.parquet."""
        from lob_rl.barrier.bar_pipeline import build_dataset

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))[:1]
        if not files:
            pytest.skip("No .dbn.zst files found")

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dataset([str(f) for f in files], n=500, output_path=tmpdir)
            parquet_path = os.path.join(tmpdir, "bars.parquet")
            assert os.path.exists(parquet_path), "bars.parquet not created"

            # Verify the parquet can be read back
            df = pd.read_parquet(parquet_path)
            assert len(df) > 0

    @pytest.mark.skipif(
        not os.path.exists("data/mes"),
        reason="No MBO data files available",
    )
    def test_writes_trade_sequences(self):
        """build_dataset with output_path creates trade_sequences/ directory."""
        from lob_rl.barrier.bar_pipeline import build_dataset

        data_dir = Path("data/mes")
        files = sorted(data_dir.glob("*.dbn.zst"))[:1]
        if not files:
            pytest.skip("No .dbn.zst files found")

        with tempfile.TemporaryDirectory() as tmpdir:
            build_dataset([str(f) for f in files], n=500, output_path=tmpdir)
            seq_dir = os.path.join(tmpdir, "trade_sequences")
            assert os.path.isdir(seq_dir), "trade_sequences/ not created"

            # Should have .npy files
            npy_files = list(Path(seq_dir).glob("*.npy"))
            assert len(npy_files) > 0, "No .npy files in trade_sequences/"


# ===========================================================================
# 10. Configurable bar sizes
# ===========================================================================


class TestConfigurableBarSize:
    """Spec default: N=500, configurable for sweep {200, 500, 1000, 2000}."""

    def test_bar_size_200(self):
        """N=200 should produce correct number of bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(600)
        bars = build_bars_from_trades(trades, n=200, session_date="2022-06-15")
        assert len(bars) == 3
        for bar in bars:
            assert len(bar.trade_prices) == 200

    def test_bar_size_1000(self):
        """N=1000 should produce correct number of bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(2500)
        bars = build_bars_from_trades(trades, n=1000, session_date="2022-06-15")
        assert len(bars) == 2
        for bar in bars:
            assert len(bar.trade_prices) == 1000

    def test_bar_size_2000(self):
        """N=2000 should produce correct number of bars."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        trades = _make_trades(4000)
        bars = build_bars_from_trades(trades, n=2000, session_date="2022-06-15")
        assert len(bars) == 2
        for bar in bars:
            assert len(bar.trade_prices) == 2000


# ===========================================================================
# 11. Stress and multi-bar consistency
# ===========================================================================


class TestMultiBarConsistency:
    """All bars in a sequence must satisfy invariants simultaneously."""

    def test_all_invariants_hold_across_20_bars(self):
        """For 20 bars: VWAP in [L, H], t_start < t_end, no overlap, volume matches."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        rng = np.random.default_rng(99)
        n_per_bar = 200
        n_bars = 20
        total = n_per_bar * n_bars

        prices = 4700.0 + rng.uniform(-5, 5, size=total)
        sizes = rng.integers(1, 20, size=total)
        trades = _make_trades_with_prices(prices, sizes)

        bars = build_bars_from_trades(trades, n=n_per_bar, session_date="2022-06-15")
        assert len(bars) == n_bars

        for i, bar in enumerate(bars):
            # Bar index
            assert bar.bar_index == i
            # OHLCV consistency
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.low <= bar.vwap <= bar.high
            # Timestamps
            assert bar.t_start < bar.t_end
            # Trade sequence
            assert len(bar.trade_prices) == n_per_bar
            assert len(bar.trade_sizes) == n_per_bar
            assert bar.trade_prices[0] == bar.open
            assert bar.trade_prices[-1] == bar.close
            assert np.max(bar.trade_prices) == bar.high
            assert np.min(bar.trade_prices) == bar.low
            # Volume
            assert bar.volume == int(np.sum(bar.trade_sizes))
            # Session date
            assert bar.session_date == "2022-06-15"

        # No temporal overlap between consecutive bars
        for i in range(len(bars) - 1):
            assert bars[i].t_end <= bars[i + 1].t_start


class TestAllTradesAccountedFor:
    """Every trade in the input that maps to a complete bar must appear in output."""

    def test_total_trade_count(self):
        """Sum of trade_prices lengths across all bars == n_bars * N."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        n = 500
        total_trades = 1723  # 3 complete bars, 223 remainder
        trades = _make_trades(total_trades)
        bars = build_bars_from_trades(trades, n=n, session_date="2022-06-15")

        total_in_bars = sum(len(bar.trade_prices) for bar in bars)
        assert total_in_bars == len(bars) * n
        assert total_in_bars == 1500  # 3 * 500

    def test_trade_prices_match_input_sequence(self):
        """Trade prices within each bar should match the input trade sequence."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [100.0 + i * 0.01 for i in range(10)]
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=5, session_date="2022-06-15")

        assert len(bars) == 2
        # First bar: trades 0-4
        np.testing.assert_array_equal(
            bars[0].trade_prices,
            np.array(prices[:5], dtype=np.float64),
        )
        # Second bar: trades 5-9
        np.testing.assert_array_equal(
            bars[1].trade_prices,
            np.array(prices[5:10], dtype=np.float64),
        )


# ===========================================================================
# 12. Varied trade sizes (non-uniform)
# ===========================================================================


class TestVariedTradeSizes:
    """Volume and VWAP must be correct with non-uniform trade sizes."""

    def test_volume_with_varying_sizes(self):
        """Volume should be sum of all trade sizes, not count of trades."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        sizes = [1, 5, 10, 20, 50, 2, 3, 7, 8, 4]  # 10 trades, total vol = 110
        prices = [100.0] * 10
        trades = _make_trades_with_prices(prices, sizes)
        bars = build_bars_from_trades(trades, n=10, session_date="2022-06-15")
        assert len(bars) == 1
        assert bars[0].volume == 110

    def test_vwap_with_varying_sizes(self):
        """VWAP with non-uniform sizes is correctly volume-weighted."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        # 2 trades: 100.0 x 90 contracts, 200.0 x 10 contracts
        # VWAP = (100*90 + 200*10) / 100 = (9000 + 2000) / 100 = 110.0
        prices = [100.0, 200.0]
        sizes = [90, 10]
        trades = _make_trades_with_prices(prices, sizes)
        bars = build_bars_from_trades(trades, n=2, session_date="2022-06-15")
        assert len(bars) == 1
        assert bars[0].vwap == pytest.approx(110.0)


# ===========================================================================
# 13. Single-price bars (all trades at same price)
# ===========================================================================


class TestSinglePriceBar:
    """Edge case: all trades at the same price."""

    def test_single_price_ohlcv(self):
        """When all trades have the same price, O=H=L=C=VWAP."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        prices = [4775.25] * 500
        trades = _make_trades_with_prices(prices)
        bars = build_bars_from_trades(trades, n=500, session_date="2022-06-15")
        assert len(bars) == 1
        bar = bars[0]
        assert bar.open == 4775.25
        assert bar.high == 4775.25
        assert bar.low == 4775.25
        assert bar.close == 4775.25
        assert bar.vwap == pytest.approx(4775.25)


# ===========================================================================
# 14. Timestamp edge cases
# ===========================================================================


class TestTimestampEdgeCases:
    """Bars with trades having identical or very close timestamps."""

    def test_single_timestamp_all_trades(self):
        """All trades at the same timestamp should still produce valid bar.

        t_start should equal t_end when all trades are simultaneous.
        Actually, spec says t_start < t_end. Let's verify the implementation
        handles this edge case — at minimum, t_start <= t_end.
        """
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        ts = _utc_ns(2022, 6, 15, 15, 0)
        trades = _make_trades(5, ts_start_ns=ts, ts_step_ns=0)
        bars = build_bars_from_trades(trades, n=5, session_date="2022-06-15")
        assert len(bars) == 1
        # With all same timestamp, t_start == t_end is acceptable
        assert bars[0].t_start <= bars[0].t_end

    def test_t_start_is_first_trade_timestamp(self):
        """t_start should be the timestamp of the first trade in the bar."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        ts_start = _utc_ns(2022, 6, 15, 14, 0)
        step = 1_000_000  # 1ms
        trades = _make_trades(10, ts_start_ns=ts_start, ts_step_ns=step)
        bars = build_bars_from_trades(trades, n=5, session_date="2022-06-15")
        assert len(bars) == 2
        assert bars[0].t_start == ts_start
        assert bars[1].t_start == ts_start + 5 * step

    def test_t_end_is_last_trade_timestamp(self):
        """t_end should be the timestamp of the last trade in the bar."""
        from lob_rl.barrier.bar_pipeline import build_bars_from_trades

        ts_start = _utc_ns(2022, 6, 15, 14, 0)
        step = 1_000_000
        trades = _make_trades(10, ts_start_ns=ts_start, ts_step_ns=step)
        bars = build_bars_from_trades(trades, n=5, session_date="2022-06-15")
        assert bars[0].t_end == ts_start + 4 * step
        assert bars[1].t_end == ts_start + 9 * step
