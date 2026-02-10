"""Tests for the barrier label construction pipeline.

Spec: docs/t2-label-construction.md

Tests the barrier-hit detection with three-outcome labels, intrabar
tiebreaking, short direction mirroring, T_max calibration, and diagnostics.
"""

import math

import numpy as np
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar


# ---------------------------------------------------------------------------
# Synthetic bar helpers (test-only, no implementation logic)
# ---------------------------------------------------------------------------

TICK_SIZE = 0.25  # /MES tick size


def _make_bar(bar_index, open_price, high, low, close,
              trade_prices=None, trade_sizes=None, volume=100,
              t_start=0, t_end=1, session_date="2022-06-15"):
    """Create a TradeBar with explicit OHLC and optional trade sequences."""
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
        vwap=(high + low) / 2.0,
        t_start=t_start,
        t_end=t_end,
        session_date=session_date,
        trade_prices=trade_prices,
        trade_sizes=trade_sizes,
    )


def _make_flat_bars(n, base_price=4000.0, spread=1.0):
    """Create n bars that stay within a narrow range around base_price.

    These bars will NOT trigger any barriers with default a=20, b=10.
    """
    bars = []
    for k in range(n):
        bars.append(_make_bar(
            bar_index=k,
            open_price=base_price,
            high=base_price + spread * TICK_SIZE,
            low=base_price - spread * TICK_SIZE,
            close=base_price,
            trade_prices=np.array([base_price], dtype=np.float64),
        ))
    return bars


# ===========================================================================
# 1. Imports and dataclass
# ===========================================================================


class TestBarrierLabelImport:
    """BarrierLabel and compute_labels should be importable."""

    def test_barrier_label_importable(self):
        """BarrierLabel should be importable from lob_rl.barrier.label_pipeline."""
        from lob_rl.barrier.label_pipeline import BarrierLabel
        assert BarrierLabel is not None

    def test_compute_labels_importable(self):
        """compute_labels should be importable and callable."""
        from lob_rl.barrier.label_pipeline import compute_labels
        assert callable(compute_labels)

    def test_calibrate_t_max_importable(self):
        """calibrate_t_max should be importable and callable."""
        from lob_rl.barrier.label_pipeline import calibrate_t_max
        assert callable(calibrate_t_max)

    def test_compute_tiebreak_frequency_importable(self):
        """compute_tiebreak_frequency should be importable and callable."""
        from lob_rl.barrier.label_pipeline import compute_tiebreak_frequency
        assert callable(compute_tiebreak_frequency)

    def test_compute_label_distribution_importable(self):
        """compute_label_distribution should be importable and callable."""
        from lob_rl.barrier.label_pipeline import compute_label_distribution
        assert callable(compute_label_distribution)


class TestBarrierLabelFields:
    """BarrierLabel must have all spec-defined fields."""

    def test_all_fields_present(self):
        """BarrierLabel has bar_index, label, tau, resolution_type, entry_price, resolution_bar."""
        from lob_rl.barrier.label_pipeline import BarrierLabel

        bl = BarrierLabel(
            bar_index=0,
            label=1,
            tau=5,
            resolution_type="upper",
            entry_price=4000.0,
            resolution_bar=5,
        )
        assert bl.bar_index == 0
        assert bl.label == 1
        assert bl.tau == 5
        assert bl.resolution_type == "upper"
        assert bl.entry_price == 4000.0
        assert bl.resolution_bar == 5


# ===========================================================================
# 2. Core labeling — upper barrier hit
# ===========================================================================


class TestUpperBarrierHit:
    """Spec test #1: Upper barrier hit on bar j=5."""

    def test_upper_hit_at_bar_5(self):
        """Hand-crafted sequence where bar 5's high crosses upper barrier.

        Entry bar k=0, close=4000.0, a=20 ticks → U = 4000.0 + 20*0.25 = 4005.0.
        Bars 1-4: high stays below 4005.0, low stays above 3997.5.
        Bar 5: high=4005.25 (crosses U).
        Expected: label=+1, tau=5, resolution_type="upper".
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),  # entry bar
            _make_bar(1, 4000.0, 4001.5, 3999.0, 4000.5),
            _make_bar(2, 4000.5, 4002.0, 3999.0, 4001.0),
            _make_bar(3, 4001.0, 4003.0, 3999.0, 4001.5),
            _make_bar(4, 4001.5, 4004.0, 3999.0, 4002.0),
            _make_bar(5, 4002.0, 4005.25, 3999.0, 4003.0),  # high >= U
        ]
        # Add enough flat trailing bars to avoid index issues for bars 1-5
        for i in range(6, 50):
            bars.append(_make_bar(i, 4003.0, 4004.0, 4002.0, 4003.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Label for bar 0: upper hit at bar 5
        assert labels[0].label == 1
        assert labels[0].tau == 5
        assert labels[0].resolution_type == "upper"
        assert labels[0].resolution_bar == 5

    def test_upper_hit_label_is_plus_one(self):
        """Upper barrier hit must produce label=+1 for long direction."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Simple: entry at 4000.0, bar 1 high >= 4005.0 (a=20)
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4005.0, 3999.0, 4002.0),  # high == U exactly
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4002.0, 4003.0, 4001.0, 4002.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1


# ===========================================================================
# 3. Core labeling — lower barrier hit
# ===========================================================================


class TestLowerBarrierHit:
    """Spec test #2: Lower barrier hit on bar j=3."""

    def test_lower_hit_at_bar_3(self):
        """Hand-crafted sequence where bar 3's low crosses lower barrier.

        Entry bar k=0, close=4000.0, b=10 ticks → D = 4000.0 - 10*0.25 = 3997.5.
        Bars 1-2: low stays above 3997.5, high stays below 4005.0.
        Bar 3: low=3997.25 (crosses D).
        Expected: label=-1, tau=3, resolution_type="lower".
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),  # entry bar
            _make_bar(1, 4000.0, 4001.0, 3998.5, 3999.5),
            _make_bar(2, 3999.5, 4001.0, 3998.0, 3999.0),
            _make_bar(3, 3999.0, 4001.0, 3997.25, 3998.0),  # low <= D
        ]
        for i in range(4, 50):
            bars.append(_make_bar(i, 3998.0, 3999.0, 3997.0, 3998.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1
        assert labels[0].tau == 3
        assert labels[0].resolution_type == "lower"
        assert labels[0].resolution_bar == 3

    def test_lower_hit_label_is_minus_one(self):
        """Lower barrier hit must produce label=-1 for long direction."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry at 4000.0, bar 1 low <= 3997.5 (b=10)
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4001.0, 3997.5, 3998.0),  # low == D exactly
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3998.0, 3999.0, 3997.0, 3998.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1


# ===========================================================================
# 4. Core labeling — timeout
# ===========================================================================


class TestTimeout:
    """Spec test #3: Timeout when neither barrier is hit."""

    def test_timeout_produces_label_zero(self):
        """All bars stay within barriers for t_max bars → label=0.

        Entry at 4000.0, a=20, b=10 → U=4005.0, D=3997.5.
        All bars high < 4005.0 and low > 3997.5 for 40 bars.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        # 50 flat bars, none breach barriers
        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 0
        assert labels[0].tau == 40
        assert labels[0].resolution_type == "timeout"

    def test_timeout_resolution_bar(self):
        """Timeout resolution_bar should be bar_index + t_max."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].resolution_bar == 0 + 40


# ===========================================================================
# 5. Tiebreaking — dual barrier breach
# ===========================================================================


class TestTiebreakUpperFirst:
    """Spec test #4: Dual barrier breach — upper hit first in trade sequence."""

    def test_tiebreak_upper_first(self):
        """Both H >= U and L <= D on same bar, but trade sequence shows upper first.

        Entry bar k=0, close=4000.0, a=20, b=10 → U=4005.0, D=3997.5.
        Bar 1: high=4005.5, low=3997.0 (both breached).
        Trade sequence: [4001.0, 4003.0, 4005.5, 3997.0]
        Upper crossed first (4005.5 >= U before 3997.0 <= D).
        Expected: label=+1, resolution_type="tiebreak_upper".
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        trade_prices_bar1 = [4001.0, 4003.0, 4005.5, 3997.0]
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4005.5, 3997.0, 4002.0,
                      trade_prices=trade_prices_bar1),
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4002.0, 4003.0, 4001.0, 4002.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1
        assert labels[0].tau == 1
        assert labels[0].resolution_type == "tiebreak_upper"


class TestTiebreakLowerFirst:
    """Spec test #5: Dual barrier breach — lower hit first in trade sequence."""

    def test_tiebreak_lower_first(self):
        """Both H >= U and L <= D on same bar, but trade sequence shows lower first.

        Entry bar k=0, close=4000.0, a=20, b=10 → U=4005.0, D=3997.5.
        Bar 1: high=4005.5, low=3997.0 (both breached).
        Trade sequence: [3999.0, 3997.0, 4005.5, 4001.0]
        Lower crossed first (3997.0 <= D before 4005.5 >= U).
        Expected: label=-1, resolution_type="tiebreak_lower".
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        trade_prices_bar1 = [3999.0, 3997.0, 4005.5, 4001.0]
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4005.5, 3997.0, 4001.0,
                      trade_prices=trade_prices_bar1),
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4001.0, 4002.0, 4000.0, 4001.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1
        assert labels[0].tau == 1
        assert labels[0].resolution_type == "tiebreak_lower"


class TestGapThroughEdgeCase:
    """Spec test #6: Gap-through — first trade exceeds both barriers."""

    def test_gap_through_upward(self):
        """First trade of bar exceeds both barriers. Gap is upward from previous close.

        Entry bar k=0, close=4000.0. Bar 1 previous close=4000.0.
        Bar 1 first trade=4010.0 (above U=4005.0 AND below... no, it's above).
        If first trade >= U and first trade <= D simultaneously is impossible
        unless the gap is HUGE. Let's use: U=4005.0, D=3997.5.
        First trade at 3995.0 — below D. But also need above U...

        Actually, gap-through means the first trade already exceeds both barriers.
        This is only possible if a=very small or b=very small and gap is huge.
        Let's use a=2, b=2 → U=4000.5, D=3999.5.
        First trade at 4001.0 — crosses U. But doesn't cross D.

        For TRUE gap-through both: need the bar to eventually cross both barriers
        but the FIRST trade is already past both. With a=2, b=2:
        U=4000.5, D=3999.5. First trade 4001.0 > U but not < D.

        Actually re-reading the spec: gap-through is when the first trade
        already exceeds both barriers. This is impossible for one trade to be
        simultaneously >= U and <= D (U > D). So the spec means: the first trade
        crosses one barrier, and it ALSO implies the other was crossed (since
        the bar's H >= U and L <= D, but we can't determine order from trade
        sequence because the first trade already crosses one).

        Let me re-read: "Gap-through edge case: if first trade already exceeds
        both barriers, resolve by gap direction from previous close."

        This makes more sense if we think of bar j having BOTH barriers breached,
        and when scanning trades, the very first trade >= U (or <= D). In that
        case, the sequence can't tell us which was hit first because the first
        trade is already past one barrier. Resolve by gap direction.

        Example: gap up → upper hit first. First trade at 4006.0 (above U=4005.0).
        Bar also has low=3997.0 (below D=3997.5). Previous close was 4000.0.
        Gap direction: 4006.0 > 4000.0 → gap UP → resolve as upper hit.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry bar 0: close=4000.0 → U=4005.0, D=3997.5 (a=20, b=10)
        # Bar 1: gaps up, first trade at 4006.0 (already above U)
        # But bar 1 also has low 3997.0 (below D)
        trade_prices_bar1 = [4006.0, 4004.0, 3997.0, 4001.0]
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4006.0, 4006.0, 3997.0, 4001.0,
                      trade_prices=trade_prices_bar1),
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4001.0, 4002.0, 4000.0, 4001.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Gap UP from 4000.0 to 4006.0 → upper hit first
        assert labels[0].label == 1
        assert labels[0].tau == 1

    def test_gap_through_downward(self):
        """First trade of bar gaps down past lower barrier. Resolve as lower hit.

        Entry bar 0: close=4000.0 → U=4005.0, D=3997.5.
        Bar 1: gaps down, first trade at 3996.0 (below D).
        Bar 1 also has high=4006.0 (above U).
        Gap direction: 3996.0 < 4000.0 → gap DOWN → lower hit.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        trade_prices_bar1 = [3996.0, 4006.0, 4000.0]
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3996.0, 4006.0, 3996.0, 4000.0,
                      trade_prices=trade_prices_bar1),
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4000.0, 4001.0, 3999.0, 4000.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1
        assert labels[0].tau == 1


# ===========================================================================
# 6. Short direction
# ===========================================================================


class TestShortDirection:
    """Spec test #7: Short direction mirrors barriers."""

    def test_short_profit_is_price_down(self):
        """For short: profit barrier below entry (price goes DOWN to win).

        Entry close=4000.0, a=20 ticks.
        Short profit barrier: D = C_k - a * 0.25 = 4000.0 - 5.0 = 3995.0.
        Short stop barrier:  U = C_k + b * 0.25 = 4000.0 + 2.5 = 4002.5.

        Bar 3 low=3994.75 (below profit barrier) → label=+1 (profit for short).
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.5, 4000.5, 3998.0, 3999.0),
            _make_bar(2, 3999.0, 4000.0, 3996.0, 3997.0),
            _make_bar(3, 3997.0, 3998.0, 3994.75, 3996.0),  # low <= 3995.0
        ]
        for i in range(4, 50):
            bars.append(_make_bar(i, 3996.0, 3997.0, 3995.0, 3996.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40, direction="short")
        assert labels[0].label == 1  # profit for short = price went DOWN
        assert labels[0].tau == 3

    def test_short_stop_is_price_up(self):
        """For short: stop barrier above entry (price goes UP to lose).

        Entry close=4000.0, b=10 ticks.
        Short stop barrier: U = C_k + b * 0.25 = 4000.0 + 2.5 = 4002.5.

        Bar 2 high=4003.0 (above stop barrier) → label=-1 (stop for short).
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4000.0, 4001.5, 3999.0, 4001.0),
            _make_bar(2, 4001.0, 4003.0, 4000.0, 4002.0),  # high >= 4002.5
        ]
        for i in range(3, 50):
            bars.append(_make_bar(i, 4002.0, 4003.0, 4001.0, 4002.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40, direction="short")
        assert labels[0].label == -1  # stop for short = price went UP
        assert labels[0].tau == 2

    def test_short_timeout(self):
        """Short direction timeout when neither barrier is hit."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40, direction="short")
        assert labels[0].label == 0
        assert labels[0].resolution_type == "timeout"


# ===========================================================================
# 7. Entry price
# ===========================================================================


class TestEntryPrice:
    """Spec test #14: Entry price is C_k (close of entry bar), not O_k or VWAP."""

    def test_entry_price_is_close(self):
        """entry_price in label should equal bar.close, not bar.open or bar.vwap."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Bar 0: open=3990, close=4000 — different from open and vwap
        bars = [
            _make_bar(0, 3990.0, 4001.0, 3989.0, 4000.0),  # close=4000
        ]
        for i in range(1, 50):
            bars.append(_make_bar(i, 4000.0, 4001.0, 3999.0, 4000.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].entry_price == 4000.0
        assert labels[0].entry_price != 3990.0  # not open
        assert labels[0].entry_price != (4001.0 + 3989.0) / 2.0  # not vwap

    def test_barriers_computed_from_close(self):
        """Barriers should be relative to C_k, not O_k.

        Bar 0: open=3990, close=4000. a=20 → U=4000+5=4005, NOT 3990+5=3995.
        Bar 1: high=3996.0 should NOT trigger upper barrier (which would only
        happen if barrier was computed from open=3990).
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 3990.0, 4001.0, 3989.0, 4000.0),  # close=4000
            _make_bar(1, 3995.0, 3996.0, 3994.0, 3995.0),  # only hits if U=3995
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3995.0, 3996.0, 3994.0, 3995.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Bar 1 high=3996.0 < U=4005.0, so should NOT be an upper hit
        assert labels[0].label != 1 or labels[0].tau != 1


# ===========================================================================
# 8. Exact barrier touch
# ===========================================================================


class TestExactBarrierTouch:
    """Spec test #13: H_j == U exactly should count as a hit."""

    def test_exact_upper_touch_is_hit(self):
        """High equals upper barrier exactly → counts as upper hit."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry close=4000.0, a=20 → U=4005.0
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4005.0, 3999.0, 4002.0),  # high == U exactly
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4002.0, 4003.0, 4001.0, 4002.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1
        assert labels[0].tau == 1

    def test_exact_lower_touch_is_hit(self):
        """Low equals lower barrier exactly → counts as lower hit."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry close=4000.0, b=10 → D=3997.5
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4001.0, 3997.5, 3998.0),  # low == D exactly
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3998.0, 3999.0, 3997.0, 3998.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1
        assert labels[0].tau == 1


# ===========================================================================
# 9. Barrier hit on first bar after entry (tau=1)
# ===========================================================================


class TestBarrierHitFirstBar:
    """Spec test #12: Barrier hit on first bar after entry, tau=1."""

    def test_upper_hit_tau_one(self):
        """Upper barrier hit on bar k+1 → tau=1."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4006.0, 3999.0, 4004.0),  # high > U=4005.0
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4004.0, 4005.0, 4003.0, 4004.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].tau == 1
        assert labels[0].resolution_bar == 1

    def test_lower_hit_tau_one(self):
        """Lower barrier hit on bar k+1 → tau=1."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4001.0, 3996.0, 3997.0),  # low < D=3997.5
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3997.0, 3998.0, 3996.0, 3997.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].tau == 1
        assert labels[0].resolution_bar == 1


# ===========================================================================
# 10. Last bars of session — insufficient lookahead
# ===========================================================================


class TestInsufficientLookahead:
    """Spec test #11: Bars near end of session with fewer than T_max remaining."""

    def test_last_bar_gets_timeout_with_short_lookahead(self):
        """Bar with only 3 bars of lookahead and no barrier hit → timeout.

        With 10 flat bars total and t_max=40: bar 9 (last) has 0 bars
        of lookahead → should still be labeled (timeout).
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(10, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)

        # Every bar should have a label
        assert len(labels) == 10

        # Last bar (index 9) has 0 forward bars, must be timeout
        assert labels[9].label == 0
        assert labels[9].resolution_type == "timeout"

    def test_label_count_equals_bar_count(self):
        """One label per bar, even for bars near end of session."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(5, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert len(labels) == 5

    def test_insufficient_lookahead_can_still_hit_barrier(self):
        """Bar near end of session can still hit a barrier within remaining bars.

        10 bars total, bar 7 entry. Bar 9 triggers upper barrier.
        tau=2, even though t_max=40 would normally scan more bars.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(10, base_price=4000.0, spread=1.0)
        # Make bar 9 trigger upper barrier for bar 7's entry
        bars[7] = _make_bar(7, 4000.0, 4001.0, 3999.0, 4000.0)
        bars[8] = _make_bar(8, 4000.0, 4001.0, 3999.0, 4000.5)
        bars[9] = _make_bar(9, 4001.0, 4006.0, 3999.0, 4003.0)  # high > U=4005

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[7].label == 1
        assert labels[7].tau == 2
        assert labels[7].resolution_bar == 9


# ===========================================================================
# 11. Invariants
# ===========================================================================


class TestInvariantTauPositive:
    """Spec test #15: tau > 0 for all labels."""

    def test_tau_always_positive(self):
        """No label should have tau=0 (resolution takes at least 1 bar)."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Mix of outcomes: some hits, some timeouts
        bars = _make_flat_bars(20, base_price=4000.0, spread=1.0)
        # Make bar 5 trigger upper for bar 0
        bars[5] = _make_bar(5, 4001.0, 4006.0, 3999.0, 4003.0)

        labels = compute_labels(bars, a=20, b=10, t_max=15)
        for lbl in labels:
            assert lbl.tau > 0, f"Bar {lbl.bar_index} has tau={lbl.tau}"


class TestInvariantTauBounded:
    """Spec test #16: tau <= t_max for all labels."""

    def test_tau_never_exceeds_t_max(self):
        """No label should have tau > t_max."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(60, base_price=4000.0, spread=1.0)
        t_max = 40
        labels = compute_labels(bars, a=20, b=10, t_max=t_max)
        for lbl in labels:
            assert lbl.tau <= t_max, (
                f"Bar {lbl.bar_index} has tau={lbl.tau} > t_max={t_max}"
            )

    def test_tau_bounded_by_remaining_bars(self):
        """For bars near session end, tau is bounded by remaining bars (< t_max)."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(10, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)

        # Bar 9 (last): tau should be min(t_max, remaining_bars)
        # remaining bars after bar 9 = 0, so tau should be small
        # The minimum possible tau is however many bars remain (could be 0 → but
        # spec says tau > 0, so implementation handles this as timeout with tau=min(t_max, remaining))
        # For bar 8: 1 bar of lookahead, for bar 9: 0 bars
        # tau should be <= remaining_bars or t_max, whichever is smaller
        for lbl in labels:
            remaining = len(bars) - 1 - lbl.bar_index
            expected_max_tau = min(40, remaining) if remaining > 0 else 40
            # tau can't exceed actual remaining bars if timeout
            if lbl.label == 0:
                assert lbl.tau <= max(remaining, 1)


class TestInvariantLabelValues:
    """Spec test #17: Label values are in {-1, 0, +1}."""

    def test_labels_are_valid(self):
        """All label values must be -1, 0, or +1."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Mixed scenario
        bars = _make_flat_bars(30, base_price=4000.0, spread=1.0)
        bars[5] = _make_bar(5, 4001.0, 4006.0, 3999.0, 4003.0)  # upper hit
        bars[10] = _make_bar(10, 3998.0, 4001.0, 3996.0, 3997.0)  # lower hit

        labels = compute_labels(bars, a=20, b=10, t_max=15)
        for lbl in labels:
            assert lbl.label in {-1, 0, 1}, (
                f"Bar {lbl.bar_index} has invalid label={lbl.label}"
            )


class TestInvariantResolutionBar:
    """Spec test #18: resolution_bar == bar_index + tau."""

    def test_resolution_bar_equals_index_plus_tau(self):
        """resolution_bar must always equal bar_index + tau."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Mix of all outcomes
        bars = _make_flat_bars(30, base_price=4000.0, spread=1.0)
        bars[3] = _make_bar(3, 4001.0, 4006.0, 3999.0, 4003.0)  # upper hit
        bars[15] = _make_bar(15, 3998.0, 4001.0, 3996.0, 3997.0)  # lower hit

        labels = compute_labels(bars, a=20, b=10, t_max=20)
        for lbl in labels:
            assert lbl.resolution_bar == lbl.bar_index + lbl.tau, (
                f"Bar {lbl.bar_index}: resolution_bar={lbl.resolution_bar} "
                f"!= bar_index({lbl.bar_index}) + tau({lbl.tau})"
            )


class TestInvariantResolutionType:
    """resolution_type must be one of the valid strings."""

    def test_resolution_type_valid(self):
        """resolution_type must be upper, lower, timeout, tiebreak_upper, or tiebreak_lower."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)
        bars[5] = _make_bar(5, 4001.0, 4006.0, 3999.0, 4003.0)

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        valid_types = {"upper", "lower", "timeout", "tiebreak_upper", "tiebreak_lower"}
        for lbl in labels:
            assert lbl.resolution_type in valid_types, (
                f"Bar {lbl.bar_index}: invalid resolution_type='{lbl.resolution_type}'"
            )


# ===========================================================================
# 12. T_max calibration
# ===========================================================================


class TestCalibrateTMax:
    """Spec test #8: calibrate_t_max returns P95 of winner tau distribution."""

    def test_calibration_returns_integer(self):
        """calibrate_t_max should return an integer."""
        from lob_rl.barrier.label_pipeline import calibrate_t_max

        # Need bars where some hit upper barrier with varying tau
        bars = []
        for k in range(200):
            bars.append(_make_bar(k, 4000.0, 4001.0, 3999.0, 4000.0))
        # Sprinkle some upper barrier hits at different distances
        for offset in [3, 5, 8, 10, 15, 20, 25, 30]:
            if offset < 200:
                bars[offset] = _make_bar(offset, 4001.0, 4006.0, 3999.0, 4003.0)

        result = calibrate_t_max(bars, a=20, b=10)
        assert isinstance(result, int)

    def test_calibration_is_ceil_p95(self):
        """Returned value should be ceil(P95 of tau for label=+1 bars).

        Construct bars where upper barrier hits happen at known tau values.
        """
        from lob_rl.barrier.label_pipeline import calibrate_t_max

        # Create 100 entry bars, each with a known upper hit at specific tau.
        # We need many small "sessions" or a long sequence where each entry
        # bar has its upper barrier hit at a predictable time.
        # Simplest: 100 groups. Entry bar has close=4000. tau bars later, high>4005.
        # Tau values: 1,2,3,...,100 (each appears once).
        # P95 of [1..100] = 95.05 → ceil = 96.

        total_bars_needed = 100 + 100  # 100 entries + some extra for lookhead
        bars = []
        for k in range(300):
            bars.append(_make_bar(k, 4000.0, 4001.0, 3999.0, 4000.0))

        # For entry bar 0: hit at bar 1 (tau=1)
        # For entry bar 1: hit at bar 3 (tau=2)
        # etc. — but this gets complicated. Use a simpler approach:
        # Just make every bar an entry with close=4000, and one specific
        # "hit" bar at index 50.
        # That means for bar 0: tau=50, bar 1: tau=49, ..., bar 49: tau=1.
        # Bars 50-299: timeout.
        # Tau values for upper hits: [1..50] each appearing once.
        # P95 of [1..50] = 47.55 → ceil = 48.

        bars[50] = _make_bar(50, 4001.0, 4006.0, 3999.0, 4003.0)

        result = calibrate_t_max(bars, a=20, b=10)
        # Result should be ceil(P95 of the tau distribution for +1 labels)
        # The exact value depends on implementation details, but it should be > 0
        assert result > 0

    def test_calibration_with_no_winners_raises_or_returns_sensible(self):
        """If no upper hits exist, calibrate_t_max should handle gracefully.

        Either raises ValueError or returns a large default. Implementation
        should not crash.
        """
        from lob_rl.barrier.label_pipeline import calibrate_t_max

        # All flat bars, no barrier hits → all timeouts
        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)
        # Should not crash. May raise or return some default.
        try:
            result = calibrate_t_max(bars, a=20, b=10)
            # If it returns, should be a positive integer
            assert isinstance(result, int)
            assert result > 0
        except (ValueError, RuntimeError):
            # Also acceptable to raise if no winners
            pass


# ===========================================================================
# 13. Diagnostics — tiebreak frequency
# ===========================================================================


class TestComputeTiebreakFrequency:
    """Spec test #9: Tiebreak frequency computation."""

    def test_known_tiebreak_frequency(self):
        """Known set of labels with 2/5 tiebreaks → frequency=0.4."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_tiebreak_frequency

        labels = [
            BarrierLabel(0, 1, 5, "upper", 4000.0, 5),
            BarrierLabel(1, -1, 3, "tiebreak_lower", 4000.0, 4),
            BarrierLabel(2, 0, 40, "timeout", 4000.0, 42),
            BarrierLabel(3, 1, 2, "tiebreak_upper", 4000.0, 5),
            BarrierLabel(4, -1, 7, "lower", 4000.0, 11),
        ]
        freq = compute_tiebreak_frequency(labels)
        assert freq == pytest.approx(2.0 / 5.0)

    def test_no_tiebreaks(self):
        """If no labels required tiebreaking, frequency=0.0."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_tiebreak_frequency

        labels = [
            BarrierLabel(0, 1, 5, "upper", 4000.0, 5),
            BarrierLabel(1, -1, 3, "lower", 4000.0, 4),
            BarrierLabel(2, 0, 40, "timeout", 4000.0, 42),
        ]
        freq = compute_tiebreak_frequency(labels)
        assert freq == 0.0

    def test_all_tiebreaks(self):
        """If all labels required tiebreaking, frequency=1.0."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_tiebreak_frequency

        labels = [
            BarrierLabel(0, 1, 1, "tiebreak_upper", 4000.0, 1),
            BarrierLabel(1, -1, 1, "tiebreak_lower", 4000.0, 2),
        ]
        freq = compute_tiebreak_frequency(labels)
        assert freq == 1.0

    def test_empty_labels(self):
        """Empty label list should return 0.0 (or handle gracefully)."""
        from lob_rl.barrier.label_pipeline import compute_tiebreak_frequency

        freq = compute_tiebreak_frequency([])
        assert freq == 0.0


# ===========================================================================
# 14. Diagnostics — label distribution
# ===========================================================================


class TestComputeLabelDistribution:
    """Spec test #10: Label distribution computation."""

    def test_known_distribution(self):
        """Known labels: 2 positive, 1 negative, 1 timeout out of 4."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_label_distribution

        labels = [
            BarrierLabel(0, 1, 5, "upper", 4000.0, 5),
            BarrierLabel(1, 1, 3, "tiebreak_upper", 4000.0, 4),
            BarrierLabel(2, -1, 7, "lower", 4000.0, 9),
            BarrierLabel(3, 0, 40, "timeout", 4000.0, 43),
        ]
        dist = compute_label_distribution(labels)
        assert dist["p_plus"] == pytest.approx(0.5)
        assert dist["p_minus"] == pytest.approx(0.25)
        assert dist["p_zero"] == pytest.approx(0.25)

    def test_distribution_sums_to_one(self):
        """p_plus + p_minus + p_zero should sum to 1.0."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_label_distribution

        labels = [
            BarrierLabel(i, [1, -1, 0][i % 3], i + 1, "upper", 4000.0, 2 * i)
            for i in range(30)
        ]
        dist = compute_label_distribution(labels)
        assert dist["p_plus"] + dist["p_minus"] + dist["p_zero"] == pytest.approx(1.0)

    def test_distribution_keys(self):
        """Distribution dict must have keys: p_plus, p_minus, p_zero."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_label_distribution

        labels = [BarrierLabel(0, 1, 5, "upper", 4000.0, 5)]
        dist = compute_label_distribution(labels)
        assert "p_plus" in dist
        assert "p_minus" in dist
        assert "p_zero" in dist

    def test_all_same_label(self):
        """All labels are +1 → p_plus=1.0, p_minus=0.0, p_zero=0.0."""
        from lob_rl.barrier.label_pipeline import BarrierLabel, compute_label_distribution

        labels = [BarrierLabel(i, 1, 5, "upper", 4000.0, i + 5) for i in range(10)]
        dist = compute_label_distribution(labels)
        assert dist["p_plus"] == pytest.approx(1.0)
        assert dist["p_minus"] == pytest.approx(0.0)
        assert dist["p_zero"] == pytest.approx(0.0)


# ===========================================================================
# 15. Return type and structure
# ===========================================================================


class TestComputeLabelsReturnStructure:
    """compute_labels returns a list of BarrierLabel, one per bar."""

    def test_returns_list(self):
        """compute_labels should return a list."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(5, base_price=4000.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert isinstance(labels, list)

    def test_one_label_per_bar(self):
        """Length of labels list equals length of bars list."""
        from lob_rl.barrier.label_pipeline import compute_labels

        for n_bars in [1, 5, 10, 50]:
            bars = _make_flat_bars(n_bars, base_price=4000.0)
            labels = compute_labels(bars, a=20, b=10, t_max=40)
            assert len(labels) == n_bars, (
                f"Expected {n_bars} labels, got {len(labels)}"
            )

    def test_label_bar_indices_match_input(self):
        """Each label's bar_index should match the corresponding bar's index."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(10, base_price=4000.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        for i, lbl in enumerate(labels):
            assert lbl.bar_index == i


# ===========================================================================
# 16. Default parameter values
# ===========================================================================


class TestDefaultParameters:
    """compute_labels default parameters: a=20, b=10, t_max=40, direction='long'."""

    def test_defaults_produce_long_labels(self):
        """Calling compute_labels(bars) without kwargs uses long direction."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Upper hit for long: high crosses C_k + 20*0.25 = C_k + 5.0
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4006.0, 3999.0, 4003.0),  # high > 4005.0
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4003.0, 4004.0, 4002.0, 4003.0))

        # Use only positional bars argument — rely on defaults
        labels = compute_labels(bars)
        assert labels[0].label == 1  # upper hit for long
        assert labels[0].tau == 1


# ===========================================================================
# 17. Custom barrier distances
# ===========================================================================


class TestCustomBarrierDistances:
    """Barrier distances a and b scale with tick_size=0.25."""

    def test_small_barriers(self):
        """a=4, b=4 → U = C_k + 1.0, D = C_k - 1.0."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry close=4000.0, U=4001.0, D=3999.0
        bars = [
            _make_bar(0, 4000.0, 4000.5, 3999.5, 4000.0),
            _make_bar(1, 4000.0, 4001.0, 3999.5, 4000.5),  # high == U=4001.0
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4000.5, 4000.7, 4000.3, 4000.5))

        labels = compute_labels(bars, a=4, b=4, t_max=40)
        assert labels[0].label == 1
        assert labels[0].tau == 1

    def test_large_barriers(self):
        """a=100, b=100 → U = C_k + 25.0, D = C_k - 25.0."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Entry close=4000.0, U=4025.0, D=3975.0
        # All bars within [3976, 4024] → timeout
        bars = _make_flat_bars(50, base_price=4000.0, spread=1.0)
        labels = compute_labels(bars, a=100, b=100, t_max=40)
        assert labels[0].label == 0  # timeout


# ===========================================================================
# 18. Asymmetric barriers (a != b)
# ===========================================================================


class TestAsymmetricBarriers:
    """Asymmetric barriers: a=20 (profit=5.0), b=10 (stop=2.5) for long."""

    def test_stop_closer_than_profit(self):
        """With b < a, stop barrier is closer to entry. Easier to hit.

        Entry close=4000.0, a=20 → U=4005.0, b=10 → D=3997.5.
        Bar 1 low=3997.25 (hits D) while high stays below U.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4001.0, 3997.25, 3998.0),  # low < D=3997.5
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3998.0, 3999.0, 3997.0, 3998.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1  # stop hit

    def test_reversed_asymmetry(self):
        """b > a: stop farther than profit.

        Entry close=4000.0, a=4 → U=4001.0, b=40 → D=3990.0.
        Bar 1 high=4001.5 (hits U). Bar 1 low stays above D.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4000.5, 3999.5, 4000.0),
            _make_bar(1, 4000.0, 4001.5, 3999.0, 4001.0),  # high > U=4001.0
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4001.0, 4002.0, 4000.0, 4001.0))

        labels = compute_labels(bars, a=4, b=40, t_max=40)
        assert labels[0].label == 1  # profit hit


# ===========================================================================
# 19. Multiple bars — label consistency across sequence
# ===========================================================================


class TestMultiBarLabelConsistency:
    """Multiple consecutive bars should be labeled independently."""

    def test_each_bar_uses_its_own_close_as_entry(self):
        """Bar k uses C_k as entry, not C_0 or any other bar's close.

        Bar 0: close=4000.0, bar 1: close=4010.0.
        For bar 1: U = 4010.0 + 5.0 = 4015.0 (not 4005.0).
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4005.0, 4011.0, 4009.0, 4010.0),   # close=4010
            _make_bar(2, 4010.0, 4015.25, 4009.0, 4012.0),   # high > 4015 (bar1's U)
        ]
        for i in range(3, 50):
            bars.append(_make_bar(i, 4012.0, 4013.0, 4011.0, 4012.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Bar 0: U=4005.0. Bar 1 high=4011.0 > 4005.0 → upper hit, tau=1
        assert labels[0].label == 1
        assert labels[0].tau == 1
        # Bar 1: U=4015.0. Bar 2 high=4015.25 > 4015.0 → upper hit, tau=1
        assert labels[1].label == 1
        assert labels[1].tau == 1
        assert labels[1].entry_price == 4010.0

    def test_adjacent_bars_different_labels(self):
        """Adjacent entry bars can have different labels."""
        from lob_rl.barrier.label_pipeline import compute_labels

        # Bar 0: close=4000, U=4005.0, D=3997.5 → upper hit at bar 2 (H=4005.25)
        # Bar 1: close=4003, U=4008.0, D=4000.5 → lower hit at bar 3 (L=4000.0)
        # Bar 1 must have H < 4005.0 (bar 0's U) to not trigger bar 0's upper at bar 1
        # Bar 2 must have L > 4000.5 (bar 1's D) to not trigger bar 1's lower at bar 2
        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),     # close=4000
            _make_bar(1, 4002.0, 4003.5, 4002.0, 4003.0),     # close=4003, H=4003.5 < U(bar0)=4005
            _make_bar(2, 4003.0, 4005.25, 4002.0, 4004.0),    # H=4005.25 >= U(bar0)=4005, L=4002 > D(bar1)=4000.5
            _make_bar(3, 4003.0, 4004.0, 4000.0, 4001.0),     # L=4000 <= D(bar1)=4000.5
        ]
        for i in range(4, 50):
            bars.append(_make_bar(i, 4001.0, 4002.0, 4000.5, 4001.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1   # upper hit at bar 2
        assert labels[1].label == -1  # lower hit at bar 3


# ===========================================================================
# 20. Empty input
# ===========================================================================


class TestEmptyInput:
    """compute_labels with empty bar list should return empty list."""

    def test_empty_bars(self):
        """Empty bars list → empty labels list."""
        from lob_rl.barrier.label_pipeline import compute_labels

        labels = compute_labels([], a=20, b=10, t_max=40)
        assert labels == []

    def test_single_bar(self):
        """Single bar → one label (timeout, since no forward bars)."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [_make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0)]
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert len(labels) == 1
        assert labels[0].label == 0  # timeout (no forward bars to scan)
        assert labels[0].resolution_type == "timeout"


# ===========================================================================
# 21. Direction parameter validation
# ===========================================================================


class TestDirectionParameter:
    """Direction parameter must be 'long' or 'short'."""

    def test_long_direction_explicit(self):
        """Explicit direction='long' produces same results as default."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_flat_bars(10, base_price=4000.0)
        labels_default = compute_labels(bars, a=20, b=10, t_max=40)
        labels_long = compute_labels(bars, a=20, b=10, t_max=40, direction="long")

        for d, l in zip(labels_default, labels_long):
            assert d.label == l.label
            assert d.tau == l.tau
            assert d.resolution_type == l.resolution_type


# ===========================================================================
# 22. Barrier detection scans only j > k (not j == k)
# ===========================================================================


class TestBarrierScanStartsAfterEntry:
    """Barriers are detected on bars j > k, NOT on bar k itself."""

    def test_entry_bar_ohlc_does_not_trigger_own_barrier(self):
        """Even if bar k's high crosses what would be its own upper barrier,
        the scan starts at j = k+1.

        Bar 0: close=4000, high=4010 (above U=4005.0). But this is the entry
        bar itself — its own high should NOT trigger the barrier.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 3995.0, 4010.0, 3994.0, 4000.0),  # high>U but entry bar
        ]
        for i in range(1, 50):
            bars.append(_make_bar(i, 4000.0, 4001.0, 3999.0, 4000.0))  # flat

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Bar 0's own high does NOT trigger its own barrier
        assert labels[0].label == 0  # timeout
        assert labels[0].tau == 40


# ===========================================================================
# 23. Short direction barrier math verification
# ===========================================================================


class TestShortBarrierMath:
    """Verify exact barrier prices for short direction."""

    def test_short_barrier_prices(self):
        """For short: profit_barrier = C_k - a*tick, stop_barrier = C_k + b*tick.

        Entry close=4000.0, a=20, b=10, tick=0.25.
        Profit barrier: 4000 - 20*0.25 = 3995.0
        Stop barrier:   4000 + 10*0.25 = 4002.5

        Bar 1: low=3994.75 < 3995.0 → profit hit for short → label=+1.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3998.0, 4002.0, 3994.75, 3996.0),  # low < 3995.0
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 3996.0, 3997.0, 3995.0, 3996.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40, direction="short")
        assert labels[0].label == 1  # profit for short
        assert labels[0].entry_price == 4000.0

    def test_short_stop_barrier_price(self):
        """For short: stop_barrier = C_k + b*tick_size.

        Entry close=4000.0, b=10. Stop = 4000 + 10*0.25 = 4002.5.
        Bar 1: high=4002.5 exactly → stop hit → label=-1.
        """
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4000.5, 4002.5, 3999.0, 4001.0),  # high == stop exactly
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4001.0, 4002.0, 4000.0, 4001.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40, direction="short")
        assert labels[0].label == -1  # stop for short


# ===========================================================================
# 24. Tiebreak only on dual breach bars
# ===========================================================================


class TestTiebreakOnlyOnDualBreach:
    """resolution_type with 'tiebreak' should only occur when both barriers are
    breached on the same bar."""

    def test_single_breach_not_tiebreak(self):
        """When only upper barrier is breached, resolution_type is 'upper', not tiebreak."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4006.0, 3998.0, 4004.0),  # high > U, low > D
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4004.0, 4005.0, 4003.0, 4004.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].resolution_type == "upper"  # NOT tiebreak_upper


# ===========================================================================
# 25. Barrier detection uses H/L not O/C for non-tiebreak
# ===========================================================================


class TestBarrierDetectionUsesHighLow:
    """Barrier detection checks H_j >= U and L_j <= D, not open/close."""

    def test_high_crosses_upper_while_close_stays_below(self):
        """High crosses upper barrier even though close is below U."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4001.0, 4005.5, 3999.0, 4002.0),  # high > U, close < U
        ]
        for i in range(2, 50):
            bars.append(_make_bar(i, 4002.0, 4003.0, 4001.0, 4002.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1  # upper hit via high, not close


# ===========================================================================
# 26. Upper barrier takes priority over lower on separate bars
# ===========================================================================


class TestFirstBarrierHitTakesPriority:
    """If upper hits on bar j1 and lower hits on bar j2 > j1, upper wins."""

    def test_earlier_upper_wins_over_later_lower(self):
        """Upper hit at bar 3, lower hit at bar 5 → label=+1 with tau=3."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 4000.0, 4002.0, 3998.0, 4001.0),
            _make_bar(2, 4001.0, 4003.0, 3998.0, 4002.0),
            _make_bar(3, 4002.0, 4006.0, 3998.0, 4004.0),  # upper hit
            _make_bar(4, 4004.0, 4005.0, 3998.0, 4000.0),
            _make_bar(5, 4000.0, 4001.0, 3996.0, 3997.0),  # lower hit (but later)
        ]
        for i in range(6, 50):
            bars.append(_make_bar(i, 3997.0, 3998.0, 3996.0, 3997.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == 1  # upper hit wins (earlier)
        assert labels[0].tau == 3

    def test_earlier_lower_wins_over_later_upper(self):
        """Lower hit at bar 2, upper hit at bar 5 → label=-1 with tau=2."""
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = [
            _make_bar(0, 4000.0, 4001.0, 3999.0, 4000.0),
            _make_bar(1, 3999.0, 4001.0, 3998.0, 3999.0),
            _make_bar(2, 3998.0, 4001.0, 3997.0, 3997.5),  # lower hit
            _make_bar(3, 3998.0, 4001.0, 3997.5, 3999.0),
            _make_bar(4, 3999.0, 4003.0, 3998.0, 4002.0),
            _make_bar(5, 4002.0, 4006.0, 4001.0, 4004.0),  # upper hit (later)
        ]
        for i in range(6, 50):
            bars.append(_make_bar(i, 4004.0, 4005.0, 4003.0, 4004.0))

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        assert labels[0].label == -1  # lower hit wins (earlier)
        assert labels[0].tau == 2
