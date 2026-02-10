"""Tests for Phase 2 Microstructure Features (Cols 17-21).

Spec: docs/phase2-microstructure-features.md

Tests the expansion from 17 to 22 features by adding 5 new LOB microstructure
features: VAMP displacement, aggressor imbalance, trade arrival rate,
cancel-to-trade ratio, and price impact per trade. Also tests the new
OrderBook.vamp() helper method.
"""

import math

import numpy as np
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar

from .conftest import (
    TICK_SIZE,
    _RTH_OPEN_NS,
    _RTH_CLOSE_NS,
    _RTH_DURATION_NS,
    make_bar as _make_bar,
    make_session_bars as _make_session_bars,
    make_mbo_dataframe as _make_mbo_dataframe,
)


# ===========================================================================
# 1. N_FEATURES constant — must be 22
# ===========================================================================


class TestNFeaturesIs22:
    """AC1: N_FEATURES == 22."""

    def test_n_features_is_22(self):
        from lob_rl.barrier import N_FEATURES

        assert N_FEATURES == 22, f"Expected N_FEATURES=22, got {N_FEATURES}"

    def test_n_features_importable_from_init(self):
        from lob_rl.barrier import N_FEATURES

        assert isinstance(N_FEATURES, int)


# ===========================================================================
# 2. _BOOK_DEFAULTS expansion — 13 values
# ===========================================================================


class TestBookDefaults13:
    """_BOOK_DEFAULTS must have 13 entries for the expanded book feature set."""

    def test_book_defaults_length_13(self):
        from lob_rl.barrier.feature_pipeline import _BOOK_DEFAULTS

        assert len(_BOOK_DEFAULTS) == 13, (
            f"Expected 13 book defaults, got {len(_BOOK_DEFAULTS)}"
        )

    def test_book_defaults_values(self):
        """Order: BBO, Depth, Cancel, Spread, OFI, DepthR, WMid, SpreadStd, VAMP, Aggr, TrdArr, C/T, Impact."""
        from lob_rl.barrier.feature_pipeline import _BOOK_DEFAULTS

        expected = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert _BOOK_DEFAULTS == expected, (
            f"Expected {expected}, got {_BOOK_DEFAULTS}"
        )

    def test_new_defaults_cols_8_to_12_all_zero(self):
        """New book feature defaults (cols 8-12) should all be 0.0."""
        from lob_rl.barrier.feature_pipeline import _BOOK_DEFAULTS

        for idx in range(8, 13):
            assert _BOOK_DEFAULTS[idx] == 0.0, (
                f"_BOOK_DEFAULTS[{idx}] = {_BOOK_DEFAULTS[idx]}, expected 0.0"
            )


# ===========================================================================
# 3. OrderBook.vamp() method
# ===========================================================================


class TestVAMPSymmetricBook:
    """Spec: Equal depth both sides -> VAMP = mid price."""

    def test_vamp_symmetric(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Symmetric book: 3 levels each side, equal qty
        for i in range(3):
            book.apply("A", "B", 4000.0 - i * TICK_SIZE, 10, order_id=100 + i)
            book.apply("A", "A", 4000.25 + i * TICK_SIZE, 10, order_id=200 + i)

        vamp = book.vamp(n=3)
        mid = book.mid_price()
        # With equal qty on both sides, VAMP should equal mid price
        assert vamp == pytest.approx(mid, rel=1e-6), (
            f"VAMP={vamp}, mid={mid} — expected equal for symmetric book"
        )


class TestVAMPAsymmetricBook:
    """Spec: Unequal depth -> VAMP skews toward heavier side."""

    def test_vamp_skews_toward_heavy_bid(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Heavy bids, light asks
        book.apply("A", "B", 4000.0, 100, order_id=1)
        book.apply("A", "B", 3999.75, 100, order_id=2)
        book.apply("A", "B", 3999.50, 100, order_id=3)
        book.apply("A", "A", 4000.25, 10, order_id=4)
        book.apply("A", "A", 4000.50, 10, order_id=5)
        book.apply("A", "A", 4000.75, 10, order_id=6)

        vamp = book.vamp(n=3)
        mid = book.mid_price()
        # VAMP should be below mid because bids have much more weight
        assert vamp < mid, (
            f"VAMP={vamp} should be < mid={mid} when bids heavier"
        )

    def test_vamp_skews_toward_heavy_ask(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Light bids, heavy asks
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "B", 3999.75, 10, order_id=2)
        book.apply("A", "B", 3999.50, 10, order_id=3)
        book.apply("A", "A", 4000.25, 100, order_id=4)
        book.apply("A", "A", 4000.50, 100, order_id=5)
        book.apply("A", "A", 4000.75, 100, order_id=6)

        vamp = book.vamp(n=3)
        mid = book.mid_price()
        # VAMP should be above mid because asks have much more weight
        assert vamp > mid, (
            f"VAMP={vamp} should be > mid={mid} when asks heavier"
        )


class TestVAMPEmptySide:
    """Spec: One side empty -> returns 0.0."""

    def test_vamp_empty_bids(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.25, 10, order_id=1)
        assert book.vamp(n=3) == 0.0

    def test_vamp_empty_asks(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        assert book.vamp(n=3) == 0.0

    def test_vamp_both_sides_empty(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        assert book.vamp(n=3) == 0.0


class TestVAMPSingleLevel:
    """Spec: Only 1 level per side -> uses that level."""

    def test_vamp_one_level_each_side(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 20, order_id=1)
        book.apply("A", "A", 4000.25, 30, order_id=2)

        vamp = book.vamp(n=3)
        # With 1 level each: VAMP = (bid*bid_qty + ask*ask_qty) / (bid_qty + ask_qty)
        expected = (4000.0 * 20 + 4000.25 * 30) / (20 + 30)
        assert vamp == pytest.approx(expected, rel=1e-6), (
            f"VAMP={vamp}, expected={expected} for single level each side"
        )

    def test_vamp_with_n_1(self):
        """vamp(n=1) should use only top level on each side."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 50, order_id=1)
        book.apply("A", "B", 3999.75, 50, order_id=2)  # L2 bid, ignored with n=1
        book.apply("A", "A", 4000.25, 50, order_id=3)
        book.apply("A", "A", 4000.50, 50, order_id=4)  # L2 ask, ignored with n=1

        vamp_n1 = book.vamp(n=1)
        expected = (4000.0 * 50 + 4000.25 * 50) / (50 + 50)
        assert vamp_n1 == pytest.approx(expected, rel=1e-6)


class TestVAMPZeroTotalQty:
    """Edge case: all quantities are zero -> returns 0.0."""

    def test_vamp_zero_qty(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # This shouldn't normally happen, but test robustness:
        # After adding and canceling all qty, levels get removed.
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "A", 4000.25, 10, order_id=2)
        book.apply("C", "B", 4000.0, 10, order_id=1)
        book.apply("C", "A", 4000.25, 10, order_id=2)
        # Now book is empty
        assert book.vamp(n=3) == 0.0


# ===========================================================================
# 4. compute_bar_features — shape now (N, 22)
# ===========================================================================


class TestComputeBarFeaturesShape22:
    """AC2: compute_bar_features() returns shape (N, 22)."""

    def test_shape_10_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars)
        assert features.shape == (10, 22), f"Got shape {features.shape}"

    def test_shape_1_bar(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        features = compute_bar_features(bars)
        assert features.shape == (1, 22)

    def test_shape_50_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(50)
        features = compute_bar_features(bars)
        assert features.shape == (50, 22)

    def test_dtype_is_float(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        assert np.issubdtype(features.dtype, np.floating)

    def test_various_bar_counts(self):
        """Shape is always (N, 22) regardless of bar count."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        for n in [1, 2, 5, 10, 50, 100]:
            bars = _make_session_bars(n)
            features = compute_bar_features(bars)
            assert features.shape == (n, 22), (
                f"Expected ({n}, 22), got {features.shape}"
            )


# ===========================================================================
# 5. No-MBO defaults for new cols 17-21
# ===========================================================================


class TestNoMboDefaultsNewCols17to21:
    """AC3: Without MBO data, new cols have correct neutral defaults (all 0.0)."""

    def test_col17_vamp_displacement_default_zero(self):
        """Col 17 (VAMP displacement) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 17], 0.0)

    def test_col18_aggressor_imbalance_default_zero(self):
        """Col 18 (aggressor imbalance) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 18], 0.0)

    def test_col19_trade_arrival_rate_default_zero(self):
        """Col 19 (trade arrival rate) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 19], 0.0)

    def test_col20_cancel_trade_ratio_default_zero(self):
        """Col 20 (cancel-to-trade ratio) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 20], 0.0)

    def test_col21_price_impact_default_zero(self):
        """Col 21 (price impact per trade) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 21], 0.0)

    def test_all_new_cols_default_zero(self):
        """Cols 17-21 all default to 0.0 without mbo_data."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        for col in range(17, 22):
            np.testing.assert_allclose(
                features[:, col], 0.0,
                err_msg=f"Col {col} should default to 0.0 without MBO data"
            )


# ===========================================================================
# 6. VAMP displacement tests (Col 17)
# ===========================================================================


class TestVAMPDisplacementPositive:
    """Spec: Book shifts up -> positive VAMP displacement."""

    def test_vamp_displacement_positive(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Start with BBO at 4000/4001 (3 levels each), then shift up to 4001/4002
        mbo = _make_mbo_dataframe([
            # Initial book: 3 bid levels, 3 ask levels
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 3999.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            {"action": "A", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 102},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 103},
            {"action": "A", "side": "A", "price": 4000.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 104},
            {"action": "A", "side": "A", "price": 4000.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 105},
            # Shift book up: cancel low levels, add higher ones
            {"action": "C", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "B", "price": 3999.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "C", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "C", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
            {"action": "C", "side": "A", "price": 4000.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 500_000_004},
            {"action": "C", "side": "A", "price": 4000.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 500_000_005},
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 7, "ts_event": _RTH_OPEN_NS + 500_000_006},
            {"action": "A", "side": "B", "price": 4000.75, "size": 100,
             "order_id": 8, "ts_event": _RTH_OPEN_NS + 500_000_007},
            {"action": "A", "side": "B", "price": 4000.50, "size": 100,
             "order_id": 9, "ts_event": _RTH_OPEN_NS + 500_000_008},
            {"action": "A", "side": "A", "price": 4001.25, "size": 100,
             "order_id": 10, "ts_event": _RTH_OPEN_NS + 500_000_009},
            {"action": "A", "side": "A", "price": 4001.50, "size": 100,
             "order_id": 11, "ts_event": _RTH_OPEN_NS + 500_000_010},
            {"action": "A", "side": "A", "price": 4001.75, "size": 100,
             "order_id": 12, "ts_event": _RTH_OPEN_NS + 500_000_011},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 17] > 0.0, (
            f"VAMP displacement should be positive when book shifts up, got {features[0, 17]}"
        )


class TestVAMPDisplacementNegative:
    """Spec: Book shifts down -> negative VAMP displacement."""

    def test_vamp_displacement_negative(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Start with BBO at 4001/4002 (3 levels), shift down to 4000/4001
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 4000.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            {"action": "A", "side": "B", "price": 4000.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 102},
            {"action": "A", "side": "A", "price": 4001.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 103},
            {"action": "A", "side": "A", "price": 4001.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 104},
            {"action": "A", "side": "A", "price": 4001.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 105},
            # Shift down
            {"action": "C", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "B", "price": 4000.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "C", "side": "B", "price": 4000.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "C", "side": "A", "price": 4001.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
            {"action": "C", "side": "A", "price": 4001.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 500_000_004},
            {"action": "C", "side": "A", "price": 4001.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 500_000_005},
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 7, "ts_event": _RTH_OPEN_NS + 500_000_006},
            {"action": "A", "side": "B", "price": 3999.75, "size": 100,
             "order_id": 8, "ts_event": _RTH_OPEN_NS + 500_000_007},
            {"action": "A", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 9, "ts_event": _RTH_OPEN_NS + 500_000_008},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 10, "ts_event": _RTH_OPEN_NS + 500_000_009},
            {"action": "A", "side": "A", "price": 4000.50, "size": 100,
             "order_id": 11, "ts_event": _RTH_OPEN_NS + 500_000_010},
            {"action": "A", "side": "A", "price": 4000.75, "size": 100,
             "order_id": 12, "ts_event": _RTH_OPEN_NS + 500_000_011},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 17] < 0.0, (
            f"VAMP displacement should be negative when book shifts down, got {features[0, 17]}"
        )


class TestVAMPDisplacementNoChange:
    """Spec: Book unchanged -> VAMP displacement = 0.0."""

    def test_vamp_displacement_zero_when_unchanged(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Book stays constant: 3 levels each side, no changes
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 3999.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            {"action": "A", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 102},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 103},
            {"action": "A", "side": "A", "price": 4000.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 104},
            {"action": "A", "side": "A", "price": 4000.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 105},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 17] == pytest.approx(0.0, abs=1e-6), (
            f"VAMP displacement should be ~0 when book unchanged, got {features[0, 17]}"
        )


class TestVAMPDisplacementEmptyBook:
    """Spec: Book empty at bar start/end -> VAMP displacement = 0.0."""

    def test_vamp_displacement_empty_book(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # No MBO messages in this bar's time window — book is empty
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS - 1_000_000},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 17] == pytest.approx(0.0, abs=1e-6), (
            f"VAMP displacement should be 0 for empty book, got {features[0, 17]}"
        )


class TestVAMPDisplacementFewerThan3Levels:
    """Spec: Fewer than 3 levels on a side -> uses however many are available."""

    def test_vamp_displacement_with_1_level_each_side(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Only 1 level each side, then shift up
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Shift up
            {"action": "C", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "A", "side": "A", "price": 4001.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # Should still compute something (using 1 level) and it should be positive
        assert features[0, 17] > 0.0, (
            f"VAMP displacement should be positive even with 1 level, got {features[0, 17]}"
        )


# ===========================================================================
# 7. Aggressor imbalance tests (Col 18)
# ===========================================================================


class TestAggressorImbalanceBuyHeavy:
    """Spec: More buy aggression -> positive imbalance."""

    def test_aggressor_imbalance_buy_heavy(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Trade messages: side indicates PASSIVE side.
        # Passive ask (aggressor is buyer): side='A'
        # More buy aggression = more trades with passive side='A'
        mbo = _make_mbo_dataframe([
            # Set up book first
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Buy aggressor trades (passive side = 'A')
            {"action": "T", "side": "A", "price": 4000.25, "size": 30,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "A", "side": "A", "price": 4000.25, "size": 30,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 201},
            {"action": "T", "side": "A", "price": 4000.25, "size": 20,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
            # One sell aggressor trade (passive side = 'B')
            {"action": "T", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 400},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # buy_vol=50, sell_vol=10 -> imbalance = (50-10)/(50+10) = 0.667
        assert features[0, 18] > 0.0, (
            f"Aggressor imbalance should be positive when buys dominate, got {features[0, 18]}"
        )


class TestAggressorImbalanceSellHeavy:
    """Spec: More sell aggression -> negative imbalance."""

    def test_aggressor_imbalance_sell_heavy(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Sell aggressor trades (passive side = 'B')
            {"action": "T", "side": "B", "price": 4000.0, "size": 30,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "A", "side": "B", "price": 4000.0, "size": 30,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 201},
            {"action": "T", "side": "B", "price": 4000.0, "size": 20,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
            # One buy aggressor trade (passive side = 'A')
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 400},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 18] < 0.0, (
            f"Aggressor imbalance should be negative when sells dominate, got {features[0, 18]}"
        )


class TestAggressorImbalanceBalanced:
    """Spec: Equal buy/sell -> near 0.0."""

    def test_aggressor_imbalance_balanced(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Equal buy and sell aggression
            {"action": "T", "side": "A", "price": 4000.25, "size": 20,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "B", "price": 4000.0, "size": 20,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert abs(features[0, 18]) < 0.1, (
            f"Aggressor imbalance should be near 0 when balanced, got {features[0, 18]}"
        )


class TestAggressorImbalanceNoTrades:
    """Spec: No trades -> aggressor imbalance = 0.0."""

    def test_aggressor_imbalance_no_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Only add and cancel messages — no trades
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            {"action": "C", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 18] == pytest.approx(0.0), (
            f"Aggressor imbalance should be 0.0 with no trades, got {features[0, 18]}"
        )


class TestAggressorImbalanceRange:
    """Spec: Aggressor imbalance clamped to [-1, +1]."""

    def test_aggressor_imbalance_range(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # All trades on one side (pure buy aggression)
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert -1.0 <= features[0, 18] <= 1.0, (
            f"Aggressor imbalance out of range: {features[0, 18]}"
        )


class TestAggressorImbalanceUsesExplicitSide:
    """Spec: Uses explicit side from MBO Trade messages, not tick rule."""

    def test_fill_messages_also_counted(self):
        """'F' (Fill) messages should be counted alongside 'T' (Trade)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Fill message (buy aggression — passive side A)
            {"action": "F", "side": "A", "price": 4000.25, "size": 40,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            # Trade message (sell aggression — passive side B)
            {"action": "T", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # buy_vol=40, sell_vol=10 -> positive
        assert features[0, 18] > 0.0, (
            f"Fill messages should count as trades, got imbalance {features[0, 18]}"
        )


# ===========================================================================
# 8. Trade arrival rate tests (Col 19)
# ===========================================================================


class TestTradeArrivalManyTrades:
    """Spec: Many trades -> higher trade arrival rate."""

    def test_trade_arrival_many_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # 20 trade messages
        records = [
            {"action": "A", "side": "A", "price": 4000.25, "size": 1000,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
        ]
        for i in range(20):
            records.append({
                "action": "T", "side": "A", "price": 4000.25, "size": 1,
                "order_id": 1, "ts_event": _RTH_OPEN_NS + 200 + i * 100,
            })
        mbo = _make_mbo_dataframe(records)

        features = compute_bar_features([bar], mbo_data=mbo)
        expected = math.log(1 + 20)
        assert features[0, 19] == pytest.approx(expected, rel=0.1), (
            f"Trade arrival rate should be ~log(21)={expected:.3f}, got {features[0, 19]}"
        )


class TestTradeArrivalFewTrades:
    """Spec: Few trades -> lower trade arrival rate."""

    def test_trade_arrival_few_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            # Just 2 trades
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        expected = math.log(1 + 2)
        assert features[0, 19] == pytest.approx(expected, rel=0.1), (
            f"Trade arrival rate should be ~log(3)={expected:.3f}, got {features[0, 19]}"
        )


class TestTradeArrivalNoTrades:
    """Spec: No trades -> trade arrival rate = 0.0."""

    def test_trade_arrival_no_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # No trades in bar
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 19] == pytest.approx(0.0), (
            f"Trade arrival rate should be 0.0 with no trades, got {features[0, 19]}"
        )


class TestTradeArrivalCountsFills:
    """Spec: Both 'T' and 'F' messages count as trades."""

    def test_trade_arrival_counts_fills(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "F", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 400},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # 3 trades (2 T + 1 F)
        expected = math.log(1 + 3)
        assert features[0, 19] == pytest.approx(expected, rel=0.1), (
            f"Trade arrival should count F messages: expected {expected:.3f}, got {features[0, 19]}"
        )


class TestTradeArrivalMonotonic:
    """More trades -> strictly higher arrival rate (log is monotonic)."""

    def test_more_trades_higher_rate(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        rates = []
        for n_trades in [0, 1, 5, 20]:
            bar = _make_bar(
                bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
                close=4000.25, volume=100,
                t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
            )
            records = [
                {"action": "A", "side": "A", "price": 4000.25, "size": 1000,
                 "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            ]
            for i in range(n_trades):
                records.append({
                    "action": "T", "side": "A", "price": 4000.25, "size": 1,
                    "order_id": 1, "ts_event": _RTH_OPEN_NS + 200 + i * 100,
                })
            mbo = _make_mbo_dataframe(records)
            features = compute_bar_features([bar], mbo_data=mbo)
            rates.append(features[0, 19])

        # Should be strictly increasing
        for i in range(1, len(rates)):
            assert rates[i] > rates[i - 1], (
                f"Trade arrival rate not monotonic: {rates}"
            )


# ===========================================================================
# 9. Cancel-to-trade ratio tests (Col 20)
# ===========================================================================


class TestCancelTradeRatioHigh:
    """Spec: Many cancels, few trades -> high ratio."""

    def test_cancel_trade_ratio_high(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # 10 cancels, 1 trade
        records = []
        for i in range(10):
            records.append({
                "action": "A", "side": "B", "price": 4000.0 - i * TICK_SIZE,
                "size": 10, "order_id": 100 + i,
                "ts_event": _RTH_OPEN_NS + i * 100,
            })
        for i in range(10):
            records.append({
                "action": "C", "side": "B",
                "price": 4000.0 - i * TICK_SIZE, "size": 10,
                "order_id": 100 + i,
                "ts_event": _RTH_OPEN_NS + 2000 + i * 100,
            })
        # Add an ask and one trade
        records.append({
            "action": "A", "side": "A", "price": 4000.25, "size": 50,
            "order_id": 200, "ts_event": _RTH_OPEN_NS + 5000,
        })
        records.append({
            "action": "T", "side": "A", "price": 4000.25, "size": 10,
            "order_id": 200, "ts_event": _RTH_OPEN_NS + 6000,
        })
        mbo = _make_mbo_dataframe(records)

        features = compute_bar_features([bar], mbo_data=mbo)
        # n_cancels=10, n_trades=1 -> log(1 + 10/1) = log(11) ≈ 2.4
        expected = math.log(1 + 10 / max(1, 1))
        assert features[0, 20] > 1.0, (
            f"Cancel-to-trade ratio should be high, got {features[0, 20]}"
        )


class TestCancelTradeRatioLow:
    """Spec: Few cancels, many trades -> low ratio."""

    def test_cancel_trade_ratio_low(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # 1 cancel, 10 trades
        records = [
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "C", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "A", "side": "A", "price": 4000.25, "size": 1000,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 300},
        ]
        for i in range(10):
            records.append({
                "action": "T", "side": "A", "price": 4000.25, "size": 1,
                "order_id": 2, "ts_event": _RTH_OPEN_NS + 400 + i * 100,
            })
        mbo = _make_mbo_dataframe(records)

        features = compute_bar_features([bar], mbo_data=mbo)
        # n_cancels=1, n_trades=10 -> log(1 + 1/10) = log(1.1) ≈ 0.095
        assert features[0, 20] < 0.5, (
            f"Cancel-to-trade ratio should be low, got {features[0, 20]}"
        )


class TestCancelTradeRatioNoCancels:
    """Spec: No cancels -> cancel-to-trade ratio = 0.0."""

    def test_cancel_trade_ratio_no_cancels(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # n_cancels=0 -> log(1 + 0/1) = log(1) = 0.0
        assert features[0, 20] == pytest.approx(0.0), (
            f"Cancel-to-trade ratio should be 0.0 with no cancels, got {features[0, 20]}"
        )


class TestCancelTradeRatioNoTrades:
    """Spec: Cancels but no trades -> log(1 + n_cancels)."""

    def test_cancel_trade_ratio_no_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # 5 cancels, 0 trades
        records = []
        for i in range(5):
            records.append({
                "action": "A", "side": "B",
                "price": 4000.0 - i * TICK_SIZE, "size": 10,
                "order_id": 100 + i,
                "ts_event": _RTH_OPEN_NS + i * 100,
            })
        for i in range(5):
            records.append({
                "action": "C", "side": "B",
                "price": 4000.0 - i * TICK_SIZE, "size": 10,
                "order_id": 100 + i,
                "ts_event": _RTH_OPEN_NS + 1000 + i * 100,
            })
        mbo = _make_mbo_dataframe(records)

        features = compute_bar_features([bar], mbo_data=mbo)
        # n_cancels=5, n_trades=0 -> log(1 + 5/max(0,1)) = log(6) ≈ 1.79
        expected = math.log(1 + 5 / 1)
        assert features[0, 20] == pytest.approx(expected, rel=0.1), (
            f"Cancel-to-trade ratio should be log(6)={expected:.3f}, got {features[0, 20]}"
        )


class TestCancelTradeRatioZeroBoth:
    """Spec: Zero trades, zero cancels -> 0.0."""

    def test_cancel_trade_ratio_zero_both(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Only adds, no cancels, no trades
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 20] == pytest.approx(0.0), (
            f"Cancel-to-trade ratio should be 0.0 with no cancels/trades, got {features[0, 20]}"
        )


# ===========================================================================
# 10. Price impact per trade tests (Col 21)
# ===========================================================================


class TestPriceImpactPositive:
    """Spec: Close > open -> positive price impact."""

    def test_price_impact_positive(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            # 5 trades
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 400},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 600},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # impact = (4001 - 4000) / (5 * 0.25) = 1.0 / 1.25 = 0.8
        assert features[0, 21] > 0.0, (
            f"Price impact should be positive when close > open, got {features[0, 21]}"
        )


class TestPriceImpactNegative:
    """Spec: Close < open -> negative price impact."""

    def test_price_impact_negative(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 21] < 0.0, (
            f"Price impact should be negative when close < open, got {features[0, 21]}"
        )


class TestPriceImpactNoChange:
    """Spec: Close == open -> price impact = 0.0."""

    def test_price_impact_zero(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 21] == pytest.approx(0.0), (
            f"Price impact should be 0.0 when close == open, got {features[0, 21]}"
        )


class TestPriceImpactManyTrades:
    """Spec: Same price move with more trades -> lower per-trade impact."""

    def test_price_impact_diluted_by_more_trades(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # Bar with +4 tick move and 2 trades
        bar_few = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )
        mbo_few = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
        ])

        # Same bar prices but 10 trades
        bar_many = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )
        records = [
            {"action": "A", "side": "A", "price": 4000.25, "size": 1000,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
        ]
        for i in range(10):
            records.append({
                "action": "T", "side": "A", "price": 4000.25, "size": 1,
                "order_id": 1, "ts_event": _RTH_OPEN_NS + 200 + i * 100,
            })
        mbo_many = _make_mbo_dataframe(records)

        features_few = compute_bar_features([bar_few], mbo_data=mbo_few)
        features_many = compute_bar_features([bar_many], mbo_data=mbo_many)

        # Same price move, but more trades → lower impact per trade
        assert features_few[0, 21] > features_many[0, 21], (
            f"Impact with 2 trades ({features_few[0, 21]}) should be > "
            f"impact with 10 trades ({features_many[0, 21]})"
        )


class TestPriceImpactHandComputed:
    """Exact hand-computed price impact."""

    def test_price_impact_exact(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # open=4000, close=4001 -> (close-open) = 1.0
        # 4 trades -> max(4,1) = 4
        # TICK_SIZE = 0.25
        # impact = 1.0 / (4 * 0.25) = 1.0
        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 300},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 400},
            {"action": "T", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        expected = (4001.0 - 4000.0) / (max(4, 1) * TICK_SIZE)
        assert features[0, 21] == pytest.approx(expected, rel=1e-6), (
            f"Price impact should be {expected}, got {features[0, 21]}"
        )


class TestPriceImpactNoTradesMBO:
    """Price impact with MBO data but no trade messages: uses n_trades=0 -> max(0,1)=1."""

    def test_price_impact_no_trades_in_mbo(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Only add messages, no trades
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # n_trades=0 -> max(0,1)=1 -> impact = (4001-4000)/(1*0.25) = 4.0
        expected = (4001.0 - 4000.0) / (1 * TICK_SIZE)
        assert features[0, 21] == pytest.approx(expected, rel=0.1), (
            f"Price impact should be {expected} with 0 trades, got {features[0, 21]}"
        )


# ===========================================================================
# 11. _compute_book_features shape expansion to (n, 13)
# ===========================================================================


class TestComputeBookFeaturesShape13:
    """_compute_book_features should return (n, 13) with cols 8-12 for new features."""

    def test_book_features_shape(self):
        from lob_rl.barrier.feature_pipeline import _compute_book_features

        bars = _make_session_bars(5)

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        result = _compute_book_features(bars, mbo)
        assert result.shape == (5, 13), (
            f"Expected (5, 13), got {result.shape}"
        )

    def test_book_features_cols_8_to_12_exist(self):
        """Cols 8-12 should be accessible and numeric."""
        from lob_rl.barrier.feature_pipeline import _compute_book_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        result = _compute_book_features([bar], mbo)
        for col in range(8, 13):
            assert np.isfinite(result[0, col]), (
                f"Book feature col {col} is not finite: {result[0, col]}"
            )


# ===========================================================================
# 12. compute_bar_features assigns book cols 17-21 from book_features[:, 8:13]
# ===========================================================================


class TestComputeBarFeaturesBookColAssignment17to21:
    """When book_features is available, cols 17-21 come from book_features cols 8-12."""

    def test_cols_17_21_populated_from_mbo(self):
        """With MBO data containing trades, cols 17-21 should not all be defaults."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4005.0, low=3995.0,
            close=4002.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Create MBO data with book changes and trades so new features activate
        mbo = _make_mbo_dataframe([
            # Book setup
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 3999.75, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            {"action": "A", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 102},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 103},
            {"action": "A", "side": "A", "price": 4000.50, "size": 100,
             "order_id": 5, "ts_event": _RTH_OPEN_NS + 104},
            {"action": "A", "side": "A", "price": 4000.75, "size": 100,
             "order_id": 6, "ts_event": _RTH_OPEN_NS + 105},
            # Trades to activate aggressor imbalance + trade arrival + impact
            {"action": "T", "side": "A", "price": 4000.25, "size": 30,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "T", "side": "A", "price": 4000.25, "size": 20,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 300},
            {"action": "T", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 400},
            # Cancels
            {"action": "C", "side": "B", "price": 3999.50, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)

        # At least trade arrival (col 19) should be non-zero (3 trades -> log(4) > 0)
        assert features[0, 19] > 0.0, (
            f"Trade arrival rate (col 19) should be > 0 with trades, got {features[0, 19]}"
        )
        # Price impact (col 21) should be non-zero (close != open)
        assert features[0, 21] != 0.0, (
            f"Price impact (col 21) should be non-zero, got {features[0, 21]}"
        )


# ===========================================================================
# 13. Integration tests
# ===========================================================================


class TestBuildFeatureMatrixShape22h:
    """AC6: build_feature_matrix returns shape (M, 22*h)."""

    def test_shape_h10(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert result.ndim == 2
        assert result.shape[1] == 22 * 10, (
            f"Expected 220 columns, got {result.shape[1]}"
        )

    def test_shape_h5(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars, h=5)
        assert result.shape[1] == 22 * 5, (
            f"Expected 110 columns, got {result.shape[1]}"
        )

    def test_finite_values(self):
        """No NaN or Inf in the final 22-feature matrix."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert np.all(np.isfinite(result)), "Final matrix has NaN/Inf"

    def test_clipped_to_5(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert np.all(result >= -5.0)
        assert np.all(result <= 5.0)


class TestBarrierEnvObsDim222:
    """AC7: BarrierEnv observation dim = 22*10 + 2 = 222 with h=10."""

    def test_obs_dim(self):
        from lob_rl.barrier import N_FEATURES
        from lob_rl.barrier.barrier_env import BarrierEnv
        from lob_rl.barrier.feature_pipeline import build_feature_matrix
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_session_bars(50)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=10)

        env = BarrierEnv(bars=bars, labels=labels, features=features)
        obs, info = env.reset()

        expected_dim = N_FEATURES * 10 + 2  # 22 * 10 + 2 = 222
        assert obs.shape == (expected_dim,), (
            f"Expected obs dim {expected_dim}, got {obs.shape}"
        )

    def test_obs_dim_value_222(self):
        """Explicit check: 22 * 10 + 2 = 222."""
        from lob_rl.barrier import N_FEATURES

        assert N_FEATURES * 10 + 2 == 222


class TestLookbackShape22:
    """assemble_lookback with 22 features produces (N-h+1, 22*h) output."""

    def test_lookback_shape(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((50, 22))
        result = assemble_lookback(normed, h=10)
        assert result.shape == (41, 220)

    def test_lookback_shape_h5(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((30, 22))
        result = assemble_lookback(normed, h=5)
        assert result.shape == (26, 110)


class TestNormalize22Features:
    """normalize_features handles 22-column input."""

    def test_normalize_22_cols(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.random.default_rng(42).standard_normal((100, 22)) * 3 + 5
        normed = normalize_features(raw, window=50)
        assert normed.shape == (100, 22)
        assert np.all(np.isfinite(normed))
        assert np.all(normed >= -5.0)
        assert np.all(normed <= 5.0)


# ===========================================================================
# 14. Existing features (cols 0-16) preserved at correct columns
# ===========================================================================


class TestExistingFeaturesPreserved22:
    """The first 17 feature columns should still compute correctly at their positions."""

    def test_trade_flow_imbalance_col0(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars)
        col = features[:, 0]
        assert np.all(col >= -1.0)
        assert np.all(col <= 1.0)

    def test_bbo_imbalance_default_col1(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 1], 0.5)

    def test_bar_range_col3(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=4000.0,
            close=4001.5, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected_range = (4002.0 - 4000.0) / TICK_SIZE
        assert features[0, 3] == pytest.approx(expected_range)

    def test_volume_log_col7(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 7] == pytest.approx(math.log(100), rel=1e-4)

    def test_cancel_asymmetry_default_col10(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 10], 0.0)

    def test_mean_spread_default_col11(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 11], 1.0)

    def test_session_age_col12(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert features[0, 12] == pytest.approx(0.0)
        assert features[20, 12] == pytest.approx(1.0)

    def test_ofi_default_col13(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 13], 0.0)

    def test_depth_ratio_default_col14(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 14], 0.5)

    def test_wmid_displacement_default_col15(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 15], 0.0)

    def test_spread_std_default_col16(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 16], 0.0)


# ===========================================================================
# 15. Conftest constants should adapt to N_FEATURES=22
# ===========================================================================


class TestConfTestConstants22:
    """conftest.py constants should reflect N_FEATURES=22."""

    def test_default_feature_dim(self):
        """DEFAULT_FEATURE_DIM should be 22 * 10 = 220."""
        from lob_rl.barrier import N_FEATURES

        expected = N_FEATURES * 10  # 220
        assert expected == 220

    def test_default_obs_dim(self):
        """DEFAULT_OBS_DIM should be 220 + 2 = 222."""
        from lob_rl.barrier import N_FEATURES

        expected = N_FEATURES * 10 + 2  # 222
        assert expected == 222


# ===========================================================================
# 16. VAMP displacement uses TICK_SIZE for normalization
# ===========================================================================


class TestVAMPDisplacementNormalization:
    """VAMP displacement should be in tick units: delta_vamp / TICK_SIZE."""

    def test_vamp_displacement_in_ticks(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Book shifts from VAMP at ~4000.125 to ~4001.125 (exactly +4 ticks)
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 101},
            # Shift up by 1.0 (4 ticks)
            {"action": "C", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "A", "side": "A", "price": 4001.25, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # VAMP shifted by ~1.0 price units = 4 ticks
        # displacement = (vamp_end - vamp_start) / TICK_SIZE ≈ 4.0
        assert features[0, 17] == pytest.approx(4.0, abs=0.5), (
            f"VAMP displacement should be ~4 ticks, got {features[0, 17]}"
        )
