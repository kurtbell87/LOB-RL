"""Tests for Phase 1 Microstructure Features (Cols 13-16).

Spec: docs/phase1-microstructure-features.md

Tests the expansion from 13 to 17 features by adding 4 new LOB microstructure
features: OFI, multi-level depth ratio, weighted mid-price displacement,
and spread dynamics (std).
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
)


# ===========================================================================
# 1. N_FEATURES constant — must be 17
# ===========================================================================


class TestNFeaturesConstant:
    """AC1: N_FEATURES == 17."""

    def test_n_features_is_17(self):
        from lob_rl.barrier import N_FEATURES

        assert N_FEATURES == 17, f"Expected N_FEATURES=17, got {N_FEATURES}"

    def test_n_features_importable_from_init(self):
        """N_FEATURES should be importable from lob_rl.barrier."""
        from lob_rl.barrier import N_FEATURES

        assert isinstance(N_FEATURES, int)


# ===========================================================================
# 2. _BOOK_DEFAULTS expansion — 8 values
# ===========================================================================


class TestBookDefaults:
    """_BOOK_DEFAULTS must have 8 entries for the expanded book feature set."""

    def test_book_defaults_length_8(self):
        from lob_rl.barrier.feature_pipeline import _BOOK_DEFAULTS

        assert len(_BOOK_DEFAULTS) == 8, (
            f"Expected 8 book defaults, got {len(_BOOK_DEFAULTS)}"
        )

    def test_book_defaults_values(self):
        """Order: BBO, Depth, Cancel, Spread, OFI, DepthR, WMid, SpreadStd."""
        from lob_rl.barrier.feature_pipeline import _BOOK_DEFAULTS

        expected = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0)
        assert _BOOK_DEFAULTS == expected, (
            f"Expected {expected}, got {_BOOK_DEFAULTS}"
        )


# ===========================================================================
# 3. compute_bar_features — shape now (N, 17)
# ===========================================================================


class TestComputeBarFeaturesShape17:
    """AC2: compute_bar_features() returns shape (N, 17)."""

    def test_shape_10_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars)
        assert features.shape == (10, 17), f"Got shape {features.shape}"

    def test_shape_1_bar(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        features = compute_bar_features(bars)
        assert features.shape == (1, 17)

    def test_shape_50_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(50)
        features = compute_bar_features(bars)
        assert features.shape == (50, 17)

    def test_dtype_is_float(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        assert np.issubdtype(features.dtype, np.floating)

    def test_various_bar_counts(self):
        """Shape is always (N, 17) regardless of bar count."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        for n in [1, 2, 5, 10, 50, 100]:
            bars = _make_session_bars(n)
            features = compute_bar_features(bars)
            assert features.shape == (n, 17), (
                f"Expected ({n}, 17), got {features.shape}"
            )


# ===========================================================================
# 4. No-MBO defaults for new cols 13-16
# ===========================================================================


class TestNoMboDefaultsNewCols:
    """AC3: Without MBO data, new cols have correct neutral defaults."""

    def test_col13_ofi_default_zero(self):
        """Col 13 (OFI) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 13], 0.0)

    def test_col14_depth_ratio_default_half(self):
        """Col 14 (depth ratio) defaults to 0.5 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 14], 0.5)

    def test_col15_wmid_displacement_default_zero(self):
        """Col 15 (weighted mid displacement) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 15], 0.0)

    def test_col16_spread_std_default_zero(self):
        """Col 16 (spread std) defaults to 0.0 when mbo_data is None."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 16], 0.0)


# ===========================================================================
# 5. OFI tests (Col 13)
# ===========================================================================


def _make_mbo_dataframe(records):
    """Build a minimal MBO DataFrame from a list of dicts.

    Each record should have: action, side, price, size, order_id, ts_event.
    Returns a pandas DataFrame compatible with _compute_book_features.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    for col in ["action", "side", "price", "size", "order_id", "ts_event"]:
        assert col in df.columns, f"Missing column: {col}"
    return df


class TestOFIPositiveBidAdds:
    """Spec: Bid adds at BBO -> positive OFI."""

    def test_ofi_positive_for_bid_adds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # Create a bar spanning a time window
        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Seed the book with initial ask-side liquidity, then add bid-side
        # Add messages at BBO on bid side → positive OFI
        mbo = _make_mbo_dataframe([
            # Initial ask to establish BBO
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            # Bid adds at best bid level
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "A", "side": "B", "price": 4000.0, "size": 30,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 13] > 0.0, (
            f"OFI should be positive for bid adds, got {features[0, 13]}"
        )


class TestOFINegativeAskAdds:
    """Spec: Ask adds at BBO -> negative OFI."""

    def test_ofi_negative_for_ask_adds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            # Initial bid to establish BBO
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            # Ask adds at best ask level
            {"action": "A", "side": "A", "price": 4001.0, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            {"action": "A", "side": "A", "price": 4001.0, "size": 30,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 13] < 0.0, (
            f"OFI should be negative for ask adds, got {features[0, 13]}"
        )


class TestOFIZeroNoAdds:
    """Spec: No Add messages in bar -> OFI = 0.0."""

    def test_ofi_zero_no_adds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Only cancel messages, no adds
        # First need to add orders (in a prior bar) so they exist to cancel.
        # Actually, the simplest: pass an empty MBO dataframe with no adds.
        # We'll use an MBO frame with zero messages matching this bar's time.
        # Use a dataframe with messages BEFORE the bar starts.
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS - 1_000_000},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 13] == pytest.approx(0.0), (
            f"OFI should be 0 with no adds in bar, got {features[0, 13]}"
        )


class TestOFIIgnoresNonBBOAdds:
    """Spec: Adds far from BBO don't count toward OFI."""

    def test_ofi_ignores_far_adds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Establish BBO at 4000/4001, then add orders far away
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            # Far-from-BBO adds (bid at 3990, ask at 4010 — 40+ ticks away)
            {"action": "A", "side": "B", "price": 3990.0, "size": 1000,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
            {"action": "A", "side": "A", "price": 4010.0, "size": 1000,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 400},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # The large far-from-BBO adds shouldn't dominate OFI.
        # If only BBO adds count: net = +10 (bid) - 10 (ask) = 0
        # If all adds count: net would be dominated by the 1000-lot far adds
        # We test that OFI is small (close to 0), not dominated by far adds
        assert abs(features[0, 13]) < 0.5, (
            f"OFI dominated by non-BBO adds: {features[0, 13]}"
        )


class TestOFIRange:
    """Spec: OFI output clamped to [-1, +1]."""

    def test_ofi_clamped(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Massive one-sided adds
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "A", "price": 4001.0, "size": 1,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "B", "price": 4000.0, "size": 100000,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert -1.0 <= features[0, 13] <= 1.0, (
            f"OFI out of range: {features[0, 13]}"
        )


class TestOFIDefaultNoMbo:
    """Spec: Without mbo_data, col 13 = 0.0."""

    def test_ofi_default(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 13], 0.0)


# ===========================================================================
# 6. Multi-level depth ratio tests (Col 14)
# ===========================================================================


class TestDepthRatioConcentrated:
    """Spec: All depth at L1 -> ratio near 1.0."""

    def test_depth_ratio_concentrated_at_bbo(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # All liquidity at L1 only — nothing at levels 2-10
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # top3/top10 = 200/200 = 1.0 since all depth is at L1
        assert features[0, 14] > 0.9, (
            f"Depth ratio should be near 1.0 for concentrated book, got {features[0, 14]}"
        )


class TestDepthRatioDispersed:
    """Spec: Depth evenly spread -> ratio < 1.0."""

    def test_depth_ratio_dispersed(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4005.0, low=3995.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Spread liquidity across 10 bid and 10 ask levels
        records = []
        oid = 1
        for i in range(10):
            records.append({
                "action": "A", "side": "B",
                "price": 4000.0 - i * TICK_SIZE,
                "size": 10,
                "order_id": oid,
                "ts_event": _RTH_OPEN_NS + oid * 100,
            })
            oid += 1
            records.append({
                "action": "A", "side": "A",
                "price": 4000.25 + i * TICK_SIZE,
                "size": 10,
                "order_id": oid,
                "ts_event": _RTH_OPEN_NS + oid * 100,
            })
            oid += 1

        mbo = _make_mbo_dataframe(records)
        features = compute_bar_features([bar], mbo_data=mbo)
        # top3 = 3*10 + 3*10 = 60, top10 = 10*10 + 10*10 = 200
        # ratio = 60/200 = 0.3
        assert features[0, 14] < 0.5, (
            f"Depth ratio should be < 0.5 for dispersed book, got {features[0, 14]}"
        )


class TestDepthRatioEmptyBook:
    """Spec: Empty book -> default 0.5."""

    def test_depth_ratio_empty_book(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # MBO with no messages in this bar's time window — book stays empty
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS - 1_000_000},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 14] == pytest.approx(0.5), (
            f"Depth ratio should be 0.5 for empty book, got {features[0, 14]}"
        )


class TestDepthRatioRange:
    """Spec: Depth ratio output in [0, 1]."""

    def test_depth_ratio_in_range(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert 0.0 <= features[0, 14] <= 1.0, (
            f"Depth ratio out of range: {features[0, 14]}"
        )


class TestDepthRatioDefaultNoMbo:
    """Spec: Without mbo_data, col 14 = 0.5."""

    def test_depth_ratio_default(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 14], 0.5)


# ===========================================================================
# 7. Weighted mid-price displacement tests (Col 15)
# ===========================================================================


class TestWMidDisplacementPositive:
    """Spec: Book shifts up -> positive displacement."""

    def test_wmid_displacement_positive(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Start with BBO at 4000/4001, then shift up to 4001/4002
        mbo = _make_mbo_dataframe([
            # Initial book: bid 4000, ask 4001
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            # Cancel old orders and add higher ones → book shifts up
            {"action": "C", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "A", "price": 4001.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "A", "side": "A", "price": 4002.0, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 15] > 0.0, (
            f"WMID displacement should be positive when book shifts up, got {features[0, 15]}"
        )


class TestWMidDisplacementNegative:
    """Spec: Book shifts down -> negative displacement."""

    def test_wmid_displacement_negative(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Start with BBO at 4001/4002, then shift down to 4000/4001
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4002.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            # Shift down
            {"action": "C", "side": "B", "price": 4001.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "C", "side": "A", "price": 4002.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_001},
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_002},
            {"action": "A", "side": "A", "price": 4001.0, "size": 100,
             "order_id": 4, "ts_event": _RTH_OPEN_NS + 500_000_003},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 15] < 0.0, (
            f"WMID displacement should be negative when book shifts down, got {features[0, 15]}"
        )


class TestWMidDisplacementNoChange:
    """Spec: Book unchanged -> 0.0."""

    def test_wmid_displacement_zero_when_unchanged(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Book stays at 4000/4001 throughout — wmid doesn't change
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 100,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 15] == pytest.approx(0.0, abs=1e-6), (
            f"WMID displacement should be ~0 when book unchanged, got {features[0, 15]}"
        )


class TestWMidDisplacementEmptyBook:
    """Spec: Empty book at bar start/end -> 0.0."""

    def test_wmid_displacement_empty_book(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # No MBO messages in this bar → book is empty at start and end
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS - 1_000_000},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 15] == pytest.approx(0.0, abs=1e-6), (
            f"WMID displacement should be 0 for empty book, got {features[0, 15]}"
        )


class TestWMidDisplacementDefaultNoMbo:
    """Spec: Without mbo_data, col 15 = 0.0."""

    def test_wmid_default(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 15], 0.0)


# ===========================================================================
# 8. Spread dynamics (std) tests (Col 16)
# ===========================================================================


class TestSpreadStdConstant:
    """Spec: Constant spread -> std approx 0."""

    def test_spread_std_constant_spread(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Multiple MBO events all maintaining spread = 1 tick (4000/4000.25)
        records = []
        oid = 1
        for i in range(10):
            records.append({
                "action": "A", "side": "B", "price": 4000.0, "size": 10,
                "order_id": oid,
                "ts_event": _RTH_OPEN_NS + (i + 1) * 100_000,
            })
            oid += 1
            records.append({
                "action": "A", "side": "A", "price": 4000.25, "size": 10,
                "order_id": oid,
                "ts_event": _RTH_OPEN_NS + (i + 1) * 100_000 + 1,
            })
            oid += 1

        mbo = _make_mbo_dataframe(records)
        features = compute_bar_features([bar], mbo_data=mbo)
        # Constant spread → std should be 0 or very close
        assert features[0, 16] == pytest.approx(0.0, abs=0.01), (
            f"Spread std should be ~0 for constant spread, got {features[0, 16]}"
        )


class TestSpreadStdVariable:
    """Spec: Variable spread -> std > 0."""

    def test_spread_std_variable_spread(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4005.0, low=3995.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Create a book with varying spread: start tight (1 tick), widen (4 ticks)
        mbo = _make_mbo_dataframe([
            # Phase 1: tight spread (1 tick)
            {"action": "A", "side": "B", "price": 4000.0, "size": 50,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            # Phase 2: widen spread by cancelling ask and adding higher
            {"action": "C", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 500_000_000},
            {"action": "A", "side": "A", "price": 4001.0, "size": 50,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 500_000_001},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        assert features[0, 16] > 0.0, (
            f"Spread std should be > 0 for variable spread, got {features[0, 16]}"
        )


class TestSpreadStdSingleSample:
    """Spec: Only 1 spread sample -> std = 0.0."""

    def test_spread_std_single_sample(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.25, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Just one MBO event that creates a valid spread
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 100},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)
        # With <= 1 valid spread sample, std should be 0
        # (The spec says >= 2 samples needed, otherwise 0.0)
        # Two adds here but the second creates the first valid spread.
        # Depending on sampling: first add → no spread (one-sided), second add → spread=1.
        # So we might get 1 sample → std = 0.0.
        # Even if 2 identical samples: std = 0.0.
        assert features[0, 16] == pytest.approx(0.0, abs=0.01), (
            f"Spread std should be 0 with <=1 sample, got {features[0, 16]}"
        )


class TestSpreadStdDefaultNoMbo:
    """Spec: Without mbo_data, col 16 = 0.0."""

    def test_spread_std_default(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 16], 0.0)


# ===========================================================================
# 9. Integration tests
# ===========================================================================


class TestBuildFeatureMatrixShape17h:
    """AC6: build_feature_matrix returns shape (M, 17*h)."""

    def test_shape_h10(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert result.ndim == 2
        assert result.shape[1] == 17 * 10, (
            f"Expected 170 columns, got {result.shape[1]}"
        )

    def test_shape_h5(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars, h=5)
        assert result.shape[1] == 17 * 5, (
            f"Expected 85 columns, got {result.shape[1]}"
        )

    def test_finite_values(self):
        """No NaN or Inf in the final 17-feature matrix."""
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


class TestBarrierEnvObsDim172:
    """AC7: BarrierEnv observation dim = 17*10 + 2 = 172 with h=10."""

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

        expected_dim = N_FEATURES * 10 + 2  # 17 * 10 + 2 = 172
        assert obs.shape == (expected_dim,), (
            f"Expected obs dim {expected_dim}, got {obs.shape}"
        )

    def test_obs_dim_value_172(self):
        """Explicit check: 17 * 10 + 2 = 172."""
        from lob_rl.barrier import N_FEATURES

        assert N_FEATURES * 10 + 2 == 172


class TestLookbackShape17:
    """assemble_lookback with 17 features produces (N-h+1, 17*h) output."""

    def test_lookback_shape(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((50, 17))
        result = assemble_lookback(normed, h=10)
        assert result.shape == (41, 170)

    def test_lookback_shape_h5(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((30, 17))
        result = assemble_lookback(normed, h=5)
        assert result.shape == (26, 85)


class TestNormalize17Features:
    """normalize_features handles 17-column input."""

    def test_normalize_17_cols(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.random.default_rng(42).standard_normal((100, 17)) * 3 + 5
        normed = normalize_features(raw, window=50)
        assert normed.shape == (100, 17)
        assert np.all(np.isfinite(normed))
        assert np.all(normed >= -5.0)
        assert np.all(normed <= 5.0)


# ===========================================================================
# 10. Existing features preserved at correct columns
# ===========================================================================


class TestExistingFeaturesPreserved:
    """The first 13 feature columns should still compute correctly."""

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

    def test_depth_imbalance_default_col2(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 2], 0.5)

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


# ===========================================================================
# 11. Conftest constants should adapt to N_FEATURES=17
# ===========================================================================


class TestConfTestConstants:
    """conftest.py constants should reflect N_FEATURES=17."""

    def test_default_feature_dim(self):
        """DEFAULT_FEATURE_DIM should be 17 * 10 = 170."""
        from lob_rl.barrier import N_FEATURES

        expected = N_FEATURES * 10  # 170
        # This tests the expectation; the conftest constant will need updating
        assert expected == 170

    def test_default_obs_dim(self):
        """DEFAULT_OBS_DIM should be 170 + 2 = 172."""
        from lob_rl.barrier import N_FEATURES

        expected = N_FEATURES * 10 + 2  # 172
        assert expected == 172


# ===========================================================================
# 12. _compute_book_features shape expansion to (n, 8)
# ===========================================================================


class TestComputeBookFeaturesShape:
    """_compute_book_features should return (n, 8) with cols 4-7 for new features."""

    def test_book_features_shape(self):
        from lob_rl.barrier.feature_pipeline import _compute_book_features

        bars = _make_session_bars(5)

        # Minimal MBO data so _compute_book_features runs
        mbo = _make_mbo_dataframe([
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
        ])

        result = _compute_book_features(bars, mbo)
        assert result.shape == (5, 8), (
            f"Expected (5, 8), got {result.shape}"
        )

    def test_book_features_cols_4_to_7_exist(self):
        """Cols 4-7 should be accessible and numeric."""
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
        for col in range(4, 8):
            assert np.isfinite(result[0, col]), (
                f"Book feature col {col} is not finite: {result[0, col]}"
            )


# ===========================================================================
# 13. compute_bar_features assigns book cols 13-16 from book_features[:, 4:8]
# ===========================================================================


class TestComputeBarFeaturesBookColAssignment:
    """When book_features is available, cols 13-16 come from book_features cols 4-7."""

    def test_cols_13_16_populated_from_mbo(self):
        """With MBO data, cols 13-16 should not all be defaults."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4005.0, low=3995.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1_000_000_000,
        )

        # Create an asymmetric book to produce non-default OFI + depth ratio
        mbo = _make_mbo_dataframe([
            # Large bid add → positive OFI
            {"action": "A", "side": "B", "price": 4000.0, "size": 200,
             "order_id": 1, "ts_event": _RTH_OPEN_NS + 100},
            {"action": "A", "side": "A", "price": 4001.0, "size": 10,
             "order_id": 2, "ts_event": _RTH_OPEN_NS + 200},
            # More bid adds for strong OFI
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 3, "ts_event": _RTH_OPEN_NS + 300},
        ])

        features = compute_bar_features([bar], mbo_data=mbo)

        # At least one of cols 13-16 should differ from default
        defaults = [0.0, 0.5, 0.0, 0.0]
        non_default_count = sum(
            1 for c, d in zip(range(13, 17), defaults)
            if features[0, c] != pytest.approx(d, abs=0.01)
        )
        assert non_default_count >= 1, (
            f"Cols 13-16 are all defaults even with MBO data: "
            f"{[features[0, c] for c in range(13, 17)]}"
        )
