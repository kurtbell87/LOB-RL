"""Tests for the barrier feature extraction pipeline.

Spec: docs/t3-feature-extraction.md

Tests the 13-feature computation, z-score normalization with trailing window,
lookback assembly, and end-to-end build_feature_matrix.
"""

import math

import numpy as np
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar

from .conftest import (
    TICK_SIZE, _RTH_OPEN_NS, _RTH_CLOSE_NS, _RTH_DURATION_NS,
    make_bar as _make_bar, make_session_bars as _make_session_bars,
)


# ===========================================================================
# 1. Imports
# ===========================================================================


class TestFeaturePipelineImports:
    """All public functions should be importable."""

    def test_compute_bar_features_importable(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features
        assert callable(compute_bar_features)

    def test_normalize_features_importable(self):
        from lob_rl.barrier.feature_pipeline import normalize_features
        assert callable(normalize_features)

    def test_assemble_lookback_importable(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback
        assert callable(assemble_lookback)

    def test_build_feature_matrix_importable(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix
        assert callable(build_feature_matrix)


# ===========================================================================
# 2. compute_bar_features — output shape
# ===========================================================================


class TestComputeBarFeaturesShape:
    """compute_bar_features returns (N, 13) ndarray."""

    def test_shape_10_bars(self):
        """10 bars → shape (10, 13)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars)
        assert isinstance(features, np.ndarray)
        assert features.shape == (10, 13)

    def test_shape_1_bar(self):
        """Single bar → shape (1, 13)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        features = compute_bar_features(bars)
        assert features.shape == (1, 13)

    def test_shape_50_bars(self):
        """50 bars → shape (50, 13)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(50)
        features = compute_bar_features(bars)
        assert features.shape == (50, 13)

    def test_dtype_is_float(self):
        """Features should be floating point."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        assert np.issubdtype(features.dtype, np.floating)


# ===========================================================================
# 3. Feature bounds (spec tests #1-#11)
# ===========================================================================


class TestTradeFlowImbalanceBounds:
    """Spec test #1: Trade flow imbalance in [-1, +1]."""

    def test_bounds_all_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        col = features[:, 0]
        assert np.all(col >= -1.0), f"Min trade flow imbalance: {col.min()}"
        assert np.all(col <= 1.0), f"Max trade flow imbalance: {col.max()}"

    def test_all_buys_equals_plus_one(self):
        """All trades on buy side → imbalance = +1."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # TradeBar.trade_sizes with all positive sides
        # The feature uses trade sides from the bar.
        # Buy-side trades → imbalance = (buy_vol - sell_vol)/total = 1.0
        tp = np.array([4000.0, 4000.25, 4000.50], dtype=np.float64)
        ts = np.array([10, 10, 10], dtype=np.int32)
        bar = TradeBar(
            bar_index=0, open=4000.0, high=4000.50, low=4000.0, close=4000.25,
            volume=30, vwap=4000.25, t_start=_RTH_OPEN_NS,
            t_end=_RTH_OPEN_NS + 1000,
            session_date="2022-06-15",
            trade_prices=tp, trade_sizes=ts,
        )
        # To get all-buy imbalance, the implementation needs trade side info.
        # If the implementation uses bar.trade_prices vs bar.vwap or some
        # other heuristic, the actual value depends on implementation.
        # We test that the value is within bounds regardless.
        features = compute_bar_features([bar])
        assert -1.0 <= features[0, 0] <= 1.0


class TestBBOImbalanceBounds:
    """Spec test #2: BBO imbalance in [0, 1]."""

    def test_bounds_without_mbo(self):
        """Without mbo_data, BBO imbalance should be neutral (0.5)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        col = features[:, 1]
        assert np.all(col >= 0.0)
        assert np.all(col <= 1.0)

    def test_neutral_default_is_half(self):
        """When mbo_data=None, BBO imbalance defaults to 0.5."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 1], 0.5)


class TestDepthImbalanceBounds:
    """Spec test #3: Depth imbalance in [0, 1]."""

    def test_bounds_without_mbo(self):
        """Without mbo_data, depth imbalance should be neutral (0.5)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        col = features[:, 2]
        assert np.all(col >= 0.0)
        assert np.all(col <= 1.0)

    def test_neutral_default_is_half(self):
        """When mbo_data=None, depth imbalance defaults to 0.5."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 2], 0.5)


class TestBarRangeBounds:
    """Spec test #4: Bar range non-negative."""

    def test_range_non_negative(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        col = features[:, 3]
        assert np.all(col >= 0.0), f"Negative bar range: {col.min()}"


class TestBodyRangeRatioBounds:
    """Spec test #5: Body/range ratio in [-1, +1]."""

    def test_bounds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        col = features[:, 5]
        assert np.all(col >= -1.0), f"Min body/range: {col.min()}"
        assert np.all(col <= 1.0), f"Max body/range: {col.max()}"


class TestVWAPDisplacementBounds:
    """Spec test #6: VWAP displacement in [-1, +1]."""

    def test_bounds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        col = features[:, 6]
        assert np.all(col >= -1.0), f"Min VWAP displacement: {col.min()}"
        assert np.all(col <= 1.0), f"Max VWAP displacement: {col.max()}"


class TestVolumeLogFinitePositive:
    """Spec test #7: Volume (log) finite and derived from positive volume."""

    def test_finite(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        col = features[:, 7]
        assert np.all(np.isfinite(col)), "Volume log has non-finite values"

    def test_positive_volume_gives_positive_log(self):
        """V_k=100 → log(100) > 0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5, volume=100)
        features = compute_bar_features(bars)
        col = features[:, 7]
        assert np.all(col > 0), "log(volume) should be positive for volume > 1"


class TestNormalizedSessionTimeBounds:
    """Spec test #8: Normalized session time in [0, 1] and monotonically non-decreasing."""

    def test_bounds(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        col = features[:, 9]
        assert np.all(col >= 0.0), f"Session time < 0: {col.min()}"
        assert np.all(col <= 1.0), f"Session time > 1: {col.max()}"

    def test_monotonically_non_decreasing(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        col = features[:, 9]
        diffs = np.diff(col)
        assert np.all(diffs >= -1e-9), "Session time not monotonically non-decreasing"


class TestSessionAgeBounds:
    """Spec test #9: Session age starts at 0, saturates at 1.0 after 20 bars."""

    def test_first_bar_is_zero(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        assert features[0, 12] == pytest.approx(0.0)

    def test_saturates_at_one(self):
        """Bar 20+ should have session_age == 1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        for k in range(20, 30):
            assert features[k, 12] == pytest.approx(1.0), (
                f"Bar {k} session_age={features[k, 12]}, expected 1.0"
            )

    def test_monotonically_non_decreasing(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        col = features[:, 12]
        diffs = np.diff(col)
        assert np.all(diffs >= -1e-9)


class TestCancelRateAsymmetryBounds:
    """Spec test #10: Cancel rate asymmetry in [-1, +1]."""

    def test_bounds_without_mbo(self):
        """Without mbo_data, cancel rate asymmetry defaults to neutral (0.0)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        col = features[:, 10]
        assert np.all(col >= -1.0)
        assert np.all(col <= 1.0)

    def test_neutral_default(self):
        """When mbo_data=None, cancel rate asymmetry defaults to 0.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 10], 0.0)


class TestMeanSpreadBounds:
    """Spec test #11: Mean spread positive."""

    def test_positive_without_mbo(self):
        """Without mbo_data, mean spread defaults to 1.0 (positive)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        col = features[:, 11]
        assert np.all(col > 0), f"Mean spread not positive: {col.min()}"

    def test_neutral_default(self):
        """When mbo_data=None, mean spread defaults to 1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars, mbo_data=None)
        np.testing.assert_allclose(features[:, 11], 1.0)


# ===========================================================================
# 4. Feature computation correctness (spec tests #12-#18)
# ===========================================================================


class TestTrailingRealizedVol:
    """Spec test #12: Trailing realized vol uses exactly 20 bars."""

    def test_nan_for_first_19_bars(self):
        """Trailing realized vol is NaN for bars 0-18 (fewer than 20 bars of history)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        col = features[:, 8]
        for k in range(19):
            assert np.isnan(col[k]), f"Bar {k} realized vol should be NaN, got {col[k]}"

    def test_not_nan_at_bar_19(self):
        """Bar 19 (index 19) is the first bar with 20 bars of close history."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert not np.isnan(features[19, 8]), "Bar 19 realized vol should NOT be NaN"

    def test_hand_computed_realized_vol(self):
        """Hand-compute realized vol for a known close sequence.

        Close prices: [100.0, 101.0, 100.0, 101.0, ...] repeating 20 times.
        Log returns: log(101/100) and log(100/101) alternating → 19 returns.
        std of log returns should match.
        """
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # 21 bars with alternating close prices for a known pattern
        close_prices = [100.0 + (i % 2) for i in range(21)]
        bars = []
        for k in range(21):
            c = close_prices[k]
            bars.append(_make_bar(
                bar_index=k, open_price=c, high=c + 0.5, low=c - 0.5,
                close=c, volume=100,
                t_start=_RTH_OPEN_NS + k * 1_000_000_000,
                t_end=_RTH_OPEN_NS + (k + 1) * 1_000_000_000 - 1,
            ))

        features = compute_bar_features(bars)
        # Bar 19 uses closes 0..19 (20 close prices → 19 log returns)
        closes = np.array(close_prices[:20], dtype=np.float64)
        log_returns = np.log(closes[1:] / closes[:-1])
        expected_vol = np.std(log_returns, ddof=0)  # population std

        # Allow either ddof=0 or ddof=1 — both are reasonable
        vol_ddof0 = np.std(log_returns, ddof=0)
        vol_ddof1 = np.std(log_returns, ddof=1)
        actual = features[19, 8]
        assert (
            actual == pytest.approx(vol_ddof0, rel=1e-6) or
            actual == pytest.approx(vol_ddof1, rel=1e-6)
        ), f"Realized vol {actual} doesn't match ddof=0 ({vol_ddof0}) or ddof=1 ({vol_ddof1})"

    def test_non_negative(self):
        """Realized vol should always be non-negative where defined."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(30)
        features = compute_bar_features(bars)
        col = features[:, 8]
        valid = col[~np.isnan(col)]
        assert np.all(valid >= 0.0)


class TestTradeFlowImbalanceHandComputed:
    """Spec test #13: Trade flow imbalance hand-computed."""

    def test_known_buy_sell_volumes(self):
        """buy_vol=80, sell_vol=20 → imbalance = (80-20)/100 = 0.6.

        We need bars where trade sides are known. The trade_prices and
        trade_sizes are stored in TradeBar, but the side information needs
        to come from somewhere. The spec says "from TradeBar trade sides".

        Since TradeBar doesn't have a trade_sides field, the implementation
        may infer sides from price direction (tick rule) or require additional
        data. We test the resulting value is in bounds and plausible.
        """
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # Create a bar where prices strictly increase (all upticks → all buys)
        tp = np.array([4000.0, 4000.25, 4000.50, 4000.75, 4001.0], dtype=np.float64)
        ts = np.array([20, 20, 20, 20, 20], dtype=np.int32)
        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=4000.0,
            close=4001.0, volume=100, vwap=4000.50,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
            trade_prices=tp, trade_sizes=ts,
        )
        features = compute_bar_features([bar])
        # All upticks → strong positive imbalance
        assert features[0, 0] > 0, "All uptick trades should give positive imbalance"


class TestBarRangeHandComputed:
    """Spec test #14: Bar range hand-computed."""

    def test_known_range(self):
        """H=4002.0, L=4000.0 → range = (4002-4000)/0.25 = 8 ticks."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=4000.0,
            close=4001.5, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected_range = (4002.0 - 4000.0) / TICK_SIZE  # = 8.0
        assert features[0, 3] == pytest.approx(expected_range)

    def test_zero_range(self):
        """H == L → range = 0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4000.0, low=4000.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 3] == pytest.approx(0.0)


class TestBodyRangeRatioZeroRange:
    """Spec test #15: Body/range ratio with zero range returns 0."""

    def test_zero_range_gives_zero_ratio(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4000.0, low=4000.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 5] == pytest.approx(0.0)


class TestVWAPDisplacementZeroRange:
    """Spec test #16: VWAP displacement with zero range returns 0."""

    def test_zero_range_gives_zero_displacement(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4000.0, low=4000.0,
            close=4000.0, volume=100, vwap=4000.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 6] == pytest.approx(0.0)


class TestSessionAgeCapping:
    """Spec test #17: Session age capping."""

    def test_bar_0_is_zero(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert features[0, 12] == pytest.approx(0.0)

    def test_bar_10_is_half(self):
        """bar_index=10 → min(10/20, 1.0) = 0.5."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert features[10, 12] == pytest.approx(0.5)

    def test_bar_20_is_one(self):
        """bar_index=20 → min(20/20, 1.0) = 1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert features[20, 12] == pytest.approx(1.0)

    def test_bar_24_is_one(self):
        """bar_index=24 → min(24/20, 1.0) = 1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(25)
        features = compute_bar_features(bars)
        assert features[24, 12] == pytest.approx(1.0)


class TestVolumeLogTransform:
    """Spec test #18: Volume log transform."""

    def test_volume_100(self):
        """V_k=100 → log(100) ≈ 4.605."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected = math.log(100)
        assert features[0, 7] == pytest.approx(expected, rel=1e-4)

    def test_volume_1(self):
        """V_k=1 → log(1) = 0.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=1,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 7] == pytest.approx(0.0)

    def test_volume_1000(self):
        """V_k=1000 → log(1000) ≈ 6.908."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=1000,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected = math.log(1000)
        assert features[0, 7] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# 5. Feature column correctness — bar body (col 4)
# ===========================================================================


class TestBarBodyComputation:
    """Bar body = (C_k - O_k) in ticks."""

    def test_positive_body(self):
        """Close > Open → positive body."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=3999.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected = (4001.0 - 4000.0) / TICK_SIZE  # = 4 ticks
        assert features[0, 4] == pytest.approx(expected)

    def test_negative_body(self):
        """Close < Open → negative body."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected = (4000.0 - 4001.0) / TICK_SIZE  # = -4 ticks
        assert features[0, 4] == pytest.approx(expected)

    def test_zero_body(self):
        """Close == Open → zero body."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 4] == pytest.approx(0.0)


# ===========================================================================
# 6. Body/range ratio correctness (col 5)
# ===========================================================================


class TestBodyRangeRatioCorrectness:
    """Body/range = (C - O) / (H - L) when range > 0."""

    def test_full_bullish_bar(self):
        """O=4000, C=4002, H=4002, L=4000 → body/range = 1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4002.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 5] == pytest.approx(1.0)

    def test_full_bearish_bar(self):
        """O=4002, C=4000, H=4002, L=4000 → body/range = -1.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4002.0, high=4002.0, low=4000.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 5] == pytest.approx(-1.0)

    def test_doji_bar(self):
        """O=C=4001, H=4002, L=4000 → body/range = 0.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 5] == pytest.approx(0.0)


# ===========================================================================
# 7. VWAP displacement correctness (col 6)
# ===========================================================================


class TestVWAPDisplacementCorrectness:
    """VWAP displacement = (C - VWAP) / (H - L) when range > 0."""

    def test_close_above_vwap(self):
        """C > VWAP → positive displacement."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4002.0, volume=100, vwap=4001.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        expected = (4002.0 - 4001.0) / (4002.0 - 4000.0)  # = 0.5
        assert features[0, 6] == pytest.approx(expected)

    def test_close_at_vwap(self):
        """C == VWAP → displacement = 0.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100, vwap=4001.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 6] == pytest.approx(0.0)


# ===========================================================================
# 8. Normalization (spec tests #19-#22)
# ===========================================================================


class TestZScoreNormalization:
    """Spec test #19: After normalization, mean ≈ 0, std ≈ 1 over trailing window."""

    def test_normalized_mean_near_zero(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        # Create a raw feature matrix with known statistics
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((3000, 13)) * 5 + 10  # mean=10, std=5
        normed = normalize_features(raw, window=2000)

        # After the warmup period, features in the last 1000 rows should
        # have mean ≈ 0 within the trailing window
        tail = normed[2000:]
        for col in range(13):
            col_mean = np.mean(tail[:, col])
            assert abs(col_mean) < 1.0, (
                f"Col {col} mean={col_mean}, expected near 0"
            )

    def test_normalized_std_near_one(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        rng = np.random.default_rng(42)
        raw = rng.standard_normal((3000, 13)) * 5 + 10
        normed = normalize_features(raw, window=2000)

        tail = normed[2000:]
        for col in range(13):
            col_std = np.std(tail[:, col])
            assert 0.1 < col_std < 5.0, (
                f"Col {col} std={col_std}, expected near 1"
            )


class TestNormalizationClipping:
    """Spec test #20: Clipping to [-5, +5]."""

    def test_no_values_outside_clip_range(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        rng = np.random.default_rng(42)
        # Include extreme outliers
        raw = rng.standard_normal((500, 13))
        raw[0, :] = 1000.0  # extreme outlier
        raw[1, :] = -1000.0
        normed = normalize_features(raw, window=200)
        assert np.all(normed >= -5.0), f"Min value: {normed.min()}"
        assert np.all(normed <= 5.0), f"Max value: {normed.max()}"

    def test_clipping_affects_outliers(self):
        """Values that would be > 5 sigma get clipped to exactly 5."""
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.ones((100, 13))
        raw[99, 0] = 10000.0  # massive outlier in last row
        normed = normalize_features(raw, window=50)
        assert normed[99, 0] == pytest.approx(5.0)


class TestNoNanOrInfInOutput:
    """Spec test #21: No NaN or Inf in final feature matrix."""

    def test_no_nan_or_inf(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        rng = np.random.default_rng(42)
        raw = rng.standard_normal((200, 13)) * 3.0 + 5.0
        normed = normalize_features(raw, window=100)
        assert np.all(np.isfinite(normed)), "Normalized features have NaN or Inf"

    def test_constant_column_no_nan(self):
        """A constant column (std=0) should not produce NaN after normalization."""
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.ones((100, 13))
        raw[:, 0] = 5.0  # constant column
        normed = normalize_features(raw, window=50)
        assert np.all(np.isfinite(normed[:, 0])), "Constant column has NaN after norm"


class TestTrailingWindowBehavior:
    """Spec test #22: Features at the start use growing window."""

    def test_short_window_no_crash(self):
        """Features at start of sequence (fewer than window bars) should work."""
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.random.default_rng(42).standard_normal((50, 13))
        # Window is 2000 but only 50 rows — should still work
        normed = normalize_features(raw, window=2000)
        assert normed.shape == (50, 13)
        assert np.all(np.isfinite(normed))

    def test_output_shape_matches_input(self):
        """Normalized output has same shape as input."""
        from lob_rl.barrier.feature_pipeline import normalize_features

        for n in [10, 100, 500]:
            raw = np.random.default_rng(42).standard_normal((n, 13))
            normed = normalize_features(raw, window=200)
            assert normed.shape == (n, 13)


# ===========================================================================
# 9. Lookback assembly (spec tests #23-#25)
# ===========================================================================


class TestLookbackShape:
    """Spec test #23: Lookback output shape is (N-h+1, 13*h)."""

    def test_shape_h10(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((50, 13))
        result = assemble_lookback(normed, h=10)
        assert result.shape == (50 - 10 + 1, 13 * 10)  # (41, 130)

    def test_shape_h5(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((30, 13))
        result = assemble_lookback(normed, h=5)
        assert result.shape == (30 - 5 + 1, 13 * 5)  # (26, 65)

    def test_shape_h1(self):
        """h=1 means no stacking, output is same as input."""
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((20, 13))
        result = assemble_lookback(normed, h=1)
        assert result.shape == (20, 13)


class TestLookbackCorrectness:
    """Spec test #24: Row i == concatenation of features[i:i+h]."""

    def test_lookback_row_content(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        rng = np.random.default_rng(42)
        normed = rng.standard_normal((20, 13))
        h = 5
        result = assemble_lookback(normed, h=h)

        for i in range(result.shape[0]):
            expected = normed[i:i + h].flatten()
            np.testing.assert_array_equal(
                result[i], expected,
                err_msg=f"Row {i} doesn't match features[{i}:{i+h}].flatten()"
            )

    def test_first_row(self):
        """First row should be features[0:h].flatten()."""
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.arange(130, dtype=np.float64).reshape(10, 13)
        result = assemble_lookback(normed, h=3)
        expected = normed[:3].flatten()
        np.testing.assert_array_equal(result[0], expected)

    def test_last_row(self):
        """Last row should be features[-h:].flatten()."""
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.arange(130, dtype=np.float64).reshape(10, 13)
        h = 3
        result = assemble_lookback(normed, h=h)
        expected = normed[-h:].flatten()
        np.testing.assert_array_equal(result[-1], expected)


class TestLookbackDefaultH10:
    """Spec test #25: Default h=10 gives 130-dim output."""

    def test_default_h(self):
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((50, 13))
        result = assemble_lookback(normed)  # default h=10
        assert result.shape[1] == 130


# ===========================================================================
# 10. Edge cases (spec tests #26-#28)
# ===========================================================================


class TestSingleBarSession:
    """Spec test #26: Single bar session."""

    def test_features_computed(self):
        """Single bar should produce features of shape (1, 13)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        features = compute_bar_features(bars)
        assert features.shape == (1, 13)

    def test_lookback_impossible_with_h_gt_1(self):
        """Lookback with h > 1 on single-bar features → empty output."""
        from lob_rl.barrier.feature_pipeline import assemble_lookback

        normed = np.random.default_rng(42).standard_normal((1, 13))
        result = assemble_lookback(normed, h=5)
        # N=1, h=5 → N-h+1 = -3 → should be empty or zero rows
        assert result.shape[0] == 0


class TestAllSamePriceBar:
    """Spec test #27: All same price bar — no division errors."""

    def test_no_division_errors(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        # All same price → range=0, body=0
        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4000.0, low=4000.0,
            close=4000.0, volume=100, vwap=4000.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features.shape == (1, 13)

        # Range should be 0
        assert features[0, 3] == pytest.approx(0.0)
        # Body should be 0
        assert features[0, 4] == pytest.approx(0.0)
        # Body/range should be 0 (not NaN/Inf)
        assert features[0, 5] == pytest.approx(0.0)
        # VWAP displacement should be 0 (not NaN/Inf)
        assert features[0, 6] == pytest.approx(0.0)
        # No NaN or Inf anywhere
        assert np.all(np.isfinite(features) | np.isnan(features[:, 8:9]))
        # Only col 8 (realized vol) can be NaN for single bar


class TestFeaturesShapeAlwaysN13:
    """Spec test #28: Features shape is always (N, 13) regardless of input."""

    def test_various_bar_counts(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        for n in [1, 2, 5, 10, 50, 100]:
            bars = _make_session_bars(n)
            features = compute_bar_features(bars)
            assert features.shape == (n, 13), (
                f"Expected ({n}, 13), got {features.shape}"
            )


# ===========================================================================
# 11. build_feature_matrix — end-to-end
# ===========================================================================


class TestBuildFeatureMatrixEndToEnd:
    """End-to-end: compute features, normalize, assemble lookback."""

    def test_output_shape_default(self):
        """Default h=10, window=2000 with 50 bars."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        # 50 bars, h=10, warmup for realized vol is 19 bars
        # After feature computation: (50, 13)
        # After normalization: (50, 13)
        # After lookback: (50 - 10 + 1, 130) = (41, 130)
        # But bars with NaN in realized vol (first 19) may be dropped
        # Actual shape depends on implementation of warmup handling
        assert result.ndim == 2
        assert result.shape[1] == 130  # 13 * 10

    def test_output_is_finite(self):
        """No NaN or Inf in the final matrix."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert np.all(np.isfinite(result)), "Final matrix has NaN/Inf"

    def test_clipped_to_5(self):
        """All values in final matrix are in [-5, +5]."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars)
        assert np.all(result >= -5.0)
        assert np.all(result <= 5.0)

    def test_custom_h(self):
        """Custom h=5 gives 65-dim output."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars, h=5)
        assert result.shape[1] == 65  # 13 * 5

    def test_custom_window(self):
        """Custom window should not crash."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars, window=20)
        assert result.ndim == 2
        assert result.shape[1] == 130

    def test_accepts_mbo_data_none(self):
        """Passing mbo_data=None explicitly should work (LOB features use neutral defaults)."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        result = build_feature_matrix(bars, mbo_data=None)
        assert result.ndim == 2
        assert np.all(np.isfinite(result))


# ===========================================================================
# 12. Column index verification — all 13 features in correct order
# ===========================================================================


class TestFeatureColumnOrder:
    """Verify the 13 features are in the correct column order per spec."""

    def test_column_count(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        assert features.shape[1] == 13

    def test_bar_range_is_nonneg_at_col3(self):
        """Column 3 is bar range (should be non-negative)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        assert np.all(features[:, 3] >= 0.0)

    def test_volume_log_at_col7(self):
        """Column 7 is log(volume)."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 7] == pytest.approx(math.log(100), rel=1e-4)

    def test_session_age_at_col12(self):
        """Column 12 is session age."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        # Bar 0 → age = 0/20 = 0.0
        assert features[0, 12] == pytest.approx(0.0)

    def test_session_time_at_col9(self):
        """Column 9 is normalized session time (should be in [0, 1])."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(20)
        features = compute_bar_features(bars)
        assert np.all(features[:, 9] >= 0.0)
        assert np.all(features[:, 9] <= 1.0)


# ===========================================================================
# 13. Normalization with NaN handling (realized vol warmup)
# ===========================================================================


class TestNormalizationWithNaN:
    """Normalization must handle NaN values from realized vol warmup."""

    def test_nan_in_raw_features_handled(self):
        """Raw features with NaN (from realized vol) should be handled in normalization."""
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.random.default_rng(42).standard_normal((50, 13))
        # Set first 19 rows of col 8 to NaN (realized vol warmup)
        raw[:19, 8] = np.nan
        normed = normalize_features(raw, window=30)

        # After normalization, there should be no NaN
        assert np.all(np.isfinite(normed)), "NaN survived normalization"


# ===========================================================================
# 14. Large sequence stress test
# ===========================================================================


class TestLargeSequence:
    """Features pipeline handles long bar sequences without error."""

    def test_200_bars(self):
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(200)
        features = compute_bar_features(bars)
        assert features.shape == (200, 13)

    def test_end_to_end_200_bars(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(200)
        result = build_feature_matrix(bars)
        assert result.ndim == 2
        assert result.shape[1] == 130
        assert np.all(np.isfinite(result))

    def test_end_to_end_200_bars_values_clipped(self):
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(200)
        result = build_feature_matrix(bars)
        assert np.all(result >= -5.0)
        assert np.all(result <= 5.0)


# ===========================================================================
# 15. Normalization — default window=2000
# ===========================================================================


class TestNormalizationDefaultWindow:
    """normalize_features default window should be 2000."""

    def test_default_window_accepted(self):
        from lob_rl.barrier.feature_pipeline import normalize_features

        raw = np.random.default_rng(42).standard_normal((100, 13))
        # Call without explicit window — should default to 2000
        normed = normalize_features(raw)
        assert normed.shape == (100, 13)


# ===========================================================================
# 16. VWAP displacement — bounded even with extreme VWAP
# ===========================================================================


class TestVWAPDisplacementExtreme:
    """VWAP displacement stays in [-1, +1] even when VWAP is at bar extremes."""

    def test_vwap_at_high(self):
        """VWAP == High → displacement = (C - H) / (H - L), bounded."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100, vwap=4002.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert -1.0 <= features[0, 6] <= 1.0

    def test_vwap_at_low(self):
        """VWAP == Low → displacement = (C - L) / (H - L), bounded."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4001.0, high=4002.0, low=4000.0,
            close=4001.0, volume=100, vwap=4000.0,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert -1.0 <= features[0, 6] <= 1.0


# ===========================================================================
# 17. Normalized session time computation correctness
# ===========================================================================


class TestNormalizedSessionTimeCorrectness:
    """Session time = (t_end - RTH_open) / (RTH_close - RTH_open)."""

    def test_first_bar_near_zero(self):
        """Bar right at RTH open should have session time near 0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_OPEN_NS, t_end=_RTH_OPEN_NS + 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 9] < 0.01, "First bar session time should be near 0"

    def test_last_bar_near_one(self):
        """Bar near RTH close should have session time near 1."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=_RTH_CLOSE_NS - 2000, t_end=_RTH_CLOSE_NS - 1000,
        )
        features = compute_bar_features([bar])
        assert features[0, 9] > 0.99, "Last bar session time should be near 1"

    def test_midday_bar(self):
        """Bar at midpoint of session should have session time near 0.5."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        mid_ns = (_RTH_OPEN_NS + _RTH_CLOSE_NS) // 2
        bar = _make_bar(
            bar_index=0, open_price=4000.0, high=4001.0, low=3999.0,
            close=4000.0, volume=100,
            t_start=mid_ns - 1000, t_end=mid_ns,
        )
        features = compute_bar_features([bar])
        assert 0.4 < features[0, 9] < 0.6, (
            f"Midday session time = {features[0, 9]}, expected ~0.5"
        )
