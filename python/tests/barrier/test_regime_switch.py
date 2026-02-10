"""Tests for the regime-switch validation module.

Spec: docs/t5-regime-switch-validation.md

Tests the synthetic regime-switch data generator, label distribution
comparison, feature distribution shifts, normalization adaptation,
and full pipeline validation.
"""

import numpy as np
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar
from lob_rl.barrier import TICK_SIZE


# ===========================================================================
# 1. Imports
# ===========================================================================


class TestRegimeSwitchImports:
    """All public functions should be importable from regime_switch module."""

    def test_generate_regime_switch_trades_importable(self):
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        assert callable(generate_regime_switch_trades)

    def test_validate_regime_switch_importable(self):
        from lob_rl.barrier.regime_switch import validate_regime_switch
        assert callable(validate_regime_switch)

    def test_compute_segment_stats_importable(self):
        from lob_rl.barrier.regime_switch import compute_segment_stats
        assert callable(compute_segment_stats)

    def test_ks_test_features_importable(self):
        from lob_rl.barrier.regime_switch import ks_test_features
        assert callable(ks_test_features)

    def test_measure_normalization_adaptation_importable(self):
        from lob_rl.barrier.regime_switch import measure_normalization_adaptation
        assert callable(measure_normalization_adaptation)


# ===========================================================================
# 2. Synthetic data generation — output shape and bar count
# ===========================================================================


class TestSyntheticDataShape:
    """Spec tests #1 and #2: total trades and bar count."""

    def test_total_trade_count(self):
        """Total trades = (n_bars_low + n_bars_high) * bar_size."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low, n_bars_high, bar_size = 100, 100, 500
        trades, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=bar_size, seed=42,
        )
        expected_trades = (n_bars_low + n_bars_high) * bar_size
        assert len(trades) == expected_trades

    def test_bar_count(self):
        """Number of bars = n_bars_low + n_bars_high."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low, n_bars_high = 100, 100
        trades, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        assert len(bars) == n_bars_low + n_bars_high

    def test_default_parameters_shape(self):
        """Default params (5000+5000 bars, 500 trades/bar) produce correct shape."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        trades, bars = generate_regime_switch_trades(seed=42)
        assert len(trades) == (5000 + 5000) * 500
        assert len(bars) == 10000


# ===========================================================================
# 3. Synthetic data generation — tick increment properties
# ===========================================================================


class TestLowVolTickIncrements:
    """Spec test #3: Low-vol regime has exactly +/- 1 tick moves."""

    def test_low_vol_increments_are_one_tick(self):
        """All trade-to-trade price changes in low-vol are exactly +/- tick_size."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low, n_bars_high, bar_size = 100, 100, 500
        trades, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=bar_size, seed=42,
        )
        n_low_trades = n_bars_low * bar_size
        low_prices = trades["price"][:n_low_trades]
        diffs = np.diff(low_prices)
        abs_diffs = np.abs(diffs)
        # All increments should be exactly 1 tick (0.25)
        np.testing.assert_allclose(abs_diffs, TICK_SIZE, atol=1e-10,
                                   err_msg="Low-vol increments are not all 1-tick")


class TestHighVolTickIncrements:
    """Spec test #4: High-vol regime includes multi-tick moves."""

    def test_high_vol_includes_multi_tick_moves(self):
        """High-vol trade-to-trade changes include 2*tick and 3*tick magnitudes."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low, n_bars_high, bar_size = 100, 200, 500
        trades, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=bar_size, seed=42,
        )
        n_low_trades = n_bars_low * bar_size
        high_prices = trades["price"][n_low_trades:]
        diffs = np.abs(np.diff(high_prices))

        # Should contain moves of magnitude 2*tick_size and 3*tick_size
        has_2tick = np.any(np.isclose(diffs, 2 * TICK_SIZE, atol=1e-10))
        has_3tick = np.any(np.isclose(diffs, 3 * TICK_SIZE, atol=1e-10))
        assert has_2tick, "High-vol should include 2-tick moves"
        assert has_3tick, "High-vol should include 3-tick moves"


# ===========================================================================
# 4. Regime boundary location
# ===========================================================================


class TestRegimeBoundary:
    """Spec test #5: Regime boundary is at correct bar index."""

    def test_boundary_at_n_bars_low(self):
        """Bar index n_bars_low is the first high-vol bar."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low = 100
        trades, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=100,
            bar_size=500, seed=42,
        )
        # Bar indices should be sequential starting from 0
        assert bars[n_bars_low].bar_index == n_bars_low
        # Bars 0..n_bars_low-1 are low-vol, bars n_bars_low.. are high-vol
        assert len(bars) == 200


# ===========================================================================
# 5. Reproducibility
# ===========================================================================


class TestReproducibility:
    """Spec test #6: Same seed produces identical trades and bars."""

    def test_same_seed_same_trades(self):
        """Two calls with seed=42 produce identical trade arrays."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        trades1, bars1 = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=42,
        )
        trades2, bars2 = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=42,
        )
        np.testing.assert_array_equal(trades1["price"], trades2["price"])
        np.testing.assert_array_equal(trades1["size"], trades2["size"])

    def test_same_seed_same_bars(self):
        """Two calls with seed=42 produce identical bars (close prices)."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        _, bars1 = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=42,
        )
        _, bars2 = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=42,
        )
        for b1, b2 in zip(bars1, bars2):
            assert b1.close == b2.close
            assert b1.high == b2.high
            assert b1.low == b2.low

    def test_different_seed_different_trades(self):
        """Different seeds produce different trade arrays."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        trades1, _ = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=42,
        )
        trades2, _ = generate_regime_switch_trades(
            n_bars_low=50, n_bars_high=50, bar_size=100, seed=99,
        )
        assert not np.array_equal(trades1["price"], trades2["price"])


# ===========================================================================
# 6. Price continuity at boundary
# ===========================================================================


class TestPriceContinuityAtBoundary:
    """Spec test #7: No gap between last low-vol trade and first high-vol trade."""

    def test_continuous_price_at_regime_boundary(self):
        """The price series is continuous across the regime boundary."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        n_bars_low, bar_size = 100, 500
        trades, _ = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=100,
            bar_size=bar_size, seed=42,
        )
        boundary_idx = n_bars_low * bar_size
        last_low = trades["price"][boundary_idx - 1]
        first_high = trades["price"][boundary_idx]
        # The first high-vol trade should continue from the last low-vol price
        # (the increment is drawn from the new regime, but starts from the same price)
        diff = abs(first_high - last_low)
        # Max single-step increment is 3*tick_size (high-vol regime)
        assert diff <= 3 * TICK_SIZE + 1e-10, (
            f"Price gap at boundary: {diff}, expected <= {3 * TICK_SIZE}"
        )


# ===========================================================================
# 7. Label distribution differences
# ===========================================================================


class TestTimeoutRateHigherInLowVol:
    """Spec test #8: Timeout rate higher in low-vol segment."""

    def test_low_vol_has_higher_timeout_rate(self):
        """low_vol['p_zero'] > high_vol['p_zero']."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["low_vol"]["p_zero"] > result["high_vol"]["p_zero"]


class TestTimeoutRatioGT2:
    """Spec test #9: Timeout ratio > 2x."""

    def test_timeout_ratio_exceeds_two(self):
        """low_vol timeout rate / high_vol timeout rate > 2.0."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["timeout_ratio"] > 2.0, (
            f"Timeout ratio {result['timeout_ratio']} not > 2.0"
        )


class TestMeanTauLongerInLowVol:
    """Spec test #10: Mean tau longer in low-vol."""

    def test_low_vol_mean_tau_greater(self):
        """low_vol['mean_tau'] > high_vol['mean_tau']."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["low_vol"]["mean_tau"] > result["high_vol"]["mean_tau"]


class TestChiSquaredRejectsH0:
    """Spec test #11: Chi-squared test rejects H0 (label distributions differ)."""

    def test_chi2_p_value_significant(self):
        """chi2_p_value < 0.01."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["chi2_p_value"] < 0.01, (
            f"Chi2 p-value {result['chi2_p_value']} not < 0.01"
        )


class TestLowVolTimeoutsNontrivial:
    """Spec test #12: Low-vol has meaningful timeout rate (>5%)."""

    def test_low_vol_p_zero_gt_5_pct(self):
        """low_vol['p_zero'] > 0.05 — timeouts are nontrivial in low-vol regime.

        With bar_size=500 and large barriers, 1-tick random walks produce
        meaningful timeouts. The absolute rate won't exceed barrier hits
        (bars span ~22 ticks so barriers are often hit intrabar), but the
        rate should be meaningfully nonzero, proving the regime distinction.
        """
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["low_vol"]["p_zero"] > 0.05, (
            f"p_zero={result['low_vol']['p_zero']} not > 0.05"
        )


class TestHighVolResolvesQuickly:
    """Spec test #13: High-vol resolves quickly (mean_tau < 5)."""

    def test_high_vol_mean_tau_lt_5(self):
        """high_vol['mean_tau'] < 5 with 3-tick moves."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["high_vol"]["mean_tau"] < 5, (
            f"High-vol mean_tau={result['high_vol']['mean_tau']} not < 5"
        )


# ===========================================================================
# 8. Feature distribution shifts
# ===========================================================================


class TestKSBarRange:
    """Spec test #14: KS test on bar range rejects H0."""

    def test_ks_bar_range_significant(self):
        """ks_bar_range_p < 0.01 (bar range distributions clearly differ)."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["ks_bar_range_p"] < 0.01, (
            f"KS bar range p={result['ks_bar_range_p']} not < 0.01"
        )


class TestKSRealizedVol:
    """Spec test #15: KS test on realized vol rejects H0."""

    def test_ks_realized_vol_significant(self):
        """ks_realized_vol_p < 0.01 (realized vol distributions clearly differ)."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["ks_realized_vol_p"] < 0.01, (
            f"KS realized vol p={result['ks_realized_vol_p']} not < 0.01"
        )


class TestBarRangeHigherInHighVol:
    """Spec test #16: Mean bar range in high-vol > low-vol."""

    def test_bar_range_higher_in_high_vol(self):
        """Compare mean bar range between segments."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        n_bars_low, n_bars_high = 500, 500
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        features = compute_bar_features(bars)
        # Column 3 is bar range
        low_range = np.mean(features[:n_bars_low, 3])
        high_range = np.mean(features[n_bars_low:, 3])
        assert high_range > low_range, (
            f"High-vol range {high_range} not > low-vol range {low_range}"
        )


class TestRealizedVolHigherInHighVol:
    """Spec test #17: Mean realized vol in high-vol > low-vol (after warmup)."""

    def test_realized_vol_higher_in_high_vol(self):
        """Compare mean realized vol between segments, skipping warmup bars."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        n_bars_low, n_bars_high = 500, 500
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        features = compute_bar_features(bars)
        # Column 8 is realized vol; skip first 19 bars (NaN warmup)
        warmup = 19
        low_vol = features[warmup:n_bars_low, 8]
        # High-vol: skip bars right after boundary where trailing vol is stale
        high_vol = features[n_bars_low + 200:, 8]
        low_mean = np.nanmean(low_vol)
        high_mean = np.nanmean(high_vol)
        assert high_mean > low_mean, (
            f"High-vol realized vol {high_mean} not > low-vol {low_mean}"
        )


# ===========================================================================
# 9. Normalization adaptation
# ===========================================================================


class TestNoDeadZone500:
    """Spec test #18: Normalization adapts within 500 bars."""

    def test_norm_adaptation_lt_500(self):
        """norm_adaptation_bars < 500."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["norm_adaptation_bars"] < 500, (
            f"Adaptation took {result['norm_adaptation_bars']} bars (>= 500)"
        )


class TestAdaptationWithin200:
    """Spec test #19: Tighter bound — adaptation within 200 bars."""

    def test_norm_adaptation_lt_200(self):
        """norm_adaptation_bars < 200."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["norm_adaptation_bars"] < 200, (
            f"Adaptation took {result['norm_adaptation_bars']} bars (>= 200)"
        )


class TestNormalizedVolShiftVisible:
    """Spec test #20: Normalized realized vol shows visible shift at boundary."""

    def test_visible_shift_after_boundary(self):
        """Mean normalized realized vol in [boundary+200, boundary+500] differs
        from [boundary-500, boundary-200] by >= 1.0 standard units."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.feature_pipeline import compute_bar_features, normalize_features

        n_bars_low, n_bars_high = 5000, 5000
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        features = compute_bar_features(bars)
        normed = normalize_features(features)

        boundary = n_bars_low
        # Column 8 is realized vol
        pre_mean = np.mean(normed[boundary - 500:boundary - 200, 8])
        post_mean = np.mean(normed[boundary + 200:boundary + 500, 8])
        diff = abs(post_mean - pre_mean)
        assert diff >= 1.0, (
            f"Normalized vol shift {diff} < 1.0 standard units"
        )


# ===========================================================================
# 10. Full pipeline validation
# ===========================================================================


class TestFullPipelinePass:
    """Spec test #21: All tests pass via validate_regime_switch."""

    def test_all_tests_pass(self):
        """validate_regime_switch(seed=42)['pass'] == True."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(seed=42)
        assert result["pass"] is True, (
            f"Pipeline validation failed. Result keys: {list(result.keys())}"
        )


class TestNoNaNInFeatures:
    """Spec test #22: No NaN in features after warmup (bar 19+)."""

    def test_no_nan_after_warmup(self):
        """Raw features have no NaN after bar index 19."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        n_bars_low, n_bars_high = 200, 200
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        features = compute_bar_features(bars)
        # After warmup (bar 19+), no NaN
        assert not np.any(np.isnan(features[19:])), (
            "Found NaN in features after warmup (bar 19+)"
        )


class TestNoInfInFeatures:
    """Spec test #23: No Inf in features anywhere."""

    def test_no_inf_anywhere(self):
        """Raw features have no Inf values."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        n_bars_low, n_bars_high = 200, 200
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        features = compute_bar_features(bars)
        assert not np.any(np.isinf(features)), "Found Inf in features"


class TestLabelsValid:
    """Spec test #24: All labels in {-1, 0, +1}, tau >= 1, tau <= t_max."""

    def test_label_values_valid(self):
        """All labels are -1, 0, or +1."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        n_bars_low, n_bars_high = 200, 200
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        for lbl in labels:
            assert lbl.label in {-1, 0, 1}, (
                f"Bar {lbl.bar_index}: invalid label={lbl.label}"
            )

    def test_tau_ge_1(self):
        """All tau >= 1."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        n_bars_low, n_bars_high = 200, 200
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        for lbl in labels:
            assert lbl.tau >= 1, f"Bar {lbl.bar_index}: tau={lbl.tau} < 1"

    def test_tau_le_t_max(self):
        """All tau <= t_max."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        t_max = 40
        n_bars_low, n_bars_high = 200, 200
        _, bars = generate_regime_switch_trades(
            n_bars_low=n_bars_low, n_bars_high=n_bars_high,
            bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=t_max)
        for lbl in labels:
            assert lbl.tau <= t_max, (
                f"Bar {lbl.bar_index}: tau={lbl.tau} > t_max={t_max}"
            )


# ===========================================================================
# 11. Edge cases
# ===========================================================================


class TestSmallRegime:
    """Spec test #25: Small regime (100 bars each) doesn't crash."""

    def test_small_regime_no_crash(self):
        """100 bars per regime produces results without error."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        assert isinstance(result, dict)
        assert "n_bars_total" in result
        assert result["n_bars_total"] == 200


class TestDeterministicSeed:
    """Spec test #26: Two calls with same seed produce identical results."""

    def test_identical_results_same_seed(self):
        """validate_regime_switch(seed=42) called twice gives identical dicts."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result1 = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        result2 = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        # Numeric fields should be identical
        for key in ["n_bars_total", "n_bars_low", "n_bars_high", "boundary_bar",
                     "timeout_ratio", "chi2_p_value", "ks_bar_range_p",
                     "ks_realized_vol_p", "norm_adaptation_bars", "pass"]:
            assert result1[key] == result2[key], (
                f"Key '{key}': {result1[key]} != {result2[key]}"
            )
        for segment in ["low_vol", "high_vol"]:
            for stat_key in ["p_plus", "p_minus", "p_zero", "mean_tau", "median_tau"]:
                assert result1[segment][stat_key] == result2[segment][stat_key], (
                    f"{segment}/{stat_key}: {result1[segment][stat_key]} != {result2[segment][stat_key]}"
                )


# ===========================================================================
# 12. compute_segment_stats helper
# ===========================================================================


class TestComputeSegmentStats:
    """Tests for the compute_segment_stats helper function."""

    def test_known_distribution(self):
        """Hand-crafted labels with known distribution."""
        from lob_rl.barrier.regime_switch import compute_segment_stats
        from lob_rl.barrier.label_pipeline import BarrierLabel

        labels = [
            BarrierLabel(bar_index=0, label=1, tau=5, resolution_type="upper",
                         entry_price=4000.0, resolution_bar=5),
            BarrierLabel(bar_index=1, label=-1, tau=3, resolution_type="lower",
                         entry_price=4000.0, resolution_bar=4),
            BarrierLabel(bar_index=2, label=0, tau=40, resolution_type="timeout",
                         entry_price=4000.0, resolution_bar=42),
            BarrierLabel(bar_index=3, label=1, tau=2, resolution_type="upper",
                         entry_price=4000.0, resolution_bar=5),
            BarrierLabel(bar_index=4, label=0, tau=40, resolution_type="timeout",
                         entry_price=4000.0, resolution_bar=44),
        ]
        stats = compute_segment_stats(labels, start=0, end=5)
        assert stats["p_plus"] == pytest.approx(2.0 / 5.0)
        assert stats["p_minus"] == pytest.approx(1.0 / 5.0)
        assert stats["p_zero"] == pytest.approx(2.0 / 5.0)
        # mean_tau = (5 + 3 + 40 + 2 + 40) / 5 = 18.0
        assert stats["mean_tau"] == pytest.approx(18.0)
        # median_tau = sorted [2,3,5,40,40] → median = 5
        assert stats["median_tau"] == pytest.approx(5.0)

    def test_subsegment_slicing(self):
        """start/end should slice the label list correctly."""
        from lob_rl.barrier.regime_switch import compute_segment_stats
        from lob_rl.barrier.label_pipeline import BarrierLabel

        labels = [
            BarrierLabel(bar_index=0, label=1, tau=5, resolution_type="upper",
                         entry_price=4000.0, resolution_bar=5),
            BarrierLabel(bar_index=1, label=0, tau=40, resolution_type="timeout",
                         entry_price=4000.0, resolution_bar=41),
            BarrierLabel(bar_index=2, label=-1, tau=3, resolution_type="lower",
                         entry_price=4000.0, resolution_bar=5),
        ]
        stats = compute_segment_stats(labels, start=1, end=3)
        # Only labels[1] and labels[2]
        assert stats["p_plus"] == pytest.approx(0.0)
        assert stats["p_minus"] == pytest.approx(0.5)
        assert stats["p_zero"] == pytest.approx(0.5)

    def test_returns_required_keys(self):
        """Stats dict has p_plus, p_minus, p_zero, mean_tau, median_tau."""
        from lob_rl.barrier.regime_switch import compute_segment_stats
        from lob_rl.barrier.label_pipeline import BarrierLabel

        labels = [
            BarrierLabel(bar_index=0, label=1, tau=5, resolution_type="upper",
                         entry_price=4000.0, resolution_bar=5),
        ]
        stats = compute_segment_stats(labels, start=0, end=1)
        for key in ["p_plus", "p_minus", "p_zero", "mean_tau", "median_tau"]:
            assert key in stats, f"Missing key: {key}"


# ===========================================================================
# 13. ks_test_features helper
# ===========================================================================


class TestKSTestFeatures:
    """Tests for the ks_test_features helper function."""

    def test_returns_p_values_for_all_columns(self):
        """Should return KS p-values for columns 0 through 12."""
        from lob_rl.barrier.regime_switch import ks_test_features

        rng = np.random.default_rng(42)
        # Two distinct distributions
        features = np.vstack([
            rng.normal(0, 1, (1000, 13)),
            rng.normal(5, 1, (1000, 13)),
        ])
        boundary = 1000
        result = ks_test_features(features, boundary, window=500)
        for col in range(13):
            key = f"ks_p_col_{col}"
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], float)

    def test_detects_distribution_shift(self):
        """When distributions are clearly different, p-values should be small."""
        from lob_rl.barrier.regime_switch import ks_test_features

        rng = np.random.default_rng(42)
        features = np.vstack([
            rng.normal(0, 1, (1000, 13)),
            rng.normal(10, 1, (1000, 13)),
        ])
        boundary = 1000
        result = ks_test_features(features, boundary, window=500)
        # All columns should show significant difference
        for col in range(13):
            assert result[f"ks_p_col_{col}"] < 0.01, (
                f"Column {col} p={result[f'ks_p_col_{col}']} not < 0.01"
            )

    def test_same_distribution_not_significant(self):
        """When distributions are the same, p-values should be large."""
        from lob_rl.barrier.regime_switch import ks_test_features

        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (2000, 13))
        boundary = 1000
        result = ks_test_features(features, boundary, window=500)
        # Most columns should NOT reject H0
        n_significant = sum(
            1 for col in range(13)
            if result[f"ks_p_col_{col}"] < 0.01
        )
        # Allow up to 2 false positives at 1% level (13 * 0.01 ≈ 0.13)
        assert n_significant <= 2, (
            f"{n_significant}/13 columns falsely significant"
        )


# ===========================================================================
# 14. measure_normalization_adaptation helper
# ===========================================================================


class TestMeasureNormalizationAdaptation:
    """Tests for the measure_normalization_adaptation helper function."""

    def test_returns_integer(self):
        """Should return an integer number of bars."""
        from lob_rl.barrier.regime_switch import measure_normalization_adaptation

        rng = np.random.default_rng(42)
        # Regime shift at bar 1000: mean jumps from 0 to 5
        normed = np.vstack([
            rng.normal(0, 1, (1000, 13)),
            rng.normal(5, 1, (2000, 13)),
        ])
        result = measure_normalization_adaptation(normed, boundary=1000, col=8)
        assert isinstance(result, int)

    def test_adapts_for_obvious_shift(self):
        """With a large distribution shift, should adapt eventually (not return -1)."""
        from lob_rl.barrier.regime_switch import measure_normalization_adaptation

        rng = np.random.default_rng(42)
        normed = np.vstack([
            rng.normal(0, 1, (1000, 13)),
            rng.normal(5, 1, (2000, 13)),
        ])
        result = measure_normalization_adaptation(normed, boundary=1000, col=8)
        assert result >= 0, "Should adapt for obvious distribution shift"

    def test_returns_minus_one_when_no_adaptation(self):
        """Returns -1 when the feature never reaches the post-boundary mean."""
        from lob_rl.barrier.regime_switch import measure_normalization_adaptation

        # Feature stays at 0 forever (no shift to adapt to)
        normed = np.zeros((2000, 13))
        # But set the "post-boundary steady-state" (boundary+500 onward) to be
        # very different from what's right after boundary — by making the tail differ
        normed[1500:, 8] = 100.0  # Only the tail has the "new" values
        # The range [1000:1500] stays at 0, which is very far from mean of 100
        # Actually, measuring adaptation checks if the range near boundary
        # reaches the steady state. If boundary+0 through boundary+499 are at 0
        # and steady state is 100, it will never adapt within available bars.
        # But wait — the function checks bars starting from boundary until they
        # enter the steady-state envelope. If none do before boundary+500, return -1.
        result = measure_normalization_adaptation(normed, boundary=1000, col=8)
        assert result == -1


# ===========================================================================
# 15. Return value structure of validate_regime_switch
# ===========================================================================


class TestValidateReturnStructure:
    """validate_regime_switch returns a dict with all required keys."""

    def test_all_top_level_keys_present(self):
        """All top-level keys from the spec are present."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        required_keys = [
            "n_bars_total", "n_bars_low", "n_bars_high", "boundary_bar",
            "low_vol", "high_vol",
            "timeout_ratio", "chi2_p_value",
            "ks_bar_range_p", "ks_realized_vol_p",
            "norm_adaptation_bars", "pass",
        ]
        for key in required_keys:
            assert key in result, f"Missing top-level key: {key}"

    def test_segment_stat_keys_present(self):
        """low_vol and high_vol sub-dicts have all required stat keys."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        for segment in ["low_vol", "high_vol"]:
            assert isinstance(result[segment], dict), (
                f"{segment} is not a dict"
            )
            for key in ["p_plus", "p_minus", "p_zero", "mean_tau", "median_tau"]:
                assert key in result[segment], (
                    f"Missing key {segment}/{key}"
                )

    def test_boundary_bar_value(self):
        """boundary_bar should equal n_bars_low."""
        from lob_rl.barrier.regime_switch import validate_regime_switch

        result = validate_regime_switch(
            n_bars_low=100, n_bars_high=100, bar_size=500, seed=42,
        )
        assert result["boundary_bar"] == 100
        assert result["n_bars_low"] == 100
        assert result["n_bars_high"] == 100
        assert result["n_bars_total"] == 200


# ===========================================================================
# 16. Trades structured array fields
# ===========================================================================


class TestTradesStructuredArray:
    """generate_regime_switch_trades returns proper structured array."""

    def test_trades_has_required_fields(self):
        """Trades array should have price, size, side, ts_event fields."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        trades, _ = generate_regime_switch_trades(
            n_bars_low=10, n_bars_high=10, bar_size=100, seed=42,
        )
        assert isinstance(trades, np.ndarray)
        assert "price" in trades.dtype.names
        assert "size" in trades.dtype.names
        assert "side" in trades.dtype.names
        assert "ts_event" in trades.dtype.names

    def test_bars_are_trade_bars(self):
        """Returned bars list should contain TradeBar instances."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        _, bars = generate_regime_switch_trades(
            n_bars_low=10, n_bars_high=10, bar_size=100, seed=42,
        )
        assert isinstance(bars, list)
        assert len(bars) > 0
        assert isinstance(bars[0], TradeBar)

    def test_start_price_applied(self):
        """First trade price should be close to start_price."""
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades

        trades, _ = generate_regime_switch_trades(
            n_bars_low=10, n_bars_high=10, bar_size=100,
            start_price=5000.0, seed=42,
        )
        # First trade should be at or very near start_price
        assert abs(trades["price"][0] - 5000.0) < 1.0
