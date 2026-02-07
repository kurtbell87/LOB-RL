"""Tests for aggregate_bars() — pure function that aggregates tick-level data into bars.

Spec: docs/bar-level-env.md (Requirement 1)

These tests verify that:
- aggregate_bars() exists and is callable
- Returns (bar_features, bar_mid_close, bar_spread_close) with correct shapes/dtypes
- Bar features have correct dimensions (13 intra-bar features per bar)
- Bar count = N // bar_size (dropping last chunk if < bar_size // 4 ticks)
- Each of the 13 intra-bar features is computed correctly:
    0: bar_return, 1: bar_range, 2: bar_volatility, 3: spread_mean,
    4: spread_close, 5: imbalance_mean, 6: imbalance_close,
    7: bid_volume_mean, 8: ask_volume_mean, 9: volume_imbalance,
    10: microprice_offset, 11: time_remaining, 12: n_ticks_norm
- Edge cases: division by zero, single-tick bars, bars with <2 ticks (vol=0)
"""

import numpy as np
import pytest

from conftest import make_realistic_obs


# ===========================================================================
# Helper: create deterministic tick data with known properties
# ===========================================================================

def _make_tick_data(n, mid_start=100.0, mid_step=0.25, spread=0.50):
    """Create (obs, mid, spread) arrays with n ticks."""
    obs, mid, spread_arr = make_realistic_obs(n, mid_start=mid_start,
                                               mid_step=mid_step, spread=spread)
    return obs, mid, spread_arr


def _hand_compute_bar(obs_chunk, mid_chunk, spread_chunk, bar_size):
    """Hand-compute expected bar features for a chunk of ticks.

    Returns a dict of expected feature values.
    """
    mid_open = mid_chunk[0]
    mid_close = mid_chunk[-1]
    mid_high = np.max(mid_chunk)
    mid_low = np.min(mid_chunk)
    n_ticks = len(mid_chunk)

    result = {}
    result['bar_return'] = (mid_close - mid_open) / mid_open
    result['bar_range'] = (mid_high - mid_low) / mid_open
    if n_ticks >= 2:
        result['bar_volatility'] = float(np.std(mid_chunk) / mid_open)
    else:
        result['bar_volatility'] = 0.0
    result['spread_mean'] = float(np.mean(spread_chunk))
    result['spread_close'] = float(spread_chunk[-1])
    result['imbalance_mean'] = float(np.mean(obs_chunk[:, 41]))
    result['imbalance_close'] = float(obs_chunk[-1, 41])

    bid_sizes = obs_chunk[:, 10:20]
    ask_sizes = obs_chunk[:, 30:40]
    result['bid_volume_mean'] = float(np.mean(np.sum(bid_sizes, axis=1)))
    result['ask_volume_mean'] = float(np.mean(np.sum(ask_sizes, axis=1)))

    sum_bid = np.sum(bid_sizes, axis=1)
    sum_ask = np.sum(ask_sizes, axis=1)
    denom = sum_bid + sum_ask
    safe_denom = np.where(denom > 0, denom, 1.0)
    vi_per_tick = np.where(denom > 0, (sum_bid - sum_ask) / safe_denom, 0.0)
    result['volume_imbalance'] = float(np.mean(vi_per_tick))

    bid0 = obs_chunk[-1, 0]
    bidsize0 = obs_chunk[-1, 10]
    ask0 = obs_chunk[-1, 20]
    asksize0 = obs_chunk[-1, 30]
    denom_mp = bidsize0 + asksize0
    if denom_mp > 0:
        microprice = (ask0 * bidsize0 + bid0 * asksize0) / denom_mp
        result['microprice_offset'] = float(microprice / mid_close - 1.0)
    else:
        result['microprice_offset'] = 0.0

    result['time_remaining'] = float(obs_chunk[-1, 42])
    result['n_ticks_norm'] = float(n_ticks / bar_size)

    return result


# ===========================================================================
# Test 1: aggregate_bars() exists and is callable
# ===========================================================================


class TestAggregateBarsExists:
    """aggregate_bars should be importable and callable."""

    def test_import(self):
        """aggregate_bars should be importable from lob_rl.bar_aggregation."""
        from lob_rl.bar_aggregation import aggregate_bars
        assert callable(aggregate_bars)

    def test_signature_accepts_required_args(self):
        """aggregate_bars(obs, mid, spread, bar_size) should not raise TypeError."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        # Should not raise
        aggregate_bars(obs, mid, spread, bar_size=10)


# ===========================================================================
# Test 2: Return types and shapes
# ===========================================================================


class TestAggregateBarsReturnShape:
    """aggregate_bars() should return (bar_features, bar_mid_close, bar_spread_close)."""

    def test_returns_3_tuple(self):
        """Should return a 3-tuple."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        result = aggregate_bars(obs, mid, spread, bar_size=10)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_bar_features_shape(self):
        """bar_features should be (B, 13) where B = N // bar_size."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features.shape == (10, 13)

    def test_bar_mid_close_shape(self):
        """bar_mid_close should be (B,)."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_mid_close.shape == (10,)

    def test_bar_spread_close_shape(self):
        """bar_spread_close should be (B,)."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        _, _, bar_spread_close = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_spread_close.shape == (10,)

    def test_bar_features_dtype_float32(self):
        """bar_features should be float32."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features.dtype == np.float32

    def test_bar_mid_close_dtype_float64(self):
        """bar_mid_close should be float64."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_mid_close.dtype == np.float64

    def test_bar_spread_close_dtype_float64(self):
        """bar_spread_close should be float64."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        _, _, bar_spread_close = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_spread_close.dtype == np.float64


# ===========================================================================
# Test 3: Bar count and partial bar dropping
# ===========================================================================


class TestBarCount:
    """Bar count should be N // bar_size, with partial bar logic."""

    def test_exact_division(self):
        """100 ticks / 10 bar_size = 10 bars (no remainder)."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features.shape[0] == 10

    def test_remainder_below_threshold_dropped(self):
        """103 ticks / 10 bar_size: remainder=3, threshold=10//4=2.
        3 >= 2, so last partial bar IS kept. => 11 bars."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(103)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        # remainder=3 >= bar_size//4=2 -> keep partial bar -> 11 bars
        assert bar_features.shape[0] == 11

    def test_remainder_at_threshold_kept(self):
        """102 ticks / 10 bar_size: remainder=2, threshold=10//4=2.
        2 >= 2 -> keep partial bar -> 11 bars."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(102)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features.shape[0] == 11

    def test_tiny_remainder_dropped(self):
        """101 ticks / 10 bar_size: remainder=1, threshold=10//4=2.
        1 < 2 -> drop partial bar -> 10 bars."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(101)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features.shape[0] == 10

    def test_bar_size_larger_than_ticks(self):
        """bar_size=200, n=50: 0 full bars, remainder=50, threshold=200//4=50.
        50 >= 50 -> keep -> 1 bar."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(50)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=200)
        assert bar_features.shape[0] == 1

    def test_bar_size_much_larger_than_ticks_dropped(self):
        """bar_size=200, n=49: remainder=49, threshold=200//4=50.
        49 < 50 -> drop -> 0 bars."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(49)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=200)
        assert bar_features.shape[0] == 0

    def test_bar_size_500_realistic(self):
        """1000 ticks / 500 bar_size = 2 bars."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(1000)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=500)
        assert bar_features.shape[0] == 2


# ===========================================================================
# Test 4: bar_return (feature index 0)
# ===========================================================================


class TestBarReturn:
    """bar_return = (mid_close - mid_open) / mid_open."""

    def test_bar_return_basic(self):
        """Hand-compute bar_return for first bar and verify."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.25)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        # Bar 0: ticks 0..9, mid_open=100.0, mid_close=100+9*0.25=102.25
        expected = (102.25 - 100.0) / 100.0
        assert bar_features[0, 0] == pytest.approx(expected, rel=1e-5)

    def test_bar_return_negative(self):
        """Decreasing mid prices should give negative bar_return."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=200.0, mid_step=-0.5)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        mid_open = 200.0
        mid_close = 200.0 + 9 * (-0.5)  # 195.5
        expected = (mid_close - mid_open) / mid_open
        assert expected < 0
        assert bar_features[0, 0] == pytest.approx(expected, rel=1e-5)

    def test_bar_return_flat(self):
        """Constant mid should give zero bar_return."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.0)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 0] == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# Test 5: bar_range (feature index 1)
# ===========================================================================


class TestBarRange:
    """bar_range = (mid_high - mid_low) / mid_open."""

    def test_bar_range_increasing(self):
        """Monotonically increasing mid: range = (mid[9] - mid[0]) / mid[0]."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.25)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        mid_open = 100.0
        mid_high = 100.0 + 9 * 0.25  # 102.25
        mid_low = 100.0
        expected = (mid_high - mid_low) / mid_open
        assert bar_features[0, 1] == pytest.approx(expected, rel=1e-5)

    def test_bar_range_flat(self):
        """Constant mid should give zero range."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.0)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 1] == pytest.approx(0.0, abs=1e-7)

    def test_bar_range_always_nonneg(self):
        """Range should always be >= 0."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100, mid_start=100.0, mid_step=-0.1)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert np.all(bar_features[:, 1] >= 0)


# ===========================================================================
# Test 6: bar_volatility (feature index 2)
# ===========================================================================


class TestBarVolatility:
    """bar_volatility = std(mid_ticks) / mid_open (0 if <2 ticks)."""

    def test_volatility_positive_for_varying_mid(self):
        """Non-constant mid should give positive volatility."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.5)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 2] > 0

    def test_volatility_zero_for_constant_mid(self):
        """Constant mid should give zero volatility."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.0)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 2] == pytest.approx(0.0, abs=1e-7)

    def test_volatility_hand_computed(self):
        """Hand-compute std(mid) / mid_open for first bar."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.25)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        mid_chunk = mid[:10]
        expected = float(np.std(mid_chunk) / mid_chunk[0])
        assert bar_features[0, 2] == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Test 7: spread_mean and spread_close (feature indices 3, 4)
# ===========================================================================


class TestSpreadFeatures:
    """spread_mean = mean(spread_ticks), spread_close = last tick spread."""

    def test_spread_mean_constant(self):
        """Constant spread should give spread_mean = spread value."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, spread=0.50)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 3] == pytest.approx(0.50, rel=1e-5)

    def test_spread_close_is_last_tick(self):
        """spread_close should be the spread at the last tick of the bar."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, spread=0.50)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 4] == pytest.approx(0.50, rel=1e-5)

    def test_spread_close_matches_bar_spread_close_output(self):
        """bar_features[b, 4] should match bar_spread_close[b]."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100, spread=0.50)
        bar_features, _, bar_spread_close = aggregate_bars(obs, mid, spread, bar_size=10)
        np.testing.assert_array_almost_equal(
            bar_features[:, 4], bar_spread_close.astype(np.float32),
            decimal=5
        )


# ===========================================================================
# Test 8: imbalance features (indices 5, 6)
# ===========================================================================


class TestImbalanceFeatures:
    """imbalance_mean = mean(obs[:, 41]), imbalance_close = obs[-1, 41]."""

    def test_imbalance_mean_computed(self):
        """imbalance_mean should be the mean of obs[:, 41] over the bar."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        expected = float(np.mean(obs[:10, 41]))
        assert bar_features[0, 5] == pytest.approx(expected, rel=1e-5)

    def test_imbalance_close_is_last_tick(self):
        """imbalance_close should be obs[-1, 41] of the bar's ticks."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        expected = float(obs[9, 41])  # last tick of bar 0
        assert bar_features[0, 6] == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Test 9: volume features (indices 7, 8, 9)
# ===========================================================================


class TestVolumeFeatures:
    """bid_volume_mean, ask_volume_mean, volume_imbalance."""

    def test_bid_volume_mean(self):
        """bid_volume_mean = mean(sum(obs[:, 10:20], axis=1))."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        bid_sums = np.sum(obs[:10, 10:20], axis=1)
        expected = float(np.mean(bid_sums))
        assert bar_features[0, 7] == pytest.approx(expected, rel=1e-4)

    def test_ask_volume_mean(self):
        """ask_volume_mean = mean(sum(obs[:, 30:40], axis=1))."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        ask_sums = np.sum(obs[:10, 30:40], axis=1)
        expected = float(np.mean(ask_sums))
        assert bar_features[0, 8] == pytest.approx(expected, rel=1e-4)

    def test_volume_imbalance_hand_computed(self):
        """volume_imbalance = mean((bid - ask) / (bid + ask)) per tick."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        bid_sums = np.sum(obs[:10, 10:20], axis=1)
        ask_sums = np.sum(obs[:10, 30:40], axis=1)
        denom = bid_sums + ask_sums
        safe_denom = np.where(denom > 0, denom, 1.0)
        vi = np.where(denom > 0, (bid_sums - ask_sums) / safe_denom, 0.0)
        expected = float(np.mean(vi))
        assert bar_features[0, 9] == pytest.approx(expected, rel=1e-4)

    def test_volume_imbalance_zero_volume(self):
        """When bid+ask=0 for all ticks, volume_imbalance should be 0."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        # Zero out all volume
        obs[:, 10:20] = 0.0
        obs[:, 30:40] = 0.0
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 9] == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# Test 10: microprice_offset (feature index 10)
# ===========================================================================


class TestMicropriceOffset:
    """microprice_offset at bar close."""

    def test_microprice_offset_hand_computed(self):
        """Hand-compute microprice offset for bar close tick."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        close_tick = obs[9]  # last tick of bar 0
        bid0 = close_tick[0]
        bidsize0 = close_tick[10]
        ask0 = close_tick[20]
        asksize0 = close_tick[30]
        denom = bidsize0 + asksize0
        microprice = (ask0 * bidsize0 + bid0 * asksize0) / denom
        expected = float(microprice / bar_mid_close[0] - 1.0)
        assert bar_features[0, 10] == pytest.approx(expected, rel=1e-4)

    def test_microprice_offset_zero_volume(self):
        """When bidsize0+asksize0=0, microprice_offset should be 0."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        obs[:, 10] = 0.0  # zero bidsize0
        obs[:, 30] = 0.0  # zero asksize0
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 10] == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# Test 11: time_remaining (feature index 11)
# ===========================================================================


class TestTimeRemaining:
    """time_remaining = obs[-1, 42] at last tick."""

    def test_time_remaining_is_last_tick(self):
        """time_remaining should be obs[last_tick, 42] of the bar."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        expected = float(obs[9, 42])
        assert bar_features[0, 11] == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Test 12: n_ticks_norm (feature index 12)
# ===========================================================================


class TestNTicksNorm:
    """n_ticks_norm = actual_ticks / bar_size."""

    def test_full_bar_is_1(self):
        """Full bar should have n_ticks_norm = 1.0."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_features[0, 12] == pytest.approx(1.0, rel=1e-5)

    def test_partial_bar_fraction(self):
        """Partial bar with 5 ticks / bar_size=10 -> n_ticks_norm = 0.5."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(15)  # 1 full bar + 5 remaining >= 10//4=2
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        # Bar 0: full (10 ticks), bar 1: partial (5 ticks, 5 >= 2 so kept)
        assert bar_features.shape[0] == 2
        assert bar_features[0, 12] == pytest.approx(1.0)
        assert bar_features[1, 12] == pytest.approx(0.5)


# ===========================================================================
# Test 13: bar_mid_close values
# ===========================================================================


class TestBarMidClose:
    """bar_mid_close should be the mid price at the last tick of each bar."""

    def test_bar_mid_close_values(self):
        """bar_mid_close[b] should equal mid[last_tick_of_bar_b]."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.25)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        assert bar_mid_close[0] == pytest.approx(mid[9])   # bar 0 close
        assert bar_mid_close[1] == pytest.approx(mid[19])  # bar 1 close

    def test_bar_mid_close_distinct_bars(self):
        """Different bars should generally have different mid_close values."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert bar_mid_close[0] != bar_mid_close[1]


# ===========================================================================
# Test 14: Full hand verification of all 13 features for one bar
# ===========================================================================


class TestFullBarVerification:
    """Verify all 13 features at once against hand computation."""

    def test_all_13_features_bar_0(self):
        """All 13 features for bar 0 should match hand computation."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.25,
                                            spread=0.50)
        bar_features, bar_mid_close, bar_spread_close = aggregate_bars(
            obs, mid, spread, bar_size=10
        )

        expected = _hand_compute_bar(obs[:10], mid[:10], spread[:10], 10)

        assert bar_features[0, 0] == pytest.approx(expected['bar_return'], rel=1e-5)
        assert bar_features[0, 1] == pytest.approx(expected['bar_range'], rel=1e-5)
        assert bar_features[0, 2] == pytest.approx(expected['bar_volatility'], rel=1e-5)
        assert bar_features[0, 3] == pytest.approx(expected['spread_mean'], rel=1e-5)
        assert bar_features[0, 4] == pytest.approx(expected['spread_close'], rel=1e-5)
        assert bar_features[0, 5] == pytest.approx(expected['imbalance_mean'], rel=1e-5)
        assert bar_features[0, 6] == pytest.approx(expected['imbalance_close'], rel=1e-5)
        assert bar_features[0, 7] == pytest.approx(expected['bid_volume_mean'], rel=1e-4)
        assert bar_features[0, 8] == pytest.approx(expected['ask_volume_mean'], rel=1e-4)
        assert bar_features[0, 9] == pytest.approx(expected['volume_imbalance'], rel=1e-4)
        assert bar_features[0, 10] == pytest.approx(expected['microprice_offset'], rel=1e-4)
        assert bar_features[0, 11] == pytest.approx(expected['time_remaining'], rel=1e-5)
        assert bar_features[0, 12] == pytest.approx(expected['n_ticks_norm'], rel=1e-5)


# ===========================================================================
# Test 15: Multiple bars are independent
# ===========================================================================


class TestMultipleBarsIndependent:
    """Each bar should be computed from its own slice of ticks."""

    def test_bar_1_uses_ticks_10_to_19(self):
        """Bar 1 should use ticks [10, 20) not [0, 10)."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        expected = _hand_compute_bar(obs[10:20], mid[10:20], spread[10:20], 10)
        assert bar_features[1, 0] == pytest.approx(expected['bar_return'], rel=1e-5)
        assert bar_features[1, 2] == pytest.approx(expected['bar_volatility'], rel=1e-5)

    def test_bars_have_different_returns(self):
        """Different bars should generally have different returns."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        # With linear ramp, bar returns should differ (different base mid_open)
        assert bar_features[0, 0] != pytest.approx(bar_features[1, 0], abs=1e-7)


# ===========================================================================
# Test 16: Edge case — all NaN/inf guards
# ===========================================================================


class TestNoNanInf:
    """aggregate_bars should not produce NaN or Inf in features."""

    def test_no_nan_in_features(self):
        """bar_features should contain no NaN values."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert not np.any(np.isnan(bar_features))

    def test_no_inf_in_features(self):
        """bar_features should contain no Inf values."""
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)
        assert not np.any(np.isinf(bar_features))
