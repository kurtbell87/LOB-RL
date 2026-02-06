"""Tests for temporal features — 10 new obs features computed in Python.

Spec: docs/temporal-features.md

These tests verify that:
- PrecomputedEnv.observation_space.shape == (54,)
- MultiDayEnv.observation_space.shape == (54,)
- 10 temporal features (mid returns, volatility, imbalance deltas,
  microprice offset, total volume imbalance, spread change) are correct
- Position is at index 53 (last element)
- For t=0, all 10 temporal features are 0.0
- For t >= 50, temporal features are non-zero (with non-constant data)
- Edge cases: zero mid, zero volumes, short episodes
- All observations are finite (no NaN/Inf)
- gymnasium check_env passes on 54-dim env
- make_env in train.py returns PrecomputedEnv (not LOBGymEnv)
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.precomputed_env import PrecomputedEnv

from conftest import (
    PRECOMPUTE_EPISODE_FILE,
    DAY_FILES,
    make_obs as _make_obs,
    make_mid as _make_mid,
    make_spread as _make_spread,
)


# ===========================================================================
# Helpers: build realistic obs arrays with distinguishable book data
# ===========================================================================

# Observation layout from C++ (43 features):
#   [0:10]   bid prices (10 levels)
#   [10:20]  bid sizes (10 levels)
#   [20:30]  ask prices (10 levels)
#   [30:40]  ask sizes (10 levels)
#   [40]     spread/mid (relative spread)
#   [41]     imbalance
#   [42]     time_left

DEPTH = 10
OBS_COLS = 43
NEW_OBS_SIZE = 54
POSITION_INDEX = 53  # spec says position moves to index 53


def make_realistic_obs(n, mid_start=100.0, mid_step=0.25, spread=0.50):
    """Build (n, 43) obs array with realistic LOB fields for temporal feature testing.

    Returns (obs, mid, spread_arr) — obs has proper bid/ask/size/imbalance structure.
    """
    obs = np.zeros((n, OBS_COLS), dtype=np.float32)
    mid = np.empty(n, dtype=np.float64)
    spread_arr = np.full(n, spread, dtype=np.float64)

    for t in range(n):
        m = mid_start + t * mid_step
        mid[t] = m
        half_spread = spread / 2.0

        # Bid prices: 10 levels descending from bid0
        bid0 = m - half_spread
        for lvl in range(DEPTH):
            obs[t, lvl] = bid0 - lvl * 0.25  # bid prices

        # Ask prices: 10 levels ascending from ask0
        ask0 = m + half_spread
        for lvl in range(DEPTH):
            obs[t, 20 + lvl] = ask0 + lvl * 0.25  # ask prices

        # Bid sizes: vary by level so total_volume_imbalance is non-trivial
        for lvl in range(DEPTH):
            obs[t, 10 + lvl] = 10.0 + t * 0.5 + lvl  # bid sizes

        # Ask sizes: slightly different so imbalance != 0
        for lvl in range(DEPTH):
            obs[t, 30 + lvl] = 8.0 + t * 0.3 + lvl  # ask sizes

        # spread/mid (relative spread)
        obs[t, 40] = spread / m

        # imbalance = (bidsize0 - asksize0) / (bidsize0 + asksize0)
        bs0 = obs[t, 10]
        as0 = obs[t, 30]
        obs[t, 41] = (bs0 - as0) / (bs0 + as0) if (bs0 + as0) > 0 else 0.0

        # time_left
        obs[t, 42] = 1.0 - t / max(n - 1, 1)

    return obs, mid, spread_arr


def make_constant_mid_obs(n, mid_val=100.0, spread=0.50):
    """Build obs with constant mid price — temporal returns should all be zero."""
    obs, mid, spread_arr = make_realistic_obs(n, mid_start=mid_val, mid_step=0.0, spread=spread)
    return obs, mid, spread_arr


# ===========================================================================
# Test 1: observation_space shape is (54,)
# ===========================================================================


class TestObservationSpaceShape:
    """PrecomputedEnv observation_space should be (54,) after temporal features."""

    def test_precomputed_env_obs_space_shape_54(self):
        """PrecomputedEnv.observation_space.shape should be (54,)."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        assert env.observation_space.shape == (NEW_OBS_SIZE,), (
            f"Expected obs space shape ({NEW_OBS_SIZE},), got {env.observation_space.shape}"
        )

    def test_precomputed_env_obs_space_dtype(self):
        """observation_space dtype should be float32."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        assert env.observation_space.dtype == np.float32

    def test_precomputed_env_obs_space_bounds(self):
        """observation_space should have -inf/+inf bounds."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        assert np.all(env.observation_space.low == -np.inf)
        assert np.all(env.observation_space.high == np.inf)


# ===========================================================================
# Test 2: MultiDayEnv observation_space shape is (54,)
# ===========================================================================


class TestMultiDayEnvObsSpaceShape:
    """MultiDayEnv observation_space should be (54,)."""

    def test_multi_day_env_obs_space_shape_54(self):
        """MultiDayEnv.observation_space.shape should be (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        assert env.observation_space.shape == (NEW_OBS_SIZE,), (
            f"Expected obs space shape ({NEW_OBS_SIZE},), got {env.observation_space.shape}"
        )

    def test_multi_day_env_reset_obs_shape_54(self):
        """MultiDayEnv.reset() should return obs with shape (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (NEW_OBS_SIZE,)

    def test_multi_day_env_step_obs_shape_54(self):
        """MultiDayEnv.step() should return obs with shape (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (NEW_OBS_SIZE,)


# ===========================================================================
# Test 3: reset/step obs shape is (54,) for PrecomputedEnv
# ===========================================================================


class TestObsOutputShape:
    """reset() and step() should return 54-float observations."""

    def test_reset_obs_shape_54(self):
        """reset() observation should have shape (54,)."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, info = env.reset()
        assert result_obs.shape == (NEW_OBS_SIZE,)

    def test_step_obs_shape_54(self):
        """step() observation should have shape (54,)."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)
        assert result_obs.shape == (NEW_OBS_SIZE,)

    def test_reset_obs_dtype_float32(self):
        """reset() observation dtype should be float32."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert result_obs.dtype == np.float32

    def test_step_obs_dtype_float32(self):
        """step() observation dtype should be float32."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)
        assert result_obs.dtype == np.float32

    def test_obs_in_observation_space_after_reset(self):
        """reset() obs should be in observation_space."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert env.observation_space.contains(result_obs)

    def test_obs_in_observation_space_after_step(self):
        """step() obs should be in observation_space."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(2)
        assert env.observation_space.contains(result_obs)


# ===========================================================================
# Test 4: position is at index 53 (moved from 43)
# ===========================================================================


class TestPositionIndex:
    """Position should be at index 53 (last element of 54-dim obs)."""

    def test_reset_position_at_index_53(self):
        """After reset(), position at index 53 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert result_obs[POSITION_INDEX] == pytest.approx(0.0)

    def test_action_0_position_neg1_at_index_53(self):
        """action=0 should set position=-1 at index 53."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(0)
        assert result_obs[POSITION_INDEX] == pytest.approx(-1.0)

    def test_action_1_position_0_at_index_53(self):
        """action=1 should set position=0 at index 53."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)
        assert result_obs[POSITION_INDEX] == pytest.approx(0.0)

    def test_action_2_position_pos1_at_index_53(self):
        """action=2 should set position=+1 at index 53."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(2)
        assert result_obs[POSITION_INDEX] == pytest.approx(1.0)


# ===========================================================================
# Test 5: t=0 — all 10 temporal features are 0.0
# ===========================================================================


class TestTemporalFeaturesAtT0:
    """At t=0 (reset), all 10 temporal features should be 0.0."""

    def test_all_temporal_features_zero_at_reset(self):
        """After reset (t=0), obs[43:53] should all be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        temporal = result_obs[43:53]
        np.testing.assert_array_equal(
            temporal, np.zeros(10, dtype=np.float32),
            err_msg=f"Temporal features at t=0 should all be 0.0, got {temporal}"
        )

    def test_each_temporal_feature_individually_zero_at_t0(self):
        """Verify each of the 10 temporal features is exactly 0.0 at t=0."""
        feature_names = [
            "mid_return_1", "mid_return_5", "mid_return_20", "mid_return_50",
            "volatility_20", "imbalance_delta_5", "imbalance_delta_20",
            "microprice_offset", "total_volume_imbalance", "spread_change_5",
        ]
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        for i, name in enumerate(feature_names):
            assert result_obs[43 + i] == pytest.approx(0.0), (
                f"{name} (index {43 + i}) should be 0.0 at t=0, got {result_obs[43 + i]}"
            )


# ===========================================================================
# Test 6: mid_return_1 correctness
# ===========================================================================


class TestMidReturn1:
    """mid_return_1 = (mid[t] - mid[t-1]) / mid[t-1] at obs index 43."""

    def test_mid_return_1_at_step_1(self):
        """After 1 step (t=1), mid_return_1 should equal (mid[1]-mid[0])/mid[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)  # now at t=1
        expected = (mid[1] - mid[0]) / mid[0]
        assert result_obs[43] == pytest.approx(float(expected), rel=1e-5), (
            f"mid_return_1 at t=1: expected {expected}, got {result_obs[43]}"
        )

    def test_mid_return_1_at_step_10(self):
        """After 10 steps (t=10), mid_return_1 = (mid[10]-mid[9])/mid[9]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(10):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[10] - mid[9]) / mid[9]
        assert result_obs[43] == pytest.approx(float(expected), rel=1e-5)

    def test_mid_return_1_zero_for_constant_mid(self):
        """With constant mid price, mid_return_1 should be 0.0."""
        obs, mid, spread = make_constant_mid_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)
        assert result_obs[43] == pytest.approx(0.0)


# ===========================================================================
# Test 7: mid_return_5 correctness
# ===========================================================================


class TestMidReturn5:
    """mid_return_5 = (mid[t] - mid[t-5]) / mid[t-5] at obs index 44."""

    def test_mid_return_5_zero_for_t_less_than_5(self):
        """For t < 5, mid_return_5 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 5):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[44] == pytest.approx(0.0), (
                f"mid_return_5 should be 0 at t={t}, got {result_obs[44]}"
            )

    def test_mid_return_5_at_t5(self):
        """At t=5, mid_return_5 = (mid[5]-mid[0])/mid[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(5):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[5] - mid[0]) / mid[0]
        assert result_obs[44] == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 8: mid_return_20 correctness
# ===========================================================================


class TestMidReturn20:
    """mid_return_20 = (mid[t] - mid[t-20]) / mid[t-20] at obs index 45."""

    def test_mid_return_20_zero_for_t_less_than_20(self):
        """For t < 20, mid_return_20 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 20):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[45] == pytest.approx(0.0), (
                f"mid_return_20 should be 0 at t={t}, got {result_obs[45]}"
            )

    def test_mid_return_20_at_t20(self):
        """At t=20, mid_return_20 = (mid[20]-mid[0])/mid[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(20):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[20] - mid[0]) / mid[0]
        assert result_obs[45] == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 9: mid_return_50 correctness
# ===========================================================================


class TestMidReturn50:
    """mid_return_50 = (mid[t] - mid[t-50]) / mid[t-50] at obs index 46."""

    def test_mid_return_50_zero_for_t_less_than_50(self):
        """For t < 50, mid_return_50 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 50):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[46] == pytest.approx(0.0), (
                f"mid_return_50 should be 0 at t={t}, got {result_obs[46]}"
            )

    def test_mid_return_50_at_t50(self):
        """At t=50, mid_return_50 = (mid[50]-mid[0])/mid[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(50):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[50] - mid[0]) / mid[0]
        assert result_obs[46] == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 10: volatility_20 correctness
# ===========================================================================


class TestVolatility20:
    """volatility_20 = rolling std of 1-step returns over last 20 steps, at index 47."""

    def test_volatility_20_zero_for_t_less_than_20(self):
        """For t < 20, volatility_20 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 20):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[47] == pytest.approx(0.0), (
                f"volatility_20 should be 0 at t={t}, got {result_obs[47]}"
            )

    def test_volatility_20_at_t20(self):
        """At t=20, volatility_20 = std of returns[0:20]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(20):
            result_obs, _, _, _, _ = env.step(1)

        # Compute expected: 1-step returns from t=1..19, then std of ret[0:20]
        ret1 = np.zeros(100, dtype=np.float64)
        ret1[1:] = (mid[1:] - mid[:-1]) / np.where(mid[:-1] != 0, mid[:-1], 1.0)
        expected = np.std(ret1[0:20])
        assert result_obs[47] == pytest.approx(float(expected), rel=1e-4), (
            f"volatility_20 at t=20: expected {expected}, got {result_obs[47]}"
        )

    def test_volatility_20_zero_for_constant_mid(self):
        """With constant mid, volatility_20 should be 0 (all returns are 0)."""
        obs, mid, spread = make_constant_mid_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(25):
            result_obs, _, _, _, _ = env.step(1)
        assert result_obs[47] == pytest.approx(0.0), (
            f"volatility_20 with constant mid should be 0, got {result_obs[47]}"
        )

    def test_volatility_20_positive_for_varying_mid(self):
        """With varying mid, volatility_20 at t >= 20 should be > 0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(25):
            result_obs, _, _, _, _ = env.step(1)
        assert result_obs[47] > 0.0, (
            f"volatility_20 should be positive for varying mid, got {result_obs[47]}"
        )


# ===========================================================================
# Test 11: imbalance_delta_5 correctness
# ===========================================================================


class TestImbalanceDelta5:
    """imbalance_delta_5 = imbalance[t] - imbalance[t-5] at index 48."""

    def test_imbalance_delta_5_zero_for_t_less_than_5(self):
        """For t < 5, imbalance_delta_5 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 5):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[48] == pytest.approx(0.0), (
                f"imbalance_delta_5 should be 0 at t={t}"
            )

    def test_imbalance_delta_5_at_t5(self):
        """At t=5, imbalance_delta_5 = imbalance[5] - imbalance[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(5):
            result_obs, _, _, _, _ = env.step(1)
        expected = obs[5, 41] - obs[0, 41]  # imbalance at index 41
        assert result_obs[48] == pytest.approx(float(expected), rel=1e-5), (
            f"imbalance_delta_5 at t=5: expected {expected}, got {result_obs[48]}"
        )


# ===========================================================================
# Test 12: imbalance_delta_20 correctness
# ===========================================================================


class TestImbalanceDelta20:
    """imbalance_delta_20 = imbalance[t] - imbalance[t-20] at index 49."""

    def test_imbalance_delta_20_zero_for_t_less_than_20(self):
        """For t < 20, imbalance_delta_20 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 20):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[49] == pytest.approx(0.0), (
                f"imbalance_delta_20 should be 0 at t={t}"
            )

    def test_imbalance_delta_20_at_t20(self):
        """At t=20, imbalance_delta_20 = imbalance[20] - imbalance[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(20):
            result_obs, _, _, _, _ = env.step(1)
        expected = obs[20, 41] - obs[0, 41]
        assert result_obs[49] == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 13: microprice_offset correctness
# ===========================================================================


class TestMicropriceOffset:
    """microprice_offset = microprice/mid - 1 at index 50.

    microprice = (ask0 * bidsize0 + bid0 * asksize0) / (bidsize0 + asksize0)
    """

    def test_microprice_offset_formula(self):
        """microprice_offset should match the formula from the spec."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        # Step to t=1 so we can check
        result_obs, _, _, _, _ = env.step(1)

        t = 1
        bid0 = obs[t, 0]
        bidsize0 = obs[t, 10]
        ask0 = obs[t, 20]
        asksize0 = obs[t, 30]
        denom = bidsize0 + asksize0
        if denom > 0:
            microprice = (ask0 * bidsize0 + bid0 * asksize0) / denom
        else:
            microprice = (bid0 + ask0) / 2.0
        expected = microprice / mid[t] - 1.0

        assert result_obs[50] == pytest.approx(float(expected), rel=1e-4), (
            f"microprice_offset at t=1: expected {expected}, got {result_obs[50]}"
        )

    def test_microprice_offset_nonzero_for_asymmetric_book(self):
        """With asymmetric bid/ask sizes, microprice_offset should be non-zero."""
        # Our make_realistic_obs creates asymmetric sizes, so at any t > 0
        # microprice != mid, giving a non-zero offset
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)
        # With different bid/ask sizes, microprice offset should be non-zero
        assert result_obs[50] != pytest.approx(0.0, abs=1e-8), (
            "microprice_offset should be non-zero for asymmetric book"
        )

    def test_microprice_offset_zero_volume_fallback(self):
        """When bidsize0 + asksize0 == 0, microprice_offset should be 0.0."""
        obs, mid, spread = make_realistic_obs(10)
        # Zero out all bid and ask sizes at t=5
        obs[5, 10:20] = 0.0
        obs[5, 30:40] = 0.0
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(5):
            result_obs, _, _, _, _ = env.step(1)
        # When denom == 0, microprice = (bid0+ask0)/2 = mid, so offset = 0
        assert result_obs[50] == pytest.approx(0.0, abs=1e-5), (
            f"microprice_offset should be ~0 when volumes are 0, got {result_obs[50]}"
        )


# ===========================================================================
# Test 14: total_volume_imbalance correctness
# ===========================================================================


class TestTotalVolumeImbalance:
    """total_volume_imbalance = (sum(bid_sizes) - sum(ask_sizes)) / (sum(bid) + sum(ask))
    at index 51."""

    def test_total_volume_imbalance_formula(self):
        """total_volume_imbalance should match spec formula using all 10 levels."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)

        t = 1
        bid_sizes_sum = obs[t, 10:20].sum()
        ask_sizes_sum = obs[t, 30:40].sum()
        total = bid_sizes_sum + ask_sizes_sum
        expected = (bid_sizes_sum - ask_sizes_sum) / total if total > 0 else 0.0

        assert result_obs[51] == pytest.approx(float(expected), rel=1e-4), (
            f"total_volume_imbalance at t=1: expected {expected}, got {result_obs[51]}"
        )

    def test_total_volume_imbalance_uses_all_10_levels(self):
        """Verify all 10 bid/ask size levels contribute, not just level 0."""
        obs, mid, spread = make_realistic_obs(10)
        # Make only levels 5-9 have large bid sizes
        obs[3, 10:15] = 1.0   # levels 0-4: small bid
        obs[3, 15:20] = 100.0  # levels 5-9: large bid
        obs[3, 30:40] = 1.0   # all ask sizes small

        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(3):
            result_obs, _, _, _, _ = env.step(1)

        # bid_sum = 5*1 + 5*100 = 505, ask_sum = 10*1 = 10
        # imbalance = (505-10)/(505+10) = 495/515 ≈ 0.961
        assert result_obs[51] > 0.9, (
            f"total_volume_imbalance should reflect all 10 levels, got {result_obs[51]}"
        )

    def test_total_volume_imbalance_zero_volumes(self):
        """When all volumes are zero, total_volume_imbalance should be 0.0."""
        obs, mid, spread = make_realistic_obs(10)
        obs[3, 10:20] = 0.0  # zero bid sizes
        obs[3, 30:40] = 0.0  # zero ask sizes
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(3):
            result_obs, _, _, _, _ = env.step(1)
        assert result_obs[51] == pytest.approx(0.0)


# ===========================================================================
# Test 15: spread_change_5 correctness
# ===========================================================================


class TestSpreadChange5:
    """spread_change_5 = rel_spread[t] - rel_spread[t-5] at index 52."""

    def test_spread_change_5_zero_for_t_less_than_5(self):
        """For t < 5, spread_change_5 should be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 5):
            result_obs, _, _, _, _ = env.step(1)
            assert result_obs[52] == pytest.approx(0.0), (
                f"spread_change_5 should be 0 at t={t}"
            )

    def test_spread_change_5_at_t5(self):
        """At t=5, spread_change_5 = rel_spread[5] - rel_spread[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(5):
            result_obs, _, _, _, _ = env.step(1)
        expected = obs[5, 40] - obs[0, 40]  # rel_spread at index 40
        assert result_obs[52] == pytest.approx(float(expected), rel=1e-5), (
            f"spread_change_5 at t=5: expected {expected}, got {result_obs[52]}"
        )


# ===========================================================================
# Test 16: for t >= 50, temporal features are non-zero (non-constant data)
# ===========================================================================


class TestTemporalFeaturesNonZeroAfterWarmup:
    """After enough history (t >= 50), temporal features should be non-zero
    for non-constant price data. (Acceptance criterion #3)"""

    def test_all_temporal_features_nonzero_at_t50(self):
        """At t=50, all 10 temporal features should be non-zero."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(50):
            result_obs, _, _, _, _ = env.step(1)

        feature_names = [
            "mid_return_1", "mid_return_5", "mid_return_20", "mid_return_50",
            "volatility_20", "imbalance_delta_5", "imbalance_delta_20",
            "microprice_offset", "total_volume_imbalance", "spread_change_5",
        ]
        for i, name in enumerate(feature_names):
            val = result_obs[43 + i]
            assert val != pytest.approx(0.0, abs=1e-10), (
                f"{name} (index {43 + i}) should be non-zero at t=50, got {val}"
            )


# ===========================================================================
# Test 17: first 43 features are unchanged from C++ obs
# ===========================================================================


class TestFirst43FeaturesUnchanged:
    """The first 43 features should still come from the C++ obs array."""

    def test_first_43_match_input_obs_at_reset(self):
        """reset() obs[:43] should match input obs[0]."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        np.testing.assert_array_almost_equal(result_obs[:43], obs[0])

    def test_first_43_match_input_obs_at_step(self):
        """After step, obs[:43] should match input obs at the current timestep."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for t in range(1, 10):
            result_obs, _, _, _, _ = env.step(1)
            np.testing.assert_array_almost_equal(
                result_obs[:43], obs[t],
                err_msg=f"First 43 features mismatch at t={t}"
            )


# ===========================================================================
# Test 18: observations are finite (no NaN or Inf)
# ===========================================================================


class TestObservationsFinite:
    """All observations should be finite (no NaN, no Inf) throughout episode."""

    def test_no_nan_in_full_episode(self):
        """No NaN values in any observation during a full episode."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert not np.any(np.isnan(result_obs)), f"NaN in reset obs"
        terminated = False
        t = 0
        while not terminated:
            result_obs, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isnan(result_obs)), f"NaN at step {t}: {result_obs}"
            t += 1

    def test_no_inf_in_full_episode(self):
        """No Inf values in any observation during a full episode."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert not np.any(np.isinf(result_obs)), f"Inf in reset obs"
        terminated = False
        t = 0
        while not terminated:
            result_obs, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isinf(result_obs)), f"Inf at step {t}: {result_obs}"
            t += 1


# ===========================================================================
# Test 19: edge case — zero mid price handled gracefully
# ===========================================================================


class TestZeroMidPrice:
    """Division by zero for mid[t-lag]==0 should be guarded, returning 0.0."""

    def test_zero_mid_at_t0_no_crash(self):
        """If mid[0] == 0, returns using it as denominator should be 0, not NaN/Inf."""
        obs, mid, spread = make_realistic_obs(10)
        mid[0] = 0.0  # zero mid at t=0
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)  # t=1, mid_return_1 uses mid[0] as denom
        # Should be 0 or finite, not NaN/Inf
        assert np.isfinite(result_obs[43]), (
            f"mid_return_1 should be finite when mid[0]==0, got {result_obs[43]}"
        )


# ===========================================================================
# Test 20: short episodes (< 50 steps) work correctly
# ===========================================================================


class TestShortEpisodes:
    """Very short episodes (< 50 steps) should work — most temporal features
    will be 0 but the env shouldn't crash."""

    def test_3_step_episode_works(self):
        """3 snapshots (2 steps) should complete without error."""
        obs, mid, spread = make_realistic_obs(3)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert result_obs.shape == (NEW_OBS_SIZE,)

        result_obs, _, terminated, _, _ = env.step(2)
        assert not terminated
        assert result_obs.shape == (NEW_OBS_SIZE,)

        result_obs, _, terminated, _, _ = env.step(1)
        assert terminated
        assert result_obs.shape == (NEW_OBS_SIZE,)

    def test_2_step_episode_minimum(self):
        """2 snapshots (1 step, the minimum) should work."""
        obs, mid, spread = make_realistic_obs(2)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert result_obs.shape == (NEW_OBS_SIZE,)

        result_obs, _, terminated, _, _ = env.step(1)
        assert terminated
        assert result_obs.shape == (NEW_OBS_SIZE,)


# ===========================================================================
# Test 21: gymnasium check_env passes on 54-dim env
# ===========================================================================


class TestCheckEnv54:
    """gymnasium.utils.env_checker.check_env() should pass on the 54-dim env."""

    def test_check_env_synthetic_54dim(self):
        """check_env() should pass on a synthetic 54-dim PrecomputedEnv."""
        from gymnasium.utils.env_checker import check_env
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        check_env(env, skip_render_check=True)

    def test_check_env_from_file_54dim(self):
        """check_env() should pass on a file-backed 54-dim PrecomputedEnv."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        check_env(env, skip_render_check=True)


# ===========================================================================
# Test 22: reward logic unchanged (still uses position and mid correctly)
# ===========================================================================


class TestRewardUnchanged:
    """Reward calculation should be unaffected by the obs shape change."""

    def test_pnl_delta_reward_correct(self):
        """Reward = position * (mid[t+1] - mid[t]) should still work."""
        obs, mid, spread = make_realistic_obs(10)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        # action=2 -> position=+1
        _, reward, _, _, _ = env.step(2)
        expected = 1.0 * (mid[1] - mid[0])
        assert reward == pytest.approx(float(expected), rel=1e-5)

    def test_flattening_penalty_still_works(self):
        """Terminal flattening penalty should still apply."""
        obs, mid, spread = make_realistic_obs(3)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        env.step(2)  # go long, non-terminal
        _, reward, terminated, _, _ = env.step(2)  # stay long, terminal
        assert terminated
        # reward = pos * delta_mid - |pos| * spread[t] / 2
        expected_pnl = 1.0 * (mid[2] - mid[1])
        expected_flat = -1.0 * spread[2] / 2.0
        expected = expected_pnl + expected_flat
        assert reward == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 23: make_env in train.py returns PrecomputedEnv
# ===========================================================================


class TestTrainMakeEnv:
    """make_env() in scripts/train.py should return a PrecomputedEnv, not LOBGymEnv."""

    def test_make_env_returns_precomputed_env(self):
        """make_env() should return a PrecomputedEnv instance."""
        import sys
        import os
        # Add scripts dir to path so we can import train
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        from train import make_env

        env = make_env(DAY_FILES[0])
        assert isinstance(env, PrecomputedEnv), (
            f"make_env() should return PrecomputedEnv, got {type(env).__name__}"
        )

    def test_make_env_obs_space_54(self):
        """make_env() result should have 54-dim observation space."""
        import sys
        import os
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        from train import make_env

        env = make_env(DAY_FILES[0])
        assert env.observation_space.shape == (NEW_OBS_SIZE,), (
            f"make_env() obs space should be ({NEW_OBS_SIZE},), "
            f"got {env.observation_space.shape}"
        )


# ===========================================================================
# Test 24: DummyVecEnv compatibility with 54-dim obs
# ===========================================================================


class TestDummyVecEnv54:
    """PrecomputedEnv with 54-dim obs should work in SB3 DummyVecEnv."""

    def test_wrappable_in_dummy_vec_env(self):
        """DummyVecEnv should accept 54-dim PrecomputedEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        obs, mid, spread = make_realistic_obs(100)

        vec_env = DummyVecEnv([lambda: PrecomputedEnv(obs.copy(), mid.copy(), spread.copy())])
        reset_obs = vec_env.reset()
        assert reset_obs.shape == (1, NEW_OBS_SIZE)

    def test_step_through_dummy_vec_env(self):
        """step() through DummyVecEnv should return correct shape."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        obs, mid, spread = make_realistic_obs(100)

        vec_env = DummyVecEnv([lambda: PrecomputedEnv(obs.copy(), mid.copy(), spread.copy())])
        vec_env.reset()
        obs_out, rewards, dones, infos = vec_env.step([1])
        assert obs_out.shape == (1, NEW_OBS_SIZE)


# ===========================================================================
# Test 25: temporal feature observation layout matches spec exactly
# ===========================================================================


class TestObservationLayout:
    """Observation indices must match the spec layout exactly."""

    def test_index_43_is_mid_return_1(self):
        """obs[43] should be mid_return_1."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        # Step to t=10 so mid_return_1 is clearly non-zero
        for _ in range(10):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[10] - mid[9]) / mid[9]
        assert result_obs[43] == pytest.approx(float(expected), rel=1e-5)

    def test_index_44_is_mid_return_5(self):
        """obs[44] should be mid_return_5."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(10):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[10] - mid[5]) / mid[5]
        assert result_obs[44] == pytest.approx(float(expected), rel=1e-5)

    def test_index_45_is_mid_return_20(self):
        """obs[45] should be mid_return_20."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(25):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[25] - mid[5]) / mid[5]
        assert result_obs[45] == pytest.approx(float(expected), rel=1e-5)

    def test_index_46_is_mid_return_50(self):
        """obs[46] should be mid_return_50."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        for _ in range(55):
            result_obs, _, _, _, _ = env.step(1)
        expected = (mid[55] - mid[5]) / mid[5]
        assert result_obs[46] == pytest.approx(float(expected), rel=1e-5)

    def test_index_53_is_position(self):
        """obs[53] should be the position (last element)."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        env.reset()
        result_obs, _, _, _, _ = env.step(2)  # go long
        assert result_obs[53] == pytest.approx(1.0)

    def test_obs_length_exactly_54(self):
        """Observation should have exactly 54 elements, no more, no less."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread)
        result_obs, _ = env.reset()
        assert len(result_obs) == NEW_OBS_SIZE


# ===========================================================================
# Test 26: execution_cost and participation_bonus still work with 54-dim obs
# ===========================================================================


class TestRewardModesWith54Dim:
    """Existing reward features should still work with the new obs layout."""

    def test_execution_cost_with_temporal_features(self):
        """execution_cost should still deduct spread/2 * |delta_pos|."""
        obs, mid, spread_arr = make_realistic_obs(10)
        env = PrecomputedEnv(obs, mid, spread_arr, execution_cost=True)
        env.reset()
        # go long from flat: delta_pos = |1-0| = 1
        _, reward, _, _, _ = env.step(2)
        pnl = 1.0 * (mid[1] - mid[0])
        exec_cost = spread_arr[0] / 2.0 * 1.0  # |1-0|
        expected = pnl - exec_cost
        assert reward == pytest.approx(float(expected), rel=1e-5)

    def test_participation_bonus_with_temporal_features(self):
        """participation_bonus should still add bonus * |position|."""
        obs, mid, spread_arr = make_realistic_obs(10)
        env = PrecomputedEnv(obs, mid, spread_arr, participation_bonus=0.01)
        env.reset()
        _, reward, _, _, _ = env.step(2)  # go long
        pnl = 1.0 * (mid[1] - mid[0])
        bonus = 0.01 * 1.0  # |position| = 1
        expected = pnl + bonus
        assert reward == pytest.approx(float(expected), rel=1e-5)


# ===========================================================================
# Test 27: MultiDayEnv forwards temporal features
# ===========================================================================


class TestMultiDayEnvTemporalFeatures:
    """MultiDayEnv should forward temporal features from inner PrecomputedEnv."""

    def test_multi_day_position_at_index_53(self):
        """MultiDayEnv obs should have position at index 53."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, _ = env.reset()
        assert obs[POSITION_INDEX] == pytest.approx(0.0)

        obs, _, _, _, _ = env.step(2)  # go long
        assert obs[POSITION_INDEX] == pytest.approx(1.0)

    def test_multi_day_obs_no_nan(self):
        """MultiDayEnv observations should have no NaN."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isnan(obs))
            if terminated:
                break

    def test_multi_day_temporal_features_present(self):
        """MultiDayEnv obs[43:53] should contain temporal features (not all zeros after warmup)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        env.reset()
        # Step far enough for temporal features to be non-zero
        for _ in range(55):
            obs, _, terminated, _, _ = env.step(1)
            if terminated:
                break
        if not terminated:
            # At least some temporal features should be non-zero at t=55
            temporal = obs[43:53]
            assert np.any(temporal != 0.0), (
                f"Temporal features should be non-zero after 55 steps, got {temporal}"
            )
