"""Tests for step_interval — coarser time sampling of precomputed data.

Spec: docs/step-interval.md

These tests verify that:
- PrecomputedEnv accepts step_interval parameter (default=1)
- step_interval=1 produces identical behavior to current code (no regression)
- step_interval>1 subsamples obs, mid, spread arrays before temporal features
- Temporal features are computed on subsampled data
- Reward uses subsampled mid prices (larger price deltas per step)
- Episode length matches subsampled array length (N//interval steps approx)
- Invalid step_interval values (0, negative) raise ValueError
- step_interval larger than data raises ValueError (< 2 rows after subsample)
- from_file() accepts and forwards step_interval
- MultiDayEnv accepts and forwards step_interval
- train.py --step-interval CLI flag exists with correct default
"""

import argparse
import inspect
import os
import sys

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.precomputed_env import PrecomputedEnv

from conftest import (
    PRECOMPUTE_EPISODE_FILE,
    DAY_FILES,
    TRAIN_SCRIPT,
    make_obs as _make_obs,
    make_mid as _make_mid,
    make_spread as _make_spread,
    make_realistic_obs,
    run_episode,
    load_train_source,
)




# ===========================================================================
# 1. Constructor accepts step_interval parameter
# ===========================================================================


class TestStepIntervalConstructor:
    """PrecomputedEnv should accept a step_interval parameter."""

    def test_default_step_interval_is_1(self):
        """step_interval should default to 1."""
        env = PrecomputedEnv(_make_obs(10), _make_mid(10), _make_spread(10))
        # Should construct without error — default is 1
        assert env is not None

    def test_explicit_step_interval_1(self):
        """step_interval=1 should be accepted without error."""
        env = PrecomputedEnv(
            _make_obs(10), _make_mid(10), _make_spread(10), step_interval=1
        )
        assert env is not None

    def test_step_interval_2(self):
        """step_interval=2 should be accepted."""
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20), step_interval=2
        )
        assert env is not None

    def test_step_interval_10(self):
        """step_interval=10 should be accepted with enough data."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=10
        )
        assert env is not None

    def test_step_interval_kwarg_accepted(self):
        """step_interval should be accepted as a keyword argument."""
        sig = inspect.signature(PrecomputedEnv.__init__)
        assert "step_interval" in sig.parameters, (
            "PrecomputedEnv.__init__() should accept step_interval parameter"
        )


# ===========================================================================
# 2. step_interval=1 is a no-op (identical behavior to current code)
# ===========================================================================


class TestStepInterval1NoOp:
    """step_interval=1 should produce identical behavior to not passing it."""

    def test_same_episode_length(self):
        """step_interval=1 should give same episode length as default."""
        obs = _make_obs(20)
        mid = _make_mid(20)
        spread = _make_spread(20)

        env_default = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy())
        env_explicit = PrecomputedEnv(
            obs.copy(), mid.copy(), spread.copy(), step_interval=1
        )

        env_default.reset()
        env_explicit.reset()

        steps_default = run_episode(env_default)
        steps_explicit = run_episode(env_explicit)

        assert steps_default == steps_explicit

    def test_same_observations(self):
        """step_interval=1 should produce same obs sequence as default."""
        obs = _make_obs(10)
        mid = _make_mid(10)
        spread = _make_spread(10)

        env_default = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy())
        env_explicit = PrecomputedEnv(
            obs.copy(), mid.copy(), spread.copy(), step_interval=1
        )

        obs_d, _ = env_default.reset()
        obs_e, _ = env_explicit.reset()
        np.testing.assert_array_equal(obs_d, obs_e)

        for _ in range(9):
            obs_d, r_d, term_d, _, _ = env_default.step(2)
            obs_e, r_e, term_e, _, _ = env_explicit.step(2)
            np.testing.assert_array_equal(obs_d, obs_e)
            assert r_d == pytest.approx(r_e)
            assert term_d == term_e

    def test_same_rewards(self):
        """step_interval=1 should produce same rewards as default."""
        obs = _make_obs(10)
        mid = _make_mid(10)
        spread = _make_spread(10)

        env_default = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy())
        env_explicit = PrecomputedEnv(
            obs.copy(), mid.copy(), spread.copy(), step_interval=1
        )

        env_default.reset()
        env_explicit.reset()

        for _ in range(9):
            _, r_d, _, _, _ = env_default.step(0)
            _, r_e, _, _, _ = env_explicit.step(0)
            assert r_d == pytest.approx(r_e)


# ===========================================================================
# 3. Subsampling reduces array length
# ===========================================================================


class TestSubsamplingReducesLength:
    """step_interval>1 should reduce the effective number of steps."""

    def test_step_interval_2_halves_steps(self):
        """step_interval=2 on 20 snapshots should give ~10 snapshots -> 9 steps."""
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20), step_interval=2
        )
        env.reset()
        steps = run_episode(env)
        # 20 snapshots with interval=2 -> 10 subsampled rows -> 9 steps
        assert steps == 9

    def test_step_interval_5_on_100(self):
        """step_interval=5 on 100 snapshots -> 20 rows -> 19 steps."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=5
        )
        env.reset()
        steps = run_episode(env)
        assert steps == 19

    def test_step_interval_10_on_100(self):
        """step_interval=10 on 100 snapshots -> 10 rows -> 9 steps."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=10
        )
        env.reset()
        steps = run_episode(env)
        assert steps == 9

    def test_step_interval_3_on_10(self):
        """step_interval=3 on 10 snapshots -> ceil(10/3)=4 subsampled rows -> 3 steps.

        array[::3] on 10 elements gives indices [0, 3, 6, 9] = 4 rows.
        """
        env = PrecomputedEnv(
            _make_obs(10), _make_mid(10), _make_spread(10), step_interval=3
        )
        env.reset()
        steps = run_episode(env)
        # np.arange(10)[::3] = [0, 3, 6, 9] -> 4 rows -> 3 steps
        assert steps == 3

    def test_step_interval_large_gives_minimum_episode(self):
        """step_interval that leaves exactly 2 rows should give 1 step."""
        # 10 snapshots, interval=5: [0, 5] = 2 rows -> 1 step
        env = PrecomputedEnv(
            _make_obs(10), _make_mid(10), _make_spread(10), step_interval=5
        )
        env.reset()
        steps = run_episode(env)
        assert steps == 1


# ===========================================================================
# 4. Subsampled obs match every Nth row of input
# ===========================================================================


class TestSubsampledObsContent:
    """After subsampling, obs should correspond to every Nth row of input."""

    def test_obs_at_reset_matches_row_0(self):
        """After reset with step_interval=2, obs[:43] should match input row 0."""
        input_obs = _make_obs(20, fill=42.0)
        env = PrecomputedEnv(
            input_obs, _make_mid(20), _make_spread(20), step_interval=2
        )
        obs, _ = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])

    def test_obs_after_step_matches_row_N(self):
        """After one step with step_interval=2, obs[:43] should match input row 2."""
        input_obs = _make_obs(20, fill=42.0)
        env = PrecomputedEnv(
            input_obs, _make_mid(20), _make_spread(20), step_interval=2
        )
        env.reset()
        obs, _, _, _, _ = env.step(1)
        # Subsampled rows are [0, 2, 4, 6, ...], so after step 0 we're at subsampled index 1 = original index 2
        np.testing.assert_array_almost_equal(obs[:43], input_obs[2])

    def test_obs_sequence_matches_every_nth_row(self):
        """Full obs sequence with step_interval=3 should use rows [0, 3, 6, 9]."""
        input_obs = _make_obs(12, fill=10.0)
        mid = _make_mid(12)
        spread = _make_spread(12)
        env = PrecomputedEnv(input_obs, mid, spread, step_interval=3)

        obs, _ = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])

        expected_rows = [3, 6, 9]
        for i, row_idx in enumerate(expected_rows):
            obs, _, _, _, _ = env.step(1)
            np.testing.assert_array_almost_equal(
                obs[:43], input_obs[row_idx],
                err_msg=f"Step {i}: expected input row {row_idx}"
            )


# ===========================================================================
# 5. Reward uses subsampled mid prices
# ===========================================================================


class TestSubsampledReward:
    """Reward should use subsampled mid prices, giving larger deltas."""

    def test_reward_uses_subsampled_mids(self):
        """With step_interval=2, reward delta should be mid[2]-mid[0], not mid[1]-mid[0]."""
        mid = np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(5), mid, _make_spread(5), step_interval=2
        )
        env.reset()
        # Subsampled mid = [100.0, 101.0, 102.0]
        # Long position: reward = +1 * (101.0 - 100.0) = 1.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.0)

    def test_reward_larger_with_interval(self):
        """step_interval>1 should produce larger per-step rewards (wider price moves)."""
        mid = np.arange(100.0, 120.0, 1.0, dtype=np.float64)  # 20 steps, +1 each
        obs = _make_obs(20)
        spread = _make_spread(20)

        env_1 = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy(), step_interval=1)
        env_5 = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy(), step_interval=5)

        env_1.reset()
        env_5.reset()

        _, r1, _, _, _ = env_1.step(2)  # reward = +1 * (101 - 100) = 1.0
        _, r5, _, _, _ = env_5.step(2)  # reward = +1 * (105 - 100) = 5.0

        assert abs(r5) > abs(r1), (
            f"step_interval=5 reward ({r5}) should be larger than step_interval=1 ({r1})"
        )

    def test_reward_exact_with_interval_3(self):
        """Verify exact reward with step_interval=3."""
        # mid = [100, 101, 102, 103, 104, 105, 106, 107, 108]
        # subsampled mid = [100, 103, 106]
        mid = np.arange(100.0, 109.0, dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(9), mid, _make_spread(9), step_interval=3
        )
        env.reset()

        # Step 0: long, reward = +1 * (103 - 100) = 3.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(3.0)


# ===========================================================================
# 6. Temporal features computed on subsampled data
# ===========================================================================


class TestTemporalFeaturesOnSubsampledData:
    """Temporal features should be computed on subsampled arrays, not full-resolution."""

    def test_temporal_features_at_t0_still_zero(self):
        """After reset, temporal features (obs[43:53]) should still be 0.0."""
        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread, step_interval=5)
        result_obs, _ = env.reset()
        temporal = result_obs[43:53]
        np.testing.assert_array_equal(
            temporal, np.zeros(10, dtype=np.float32),
            err_msg=f"Temporal features at t=0 should be 0.0, got {temporal}"
        )

    def test_mid_return_1_uses_subsampled_mid(self):
        """mid_return_1 at step 1 should be (sub_mid[1]-sub_mid[0])/sub_mid[0]."""
        obs, mid, spread = make_realistic_obs(100)
        interval = 5
        env = PrecomputedEnv(obs, mid, spread, step_interval=interval)
        env.reset()
        result_obs, _, _, _, _ = env.step(1)

        # Subsampled mid indices: [0, 5, 10, ...]
        sub_mid = mid[::interval]
        expected = (sub_mid[1] - sub_mid[0]) / sub_mid[0]
        assert result_obs[43] == pytest.approx(float(expected), rel=1e-4), (
            f"mid_return_1 with interval={interval}: expected {expected}, got {result_obs[43]}"
        )

    def test_temporal_features_differ_with_interval(self):
        """Temporal features should differ between step_interval=1 and step_interval=5."""
        obs, mid, spread = make_realistic_obs(200)

        env_1 = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy(), step_interval=1)
        env_5 = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy(), step_interval=5)

        env_1.reset()
        env_5.reset()

        # Step enough times to get non-trivial temporal features
        for _ in range(10):
            obs_1, _, _, _, _ = env_1.step(1)
        for _ in range(10):
            obs_5, _, _, _, _ = env_5.step(1)

        # At least one temporal feature should differ
        temporal_1 = obs_1[43:53]
        temporal_5 = obs_5[43:53]
        assert not np.allclose(temporal_1, temporal_5), (
            "Temporal features should differ between step_interval=1 and step_interval=5"
        )


# ===========================================================================
# 7. Validation: invalid step_interval values
# ===========================================================================


class TestStepIntervalValidation:
    """Invalid step_interval values should raise ValueError."""

    def test_step_interval_0_raises(self):
        """step_interval=0 should raise ValueError."""
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(10), _make_mid(10), _make_spread(10), step_interval=0
            )

    def test_step_interval_negative_raises(self):
        """step_interval=-1 should raise ValueError."""
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(10), _make_mid(10), _make_spread(10), step_interval=-1
            )

    def test_step_interval_negative_large_raises(self):
        """step_interval=-100 should raise ValueError."""
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(10), _make_mid(10), _make_spread(10), step_interval=-100
            )

    def test_step_interval_too_large_for_data(self):
        """step_interval that leaves < 2 rows should raise ValueError."""
        # 3 snapshots with interval=3: [0] -> only 1 row -> ValueError
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(3), _make_mid(3), _make_spread(3), step_interval=3
            )

    def test_step_interval_equals_data_length(self):
        """step_interval == data length should raise ValueError (only 1 row)."""
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(5), _make_mid(5), _make_spread(5), step_interval=5
            )

    def test_step_interval_exceeds_data_length(self):
        """step_interval > data length should raise ValueError."""
        with pytest.raises(ValueError):
            PrecomputedEnv(
                _make_obs(5), _make_mid(5), _make_spread(5), step_interval=10
            )

    def test_step_interval_leaves_exactly_2_rows_ok(self):
        """step_interval that leaves exactly 2 rows should be accepted."""
        # 4 snapshots with interval=2: [0, 2] -> 2 rows -> OK
        env = PrecomputedEnv(
            _make_obs(4), _make_mid(4), _make_spread(4), step_interval=2
        )
        assert env is not None


# ===========================================================================
# 8. Observation space unchanged (still 54 dims)
# ===========================================================================


class TestObsSpaceUnchanged:
    """Observation space should still be (54,) regardless of step_interval."""

    def test_obs_space_shape_with_interval(self):
        """observation_space.shape should be (54,) with step_interval > 1."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=5
        )
        assert env.observation_space.shape == (54,)

    def test_action_space_unchanged(self):
        """action_space should still be Discrete(3)."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=5
        )
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_reset_obs_shape_with_interval(self):
        """reset() obs shape should be (54,) with step_interval."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=5
        )
        obs, _ = env.reset()
        assert obs.shape == (54,)
        assert obs.dtype == np.float32

    def test_step_obs_shape_with_interval(self):
        """step() obs shape should be (54,) with step_interval."""
        env = PrecomputedEnv(
            _make_obs(100), _make_mid(100), _make_spread(100), step_interval=5
        )
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (54,)
        assert obs.dtype == np.float32


# ===========================================================================
# 9. Position tracking still works with step_interval
# ===========================================================================


class TestPositionWithInterval:
    """Position at index 53 should still work correctly with step_interval."""

    def test_position_at_reset_is_zero(self):
        """Position should be 0.0 at reset with step_interval."""
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20), step_interval=2
        )
        obs, _ = env.reset()
        assert obs[53] == pytest.approx(0.0)

    def test_action_mapping_with_interval(self):
        """Action mapping {0->-1, 1->0, 2->+1} should still work."""
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20), step_interval=2
        )
        env.reset()

        obs, _, _, _, _ = env.step(0)
        assert obs[53] == pytest.approx(-1.0)

        obs, _, _, _, _ = env.step(2)
        assert obs[53] == pytest.approx(1.0)

        obs, _, _, _, _ = env.step(1)
        assert obs[53] == pytest.approx(0.0)


# ===========================================================================
# 10. Flattening penalty works with step_interval
# ===========================================================================


class TestFlatteningWithInterval:
    """Flattening penalty at terminal step should work with step_interval."""

    def test_flattening_penalty_uses_subsampled_spread(self):
        """Forced flatten uses spread from subsampled array."""
        # 6 snapshots, interval=2: subsampled = [0, 2, 4] -> 3 rows, 2 steps
        mid = np.array([100.0, 100.0, 101.0, 101.0, 102.0, 102.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0], dtype=np.float64)
        # subsampled: mid=[100, 101, 102], spread=[0.5, 1.0, 2.0]

        env = PrecomputedEnv(
            _make_obs(6), mid, spread, step_interval=2
        )
        env.reset()

        # Step 0: go long (non-terminal)
        _, _, terminated, _, _ = env.step(2)
        assert not terminated

        # Step 1: terminal — forced flatten
        # close_cost = subsampled_spread[2]/2 * |prev_position| = 2.0/2 * 1 = 1.0
        # reward = -1.0
        _, reward, terminated, _, info = env.step(2)
        assert terminated
        assert reward == pytest.approx(-1.0)
        assert info["forced_flatten"] is True


# ===========================================================================
# 11. Execution cost and participation bonus work with step_interval
# ===========================================================================


class TestRewardModesWithInterval:
    """Existing reward features should still work with step_interval."""

    def test_execution_cost_with_interval(self):
        """execution_cost should work correctly with step_interval."""
        mid = np.arange(100.0, 110.0, dtype=np.float64)
        spread = np.full(10, 0.5, dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(10), mid, spread, execution_cost=True, step_interval=2
        )
        env.reset()
        # go long from flat: delta_pos = |1-0| = 1
        # subsampled mid = [100, 102, 104, 106, 108]
        # subsampled spread = [0.5, 0.5, 0.5, 0.5, 0.5]
        # pnl = +1 * (102 - 100) = 2.0
        # exec cost = 0.5 / 2 * 1 = 0.25
        # reward = 2.0 - 0.25 = 1.75
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.75)

    def test_participation_bonus_with_interval(self):
        """participation_bonus should work correctly with step_interval."""
        mid = np.arange(100.0, 110.0, dtype=np.float64)
        spread = np.full(10, 0.5, dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(10), mid, spread, participation_bonus=0.01, step_interval=2
        )
        env.reset()
        # go long: subsampled mid delta = 102 - 100 = 2.0
        # pnl = +1 * 2.0 = 2.0, bonus = 0.01 * 1.0 = 0.01
        # reward = 2.0 + 0.01 = 2.01
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(2.01)

    def test_penalized_mode_with_interval(self):
        """pnl_delta_penalized should work correctly with step_interval."""
        mid = np.arange(100.0, 110.0, dtype=np.float64)
        spread = np.full(10, 0.5, dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(10), mid, spread,
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            step_interval=2,
        )
        env.reset()
        # pnl = +1 * (102 - 100) = 2.0, penalty = 0.5 * 1 = 0.5
        # reward = 2.0 - 0.5 = 1.5
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.5)


# ===========================================================================
# 12. from_file() accepts and forwards step_interval
# ===========================================================================


class TestFromFileStepInterval:
    """PrecomputedEnv.from_file() should accept and forward step_interval."""

    def test_from_file_signature_has_step_interval(self):
        """from_file() should have step_interval parameter."""
        sig = inspect.signature(PrecomputedEnv.from_file)
        assert "step_interval" in sig.parameters, (
            "from_file() should accept step_interval parameter"
        )

    def test_from_file_default_step_interval_1(self):
        """from_file() with default step_interval should work normally."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        env.reset()
        steps_default = run_episode(env)
        assert steps_default > 0

    def test_from_file_step_interval_reduces_steps(self):
        """from_file(step_interval=10) should produce fewer steps."""
        env_1 = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, step_interval=1)
        env_10 = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, step_interval=10)

        env_1.reset()
        env_10.reset()

        steps_1 = run_episode(env_1)
        steps_10 = run_episode(env_10)

        # With interval=10, should have roughly 1/10th the steps
        assert steps_10 < steps_1, (
            f"step_interval=10 ({steps_10} steps) should give fewer steps than "
            f"step_interval=1 ({steps_1} steps)"
        )
        # Roughly 1/10th (within a factor of 2 tolerance due to rounding)
        assert steps_10 <= steps_1 // 5, (
            f"step_interval=10 ({steps_10}) should be roughly 1/10th of "
            f"step_interval=1 ({steps_1})"
        )

    def test_from_file_step_interval_valid_env(self):
        """from_file(step_interval=5) should produce a valid gymnasium env."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, step_interval=5)
        obs, info = env.reset()
        assert obs.shape == (54,)
        assert obs.dtype == np.float32
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (54,)
        assert isinstance(reward, (float, np.floating))


# ===========================================================================
# 13. MultiDayEnv accepts and forwards step_interval
# ===========================================================================


class TestMultiDayEnvStepInterval:
    """MultiDayEnv should accept and forward step_interval to inner env."""

    def test_multi_day_signature_has_step_interval(self):
        """MultiDayEnv.__init__() should have step_interval parameter."""
        from lob_rl.multi_day_env import MultiDayEnv
        sig = inspect.signature(MultiDayEnv.__init__)
        assert "step_interval" in sig.parameters, (
            "MultiDayEnv.__init__() should accept step_interval parameter"
        )

    def test_multi_day_default_step_interval(self):
        """MultiDayEnv with default step_interval should work normally."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, _ = env.reset()
        assert obs.shape == (54,)
        steps = run_episode(env)
        assert steps > 0

    def test_multi_day_step_interval_reduces_steps(self):
        """MultiDayEnv(step_interval=10) should produce fewer steps per episode."""
        from lob_rl.multi_day_env import MultiDayEnv

        env_1 = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, step_interval=1
        )
        env_10 = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, step_interval=10
        )

        env_1.reset()
        env_10.reset()

        steps_1 = run_episode(env_1)
        steps_10 = run_episode(env_10)

        assert steps_10 < steps_1, (
            f"step_interval=10 ({steps_10}) should give fewer steps than "
            f"step_interval=1 ({steps_1})"
        )

    def test_multi_day_step_interval_valid_obs(self):
        """MultiDayEnv with step_interval should produce valid observations."""
        from lob_rl.multi_day_env import MultiDayEnv

        env = MultiDayEnv(
            file_paths=DAY_FILES[:2], shuffle=False, step_interval=5
        )
        obs, _ = env.reset()
        assert obs.shape == (54,)
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))

    def test_multi_day_step_interval_multiple_days(self):
        """MultiDayEnv should apply step_interval consistently across day switches."""
        from lob_rl.multi_day_env import MultiDayEnv

        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], shuffle=False, step_interval=10
        )

        for day in range(3):
            obs, _ = env.reset()
            assert obs.shape == (54,)
            steps = run_episode(env)
            assert steps > 0


# ===========================================================================
# 14. train.py --step-interval CLI flag
# ===========================================================================


class TestTrainScriptCLI:
    """train.py should have --step-interval CLI flag."""

    def test_step_interval_flag_in_source(self):
        """train.py should contain --step-interval argument definition."""
        source = load_train_source()
        assert "--step-interval" in source, (
            "train.py should define --step-interval CLI flag"
        )

    def test_step_interval_default_is_1(self):
        """--step-interval should default to 1."""
        source = load_train_source()
        # Should contain default=1 near --step-interval
        assert "default=1" in source or "default = 1" in source, (
            "--step-interval should have default=1"
        )

    def test_step_interval_type_is_int(self):
        """--step-interval should have type=int in the same add_argument call."""
        source = load_train_source()
        # Find the add_argument line for --step-interval and verify type=int
        import re
        pattern = r"add_argument\(['\"]--step-interval['\"].*?type\s*=\s*int"
        assert re.search(pattern, source, re.DOTALL), (
            "--step-interval add_argument call should include type=int"
        )


# ===========================================================================
# 15. make_env and make_train_env forward step_interval
# ===========================================================================


class TestTrainFunctionsForwardInterval:
    """make_env() and make_train_env() should accept and forward step_interval."""

    def test_make_env_has_step_interval_param(self):
        """make_env() in train.py should accept step_interval."""
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        from train import make_env
        sig = inspect.signature(make_env)
        assert "step_interval" in sig.parameters, (
            "make_env() should accept step_interval parameter"
        )

    def test_make_train_env_has_step_interval_param(self):
        """make_train_env() in train.py should accept step_interval."""
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        from train import make_train_env
        sig = inspect.signature(make_train_env)
        assert "step_interval" in sig.parameters, (
            "make_train_env() should accept step_interval parameter"
        )

    def test_evaluate_sortino_has_step_interval_param(self):
        """evaluate_sortino() should accept step_interval."""
        scripts_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        from train import evaluate_sortino
        sig = inspect.signature(evaluate_sortino)
        assert "step_interval" in sig.parameters, (
            "evaluate_sortino() should accept step_interval parameter"
        )


# ===========================================================================
# 16. Gymnasium check_env passes with step_interval
# ===========================================================================


class TestCheckEnvWithInterval:
    """gymnasium check_env should pass with step_interval > 1."""

    def test_check_env_synthetic_with_interval(self):
        """check_env() should pass with step_interval=5 on synthetic data."""
        from gymnasium.utils.env_checker import check_env

        obs, mid, spread = make_realistic_obs(100)
        env = PrecomputedEnv(obs, mid, spread, step_interval=5)
        check_env(env, skip_render_check=True)

    def test_check_env_from_file_with_interval(self):
        """check_env() should pass with step_interval=10 on file-backed data."""
        from gymnasium.utils.env_checker import check_env

        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, step_interval=10)
        check_env(env, skip_render_check=True)


# ===========================================================================
# 17. Acceptance criterion: step_interval=10 on large data
# ===========================================================================


class TestAcceptanceCriteria:
    """Key acceptance criteria from the spec."""

    def test_interval_10_reduces_to_tenth(self):
        """step_interval=10 on N snapshots should give ~N/10 subsampled rows."""
        N = 1000
        obs = _make_obs(N)
        mid = _make_mid(N)
        spread = _make_spread(N)

        env = PrecomputedEnv(obs, mid, spread, step_interval=10)
        env.reset()
        steps = run_episode(env)

        # N=1000, interval=10: subsampled = 100 rows -> 99 steps
        assert steps == 99

    def test_no_nan_in_subsampled_episode(self):
        """No NaN values should appear in any obs during a subsampled episode."""
        obs, mid, spread = make_realistic_obs(200)
        env = PrecomputedEnv(obs, mid, spread, step_interval=5)
        result_obs, _ = env.reset()
        assert not np.any(np.isnan(result_obs))

        terminated = False
        while not terminated:
            result_obs, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isnan(result_obs))

    def test_no_inf_in_subsampled_episode(self):
        """No Inf values should appear in any obs during a subsampled episode."""
        obs, mid, spread = make_realistic_obs(200)
        env = PrecomputedEnv(obs, mid, spread, step_interval=5)
        result_obs, _ = env.reset()
        assert not np.any(np.isinf(result_obs))

        terminated = False
        while not terminated:
            result_obs, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isinf(result_obs))


# ===========================================================================
# 18. Deterministic behavior with step_interval
# ===========================================================================


class TestDeterminismWithInterval:
    """Two episodes with same actions and step_interval should be identical."""

    def test_deterministic_with_interval(self):
        """Replaying same actions with step_interval=3 gives identical results."""
        obs = _make_obs(30)
        mid = _make_mid(30)
        spread = _make_spread(30)

        env = PrecomputedEnv(obs.copy(), mid.copy(), spread.copy(), step_interval=3)
        actions = [2, 0, 1, 2, 0, 1, 2, 0, 1]  # 9 steps for 10 subsampled rows

        # Episode 1
        obs1, _ = env.reset()
        results1 = []
        for a in actions:
            o, r, t, _, _ = env.step(a)
            results1.append((o.copy(), r, t))
            if t:
                break

        # Episode 2
        obs2, _ = env.reset()
        results2 = []
        for a in actions:
            o, r, t, _, _ = env.step(a)
            results2.append((o.copy(), r, t))
            if t:
                break

        np.testing.assert_array_equal(obs1, obs2)
        for i, ((o1, r1, t1), (o2, r2, t2)) in enumerate(zip(results1, results2)):
            np.testing.assert_array_equal(o1, o2, err_msg=f"obs differ at step {i}")
            assert r1 == pytest.approx(r2), f"rewards differ at step {i}"
            assert t1 == t2, f"terminated differ at step {i}"


# ===========================================================================
# 19. Reset works correctly with step_interval
# ===========================================================================


class TestResetWithInterval:
    """reset() should work correctly with step_interval."""

    def test_reset_mid_episode_with_interval(self):
        """reset() after partial episode should return to t=0 of subsampled data."""
        input_obs = _make_obs(30, fill=7.77)
        env = PrecomputedEnv(
            input_obs, _make_mid(30), _make_spread(30), step_interval=3
        )
        env.reset()
        env.step(2)
        env.step(0)

        obs, _ = env.reset()
        # Should return to subsampled row 0, which is original row 0
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])
        assert obs[53] == pytest.approx(0.0)

    def test_multiple_episodes_with_interval(self):
        """Multiple reset/episode cycles should work with step_interval."""
        obs = _make_obs(50)
        mid = _make_mid(50)
        spread = _make_spread(50)

        env = PrecomputedEnv(obs, mid, spread, step_interval=5)

        for _ in range(3):
            o, _ = env.reset()
            assert o.shape == (54,)
            assert o[53] == pytest.approx(0.0)
            steps = run_episode(env)
            assert steps > 0
