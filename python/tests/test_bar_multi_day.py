"""Tests for MultiDayEnv with bar_size parameter.

Spec: docs/bar-level-env.md (Requirement 3)

These tests verify that:
- MultiDayEnv accepts bar_size parameter (default 0 = tick-level)
- bar_size=0 preserves existing tick-level behavior (backward compatible)
- bar_size>0 creates BarLevelEnv instances instead of PrecomputedEnv
- observation_space matches inner env (21 dims for bar, 54 for tick)
- step_interval is ignored when bar_size>0
- steps_per_episode is respected with bar_size>0
- Episode length is ~N/bar_size bars instead of N ticks
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.multi_day_env import MultiDayEnv

from conftest import make_realistic_obs, create_synthetic_cache_dir


# ===========================================================================
# Test 1: MultiDayEnv accepts bar_size parameter
# ===========================================================================


class TestMultiDayEnvBarSizeParam:
    """MultiDayEnv should accept a bar_size parameter."""

    def test_bar_size_param_accepted(self):
        """MultiDayEnv(cache_dir=..., bar_size=100) should not raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert env is not None

    def test_bar_size_default_is_zero(self):
        """Default bar_size should be 0 (tick-level, existing behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=50)
            env = MultiDayEnv(cache_dir=cache_dir)
            # Default = 0 means tick-level, obs shape should be (54,)
            assert env.observation_space.shape == (54,)


# ===========================================================================
# Test 2: bar_size=0 preserves tick-level behavior
# ===========================================================================


class TestBarSizeZeroBackwardCompat:
    """bar_size=0 should produce identical behavior to current (no bar_size)."""

    def test_tick_level_obs_shape(self):
        """bar_size=0 should give observation_space shape (54,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=50)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=0)
            assert env.observation_space.shape == (54,)

    def test_tick_level_episode_length(self):
        """bar_size=0 should give tick-level episode length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1, n_rows=50)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=0)
            env.reset()
            steps = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(1)
                steps += 1
            # Tick-level: should be 49 steps for 50 rows
            assert steps == 49


# ===========================================================================
# Test 3: bar_size>0 uses BarLevelEnv
# ===========================================================================


class TestBarSizePositiveUsesBarEnv:
    """bar_size>0 should create BarLevelEnv instances."""

    def test_bar_level_obs_shape_21(self):
        """bar_size>0 should give observation_space shape (21,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert env.observation_space.shape == (21,)

    def test_bar_level_action_space(self):
        """bar_size>0 should still have Discrete(3) action space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert isinstance(env.action_space, spaces.Discrete)
            assert env.action_space.n == 3

    def test_reset_returns_correct_shape(self):
        """reset() with bar_size>0 should return (21,) obs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            obs, info = env.reset()
            assert obs.shape == (21,)

    def test_step_returns_correct_shape(self):
        """step() with bar_size>0 should return (21,) obs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            env.reset()
            obs, _, _, _, _ = env.step(1)
            assert obs.shape == (21,)

    def test_bar_level_episode_shorter_than_tick(self):
        """bar_size>0 episodes should be much shorter than tick-level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1, n_rows=500)

            # Tick-level
            env_tick = MultiDayEnv(cache_dir=cache_dir, bar_size=0)
            env_tick.reset()
            tick_steps = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env_tick.step(1)
                tick_steps += 1

            # Bar-level with bar_size=100
            env_bar = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            env_bar.reset()
            bar_steps = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env_bar.step(1)
                bar_steps += 1

            assert bar_steps < tick_steps, (
                f"Bar-level ({bar_steps} steps) should be shorter "
                f"than tick-level ({tick_steps} steps)"
            )
            # With 500 ticks / 100 bar_size = 5 bars -> 4 steps
            assert bar_steps == 4, f"Expected 4 bar steps, got {bar_steps}"


# ===========================================================================
# Test 4: step_interval ignored when bar_size>0
# ===========================================================================


class TestStepIntervalIgnoredWithBarSize:
    """step_interval should be ignored when bar_size>0."""

    def test_step_interval_ignored(self):
        """bar_size=100 with step_interval=10 should behave same as step_interval=1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1, n_rows=500)

            env1 = MultiDayEnv(cache_dir=cache_dir, bar_size=100, step_interval=1)
            env10 = MultiDayEnv(cache_dir=cache_dir, bar_size=100, step_interval=10)

            obs1, _ = env1.reset(seed=42)
            obs10, _ = env10.reset(seed=42)

            # Both should produce same obs (step_interval ignored for bar-level)
            np.testing.assert_array_almost_equal(obs1, obs10)

    def test_step_interval_warning(self):
        """Setting step_interval with bar_size>0 should produce a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                env = MultiDayEnv(cache_dir=cache_dir, bar_size=100,
                                  step_interval=10)
                # Check that a warning was issued about step_interval being ignored
                warning_messages = [str(warning.message) for warning in w]
                assert any("step_interval" in msg.lower() or "bar_size" in msg.lower()
                           for msg in warning_messages), (
                    f"Expected warning about step_interval being ignored with bar_size>0, "
                    f"got warnings: {warning_messages}"
                )


# ===========================================================================
# Test 5: Multiple days cycle correctly with bar_size
# ===========================================================================


class TestMultiDayBarCycling:
    """MultiDayEnv with bar_size should cycle through days on reset."""

    def test_cycles_through_days(self):
        """Multiple resets should cycle through different days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3, n_rows=200)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=50)

            obs_list = []
            for _ in range(3):
                obs, info = env.reset()
                obs_list.append(obs.copy())
                assert obs.shape == (21,)

            # At least some should differ (different days have different mid prices)
            all_same = all(np.array_equal(obs_list[0], obs_list[i])
                           for i in range(1, 3))
            assert not all_same, "Different days should produce different observations"

    def test_day_index_in_info(self):
        """reset() info should contain day_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=3, n_rows=200)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=50)
            _, info = env.reset()
            assert "day_index" in info


# ===========================================================================
# Test 6: Reward parameters forwarded to BarLevelEnv
# ===========================================================================


class TestRewardParamsForwarded:
    """Reward parameters should be forwarded from MultiDayEnv to inner env."""

    def test_execution_cost_forwarded(self):
        """execution_cost=True should affect rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1, n_rows=200)

            env_no_cost = MultiDayEnv(cache_dir=cache_dir, bar_size=50,
                                       execution_cost=False)
            env_cost = MultiDayEnv(cache_dir=cache_dir, bar_size=50,
                                    execution_cost=True)

            env_no_cost.reset(seed=42)
            env_cost.reset(seed=42)

            _, r_no, _, _, _ = env_no_cost.step(2)  # go long
            _, r_yes, _, _, _ = env_cost.step(2)

            # With execution cost, reward should be lower
            assert r_yes <= r_no, (
                f"Execution cost should reduce reward: "
                f"without={r_no}, with={r_yes}"
            )

    def test_participation_bonus_forwarded(self):
        """participation_bonus>0 should affect rewards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=1, n_rows=200)

            env_no = MultiDayEnv(cache_dir=cache_dir, bar_size=50,
                                  participation_bonus=0.0)
            env_yes = MultiDayEnv(cache_dir=cache_dir, bar_size=50,
                                   participation_bonus=1.0)

            env_no.reset(seed=42)
            env_yes.reset(seed=42)

            _, r_no, _, _, _ = env_no.step(2)
            _, r_yes, _, _, _ = env_yes.step(2)

            # With bonus, reward should be higher for positioned agents
            assert r_yes > r_no, (
                f"Participation bonus should increase reward: "
                f"without={r_no}, with={r_yes}"
            )


# ===========================================================================
# Test 7: bar_size too large for day — skip with warning
# ===========================================================================


class TestBarSizeLargerThanDay:
    """bar_size larger than total ticks should skip days with warning."""

    def test_all_days_skipped_raises(self):
        """If all days produce 0 bars, should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 10 ticks per day, bar_size=100: 10 < 100//4=25, so 0 bars
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=2, n_rows=10)
            with pytest.raises(ValueError):
                MultiDayEnv(cache_dir=cache_dir, bar_size=100)

    def test_some_days_skipped(self):
        """Days with too few ticks should be skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Day 1: 10 ticks (too few for bar_size=100)
            obs1, mid1, spread1 = make_realistic_obs(10)
            np.savez(os.path.join(cache_dir, "2025-01-01.npz"),
                     obs=obs1, mid=mid1, spread=spread1)

            # Day 2: 500 ticks (enough for bar_size=100)
            obs2, mid2, spread2 = make_realistic_obs(500)
            np.savez(os.path.join(cache_dir, "2025-01-02.npz"),
                     obs=obs2, mid=mid2, spread=spread2)

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
                # Should construct successfully (day 2 has enough data)
                obs, _ = env.reset()
                assert obs.shape == (21,)


# ===========================================================================
# Test 8: Is a valid gymnasium.Env
# ===========================================================================


class TestMultiDayBarIsGymEnv:
    """MultiDayEnv with bar_size should be a valid gymnasium.Env."""

    def test_is_gym_env(self):
        """Should be an instance of gymnasium.Env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert isinstance(env, gym.Env)

    def test_has_observation_space(self):
        """Should have observation_space attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert hasattr(env, 'observation_space')

    def test_has_action_space(self):
        """Should have action_space attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_rows=500)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=100)
            assert hasattr(env, 'action_space')


# ===========================================================================
# Test 9: Shuffle works with bar_size
# ===========================================================================


class TestMultiDayBarShuffle:
    """Shuffle should work with bar_size>0."""

    def test_shuffle_produces_different_order(self):
        """With shuffle=True, day order should differ across epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=200)
            env = MultiDayEnv(cache_dir=cache_dir, bar_size=50, shuffle=True,
                              seed=42)

            day_indices = []
            for _ in range(5):
                _, info = env.reset()
                day_indices.append(info['day_index'])

            # Not all the same day
            assert len(set(day_indices)) > 1, (
                f"Shuffle should produce different day indices: {day_indices}"
            )
