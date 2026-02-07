"""Tests for MultiDayEnv with cache_dir parameter.

Spec: docs/precompute-cache.md (Requirement 3)

These tests verify that:
- Exactly one of file_paths or cache_dir must be provided (ValueError if both or neither)
- cache_dir globs for *.npz files sorted by name (date order)
- cache_dir loads via PrecomputedEnv.from_cache() instead of from_file()
- .npz files with wrong keys are skipped (same as current .bin skip pattern)
- cache_dir with empty directory raises error "No .npz files found"
- MultiDayEnv(cache_dir=...) produces identical obs/rewards as file_paths version
  (acceptance criterion 3)
- Existing file_paths workflow is unchanged (backward compatible)
"""

import os
import tempfile
import warnings

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

import lob_rl_core
from lob_rl.multi_day_env import MultiDayEnv
from lob_rl.precomputed_env import PrecomputedEnv

from conftest import FIXTURE_DIR, DAY_FILES, run_episode, make_realistic_obs


def _create_cache_dir(tmpdir, day_files, dates=None):
    """Create a cache directory with .npz files from real .bin fixtures."""
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    if dates is None:
        dates = [f"2025-01-{i + 10:02d}" for i in range(len(day_files))]

    cfg = lob_rl_core.SessionConfig.default_rth()
    for date, bin_path in zip(dates, day_files):
        obs, mid, spread, num_steps = lob_rl_core.precompute(bin_path, cfg)
        if num_steps >= 2:
            np.savez(os.path.join(cache_dir, f"{date}.npz"),
                     obs=obs, mid=mid, spread=spread)

    return cache_dir, dates


def _create_synthetic_cache_dir(tmpdir, n_days=3, n_rows=50):
    """Create a cache directory with synthetic .npz files."""
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    dates = [f"2025-01-{i + 10:02d}" for i in range(n_days)]
    for i, date in enumerate(dates):
        obs, mid, spread = make_realistic_obs(n_rows, mid_start=100.0 + i * 10)
        np.savez(os.path.join(cache_dir, f"{date}.npz"),
                 obs=obs, mid=mid, spread=spread)

    return cache_dir, dates


# ===========================================================================
# Test 1: Mutual exclusivity — exactly one of file_paths or cache_dir
# ===========================================================================


class TestMutualExclusivity:
    """Exactly one of file_paths or cache_dir must be provided."""

    def test_both_raises_value_error(self):
        """Providing both file_paths and cache_dir should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            with pytest.raises(ValueError):
                MultiDayEnv(file_paths=DAY_FILES[:1], cache_dir=cache_dir)

    def test_neither_raises_value_error(self):
        """Providing neither file_paths nor cache_dir should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            MultiDayEnv(file_paths=None, cache_dir=None)

    def test_file_paths_only_works(self):
        """Providing only file_paths should work (backward compatible)."""
        env = MultiDayEnv(file_paths=DAY_FILES[:1], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_cache_dir_only_works(self):
        """Providing only cache_dir should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            obs, info = env.reset()
            assert obs.shape == (54,)


# ===========================================================================
# Test 2: cache_dir loads .npz files sorted by name
# ===========================================================================


class TestCacheDirSorting:
    """cache_dir should glob for *.npz files sorted by name (date order)."""

    def test_npz_files_loaded_in_sorted_order(self):
        """Days from cache_dir should be visited in sorted filename order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with dates that sort differently than insertion order
            dates = ["2025-01-15", "2025-01-10", "2025-01-12"]
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_days=3)
            # Rename to use specific dates
            existing_files = sorted(os.listdir(cache_dir))
            for old_name, new_date in zip(existing_files, dates):
                old_path = os.path.join(cache_dir, old_name)
                new_path = os.path.join(cache_dir, f"{new_date}.npz")
                os.rename(old_path, new_path)

            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            # Visit all 3 days and collect day_index info
            day_indices = []
            for _ in range(3):
                _, info = env.reset()
                day_indices.append(info["day_index"])
                run_episode(env)

            # Sequential mode: should visit 0, 1, 2 (which maps to sorted filenames)
            assert day_indices == [0, 1, 2], (
                f"Expected sequential [0, 1, 2], got {day_indices}"
            )


# ===========================================================================
# Test 3: cache_dir is a valid gymnasium env
# ===========================================================================


class TestCacheDirEnv:
    """MultiDayEnv with cache_dir should be a valid gymnasium env."""

    def test_observation_space_shape(self):
        """observation_space should be (54,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert env.observation_space.shape == (54,)

    def test_action_space_n_is_3(self):
        """action_space should be Discrete(3)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert env.action_space.n == 3

    def test_is_gymnasium_env(self):
        """Instance should be a gymnasium.Env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            assert isinstance(env, gym.Env)

    def test_reset_returns_correct_tuple(self):
        """reset() should return (obs, info)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            result = env.reset()
            assert isinstance(result, tuple)
            assert len(result) == 2
            obs, info = result
            assert obs.shape == (54,)
            assert isinstance(info, dict)

    def test_step_returns_correct_tuple(self):
        """step() should return (obs, reward, terminated, truncated, info)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            env.reset()
            result = env.step(1)
            assert isinstance(result, tuple)
            assert len(result) == 5

    def test_full_episode_runs(self):
        """A full episode should complete without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            env.reset()
            steps = run_episode(env)
            assert steps > 0

    def test_check_env_passes(self):
        """gymnasium check_env should pass."""
        from gymnasium.utils.env_checker import check_env
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_rows=30)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            check_env(env, skip_render_check=True)


# ===========================================================================
# Test 4: cache_dir with empty directory
# ===========================================================================


class TestEmptyCacheDir:
    """cache_dir with no .npz files should raise error."""

    def test_empty_dir_raises(self):
        """An empty cache directory should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = os.path.join(tmpdir, "empty_cache")
            os.makedirs(empty_dir)
            with pytest.raises((ValueError, FileNotFoundError)):
                MultiDayEnv(cache_dir=empty_dir, shuffle=False)

    def test_dir_with_non_npz_files_raises(self):
        """A directory with only non-.npz files should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)
            # Create some non-npz files
            with open(os.path.join(cache_dir, "readme.txt"), "w") as f:
                f.write("not an npz file")
            with pytest.raises((ValueError, FileNotFoundError)):
                MultiDayEnv(cache_dir=cache_dir, shuffle=False)


# ===========================================================================
# Test 5: .npz with wrong keys are skipped
# ===========================================================================


class TestSkipsInvalidNpz:
    """.npz files with wrong keys should be skipped (not crash)."""

    def test_bad_npz_skipped_good_npz_used(self):
        """A mix of valid and invalid .npz files should skip invalid ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)

            # Create one valid .npz
            obs, mid, spread = make_realistic_obs(30)
            np.savez(os.path.join(cache_dir, "2025-01-10.npz"),
                     obs=obs, mid=mid, spread=spread)

            # Create one invalid .npz (wrong keys)
            np.savez(os.path.join(cache_dir, "2025-01-11.npz"),
                     wrong_key=np.array([1.0]))

            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            obs_out, info = env.reset()
            assert obs_out.shape == (54,)

    def test_all_bad_npz_raises(self):
        """If ALL .npz files are invalid, should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "cache")
            os.makedirs(cache_dir)

            # Create only invalid .npz files
            np.savez(os.path.join(cache_dir, "2025-01-10.npz"),
                     wrong_key=np.array([1.0]))
            np.savez(os.path.join(cache_dir, "2025-01-11.npz"),
                     another_key=np.array([2.0]))

            with pytest.raises(ValueError):
                MultiDayEnv(cache_dir=cache_dir, shuffle=False)


# ===========================================================================
# Test 6: cache_dir matches file_paths behavior (acceptance criterion 3)
# ===========================================================================


class TestCacheDirMatchesFilePaths:
    """MultiDayEnv(cache_dir=...) should produce identical results as file_paths."""

    def test_same_reset_obs(self):
        """First reset obs from cache_dir should match file_paths version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_cache_dir(tmpdir, DAY_FILES[:1])

            env_files = MultiDayEnv(file_paths=DAY_FILES[:1], shuffle=False)
            env_cache = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            obs_files, _ = env_files.reset()
            obs_cache, _ = env_cache.reset()

            np.testing.assert_array_almost_equal(
                obs_files, obs_cache,
                err_msg="cache_dir and file_paths produce different reset obs"
            )

    def test_same_rewards(self):
        """Rewards from cache_dir and file_paths should match for same actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_cache_dir(tmpdir, DAY_FILES[:1])

            env_files = MultiDayEnv(file_paths=DAY_FILES[:1], shuffle=False)
            env_cache = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            env_files.reset()
            env_cache.reset()

            for i, action in enumerate([2, 0, 1, 2, 0]):
                _, r_files, _, _, _ = env_files.step(action)
                _, r_cache, _, _, _ = env_cache.step(action)
                assert r_files == pytest.approx(r_cache), (
                    f"Reward mismatch at step {i}: files={r_files}, cache={r_cache}"
                )


# ===========================================================================
# Test 7: cache_dir with shuffle and seed
# ===========================================================================


class TestCacheDirShuffle:
    """cache_dir should support shuffle=True and seed for determinism."""

    def test_shuffle_with_seed_deterministic(self):
        """Same seed should produce same day ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_days=5, n_rows=30)

            def collect_indices(seed):
                env = MultiDayEnv(cache_dir=cache_dir, shuffle=True, seed=seed)
                indices = []
                for _ in range(5):
                    _, info = env.reset()
                    indices.append(info["day_index"])
                    run_episode(env)
                return indices

            a = collect_indices(42)
            b = collect_indices(42)
            assert a == b, f"Same seed should give same order: {a} vs {b}"

    def test_sequential_visits_all_days(self):
        """Sequential mode should visit all days in order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_days=3, n_rows=30)
            env = MultiDayEnv(cache_dir=cache_dir, shuffle=False)

            day_indices = []
            for _ in range(3):
                _, info = env.reset()
                day_indices.append(info["day_index"])
                run_episode(env)

            assert sorted(day_indices) == [0, 1, 2], (
                f"Expected all days visited, got {day_indices}"
            )


# ===========================================================================
# Test 8: cache_dir forwards parameters to PrecomputedEnv
# ===========================================================================


class TestCacheDirForwardsParams:
    """cache_dir mode should forward reward_mode, execution_cost, etc."""

    def test_forwards_reward_mode(self):
        """reward_mode should be forwarded to inner env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(
                cache_dir=cache_dir, shuffle=False,
                reward_mode="pnl_delta_penalized", lambda_=0.5,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_forwards_execution_cost(self):
        """execution_cost should be forwarded to inner env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(
                cache_dir=cache_dir, shuffle=False, execution_cost=True,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_forwards_step_interval(self):
        """step_interval should be forwarded to inner env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_rows=100)

            env_1 = MultiDayEnv(cache_dir=cache_dir, shuffle=False, step_interval=1)
            env_10 = MultiDayEnv(cache_dir=cache_dir, shuffle=False, step_interval=10)

            env_1.reset()
            env_10.reset()

            steps_1 = run_episode(env_1)
            steps_10 = run_episode(env_10)

            assert steps_10 < steps_1, (
                f"step_interval=10 ({steps_10}) should have fewer steps "
                f"than step_interval=1 ({steps_1})"
            )

    def test_forwards_participation_bonus(self):
        """participation_bonus should be forwarded to inner env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            env = MultiDayEnv(
                cache_dir=cache_dir, shuffle=False, participation_bonus=0.01,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))


# ===========================================================================
# Test 9: session_config is ignored when cache_dir is set
# ===========================================================================


class TestSessionConfigIgnored:
    """session_config should be ignored when cache_dir is set."""

    def test_session_config_ignored_with_cache_dir(self):
        """Passing session_config with cache_dir should not crash or error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir)
            # This should not raise — session_config should be silently ignored
            env = MultiDayEnv(
                cache_dir=cache_dir, shuffle=False,
                session_config={"rth_open_ns": 48_600_000_000_000,
                                "rth_close_ns": 72_000_000_000_000},
            )
            obs, info = env.reset()
            assert obs.shape == (54,)


# ===========================================================================
# Test 10: Backward compatibility — file_paths still works unchanged
# ===========================================================================


class TestBackwardCompatibility:
    """Existing file_paths workflow should be completely unchanged."""

    def test_file_paths_still_works(self):
        """MultiDayEnv(file_paths=...) should still work as before."""
        env = MultiDayEnv(file_paths=DAY_FILES[:2], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)
        assert "day_index" in info

    def test_file_paths_accepts_all_params(self):
        """file_paths mode should accept all existing parameters."""
        env = MultiDayEnv(
            file_paths=DAY_FILES[:2],
            session_config=None,
            steps_per_episode=50,
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False,
            participation_bonus=0.0,
            step_interval=1,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_file_paths_full_episode(self):
        """Full episode with file_paths should complete as before."""
        env = MultiDayEnv(file_paths=DAY_FILES[:1], shuffle=False)
        env.reset()
        steps = run_episode(env)
        assert steps > 0


# ===========================================================================
# Test 11: DummyVecEnv compatibility with cache_dir
# ===========================================================================


class TestDummyVecEnvCompatibility:
    """MultiDayEnv(cache_dir=...) should work inside SB3's DummyVecEnv."""

    def test_wrappable_in_dummy_vec_env(self):
        """DummyVecEnv should wrap cache_dir-based MultiDayEnv."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_rows=30)
            vec_env = DummyVecEnv([
                lambda: MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            ])
            obs = vec_env.reset()
            assert obs.shape == (1, 54)

    def test_dummy_vec_env_step(self):
        """step() through DummyVecEnv should work."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir, _ = _create_synthetic_cache_dir(tmpdir, n_rows=30)
            vec_env = DummyVecEnv([
                lambda: MultiDayEnv(cache_dir=cache_dir, shuffle=False)
            ])
            vec_env.reset()
            obs, rewards, dones, infos = vec_env.step([1])
            assert obs.shape == (1, 54)
            assert rewards.shape == (1,)
