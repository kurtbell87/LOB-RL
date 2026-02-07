"""Tests for PrecomputedEnv.from_cache() classmethod.

Spec: docs/precompute-cache.md (Requirement 2)

These tests verify that:
- from_cache(npz_path) loads obs, mid, spread from an .npz file
- from_cache produces a valid PrecomputedEnv (gymnasium.Env)
- Loaded env behaves identically to one constructed from raw arrays
- from_cache accepts reward_mode, lambda_, execution_cost, participation_bonus, step_interval
- from_cache is fast (numpy load + array slicing only)
- Missing or wrong keys in .npz raise appropriate errors
- Cache files are reusable across step_interval values (acceptance criterion 5)
"""

import os
import tempfile

import numpy as np
import pytest
import gymnasium as gym

from lob_rl.precomputed_env import PrecomputedEnv

from conftest import make_obs, make_mid, make_spread, make_realistic_obs

# Fixture paths
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DAY_FILES = [os.path.join(FIXTURE_DIR, f"day{i}.bin") for i in range(5)]


def _create_npz(tmpdir, n=100, name="test.npz"):
    """Create a valid .npz cache file with synthetic data and return path."""
    obs, mid, spread_arr = make_realistic_obs(n)
    path = os.path.join(tmpdir, name)
    np.savez(path, obs=obs, mid=mid, spread=spread_arr)
    return path, obs, mid, spread_arr


def _create_npz_from_file(tmpdir, bin_path, name="day.npz"):
    """Create .npz by precomputing a real .bin file."""
    import lob_rl_core
    cfg = lob_rl_core.SessionConfig.default_rth()
    obs, mid, spread, _ = lob_rl_core.precompute(bin_path, cfg)
    path = os.path.join(tmpdir, name)
    np.savez(path, obs=obs, mid=mid, spread=spread)
    return path, obs, mid, spread


# ===========================================================================
# Test 1: from_cache() exists and is callable
# ===========================================================================


class TestFromCacheExists:
    """PrecomputedEnv should have a from_cache classmethod."""

    def test_from_cache_is_classmethod(self):
        """from_cache should be a classmethod on PrecomputedEnv."""
        assert hasattr(PrecomputedEnv, "from_cache"), (
            "PrecomputedEnv does not have from_cache classmethod"
        )
        assert callable(PrecomputedEnv.from_cache), (
            "PrecomputedEnv.from_cache is not callable"
        )


# ===========================================================================
# Test 2: from_cache() returns PrecomputedEnv
# ===========================================================================


class TestFromCacheReturnsEnv:
    """from_cache() should return a valid PrecomputedEnv instance."""

    def test_returns_precomputed_env(self):
        """from_cache(npz_path) should return a PrecomputedEnv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            assert isinstance(env, PrecomputedEnv)

    def test_is_gymnasium_env(self):
        """from_cache() result should be a gymnasium.Env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            assert isinstance(env, gym.Env)


# ===========================================================================
# Test 3: from_cache() loads correct data
# ===========================================================================


class TestFromCacheLoadsData:
    """from_cache() should load arrays from the .npz and use them correctly."""

    def test_reset_returns_correct_shape(self):
        """reset() on cached env should return obs of shape (54,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            obs, info = env.reset()
            assert obs.shape == (54,)

    def test_reset_obs_dtype_float32(self):
        """reset() obs should be float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            obs, _ = env.reset()
            assert obs.dtype == np.float32

    def test_first_43_match_npz_obs_row_0(self):
        """reset() obs[:43] should match the first row of the cached obs array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, obs_arr, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            obs, _ = env.reset()
            np.testing.assert_array_almost_equal(obs[:43], obs_arr[0])

    def test_step_works(self):
        """step() on cached env should return valid 5-tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path)
            env.reset()
            obs, reward, terminated, truncated, info = env.step(1)
            assert obs.shape == (54,)
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))

    def test_episode_length_matches(self):
        """Episode from cached env should have N-1 steps (N = number of snapshots)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 20
            npz_path, _, _, _ = _create_npz(tmpdir, n=n)
            env = PrecomputedEnv.from_cache(npz_path)
            env.reset()
            steps = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(1)
                steps += 1
            assert steps == n - 1, f"Expected {n-1} steps, got {steps}"


# ===========================================================================
# Test 4: from_cache() accepts all optional parameters
# ===========================================================================


class TestFromCacheParameters:
    """from_cache() should accept reward_mode, lambda_, execution_cost, etc."""

    def test_accepts_reward_mode(self):
        """from_cache should accept reward_mode parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(
                npz_path, reward_mode="pnl_delta_penalized", lambda_=0.5
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_accepts_execution_cost(self):
        """from_cache should accept execution_cost parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path, execution_cost=True)
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_accepts_participation_bonus(self):
        """from_cache should accept participation_bonus parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir)
            env = PrecomputedEnv.from_cache(npz_path, participation_bonus=0.01)
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_accepts_step_interval(self):
        """from_cache should accept step_interval parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir, n=100)
            env = PrecomputedEnv.from_cache(npz_path, step_interval=5)
            env.reset()
            _, reward, _, _, _ = env.step(1)
            assert isinstance(reward, (float, np.floating))


# ===========================================================================
# Test 5: from_cache() matches from_file() behavior (acceptance criterion 2)
# ===========================================================================


class TestFromCacheMatchesFromFile:
    """from_cache() should produce identical behavior to from_file() for the same day."""

    def test_same_reset_obs(self):
        """from_cache and from_file should produce same obs on reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, obs, mid, spread = _create_npz_from_file(tmpdir, DAY_FILES[0])

            env_cache = PrecomputedEnv.from_cache(npz_path)
            env_file = PrecomputedEnv.from_file(DAY_FILES[0])

            obs_cache, _ = env_cache.reset()
            obs_file, _ = env_file.reset()

            np.testing.assert_array_almost_equal(
                obs_cache, obs_file,
                err_msg="from_cache and from_file produce different reset obs"
            )

    def test_same_rewards_sequence(self):
        """from_cache and from_file should produce identical rewards for same actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz_from_file(tmpdir, DAY_FILES[0])

            env_cache = PrecomputedEnv.from_cache(npz_path)
            env_file = PrecomputedEnv.from_file(DAY_FILES[0])

            env_cache.reset()
            env_file.reset()

            actions = [2, 0, 1, 2, 0]  # sequence of actions
            for i, action in enumerate(actions):
                _, r_cache, t_cache, _, _ = env_cache.step(action)
                _, r_file, t_file, _, _ = env_file.step(action)

                assert r_cache == pytest.approx(r_file), (
                    f"Reward differs at step {i}: cache={r_cache}, file={r_file}"
                )
                assert t_cache == t_file, (
                    f"Terminated differs at step {i}: cache={t_cache}, file={t_file}"
                )

    def test_same_obs_at_each_step(self):
        """Observations should be identical at each step for both envs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz_from_file(tmpdir, DAY_FILES[0])

            env_cache = PrecomputedEnv.from_cache(npz_path)
            env_file = PrecomputedEnv.from_file(DAY_FILES[0])

            env_cache.reset()
            env_file.reset()

            for step in range(10):
                obs_cache, _, t_cache, _, _ = env_cache.step(step % 3)
                obs_file, _, t_file, _, _ = env_file.step(step % 3)

                np.testing.assert_array_almost_equal(
                    obs_cache, obs_file,
                    err_msg=f"Obs differs at step {step}"
                )
                if t_cache:
                    break


# ===========================================================================
# Test 6: Cache reusable across step_interval (acceptance criterion 5)
# ===========================================================================


class TestCacheReusableAcrossStepInterval:
    """Same .npz cache should work with different step_interval values."""

    def test_step_interval_1_and_5(self):
        """Same cache file should produce different episode lengths with different intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir, n=100)

            env_1 = PrecomputedEnv.from_cache(npz_path, step_interval=1)
            env_5 = PrecomputedEnv.from_cache(npz_path, step_interval=5)

            env_1.reset()
            env_5.reset()

            steps_1 = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env_1.step(1)
                steps_1 += 1

            steps_5 = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env_5.step(1)
                steps_5 += 1

            # step_interval=5 should have ~5x fewer steps
            assert steps_5 < steps_1, (
                f"step_interval=5 ({steps_5} steps) should have fewer steps "
                f"than step_interval=1 ({steps_1} steps)"
            )
            expected_ratio = steps_1 / steps_5
            assert 4.0 < expected_ratio < 6.0, (
                f"Expected ~5x ratio, got {expected_ratio}"
            )

    def test_step_interval_10(self):
        """Cache with 100 rows and step_interval=10 should give ~10 steps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir, n=100)
            env = PrecomputedEnv.from_cache(npz_path, step_interval=10)
            env.reset()

            steps = 0
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(1)
                steps += 1

            # 100 rows / 10 interval = 10 rows => 9 steps
            assert steps == 9, f"Expected 9 steps, got {steps}"


# ===========================================================================
# Test 7: Invalid .npz files
# ===========================================================================


class TestInvalidNpz:
    """from_cache() should handle invalid .npz files gracefully."""

    def test_missing_obs_key_raises(self):
        """An .npz missing the 'obs' key should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez(path, mid=np.array([1.0]), spread=np.array([1.0]))
            with pytest.raises((KeyError, ValueError)):
                PrecomputedEnv.from_cache(path)

    def test_missing_mid_key_raises(self):
        """An .npz missing the 'mid' key should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez(path, obs=make_obs(5), spread=np.array([1.0] * 5))
            with pytest.raises((KeyError, ValueError)):
                PrecomputedEnv.from_cache(path)

    def test_missing_spread_key_raises(self):
        """An .npz missing the 'spread' key should raise an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            np.savez(path, obs=make_obs(5), mid=np.array([1.0] * 5))
            with pytest.raises((KeyError, ValueError)):
                PrecomputedEnv.from_cache(path)

    def test_nonexistent_path_raises(self):
        """from_cache with a nonexistent path should raise an error."""
        with pytest.raises((FileNotFoundError, OSError)):
            PrecomputedEnv.from_cache("/nonexistent/path/test.npz")


# ===========================================================================
# Test 8: from_cache passes gymnasium check_env
# ===========================================================================


class TestFromCacheCheckEnv:
    """PrecomputedEnv created via from_cache should pass gymnasium's check_env."""

    def test_check_env_passes(self):
        """check_env() should pass on a cached env."""
        from gymnasium.utils.env_checker import check_env
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz(tmpdir, n=50)
            env = PrecomputedEnv.from_cache(npz_path)
            check_env(env, skip_render_check=True)


# ===========================================================================
# Test 9: from_cache with execution_cost matches from_file with execution_cost
# ===========================================================================


class TestFromCacheWithExecutionCost:
    """from_cache(execution_cost=True) should match from_file(execution_cost=True)."""

    def test_execution_cost_rewards_match(self):
        """With execution_cost=True, rewards from cache and file should match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path, _, _, _ = _create_npz_from_file(tmpdir, DAY_FILES[0])

            env_cache = PrecomputedEnv.from_cache(npz_path, execution_cost=True)
            env_file = PrecomputedEnv.from_file(DAY_FILES[0], execution_cost=True)

            env_cache.reset()
            env_file.reset()

            # Take actions that change position (generates execution cost)
            for action in [2, 0, 1, 2]:
                _, r_cache, _, _, _ = env_cache.step(action)
                _, r_file, _, _, _ = env_file.step(action)
                assert r_cache == pytest.approx(r_file), (
                    f"Execution cost reward mismatch: cache={r_cache}, file={r_file}"
                )
