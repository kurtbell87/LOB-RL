"""Tests for PrecomputedMultiDayEnv — precomputed multi-day gymnasium environment.

Spec: docs/precomputed-multi-day.md

These tests verify that:
- MultiDayEnv precomputes all days at construction (C++ calls happen once)
- step() and reset() use pure numpy (no C++ calls during training)
- Sequential and shuffle modes work correctly
- Seed determinism: same seed produces same day ordering
- Passes gymnasium.utils.env_checker.check_env()
- Passes SB3 DummyVecEnv compatibility
- Day files with < 2 BBO changes are skipped (with warning)
- Empty file_paths or all-invalid files raises ValueError
- info dict from reset() includes day_index
- steps_per_episode is accepted but ignored (full session always runs)
- session_config=None uses default_rth()
- Single day file works (cycles same day)
"""

import os

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from conftest import FIXTURE_DIR, DAY_FILES, run_episode


# ===========================================================================
# Import & Class Identity
# ===========================================================================


class TestImportAndIdentity:
    """MultiDayEnv should be importable and a gymnasium.Env subclass."""

    def test_import_from_module(self):
        """MultiDayEnv should be importable from lob_rl.multi_day_env."""
        from lob_rl.multi_day_env import MultiDayEnv
        assert MultiDayEnv is not None

    def test_is_gymnasium_env_subclass(self):
        """MultiDayEnv should be a subclass of gymnasium.Env."""
        from lob_rl.multi_day_env import MultiDayEnv
        assert issubclass(MultiDayEnv, gym.Env)

    def test_instance_is_gymnasium_env(self):
        """An instance of MultiDayEnv should be an instance of gymnasium.Env."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert isinstance(env, gym.Env)

    def test_import_from_package(self):
        """MultiDayEnv should be importable from the lob_rl package."""
        import lob_rl
        assert hasattr(lob_rl, "MultiDayEnv")


# ===========================================================================
# Spaces
# ===========================================================================


class TestSpaces:
    """observation_space and action_space should match PrecomputedEnv."""

    def test_observation_space_is_box(self):
        """observation_space should be a gymnasium.spaces.Box."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert isinstance(env.observation_space, spaces.Box)

    def test_observation_space_shape(self):
        """observation_space shape should be (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert env.observation_space.shape == (54,)

    def test_observation_space_dtype(self):
        """observation_space dtype should be float32."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert env.observation_space.dtype == np.float32

    def test_action_space_is_discrete(self):
        """action_space should be a gymnasium.spaces.Discrete."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_n_is_3(self):
        """action_space.n should be 3 (short / flat / long)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert env.action_space.n == 3


# ===========================================================================
# Constructor & Precomputation
# ===========================================================================


class TestConstructor:
    """Constructor should precompute all days and validate inputs."""

    def test_empty_file_paths_raises_value_error(self):
        """Empty file_paths list should raise ValueError."""
        from lob_rl.multi_day_env import MultiDayEnv
        with pytest.raises(ValueError):
            MultiDayEnv(file_paths=[])

    def test_single_file_works(self):
        """Single file should work — precomputes one day."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=[DAY_FILES[0]], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)
        assert isinstance(info, dict)

    def test_accepts_session_config(self):
        """session_config dict should be forwarded to precompute()."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES,
            session_config={
                "rth_open_ns": 48_600_000_000_000,
                "rth_close_ns": 72_000_000_000_000,
            },
            shuffle=False,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_session_config_none_uses_default(self):
        """session_config=None should use SessionConfig.default_rth()."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES,
            session_config=None,
            shuffle=False,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_steps_per_episode_accepted_but_ignored(self):
        """steps_per_episode should be accepted but ignored — full session runs.

        The precomputed env always runs the full session (all BBO snapshots).
        Passing steps_per_episode should not truncate the episode.
        """
        from lob_rl.multi_day_env import MultiDayEnv

        # Create two envs: one with steps_per_episode=10, one without
        env_with = MultiDayEnv(
            file_paths=[DAY_FILES[0]], steps_per_episode=10, shuffle=False,
        )
        env_without = MultiDayEnv(
            file_paths=[DAY_FILES[0]], steps_per_episode=0, shuffle=False,
        )

        env_with.reset()
        env_without.reset()

        steps_with = run_episode(env_with)
        steps_without = run_episode(env_without)

        # Both should run the full session — same step count
        assert steps_with == steps_without, (
            f"steps_per_episode should be ignored: "
            f"with={steps_with}, without={steps_without}"
        )

    def test_accepts_reward_mode(self):
        """reward_mode should be forwarded to inner PrecomputedEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES,
            reward_mode="pnl_delta_penalized",
            lambda_=0.001,
            shuffle=False,
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)
        assert isinstance(reward, (float, np.floating))

    def test_nonexistent_file_raises(self):
        """A file path that doesn't exist should raise immediately at construction."""
        from lob_rl.multi_day_env import MultiDayEnv
        with pytest.raises(Exception):
            MultiDayEnv(
                file_paths=["/nonexistent/fake_day.bin"],
                shuffle=False,
            )


# ===========================================================================
# Precomputation at Construction Time (C++ calls happen once)
# ===========================================================================


class TestPrecomputationAtConstruction:
    """Precomputation should happen at construction, not during step/reset.

    This is the key behavioral change: the new MultiDayEnv calls
    lob_rl_core.precompute() for each file at __init__ time, storing
    numpy arrays. step() and reset() should then use pure numpy.
    """

    def test_inner_env_is_precomputed_env(self):
        """After reset(), the inner env should be a PrecomputedEnv, not LOBGymEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        from lob_rl.precomputed_env import PrecomputedEnv

        env = MultiDayEnv(file_paths=DAY_FILES[:1], shuffle=False)
        env.reset()

        # The internal implementation should use PrecomputedEnv
        assert hasattr(env, "_inner_env") or hasattr(env, "_current_env"), (
            "MultiDayEnv should have an inner env attribute"
        )
        inner = getattr(env, "_inner_env", None) or getattr(env, "_current_env", None)
        assert isinstance(inner, PrecomputedEnv), (
            f"Inner env should be PrecomputedEnv, got {type(inner).__name__}"
        )

    def test_no_lob_gym_env_import_needed(self):
        """The new MultiDayEnv should NOT import or use LOBGymEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        import lob_rl.multi_day_env as module

        # LOBGymEnv should NOT be referenced in the module
        source = open(module.__file__).read()
        assert "LOBGymEnv" not in source, (
            "Precomputed MultiDayEnv should not import or reference LOBGymEnv"
        )


# ===========================================================================
# Skipping Invalid Day Files
# ===========================================================================


class TestSkipInvalidDayFiles:
    """Day files producing < 2 BBO snapshots should be skipped with warning."""

    def test_all_invalid_files_raises_value_error(self):
        """If ALL day files produce < 2 BBO snapshots, raise ValueError."""
        from lob_rl.multi_day_env import MultiDayEnv
        # Use the episode_200records.bin file which has epoch-era timestamps
        # that won't produce RTH data — should get < 2 BBO changes
        bad_file = os.path.join(FIXTURE_DIR, "episode_200records.bin")
        with pytest.raises(ValueError):
            MultiDayEnv(file_paths=[bad_file], shuffle=False)

    def test_some_invalid_files_skipped(self):
        """Files with < 2 BBO snapshots should be skipped, rest used."""
        from lob_rl.multi_day_env import MultiDayEnv
        bad_file = os.path.join(FIXTURE_DIR, "episode_200records.bin")
        # Mix valid and invalid files
        mixed = [bad_file] + DAY_FILES[:2]
        env = MultiDayEnv(file_paths=mixed, shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)


# ===========================================================================
# reset() Behavior
# ===========================================================================


class TestReset:
    """reset() should return valid obs and advance to next day."""

    def test_reset_returns_tuple(self):
        """reset() should return (obs, info) tuple."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_is_ndarray(self):
        """reset() first element should be a numpy ndarray."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_shape(self):
        """reset() observation shape should be (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_reset_obs_dtype(self):
        """reset() observation dtype should be float32."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs.dtype == np.float32

    def test_reset_info_is_dict(self):
        """reset() second element should be a dict."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_info_has_day_index(self):
        """reset() info dict should include 'day_index' key."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert "day_index" in info, (
            f"info dict should contain 'day_index', got keys: {list(info.keys())}"
        )

    def test_reset_day_index_is_int(self):
        """reset() info['day_index'] should be an int."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert isinstance(info["day_index"], int), (
            f"day_index should be int, got {type(info['day_index']).__name__}"
        )

    def test_reset_day_index_valid_range(self):
        """reset() info['day_index'] should be a valid index into the file list."""
        from lob_rl.multi_day_env import MultiDayEnv
        n = len(DAY_FILES)
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert 0 <= info["day_index"] < n, (
            f"day_index {info['day_index']} out of range [0, {n})"
        )

    def test_reset_obs_in_observation_space(self):
        """reset() observation should be within observation_space."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

    def test_reset_position_is_zero(self):
        """reset() should return obs with position=0 (index 53)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs[53] == pytest.approx(0.0)

    def test_reset_accepts_seed_kwarg(self):
        """reset(seed=42) should be accepted without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=True)
        obs, info = env.reset(seed=42)
        assert obs.shape == (54,)

    def test_reset_accepts_options_kwarg(self):
        """reset(options={}) should be accepted without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset(options={})
        assert obs.shape == (54,)

    def test_reset_no_nan(self):
        """Observation after reset should contain no NaN values."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), f"NaN in reset obs: {obs}"

    def test_reset_no_inf(self):
        """Observation after reset should contain no infinite values."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, _ = env.reset()
        assert not np.any(np.isinf(obs)), f"Inf in reset obs: {obs}"


# ===========================================================================
# step() Delegation
# ===========================================================================


class TestStep:
    """step() should delegate to the current inner PrecomputedEnv."""

    def test_step_returns_five_tuple(self):
        """step() should return (obs, reward, terminated, truncated, info)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        result = env.step(1)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_is_ndarray(self):
        """step() observation should be a numpy ndarray."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert isinstance(obs, np.ndarray)

    def test_step_obs_shape(self):
        """step() observation shape should be (54,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (54,)

    def test_step_obs_dtype(self):
        """step() observation dtype should be float32."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.dtype == np.float32

    def test_step_reward_is_float(self):
        """step() reward should be a float."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert isinstance(reward, (float, np.floating))

    def test_step_terminated_is_bool(self):
        """step() terminated should be a bool."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        _, _, terminated, _, _ = env.step(1)
        assert isinstance(terminated, (bool, np.bool_))

    def test_step_truncated_is_bool(self):
        """step() truncated should be a bool."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        _, _, _, truncated, _ = env.step(1)
        assert isinstance(truncated, (bool, np.bool_))

    def test_step_truncated_is_false(self):
        """step() truncated should always be False."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        _, _, _, truncated, _ = env.step(1)
        assert truncated is False

    def test_step_info_is_dict(self):
        """step() info should be a dict."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        _, _, _, _, info = env.step(1)
        assert isinstance(info, dict)

    def test_step_obs_in_observation_space(self):
        """step() observation should be within observation_space."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert env.observation_space.contains(obs)

    def test_action_0_short_position(self):
        """Action 0 should set position to -1."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs[53] == pytest.approx(-1.0)

    def test_action_1_flat_position(self):
        """Action 1 should set position to 0."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs[53] == pytest.approx(0.0)

    def test_action_2_long_position(self):
        """Action 2 should set position to +1."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(2)
        assert obs[53] == pytest.approx(1.0)

    def test_full_episode_no_crash(self):
        """Complete a full episode without crashing."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        steps = run_episode(env)
        assert steps > 0

    def test_reward_is_finite_during_episode(self):
        """Rewards during episode should be finite (not NaN or inf)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            _, reward, terminated, _, _ = env.step(steps % 3)
            assert np.isfinite(reward), f"Non-finite reward {reward} at step {steps}"
            steps += 1

    def test_obs_no_nan_during_episode(self):
        """Observations during episode should contain no NaN."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            obs, _, terminated, _, _ = env.step(steps % 3)
            assert not np.any(np.isnan(obs)), f"NaN at step {steps}: {obs}"
            steps += 1


# ===========================================================================
# Sequential Cycling (shuffle=False)
# ===========================================================================


class TestSequentialCycling:
    """With shuffle=False, days should be visited in order, wrapping around."""

    def test_advances_to_next_day_on_reset(self):
        """Each reset() should advance to the next day's file.

        We verify via the day_index in the info dict.
        """
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], shuffle=False,
        )

        day_indices = []
        for _ in range(3):
            _, info = env.reset()
            day_indices.append(info["day_index"])
            run_episode(env)

        # Sequential: should visit days 0, 1, 2 in order
        assert day_indices == [0, 1, 2], (
            f"Expected sequential [0, 1, 2], got {day_indices}"
        )

    def test_wraps_around_after_all_days(self):
        """After visiting all days, should wrap back to the first."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 3
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )

        # Visit all days once
        first_obs, first_info = env.reset()
        run_episode(env)

        for _ in range(n_days - 1):
            env.reset()
            run_episode(env)

        # This should wrap back to day 0
        wrap_obs, wrap_info = env.reset()
        assert wrap_info["day_index"] == first_info["day_index"], (
            f"Wrap-around day_index should match first: "
            f"first={first_info['day_index']}, wrap={wrap_info['day_index']}"
        )

        np.testing.assert_array_almost_equal(
            first_obs, wrap_obs,
            err_msg="Observation after wrap-around should match first day",
        )

    def test_sequential_order_is_deterministic(self):
        """Two independent sequential envs should visit days in same order."""
        from lob_rl.multi_day_env import MultiDayEnv

        def collect_day_indices(files, n_resets):
            env = MultiDayEnv(
                file_paths=files, shuffle=False,
            )
            indices = []
            for _ in range(n_resets):
                _, info = env.reset()
                indices.append(info["day_index"])
                run_episode(env)
            return indices

        a = collect_day_indices(DAY_FILES[:3], 6)
        b = collect_day_indices(DAY_FILES[:3], 6)
        assert a == b, f"Sequential order should be deterministic: {a} vs {b}"

    def test_cycles_through_all_files(self):
        """Over N resets (for N files), each file should be visited exactly once."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )

        day_indices = []
        for _ in range(n_days):
            _, info = env.reset()
            day_indices.append(info["day_index"])
            run_episode(env)

        # Should visit each day exactly once
        assert sorted(day_indices) == list(range(n_days)), (
            f"Expected all days visited: {sorted(day_indices)} vs {list(range(n_days))}"
        )

    def test_single_file_cycles_to_same_data(self):
        """With a single file, every reset should replay the same data."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False,
        )

        obs1, info1 = env.reset()
        run_episode(env)
        obs2, info2 = env.reset()

        np.testing.assert_array_almost_equal(
            obs1, obs2,
            err_msg="Single-file env should produce same obs on each reset",
        )

    def test_day_index_sequential_pattern(self):
        """Sequential day_index should follow 0, 1, 2, 0, 1, 2, ..."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 3
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )

        day_indices = []
        for _ in range(n_days * 2):
            _, info = env.reset()
            day_indices.append(info["day_index"])
            run_episode(env)

        expected = [i % n_days for i in range(n_days * 2)]
        assert day_indices == expected, (
            f"Expected cycling pattern {expected}, got {day_indices}"
        )


# ===========================================================================
# Shuffle Mode (shuffle=True)
# ===========================================================================


class TestShuffleMode:
    """With shuffle=True, day order should be randomized."""

    def test_shuffle_visits_all_days_in_epoch(self):
        """In shuffle mode, all days should still be visited within an epoch."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            shuffle=True,
            seed=42,
        )

        day_indices = []
        for _ in range(n_days):
            _, info = env.reset()
            day_indices.append(info["day_index"])
            run_episode(env)

        # All N days should be visited (though in shuffled order)
        assert sorted(day_indices) == list(range(n_days)), (
            f"Expected all days visited: sorted={sorted(day_indices)}"
        )

    def test_shuffle_order_differs_from_sequential(self):
        """Shuffled order should (with high probability) differ from sequential order.

        We test with enough days that the probability of accidental match is tiny.
        """
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        # Sequential
        seq_env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )
        seq_indices = []
        for _ in range(n_days):
            _, info = seq_env.reset()
            seq_indices.append(info["day_index"])
            run_episode(seq_env)

        # Shuffled (seed chosen to very likely differ from 0,1,2,3,4)
        shuf_env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            shuffle=True,
            seed=12345,
        )
        shuf_indices = []
        for _ in range(n_days):
            _, info = shuf_env.reset()
            shuf_indices.append(info["day_index"])
            run_episode(shuf_env)

        # At least one position should differ
        assert seq_indices != shuf_indices, (
            "Shuffled order matched sequential order exactly — "
            "shuffle may not be working"
        )

    def test_reshuffle_at_epoch_boundary(self):
        """At the start of each epoch, day order should be re-shuffled.

        We run 2 full epochs and check that the order differs between them.
        With 5 days, probability of identical shuffles is 1/120 = <1%.
        """
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            shuffle=True,
            seed=99,
        )

        # Epoch 1
        epoch1_indices = []
        for _ in range(n_days):
            _, info = env.reset()
            epoch1_indices.append(info["day_index"])
            run_episode(env)

        # Epoch 2
        epoch2_indices = []
        for _ in range(n_days):
            _, info = env.reset()
            epoch2_indices.append(info["day_index"])
            run_episode(env)

        # Orders should differ (with high probability)
        assert epoch1_indices != epoch2_indices, (
            "Epoch 1 and Epoch 2 had identical day orders — "
            "re-shuffle at epoch boundary may not be working"
        )


# ===========================================================================
# Seed Determinism
# ===========================================================================


class TestSeedDeterminism:
    """Seeded shuffle should produce deterministic, reproducible day orderings."""

    def test_same_seed_same_order(self):
        """Two envs with the same seed should visit days in the same order."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        def collect_indices(seed):
            env = MultiDayEnv(
                file_paths=DAY_FILES[:n_days],
                shuffle=True,
                seed=seed,
            )
            indices = []
            for _ in range(n_days):
                _, info = env.reset()
                indices.append(info["day_index"])
                run_episode(env)
            return indices

        a = collect_indices(42)
        b = collect_indices(42)
        assert a == b, f"Same-seed day orderings differ: {a} vs {b}"

    def test_different_seeds_different_order(self):
        """Two envs with different seeds should (very likely) have different orders."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        def collect_indices(seed):
            env = MultiDayEnv(
                file_paths=DAY_FILES[:n_days],
                shuffle=True,
                seed=seed,
            )
            indices = []
            for _ in range(n_days):
                _, info = env.reset()
                indices.append(info["day_index"])
                run_episode(env)
            return indices

        a = collect_indices(42)
        b = collect_indices(9999)
        assert a != b, "Different seeds produced identical day orderings"

    def test_reset_seed_overrides_constructor_seed(self):
        """reset(seed=X) should seed the shuffle RNG for reproducibility."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        # Create env with one seed, but override with reset(seed=...)
        env1 = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            shuffle=True,
            seed=1,
        )
        env2 = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            shuffle=True,
            seed=2,  # different constructor seed
        )

        # Both reset with the same seed — should produce same day
        _, info1 = env1.reset(seed=777)
        _, info2 = env2.reset(seed=777)
        assert info1["day_index"] == info2["day_index"], (
            f"reset(seed=X) should override constructor seed: "
            f"env1 day={info1['day_index']}, env2 day={info2['day_index']}"
        )


# ===========================================================================
# Multi-Episode Lifecycle
# ===========================================================================


class TestMultiEpisodeLifecycle:
    """Running many episodes across multiple days should work correctly."""

    def test_run_two_full_epochs(self):
        """Two full epochs (N resets each) should complete without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 3
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )

        for epoch in range(2):
            for day in range(n_days):
                obs, info = env.reset()
                assert obs.shape == (54,), f"Epoch {epoch}, day {day}: bad obs shape"
                assert obs.dtype == np.float32
                assert isinstance(info, dict)
                assert "day_index" in info
                steps = run_episode(env)
                assert steps > 0, f"Epoch {epoch}, day {day}: zero steps"

    def test_position_resets_each_episode(self):
        """Position (obs[53]) should be 0 at the start of each new episode/day."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], shuffle=False,
        )

        for i in range(6):  # 2 full epochs
            obs, _ = env.reset()
            assert obs[53] == pytest.approx(0.0), (
                f"Position not 0 at start of episode {i}: {obs[53]}"
            )
            # Take a position-changing action, then let episode end
            env.step(2)  # go long
            run_episode(env)

    def test_action_space_sample_works_across_days(self):
        """Sampling from action_space and using in step() should work across days."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], shuffle=False,
        )

        for _ in range(3):
            env.reset()
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == (54,)
                if terminated:
                    break

    def test_day_index_tracks_across_episodes(self):
        """day_index in info should accurately track which day is active."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 3
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], shuffle=False,
        )

        for expected_day in range(n_days):
            _, info = env.reset()
            assert info["day_index"] == expected_day, (
                f"Expected day_index={expected_day}, got {info['day_index']}"
            )
            run_episode(env)


# ===========================================================================
# Full Session Runs (steps_per_episode ignored)
# ===========================================================================


class TestFullSessionRuns:
    """Precomputed env should always run full sessions, ignoring steps_per_episode."""

    def test_episode_runs_full_session(self):
        """Episode should run until all precomputed BBO snapshots are exhausted."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=[DAY_FILES[0]], shuffle=False)
        env.reset()
        steps = run_episode(env)
        # The day files have many BBO snapshots — should be more than any
        # typical steps_per_episode value like 10 or 50
        assert steps > 1, f"Episode should have multiple steps, got {steps}"

    def test_episode_terminates_not_truncates(self):
        """Episode should end with terminated=True, truncated=False."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=[DAY_FILES[0]], shuffle=False)
        env.reset()
        terminated = False
        truncated = False
        while not terminated:
            _, _, terminated, truncated, _ = env.step(1)
        assert terminated is True
        assert truncated is False


# ===========================================================================
# gymnasium.utils.env_checker.check_env()
# ===========================================================================


class TestCheckEnv:
    """gymnasium.utils.env_checker.check_env() should pass for MultiDayEnv."""

    def test_check_env_sequential(self):
        """check_env() should pass for sequential MultiDayEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], shuffle=False,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_shuffled(self):
        """check_env() should pass for shuffled MultiDayEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3],
            shuffle=True,
            seed=42,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_single_file(self):
        """check_env() should pass for single-file MultiDayEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False,
        )
        check_env(env, skip_render_check=True)


# ===========================================================================
# DummyVecEnv Compatibility (SB3)
# ===========================================================================


class TestDummyVecEnvCompatibility:
    """MultiDayEnv should work inside SB3's DummyVecEnv."""

    def test_wrappable_in_dummy_vec_env(self):
        """DummyVecEnv([lambda: MultiDayEnv(...)]) should work."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from lob_rl.multi_day_env import MultiDayEnv

        vec_env = DummyVecEnv([
            lambda: MultiDayEnv(
                file_paths=DAY_FILES[:3], shuffle=False,
            )
        ])
        obs = vec_env.reset()
        assert obs.shape == (1, 54)

    def test_dummy_vec_env_step(self):
        """step() through DummyVecEnv should return valid results."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from lob_rl.multi_day_env import MultiDayEnv

        vec_env = DummyVecEnv([
            lambda: MultiDayEnv(
                file_paths=DAY_FILES[:3], shuffle=False,
            )
        ])
        vec_env.reset()
        obs, rewards, dones, infos = vec_env.step([1])
        assert obs.shape == (1, 54)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)


# ===========================================================================
# Backward Compatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Precomputed MultiDayEnv should be backward-compatible with old constructor args."""

    def test_all_constructor_args_accepted(self):
        """Constructor should accept all args from the old MultiDayEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:2],
            session_config=None,
            steps_per_episode=50,
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_constructor_with_all_optional_args(self):
        """Constructor should work with all optional args specified."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:2],
            session_config={
                "rth_open_ns": 48_600_000_000_000,
                "rth_close_ns": 72_000_000_000_000,
                "warmup_messages": -1,
            },
            steps_per_episode=100,
            reward_mode="pnl_delta_penalized",
            lambda_=0.01,
            shuffle=True,
            seed=42,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)
        assert "day_index" in info
