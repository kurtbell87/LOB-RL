"""Tests for Step 4: Multi-Day Training Environment.

Spec: docs/step4-multi-day-env.md

These tests verify that:
- MultiDayEnv is a valid gymnasium.Env subclass
- It cycles through multiple data files (one per day)
- Sequential mode (shuffle=False) visits files in order, wrapping around
- Shuffle mode randomizes day order and re-shuffles at epoch boundaries
- Seeded shuffle produces deterministic day ordering
- step() delegates to the current inner LOBGymEnv
- Empty file_paths raises ValueError
- Single file works like a regular LOBGymEnv
- observation_space and action_space match LOBGymEnv (Box(44,), Discrete(3))
- check_env() passes
"""

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
    """observation_space and action_space should match LOBGymEnv."""

    def test_observation_space_is_box(self):
        """observation_space should be a gymnasium.spaces.Box."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert isinstance(env.observation_space, spaces.Box)

    def test_observation_space_shape(self):
        """observation_space shape should be (44,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        assert env.observation_space.shape == (44,)

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
# Constructor & Configuration
# ===========================================================================


class TestConstructor:
    """Constructor should validate inputs and forward config to inner env."""

    def test_empty_file_paths_raises_value_error(self):
        """Empty file_paths list should raise ValueError."""
        from lob_rl.multi_day_env import MultiDayEnv
        with pytest.raises(ValueError):
            MultiDayEnv(file_paths=[])

    def test_single_file_works(self):
        """Single file should work like a regular LOBGymEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=[DAY_FILES[0]], shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (44,)
        assert isinstance(info, dict)

    def test_accepts_session_config(self):
        """session_config dict should be forwarded to inner LOBGymEnv."""
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
        assert obs.shape == (44,)

    def test_accepts_steps_per_episode(self):
        """steps_per_episode should be forwarded to inner LOBGymEnv."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES,
            steps_per_episode=10,
            shuffle=False,
        )
        env.reset()
        steps = run_episode(env)
        assert steps <= 15, f"Expected ~10 steps, got {steps}"

    def test_accepts_reward_mode(self):
        """reward_mode should be forwarded to inner LOBGymEnv."""
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
        """A file path that doesn't exist should raise immediately."""
        from lob_rl.multi_day_env import MultiDayEnv
        with pytest.raises(Exception):
            env = MultiDayEnv(
                file_paths=["/nonexistent/fake_day.bin"],
                shuffle=False,
            )
            env.reset()


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
        """reset() observation shape should be (44,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (44,)

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

    def test_reset_obs_in_observation_space(self):
        """reset() observation should be within observation_space."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)

    def test_reset_position_is_zero(self):
        """reset() should return obs with position=0 (index 43)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs[43] == pytest.approx(0.0)

    def test_reset_accepts_seed_kwarg(self):
        """reset(seed=42) should be accepted without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=True)
        obs, info = env.reset(seed=42)
        assert obs.shape == (44,)

    def test_reset_accepts_options_kwarg(self):
        """reset(options={}) should be accepted without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset(options={})
        assert obs.shape == (44,)

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
    """step() should delegate to the current inner LOBGymEnv."""

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
        """step() observation shape should be (44,)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (44,)

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
        assert obs[43] == pytest.approx(-1.0)

    def test_action_1_flat_position(self):
        """Action 1 should set position to 0."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs[43] == pytest.approx(0.0)

    def test_action_2_long_position(self):
        """Action 2 should set position to +1."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        env.reset()
        obs, _, _, _, _ = env.step(2)
        assert obs[43] == pytest.approx(1.0)

    def test_full_episode_no_crash(self):
        """Complete a full episode without crashing."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, steps_per_episode=10, shuffle=False,
        )
        env.reset()
        steps = run_episode(env)
        assert steps > 0

    def test_reward_is_finite_during_episode(self):
        """Rewards during episode should be finite (not NaN or inf)."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, steps_per_episode=10, shuffle=False,
        )
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 50:
            _, reward, terminated, _, _ = env.step(steps % 3)
            assert np.isfinite(reward), f"Non-finite reward {reward} at step {steps}"
            steps += 1

    def test_obs_no_nan_during_episode(self):
        """Observations during episode should contain no NaN."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, steps_per_episode=10, shuffle=False,
        )
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 50:
            obs, _, terminated, _, _ = env.step(steps % 3)
            assert not np.any(np.isnan(obs)), f"NaN at step {steps}: {obs}"
            steps += 1


# ===========================================================================
# Sequential Cycling (shuffle=False)
# ===========================================================================


class TestSequentialCycling:
    """With shuffle=False, days should be visited in order, wrapping around."""

    def test_advances_to_next_day_on_reset(self):
        """Each reset() should advance to the next day's file."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3], steps_per_episode=10, shuffle=False,
        )

        # Collect first obs from each reset to detect day changes.
        # Each day has a distinct price range, so obs should differ.
        observations = []
        for _ in range(3):
            obs, _ = env.reset()
            observations.append(obs.copy())
            run_episode(env)

        # At least two of the three resets should yield different observations.
        # (If all three obs were identical, the env isn't switching days.)
        all_same = all(
            np.allclose(observations[0], observations[i])
            for i in range(1, len(observations))
        )
        assert not all_same, (
            "All resets returned identical observations — "
            "env is not cycling through different day files"
        )

    def test_wraps_around_after_all_days(self):
        """After visiting all days, should wrap back to the first."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 3
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], steps_per_episode=10, shuffle=False,
        )

        # Visit all days once + one more reset to wrap around
        first_obs, _ = env.reset()
        run_episode(env)

        for _ in range(n_days - 1):
            env.reset()
            run_episode(env)

        # This should wrap back to day 0
        wrap_obs, _ = env.reset()

        np.testing.assert_array_almost_equal(
            first_obs, wrap_obs,
            err_msg="Observation after wrap-around should match first day",
        )

    def test_sequential_order_is_deterministic(self):
        """Two independent sequential envs should visit days in same order."""
        from lob_rl.multi_day_env import MultiDayEnv

        def collect_first_obs(files, n_resets):
            env = MultiDayEnv(
                file_paths=files, steps_per_episode=10, shuffle=False,
            )
            obs_list = []
            for _ in range(n_resets):
                obs, _ = env.reset()
                obs_list.append(obs.copy())
                run_episode(env)
            return obs_list

        obs_a = collect_first_obs(DAY_FILES[:3], 6)
        obs_b = collect_first_obs(DAY_FILES[:3], 6)

        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_array_almost_equal(
                a, b, err_msg=f"Obs mismatch at reset {i}",
            )

    def test_cycles_through_all_files(self):
        """Over N resets (for N files), each file should be visited exactly once."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5
        env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days], steps_per_episode=10, shuffle=False,
        )

        observations = []
        for _ in range(n_days):
            obs, _ = env.reset()
            observations.append(obs.copy())
            run_episode(env)

        # All N observations should be pairwise distinct
        for i in range(n_days):
            for j in range(i + 1, n_days):
                assert not np.allclose(observations[i], observations[j]), (
                    f"Days {i} and {j} produced identical observations — "
                    "not all files were visited"
                )

    def test_single_file_cycles_to_same_data(self):
        """With a single file, every reset should replay the same data."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=[DAY_FILES[0]], steps_per_episode=10, shuffle=False,
        )

        obs1, _ = env.reset()
        run_episode(env)
        obs2, _ = env.reset()

        np.testing.assert_array_almost_equal(
            obs1, obs2,
            err_msg="Single-file env should produce same obs on each reset",
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
            steps_per_episode=10,
            shuffle=True,
            seed=42,
        )

        observations = []
        for _ in range(n_days):
            obs, _ = env.reset()
            observations.append(obs.copy())
            run_episode(env)

        # All N observations should be pairwise distinct (all files visited)
        for i in range(n_days):
            for j in range(i + 1, n_days):
                assert not np.allclose(observations[i], observations[j]), (
                    f"Shuffle epoch: days {i} and {j} produced identical obs — "
                    "not all files were visited"
                )

    def test_shuffle_order_differs_from_sequential(self):
        """Shuffled order should (with high probability) differ from sequential order.

        We test with enough days that the probability of accidental match is tiny.
        """
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        # Sequential
        seq_env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            steps_per_episode=10,
            shuffle=False,
        )
        seq_obs = []
        for _ in range(n_days):
            obs, _ = seq_env.reset()
            seq_obs.append(obs.copy())
            run_episode(seq_env)

        # Shuffled (seed chosen to very likely differ from 0,1,2,3,4)
        shuf_env = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            steps_per_episode=10,
            shuffle=True,
            seed=12345,
        )
        shuf_obs = []
        for _ in range(n_days):
            obs, _ = shuf_env.reset()
            shuf_obs.append(obs.copy())
            run_episode(shuf_env)

        # At least one position should differ
        any_differ = any(
            not np.allclose(seq_obs[i], shuf_obs[i])
            for i in range(n_days)
        )
        assert any_differ, (
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
            steps_per_episode=10,
            shuffle=True,
            seed=99,
        )

        # Epoch 1
        epoch1_obs = []
        for _ in range(n_days):
            obs, _ = env.reset()
            epoch1_obs.append(obs.copy())
            run_episode(env)

        # Epoch 2
        epoch2_obs = []
        for _ in range(n_days):
            obs, _ = env.reset()
            epoch2_obs.append(obs.copy())
            run_episode(env)

        # Orders should differ in at least one position
        any_differ = any(
            not np.allclose(epoch1_obs[i], epoch2_obs[i])
            for i in range(n_days)
        )
        assert any_differ, (
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

        def collect_obs(seed):
            env = MultiDayEnv(
                file_paths=DAY_FILES[:n_days],
                steps_per_episode=10,
                shuffle=True,
                seed=seed,
            )
            obs_list = []
            for _ in range(n_days):
                obs, _ = env.reset()
                obs_list.append(obs.copy())
                run_episode(env)
            return obs_list

        obs_a = collect_obs(42)
        obs_b = collect_obs(42)

        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_array_almost_equal(
                a, b, err_msg=f"Same-seed obs differ at reset {i}",
            )

    def test_different_seeds_different_order(self):
        """Two envs with different seeds should (very likely) have different orders."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        def collect_obs(seed):
            env = MultiDayEnv(
                file_paths=DAY_FILES[:n_days],
                steps_per_episode=10,
                shuffle=True,
                seed=seed,
            )
            obs_list = []
            for _ in range(n_days):
                obs, _ = env.reset()
                obs_list.append(obs.copy())
                run_episode(env)
            return obs_list

        obs_a = collect_obs(42)
        obs_b = collect_obs(9999)

        any_differ = any(
            not np.allclose(obs_a[i], obs_b[i])
            for i in range(n_days)
        )
        assert any_differ, "Different seeds produced identical day orderings"

    def test_reset_seed_overrides_constructor_seed(self):
        """reset(seed=X) should seed the shuffle RNG for reproducibility."""
        from lob_rl.multi_day_env import MultiDayEnv
        n_days = 5

        # Create env with one seed, but override with reset(seed=...)
        env1 = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            steps_per_episode=10,
            shuffle=True,
            seed=1,
        )
        env2 = MultiDayEnv(
            file_paths=DAY_FILES[:n_days],
            steps_per_episode=10,
            shuffle=True,
            seed=2,  # different constructor seed
        )

        # Both reset with the same seed — should produce same ordering
        obs1, _ = env1.reset(seed=777)
        obs2, _ = env2.reset(seed=777)
        np.testing.assert_array_almost_equal(
            obs1, obs2,
            err_msg="reset(seed=X) should override constructor seed",
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
            file_paths=DAY_FILES[:n_days],
            steps_per_episode=10,
            shuffle=False,
        )

        for epoch in range(2):
            for day in range(n_days):
                obs, info = env.reset()
                assert obs.shape == (44,), f"Epoch {epoch}, day {day}: bad obs shape"
                assert obs.dtype == np.float32
                assert isinstance(info, dict)
                steps = run_episode(env)
                assert steps > 0, f"Epoch {epoch}, day {day}: zero steps"

    def test_position_resets_each_episode(self):
        """Position (obs[43]) should be 0 at the start of each new episode/day."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3],
            steps_per_episode=10,
            shuffle=False,
        )

        for i in range(6):  # 2 full epochs
            obs, _ = env.reset()
            assert obs[43] == pytest.approx(0.0), (
                f"Position not 0 at start of episode {i}: {obs[43]}"
            )
            # Take a position-changing action, then let episode end
            env.step(2)  # go long
            run_episode(env)

    def test_action_space_sample_works_across_days(self):
        """Sampling from action_space and using in step() should work across days."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3],
            steps_per_episode=10,
            shuffle=False,
        )

        for _ in range(3):
            env.reset()
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert obs.shape == (44,)
                if terminated:
                    break


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
            file_paths=DAY_FILES[:3],
            steps_per_episode=10,
            shuffle=False,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_shuffled(self):
        """check_env() should pass for shuffled MultiDayEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES[:3],
            steps_per_episode=10,
            shuffle=True,
            seed=42,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_single_file(self):
        """check_env() should pass for single-file MultiDayEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=[DAY_FILES[0]],
            steps_per_episode=10,
            shuffle=False,
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
                file_paths=DAY_FILES[:3],
                steps_per_episode=10,
                shuffle=False,
            )
        ])
        obs = vec_env.reset()
        assert obs.shape == (1, 44)

    def test_dummy_vec_env_step(self):
        """step() through DummyVecEnv should return valid results."""
        from stable_baselines3.common.vec_env import DummyVecEnv
        from lob_rl.multi_day_env import MultiDayEnv

        vec_env = DummyVecEnv([
            lambda: MultiDayEnv(
                file_paths=DAY_FILES[:3],
                steps_per_episode=10,
                shuffle=False,
            )
        ])
        vec_env.reset()
        obs, rewards, dones, infos = vec_env.step([1])
        assert obs.shape == (1, 44)
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
