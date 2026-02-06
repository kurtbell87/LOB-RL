"""Tests for Step 3c: Gym Wrapper (gymnasium.Env subclass).

Spec: docs/step3c-gym-wrapper.md

These tests verify that:
- LOBGymEnv is a valid gymnasium.Env subclass
- observation_space is Box(shape=(44,), dtype=float32)
- action_space is Discrete(3)
- reset() returns (ndarray, dict) tuple
- step() returns 5-tuple (obs, reward, terminated, truncated, info)
- Observation is numpy array of shape (44,) and dtype float32
- check_env() passes
- All constructor configurations work (synthetic, file, session, reward mode)
- Multiple episodes (reset after done) work
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.gym_env import LOBGymEnv
from conftest import EPISODE_FILE, SESSION_FILE


# ===========================================================================
# Import & Class Identity
# ===========================================================================


class TestImportAndIdentity:
    """LOBGymEnv should be importable and a gymnasium.Env subclass."""

    def test_import_lob_gym_env(self):
        """LOBGymEnv should be importable from lob_rl.gym_env."""
        assert LOBGymEnv is not None

    def test_is_gymnasium_env_subclass(self):
        """LOBGymEnv should be a subclass of gymnasium.Env."""
        assert issubclass(LOBGymEnv, gym.Env)

    def test_instance_is_gymnasium_env(self):
        """An instance of LOBGymEnv should be an instance of gymnasium.Env."""
        env = LOBGymEnv()
        assert isinstance(env, gym.Env)


# ===========================================================================
# Default Construction (SyntheticSource)
# ===========================================================================


class TestDefaultConstruction:
    """LOBGymEnv() with no args should create a valid env with SyntheticSource."""

    def test_default_constructor_no_args(self):
        """LOBGymEnv() should be constructible with no arguments."""
        env = LOBGymEnv()
        assert env is not None

    def test_default_has_observation_space(self):
        """Default env should have an observation_space attribute."""
        env = LOBGymEnv()
        assert hasattr(env, "observation_space")

    def test_default_has_action_space(self):
        """Default env should have an action_space attribute."""
        env = LOBGymEnv()
        assert hasattr(env, "action_space")


# ===========================================================================
# Observation Space
# ===========================================================================


class TestObservationSpace:
    """observation_space should be Box(shape=(44,), dtype=float32)."""

    def test_observation_space_is_box(self):
        """observation_space should be a gymnasium.spaces.Box."""
        env = LOBGymEnv()
        assert isinstance(env.observation_space, spaces.Box)

    def test_observation_space_shape(self):
        """observation_space shape should be (44,)."""
        env = LOBGymEnv()
        assert env.observation_space.shape == (44,)

    def test_observation_space_dtype(self):
        """observation_space dtype should be float32."""
        env = LOBGymEnv()
        assert env.observation_space.dtype == np.float32

    def test_observation_space_low_is_neg_inf(self):
        """observation_space low bounds should be -inf."""
        env = LOBGymEnv()
        assert np.all(env.observation_space.low == -np.inf)

    def test_observation_space_high_is_pos_inf(self):
        """observation_space high bounds should be +inf."""
        env = LOBGymEnv()
        assert np.all(env.observation_space.high == np.inf)


# ===========================================================================
# Action Space
# ===========================================================================


class TestActionSpace:
    """action_space should be Discrete(3)."""

    def test_action_space_is_discrete(self):
        """action_space should be a gymnasium.spaces.Discrete."""
        env = LOBGymEnv()
        assert isinstance(env.action_space, spaces.Discrete)

    def test_action_space_n_is_3(self):
        """action_space.n should be 3."""
        env = LOBGymEnv()
        assert env.action_space.n == 3


# ===========================================================================
# reset() Return Format
# ===========================================================================


class TestResetReturnFormat:
    """reset() should return (ndarray, dict) per gymnasium API."""

    def test_reset_returns_tuple(self):
        """reset() should return a tuple."""
        env = LOBGymEnv()
        result = env.reset()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"

    def test_reset_returns_two_elements(self):
        """reset() should return a 2-tuple."""
        env = LOBGymEnv()
        result = env.reset()
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    def test_reset_obs_is_ndarray(self):
        """reset() first element should be a numpy ndarray."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"

    def test_reset_obs_shape(self):
        """reset() observation shape should be (44,)."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert obs.shape == (44,), f"Expected shape (44,), got {obs.shape}"

    def test_reset_obs_dtype(self):
        """reset() observation dtype should be float32."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_reset_info_is_dict(self):
        """reset() second element should be a dict."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"

    def test_reset_info_is_empty(self):
        """reset() info should be an empty dict."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert info == {}, f"Expected empty dict, got {info}"

    def test_reset_accepts_seed_kwarg(self):
        """reset(seed=42) should be accepted without error."""
        env = LOBGymEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (44,)

    def test_reset_accepts_options_kwarg(self):
        """reset(options={}) should be accepted without error."""
        env = LOBGymEnv()
        obs, info = env.reset(options={})
        assert obs.shape == (44,)

    def test_reset_accepts_seed_and_options(self):
        """reset(seed=42, options={}) should be accepted without error."""
        env = LOBGymEnv()
        obs, info = env.reset(seed=42, options={})
        assert obs.shape == (44,)

    def test_reset_obs_in_observation_space(self):
        """reset() observation should be within the observation space."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert env.observation_space.contains(obs), (
            f"Observation {obs} not in observation_space"
        )

    def test_reset_initial_position_is_zero(self):
        """reset() should return obs with position=0 (index 43)."""
        env = LOBGymEnv()
        obs, info = env.reset()
        assert obs[43] == pytest.approx(0.0), (
            f"Initial position should be 0, got {obs[43]}"
        )


# ===========================================================================
# step() Return Format
# ===========================================================================


class TestStepReturnFormat:
    """step() should return (obs, reward, terminated, truncated, info) 5-tuple."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        self.env = LOBGymEnv()
        self.env.reset()

    def test_step_returns_tuple(self):
        """step() should return a tuple."""
        result = self.env.step(1)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"

    def test_step_returns_five_elements(self):
        """step() should return a 5-tuple."""
        result = self.env.step(1)
        assert len(result) == 5, f"Expected 5 elements, got {len(result)}"

    def test_step_obs_is_ndarray(self):
        """step() observation should be a numpy ndarray."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert isinstance(obs, np.ndarray), f"Expected ndarray, got {type(obs)}"

    def test_step_obs_shape(self):
        """step() observation shape should be (44,)."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert obs.shape == (44,), f"Expected shape (44,), got {obs.shape}"

    def test_step_obs_dtype(self):
        """step() observation dtype should be float32."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_step_reward_is_float(self):
        """step() reward should be a float."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert isinstance(reward, (float, np.floating)), (
            f"Expected float, got {type(reward)}"
        )

    def test_step_terminated_is_bool(self):
        """step() terminated should be a bool."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert isinstance(terminated, (bool, np.bool_)), (
            f"Expected bool, got {type(terminated)}"
        )

    def test_step_truncated_is_bool(self):
        """step() truncated should be a bool."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert isinstance(truncated, (bool, np.bool_)), (
            f"Expected bool, got {type(truncated)}"
        )

    def test_step_truncated_is_always_false(self):
        """step() truncated should always be False."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert truncated is False, f"Expected truncated=False, got {truncated}"

    def test_step_info_is_dict(self):
        """step() info should be a dict."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"

    def test_step_obs_in_observation_space(self):
        """step() observation should be within the observation space."""
        obs, reward, terminated, truncated, info = self.env.step(1)
        assert self.env.observation_space.contains(obs), (
            f"Observation {obs} not in observation_space"
        )


# ===========================================================================
# Action Mapping Through Gym Interface
# ===========================================================================


class TestActionMapping:
    """Actions 0/1/2 should map to short/flat/long positions."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        self.env = LOBGymEnv()
        self.env.reset()

    def test_action_0_short_position(self):
        """Action 0 through gym step should set position to -1."""
        obs, _, _, _, _ = self.env.step(0)
        assert obs[43] == pytest.approx(-1.0), (
            f"Expected position=-1, got {obs[43]}"
        )

    def test_action_1_flat_position(self):
        """Action 1 through gym step should set position to 0."""
        obs, _, _, _, _ = self.env.step(1)
        assert obs[43] == pytest.approx(0.0), (
            f"Expected position=0, got {obs[43]}"
        )

    def test_action_2_long_position(self):
        """Action 2 through gym step should set position to +1."""
        obs, _, _, _, _ = self.env.step(2)
        assert obs[43] == pytest.approx(1.0), (
            f"Expected position=+1, got {obs[43]}"
        )


# ===========================================================================
# Episode Lifecycle
# ===========================================================================


class TestEpisodeLifecycle:
    """Full episode runs and multi-episode resets should work."""

    def test_full_episode_no_crash(self):
        """Complete a full episode through gym interface without crashing."""
        env = LOBGymEnv()
        obs, info = env.reset()
        step_count = 0
        terminated = False

        while not terminated:
            action = step_count % 3
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            assert obs.shape == (44,)
            assert obs.dtype == np.float32
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, (bool, np.bool_))
            assert truncated == False
            assert isinstance(info, dict)

            if step_count > 500:
                pytest.fail("Episode did not terminate within 500 steps")

        assert step_count > 0, "Episode should have at least one step"

    def test_episode_terminates(self):
        """Stepping through an episode should eventually set terminated=True."""
        env = LOBGymEnv()
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            _, _, terminated, _, _ = env.step(1)
            steps += 1
        assert terminated, f"Episode did not terminate after {steps} steps"

    def test_truncated_never_true_during_episode(self):
        """truncated should remain False throughout an entire episode."""
        env = LOBGymEnv()
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            _, _, terminated, truncated, _ = env.step(1)
            assert truncated == False, f"truncated was True at step {steps}"
            steps += 1

    def test_reset_after_done(self):
        """After episode ends, reset should start a new episode."""
        env = LOBGymEnv()
        env.reset()

        # Run to completion
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(1)

        # Reset and verify
        obs, info = env.reset()
        assert obs.shape == (44,)
        assert obs.dtype == np.float32
        assert obs[43] == pytest.approx(0.0), "Position should be 0 after reset"
        assert isinstance(info, dict)

    def test_multiple_episodes(self):
        """Should be able to run multiple episodes back-to-back."""
        env = LOBGymEnv()

        for episode in range(3):
            obs, info = env.reset()
            assert obs.shape == (44,)
            assert obs[43] == pytest.approx(0.0), (
                f"Position not 0 at start of episode {episode}"
            )

            terminated = False
            steps = 0
            while not terminated and steps < 500:
                _, _, terminated, truncated, _ = env.step(steps % 3)
                assert truncated == False
                steps += 1

            assert terminated, f"Episode {episode} did not terminate"

    def test_multiple_episodes_deterministic(self):
        """Two episodes with same actions should produce identical observations."""
        env = LOBGymEnv()

        # Episode 1
        obs1_first, _ = env.reset()
        obs1_list = [obs1_first.copy()]
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(1)
            obs1_list.append(obs.copy())

        # Episode 2
        obs2_first, _ = env.reset()
        obs2_list = [obs2_first.copy()]
        terminated = False
        while not terminated:
            obs, _, terminated, _, _ = env.step(1)
            obs2_list.append(obs.copy())

        assert len(obs1_list) == len(obs2_list), "Episodes should have same length"
        for i, (o1, o2) in enumerate(zip(obs1_list, obs2_list)):
            np.testing.assert_array_almost_equal(
                o1, o2, err_msg=f"Obs mismatch at step {i}"
            )


# ===========================================================================
# Constructor Variants
# ===========================================================================


class TestConstructorVariants:
    """LOBGymEnv should support all constructor configurations."""

    def test_file_path_constructor(self):
        """LOBGymEnv(file_path='...') should create file-backed env."""
        env = LOBGymEnv(file_path=EPISODE_FILE)
        obs, info = env.reset()
        assert obs.shape == (44,)
        assert isinstance(info, dict)

    def test_file_path_step_works(self):
        """File-backed LOBGymEnv should support step()."""
        env = LOBGymEnv(file_path=EPISODE_FILE)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (44,)
        assert truncated == False

    def test_file_path_full_episode(self):
        """File-backed LOBGymEnv should run a complete episode."""
        env = LOBGymEnv(file_path=EPISODE_FILE)
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            _, _, terminated, _, _ = env.step(steps % 3)
            steps += 1
        assert steps > 0

    def test_steps_per_episode_kwarg(self):
        """LOBGymEnv(steps_per_episode=10) should configure episode length."""
        env = LOBGymEnv(steps_per_episode=10)
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 500:
            _, _, terminated, _, _ = env.step(1)
            steps += 1
        assert steps <= 15, f"Expected ~10 steps, got {steps}"

    def test_session_config_dict(self):
        """LOBGymEnv(file_path=..., session_config={...}) should work."""
        env = LOBGymEnv(
            file_path=SESSION_FILE,
            session_config={
                "rth_open_ns": 48_600_000_000_000,
                "rth_close_ns": 72_000_000_000_000,
            },
            steps_per_episode=30,
        )
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_session_config_step_works(self):
        """Session-configured LOBGymEnv should support step()."""
        env = LOBGymEnv(
            file_path=SESSION_FILE,
            session_config={
                "rth_open_ns": 48_600_000_000_000,
                "rth_close_ns": 72_000_000_000_000,
            },
            steps_per_episode=30,
        )
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (44,)
        assert truncated == False

    def test_reward_mode_pnl_delta(self):
        """LOBGymEnv(reward_mode='pnl_delta') should work."""
        env = LOBGymEnv(reward_mode="pnl_delta")
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert isinstance(reward, (float, np.floating))

    def test_reward_mode_pnl_delta_penalized(self):
        """LOBGymEnv(reward_mode='pnl_delta_penalized', lambda_=0.001) should work."""
        env = LOBGymEnv(
            reward_mode="pnl_delta_penalized",
            lambda_=0.001,
        )
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(2)
        assert isinstance(reward, (float, np.floating))

    def test_all_kwargs_combined(self):
        """LOBGymEnv with all kwargs should work."""
        env = LOBGymEnv(
            file_path=SESSION_FILE,
            session_config={
                "rth_open_ns": 48_600_000_000_000,
                "rth_close_ns": 72_000_000_000_000,
            },
            steps_per_episode=30,
            reward_mode="pnl_delta_penalized",
            lambda_=0.001,
        )
        obs, info = env.reset()
        assert obs.shape == (44,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)


# ===========================================================================
# gymnasium.utils.env_checker.check_env()
# ===========================================================================


class TestCheckEnv:
    """gymnasium.utils.env_checker.check_env() should pass."""

    def test_check_env_default(self):
        """check_env() should pass on default LOBGymEnv."""
        from gymnasium.utils.env_checker import check_env
        env = LOBGymEnv()
        # check_env raises on failure; no exception = pass
        check_env(env, skip_render_check=True)

    def test_check_env_file_backed(self):
        """check_env() should pass on file-backed LOBGymEnv."""
        from gymnasium.utils.env_checker import check_env
        env = LOBGymEnv(file_path=EPISODE_FILE)
        check_env(env, skip_render_check=True)


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_step_before_reset_behavior(self):
        """step() before reset() should either raise or handle gracefully.

        The gymnasium API requires reset() before step(). The implementation
        should either raise an error or return a reasonable default.
        """
        env = LOBGymEnv()
        # Either it raises an exception or handles it; both are acceptable
        try:
            result = env.step(1)
            # If it doesn't raise, should still return valid format
            assert len(result) == 5
        except Exception:
            pass  # Raising is acceptable behavior

    def test_obs_no_nan_after_reset(self):
        """Observation after reset should contain no NaN values.

        NaN in observations can crash neural networks.
        """
        env = LOBGymEnv()
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs)), f"NaN found in reset observation: {obs}"

    def test_obs_no_inf_after_reset(self):
        """Observation after reset should contain no infinite values."""
        env = LOBGymEnv()
        obs, _ = env.reset()
        assert not np.any(np.isinf(obs)), f"Inf found in reset observation: {obs}"

    def test_obs_no_nan_during_episode(self):
        """Observations during episode should contain no NaN values."""
        env = LOBGymEnv()
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 50:
            obs, _, terminated, _, _ = env.step(steps % 3)
            assert not np.any(np.isnan(obs)), (
                f"NaN found in step {steps} observation: {obs}"
            )
            steps += 1

    def test_reward_is_finite(self):
        """Rewards should always be finite (not NaN or inf)."""
        env = LOBGymEnv()
        env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 50:
            _, reward, terminated, _, _ = env.step(steps % 3)
            assert np.isfinite(reward), (
                f"Non-finite reward {reward} at step {steps}"
            )
            steps += 1

    def test_action_space_sample_works(self):
        """Sampling from action_space and using it in step() should work."""
        env = LOBGymEnv()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (44,)
