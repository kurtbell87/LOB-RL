"""Tests for the vectorized barrier environment helpers.

Spec: docs/t9-ppo-training.md — Module 1: barrier_vec_env.py

Helper functions for creating SB3-compatible vectorized environments:
make_barrier_env_fn() and make_barrier_vec_env().
"""

import numpy as np
import pytest

from lob_rl.barrier.reward_accounting import ACTION_FLAT

from .conftest import (
    make_session_data_list,
    DEFAULT_H,
    DEFAULT_OBS_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H = DEFAULT_H
_OBS_DIM = DEFAULT_OBS_DIM


# ===========================================================================
# 1. make_barrier_env_fn (spec test 13)
# ===========================================================================


class TestMakeBarrierEnvFn:
    """Verify make_barrier_env_fn returns a callable that creates valid envs."""

    def test_make_barrier_env_fn_callable(self):
        """make_barrier_env_fn() returns a callable that creates a valid env."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_env_fn

        sessions = make_session_data_list(n_sessions=3)
        fn = make_barrier_env_fn(sessions, seed=42)

        assert callable(fn), "make_barrier_env_fn must return a callable"

        env = fn()
        assert hasattr(env, "reset"), "Created env must have reset()"
        assert hasattr(env, "step"), "Created env must have step()"
        assert hasattr(env, "action_masks"), "Created env must have action_masks()"

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (_OBS_DIM,)


# ===========================================================================
# 2. make_barrier_vec_env with DummyVecEnv (spec tests 14–15)
# ===========================================================================


class TestMakeBarrierVecEnvDummy:
    """Verify DummyVecEnv creation and num_envs."""

    def test_make_barrier_vec_env_dummy(self):
        """make_barrier_vec_env(use_subprocess=False) creates a working DummyVecEnv."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        sessions = make_session_data_list(n_sessions=3)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )
        assert vec_env is not None
        # VecEnv should be usable
        obs = vec_env.reset()
        assert isinstance(obs, np.ndarray)
        vec_env.close()

    def test_make_barrier_vec_env_n_envs(self):
        """Created VecEnv has correct num_envs."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        sessions = make_session_data_list(n_sessions=3)
        n_envs = 3
        vec_env = make_barrier_vec_env(
            sessions, n_envs=n_envs, use_subprocess=False, seed=42,
        )
        assert vec_env.num_envs == n_envs, (
            f"Expected {n_envs} envs, got {vec_env.num_envs}"
        )
        vec_env.close()


# ===========================================================================
# 3. VecEnv step/reset shapes (spec tests 16–17)
# ===========================================================================


class TestVecEnvShapes:
    """Verify step and reset return correct batch dimensions."""

    def test_vec_env_step_returns_correct_shapes(self):
        """Step returns arrays with correct batch dimensions."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        n_envs = 2
        sessions = make_session_data_list(n_sessions=3)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=n_envs, use_subprocess=False, seed=42,
        )
        vec_env.reset()

        # Step with flat actions for all envs
        actions = np.array([ACTION_FLAT] * n_envs)
        obs, rewards, dones, infos = vec_env.step(actions)

        assert obs.shape == (n_envs, _OBS_DIM), (
            f"Expected obs shape ({n_envs}, {_OBS_DIM}), got {obs.shape}"
        )
        assert rewards.shape == (n_envs,), (
            f"Expected rewards shape ({n_envs},), got {rewards.shape}"
        )
        assert dones.shape == (n_envs,), (
            f"Expected dones shape ({n_envs},), got {dones.shape}"
        )
        assert len(infos) == n_envs
        vec_env.close()

    def test_vec_env_reset_returns_obs(self):
        """Reset returns observation array of shape (n_envs, 132)."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        n_envs = 2
        sessions = make_session_data_list(n_sessions=3)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=n_envs, use_subprocess=False, seed=42,
        )
        obs = vec_env.reset()
        assert obs.shape == (n_envs, _OBS_DIM), (
            f"Expected obs shape ({n_envs}, {_OBS_DIM}), got {obs.shape}"
        )
        vec_env.close()


# ===========================================================================
# 4. Random agent on vec env (spec test 18)
# ===========================================================================


class TestVecEnvRandomAgent:
    """Verify random agent completes episodes on vectorized env."""

    def test_vec_env_random_agent_completes(self):
        """Random agent runs 5 episodes on vec env without error."""
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        n_envs = 2
        sessions = make_session_data_list(n_sessions=3, n_bars=30)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=n_envs, use_subprocess=False, seed=42,
        )
        obs = vec_env.reset()
        rng = np.random.default_rng(42)

        episodes_completed = 0
        max_steps = 5000
        steps = 0

        while episodes_completed < 5 and steps < max_steps:
            # Random actions for each env
            actions = rng.integers(0, 4, size=n_envs)
            obs, rewards, dones, infos = vec_env.step(actions)
            episodes_completed += sum(dones)
            steps += 1

        vec_env.close()
        assert episodes_completed >= 5, (
            f"Expected >= 5 completed episodes, got {episodes_completed} in {steps} steps"
        )
