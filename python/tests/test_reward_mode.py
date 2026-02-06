"""Tests for Step 3b: RewardCalculator Python bindings.

Spec: docs/step3b-reward-calculator.md

These tests verify that:
- LOBEnv constructors accept optional reward_mode and lambda_ parameters
- reward_mode accepts "pnl_delta" and "pnl_delta_penalized" strings
- PnLDeltaPenalized produces lower rewards than PnLDelta when position != 0
- Default reward mode is PnLDelta (backward compatible)
"""

import pytest

import lob_rl_core

from conftest import EPISODE_FILE, SESSION_FILE


# ===========================================================================
# LOBEnv: Default constructor with reward_mode (SyntheticSource)
# ===========================================================================


class TestRewardModeDefaultConstructor:
    """LOBEnv() should accept optional reward_mode and lambda_ kwargs."""

    def test_default_constructor_still_works(self):
        """LOBEnv() with no reward args should use PnLDelta (backward compat)."""
        env = lob_rl_core.LOBEnv()
        obs = env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(reward, float)

    def test_pnl_delta_mode_string(self):
        """LOBEnv(reward_mode='pnl_delta') should work."""
        env = lob_rl_core.LOBEnv(reward_mode="pnl_delta")
        obs = env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(reward, float)

    def test_pnl_delta_penalized_mode_string(self):
        """LOBEnv(reward_mode='pnl_delta_penalized') should work."""
        env = lob_rl_core.LOBEnv(reward_mode="pnl_delta_penalized", lambda_=0.1)
        obs = env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(reward, float)

    def test_lambda_parameter(self):
        """LOBEnv should accept a lambda_ float parameter."""
        env = lob_rl_core.LOBEnv(reward_mode="pnl_delta_penalized", lambda_=0.5)
        obs = env.reset()
        obs, reward, done = env.step(2)  # long position
        assert isinstance(reward, float)

    def test_invalid_reward_mode_raises(self):
        """LOBEnv with invalid reward_mode string should raise."""
        with pytest.raises(Exception):
            lob_rl_core.LOBEnv(reward_mode="invalid_mode")


# ===========================================================================
# LOBEnv: SyntheticSource with steps + reward mode
# ===========================================================================


class TestRewardModeSyntheticSteps:
    """LOBEnv(steps_per_episode, reward_mode, lambda_) should work."""

    def test_synthetic_with_steps_and_reward_mode(self):
        """LOBEnv(steps, reward_mode) should configure both."""
        env = lob_rl_core.LOBEnv(
            steps_per_episode=10,
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
        )
        env.reset()
        obs, reward, done = env.step(2)
        assert isinstance(reward, float)

    def test_synthetic_penalized_reward_lower_than_pnl_delta(self):
        """Penalized mode should produce lower cumulative reward when holding."""
        env_pnl = lob_rl_core.LOBEnv(steps_per_episode=10)
        env_pen = lob_rl_core.LOBEnv(
            steps_per_episode=10,
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
        )

        env_pnl.reset()
        env_pen.reset()

        total_pnl = 0.0
        total_pen = 0.0

        for _ in range(10):
            obs_pnl, reward_pnl, done_pnl = env_pnl.step(2)  # long
            obs_pen, reward_pen, done_pen = env_pen.step(2)    # long
            total_pnl += reward_pnl
            total_pen += reward_pen
            if done_pnl or done_pen:
                break

        assert total_pen < total_pnl, (
            f"Penalized total ({total_pen}) should be less than "
            f"PnLDelta total ({total_pnl}) when holding a position"
        )


# ===========================================================================
# LOBEnv: File-based constructor with reward mode
# ===========================================================================


class TestRewardModeFileConstructor:
    """LOBEnv(file_path, ..., reward_mode, lambda_) should work."""

    def test_file_with_reward_mode(self):
        """LOBEnv(file_path, reward_mode=...) should accept reward mode."""
        env = lob_rl_core.LOBEnv(
            EPISODE_FILE,
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
        )
        obs = env.reset()
        obs, reward, done = env.step(2)
        assert isinstance(reward, float)

    def test_file_with_steps_and_reward_mode(self):
        """LOBEnv(file_path, steps, reward_mode, lambda_) should work."""
        env = lob_rl_core.LOBEnv(
            EPISODE_FILE,
            20,
            reward_mode="pnl_delta_penalized",
            lambda_=0.2,
        )
        env.reset()
        obs, reward, done = env.step(2)
        assert isinstance(reward, float)


# ===========================================================================
# LOBEnv: Session constructor with reward mode
# ===========================================================================


class TestRewardModeSessionConstructor:
    """LOBEnv(file_path, session_config, steps, reward_mode, lambda_)."""

    def test_session_with_reward_mode(self):
        """Session constructor should accept reward_mode and lambda_."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(
            SESSION_FILE,
            cfg,
            30,
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
        )
        obs = env.reset()
        assert isinstance(obs, (list, tuple))
        assert len(obs) == 44

    def test_session_pnl_delta_default(self):
        """Session constructor without reward_mode should default to PnLDelta."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 30)
        obs = env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(reward, float)

    def test_session_penalized_full_episode(self):
        """Session + penalized mode should run a complete episode."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(
            SESSION_FILE,
            cfg,
            30,
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
        )
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            obs, reward, done = env.step(steps % 3)
            steps += 1
        assert steps > 0


# ===========================================================================
# LOBEnv: Flat position — penalized mode should match PnLDelta
# ===========================================================================


class TestRewardModeFlatPosition:
    """When position is flat, both modes should give identical rewards."""

    def test_flat_position_identical_rewards(self):
        """Flat position (action=1) should give same reward in both modes."""
        env_pnl = lob_rl_core.LOBEnv(steps_per_episode=5)
        env_pen = lob_rl_core.LOBEnv(
            steps_per_episode=5,
            reward_mode="pnl_delta_penalized",
            lambda_=1.0,  # Large lambda to make any difference obvious
        )

        env_pnl.reset()
        env_pen.reset()

        for _ in range(5):
            _, reward_pnl, done_pnl = env_pnl.step(1)  # flat
            _, reward_pen, done_pen = env_pen.step(1)    # flat
            assert reward_pnl == pytest.approx(reward_pen), (
                "Flat position should give identical rewards regardless of mode"
            )
            if done_pnl or done_pen:
                break
