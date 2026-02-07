"""Tests for BarLevelEnv — bar-level gymnasium environment.

Spec: docs/bar-level-env.md (Requirement 2)

These tests verify that:
- BarLevelEnv is a valid gymnasium.Env with correct spaces
- observation_space: Box(-inf, inf, shape=(21,), float32)
- action_space: Discrete(3)
- Constructor calls aggregate_bars() and computes cross-bar temporal features
- Cross-bar temporal features (7 dims) are correct at each bar index
- reset() returns obs_for_bar[0] with position=0
- step() advances bar_index, computes reward, terminates correctly
- Reward = position * (bar_mid_close[t] - bar_mid_close[t-1])
- Execution cost: spread/2 * |position - prev_position|
- Participation bonus: bonus * |position|
- Flattening penalty at terminal step
- from_cache() and from_file() classmethods work
- Passes gymnasium check_env
"""

import os
import tempfile

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from conftest import make_realistic_obs, PRECOMPUTE_EPISODE_FILE


# ===========================================================================
# Helper
# ===========================================================================


def _make_tick_data(n, mid_start=100.0, mid_step=0.25, spread=0.50):
    """Create (obs, mid, spread) arrays with n ticks."""
    obs, mid, spread_arr = make_realistic_obs(n, mid_start=mid_start,
                                               mid_step=mid_step, spread=spread)
    return obs, mid, spread_arr


# ===========================================================================
# Test 1: BarLevelEnv exists and is importable
# ===========================================================================


class TestBarLevelEnvExists:
    """BarLevelEnv should be importable from lob_rl.bar_level_env."""

    def test_import(self):
        """BarLevelEnv should be importable."""
        from lob_rl.bar_level_env import BarLevelEnv
        assert BarLevelEnv is not None

    def test_is_gymnasium_env_subclass(self):
        """BarLevelEnv should be a subclass of gymnasium.Env."""
        from lob_rl.bar_level_env import BarLevelEnv
        assert issubclass(BarLevelEnv, gym.Env)


# ===========================================================================
# Test 2: Constructor and spaces
# ===========================================================================


class TestBarLevelEnvConstructor:
    """BarLevelEnv constructor should accept tick-level arrays + bar_size."""

    def test_constructor_minimal(self):
        """BarLevelEnv(obs, mid, spread) should construct without error."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        assert env is not None

    def test_observation_space_shape_21(self):
        """observation_space should have shape (21,)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        assert env.observation_space.shape == (21,)

    def test_observation_space_dtype_float32(self):
        """observation_space dtype should be float32."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds_infinite(self):
        """observation_space should have -inf/+inf bounds."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        assert np.all(env.observation_space.low == -np.inf)
        assert np.all(env.observation_space.high == np.inf)

    def test_action_space_discrete_3(self):
        """action_space should be Discrete(3)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_default_bar_size(self):
        """Default bar_size should be 500."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(1000)
        env = BarLevelEnv(obs, mid, spread)
        # 1000 ticks / 500 = 2 bars
        env.reset()
        # If default is 500, we should be able to step (2 bars -> 1 step)
        _, _, terminated, _, _ = env.step(1)
        assert terminated  # only 1 step with 2 bars

    def test_accepts_reward_mode(self):
        """Should accept reward_mode parameter."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10,
                          reward_mode="pnl_delta_penalized", lambda_=0.5)
        assert env is not None

    def test_accepts_execution_cost(self):
        """Should accept execution_cost parameter."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, execution_cost=True)
        assert env is not None

    def test_accepts_participation_bonus(self):
        """Should accept participation_bonus parameter."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, participation_bonus=0.01)
        assert env is not None


# ===========================================================================
# Test 3: reset()
# ===========================================================================


class TestBarLevelEnvReset:
    """reset() should return (obs_21, info) with position=0."""

    def test_returns_2_tuple(self):
        """reset() should return (obs, info)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_obs_shape_21(self):
        """reset() observation should have shape (21,)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert obs_out.shape == (21,)

    def test_obs_dtype_float32(self):
        """reset() observation should be float32."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert obs_out.dtype == np.float32

    def test_position_is_zero(self):
        """reset() obs should have position=0 at last index (index 20)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert obs_out[20] == pytest.approx(0.0)

    def test_obs_in_observation_space(self):
        """reset() obs should be within observation_space."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert env.observation_space.contains(obs_out)

    def test_reset_sets_bar_index_0(self):
        """After reset, next step should use bar index 0->1."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # Take some steps
        env.step(2)
        env.step(1)
        # Reset should go back to bar 0
        obs_out, _ = env.reset()
        assert obs_out[20] == pytest.approx(0.0)  # position reset

    def test_info_is_dict(self):
        """reset() info should be a dict."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        _, info = env.reset()
        assert isinstance(info, dict)


# ===========================================================================
# Test 4: step() format
# ===========================================================================


class TestBarLevelEnvStepFormat:
    """step() should return (obs, reward, terminated, truncated, info)."""

    def test_step_returns_5_tuple(self):
        """step() should return exactly 5 elements."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        result = env.step(1)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_shape_21(self):
        """step() obs should have shape (21,)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(1)
        assert obs_out.shape == (21,)

    def test_step_reward_is_float(self):
        """step() reward should be a float."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert isinstance(reward, (float, np.floating))

    def test_step_terminated_is_bool(self):
        """step() terminated should be a bool."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, _, terminated, _, _ = env.step(1)
        assert isinstance(terminated, (bool, np.bool_))

    def test_step_truncated_always_false(self):
        """step() truncated should always be False."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, _, _, truncated, _ = env.step(1)
        assert truncated is False

    def test_step_obs_in_observation_space(self):
        """step() obs should be within observation_space."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(1)
        assert env.observation_space.contains(obs_out)


# ===========================================================================
# Test 5: Action mapping
# ===========================================================================


class TestBarLevelEnvActionMapping:
    """Actions: 0=short(-1), 1=flat(0), 2=long(+1) reflected in obs[20]."""

    def test_action_0_short(self):
        """action=0 -> position=-1 at obs[20]."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(0)
        assert obs_out[20] == pytest.approx(-1.0)

    def test_action_1_flat(self):
        """action=1 -> position=0 at obs[20]."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(1)
        assert obs_out[20] == pytest.approx(0.0)

    def test_action_2_long(self):
        """action=2 -> position=+1 at obs[20]."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(2)
        assert obs_out[20] == pytest.approx(1.0)

    def test_position_changes(self):
        """Position should reflect most recent action."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(2)
        assert obs_out[20] == pytest.approx(1.0)
        obs_out, _, _, _, _ = env.step(0)
        assert obs_out[20] == pytest.approx(-1.0)


# ===========================================================================
# Test 6: Reward — close-to-close PnL
# ===========================================================================


class TestBarLevelEnvReward:
    """Reward = position * (bar_mid_close[t] - bar_mid_close[t-1])."""

    def test_long_reward_positive_move(self):
        """Long position with bars going up should give positive reward."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # action=2 -> position=+1
        _, reward, _, _, _ = env.step(2)
        expected = 1.0 * (bar_mid_close[1] - bar_mid_close[0])
        assert reward == pytest.approx(expected, rel=1e-5)

    def test_short_reward_negative_move(self):
        """Short position with bars going up should give negative reward."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # action=0 -> position=-1
        _, reward, _, _, _ = env.step(0)
        expected = -1.0 * (bar_mid_close[1] - bar_mid_close[0])
        assert reward == pytest.approx(expected, rel=1e-5)

    def test_flat_reward_zero(self):
        """Flat position should give zero reward."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.25)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0, abs=1e-7)

    def test_consecutive_rewards_use_correct_bars(self):
        """Each step uses bar_mid_close[t] - bar_mid_close[t-1]."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(40, mid_start=100.0, mid_step=0.25)
        _, bar_mid_close, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # Step 0: bar 0 -> bar 1
        _, r0, _, _, _ = env.step(2)
        expected_r0 = 1.0 * (bar_mid_close[1] - bar_mid_close[0])
        assert r0 == pytest.approx(expected_r0, rel=1e-5)

        # Step 1: bar 1 -> bar 2
        _, r1, _, _, _ = env.step(2)
        expected_r1 = 1.0 * (bar_mid_close[2] - bar_mid_close[1])
        assert r1 == pytest.approx(expected_r1, rel=1e-5)


# ===========================================================================
# Test 7: Execution cost
# ===========================================================================


class TestBarLevelEnvExecutionCost:
    """execution_cost: spread/2 * |position - prev_position|."""

    def test_execution_cost_on_position_change(self):
        """Going from flat to long should incur spread/2 * 1."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.0,
                                            spread=0.50)
        _, bar_mid_close, bar_spread_close = aggregate_bars(obs, mid, spread,
                                                             bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10, execution_cost=True)
        env.reset()
        # action=2: flat->long, cost = bar_spread_close[0]/2 * |1-0| = 0.25
        _, reward, _, _, _ = env.step(2)
        # pnl = 1 * (bar_mid_close[1] - bar_mid_close[0]) = 0 (mid_step=0)
        # cost = bar_spread_close[0] / 2 * 1
        expected_cost = bar_spread_close[0] / 2.0 * 1.0
        assert reward == pytest.approx(-expected_cost, rel=1e-5)

    def test_no_cost_when_staying_same_position(self):
        """Staying long should incur no execution cost."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(40, mid_start=100.0, mid_step=0.0,
                                            spread=0.50)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, execution_cost=True)
        env.reset()
        env.step(2)  # flat -> long (has cost)
        _, reward, _, _, _ = env.step(2)  # long -> long (no cost, no pnl)
        assert reward == pytest.approx(0.0, abs=1e-6)

    def test_cost_doubles_on_reversal(self):
        """Going from long(+1) to short(-1) should cost spread/2 * 2."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(40, mid_start=100.0, mid_step=0.0,
                                            spread=0.50)
        _, _, bar_spread_close = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10, execution_cost=True)
        env.reset()
        env.step(2)  # flat -> long
        _, reward, _, _, _ = env.step(0)  # long -> short, delta=2
        # cost = bar_spread_close[1] / 2 * |(-1) - 1| = bar_spread_close[1]
        expected_cost = bar_spread_close[1] / 2.0 * 2.0
        assert reward == pytest.approx(-expected_cost, rel=1e-5)

    def test_no_cost_when_disabled(self):
        """Without execution_cost, no cost should be charged."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.0,
                                            spread=0.50)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, execution_cost=False)
        env.reset()
        _, reward, _, _, _ = env.step(2)
        # pnl=0, no cost
        assert reward == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Test 8: Participation bonus
# ===========================================================================


class TestBarLevelEnvParticipationBonus:
    """participation_bonus: bonus * |position|."""

    def test_bonus_with_position(self):
        """Long position with bonus=0.01 should get 0.01 added."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.0)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, participation_bonus=0.01)
        env.reset()
        _, reward, _, _, _ = env.step(2)
        # pnl = 0, bonus = 0.01 * |1| = 0.01
        assert reward == pytest.approx(0.01, rel=1e-5)

    def test_no_bonus_when_flat(self):
        """Flat position should get no bonus."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.0)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, participation_bonus=0.01)
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0, abs=1e-7)

    def test_short_gets_bonus(self):
        """Short position should also get bonus."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(30, mid_start=100.0, mid_step=0.0)
        env = BarLevelEnv(obs, mid, spread, bar_size=10, participation_bonus=0.01)
        env.reset()
        _, reward, _, _, _ = env.step(0)
        # pnl = 0, bonus = 0.01 * |-1| = 0.01
        assert reward == pytest.approx(0.01, rel=1e-5)


# ===========================================================================
# Test 9: Episode length and termination
# ===========================================================================


class TestBarLevelEnvEpisodeLength:
    """B bars -> B-1 steps before termination."""

    def test_10_bars_gives_9_steps(self):
        """100 ticks / 10 bar_size = 10 bars -> 9 steps."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()

        steps = 0
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(1)
            steps += 1
        assert steps == 9

    def test_2_bars_gives_1_step(self):
        """20 ticks / 10 bar_size = 2 bars -> 1 step (immediately terminal)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(20)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, _, terminated, _, _ = env.step(1)
        assert terminated

    def test_not_terminated_before_last_step(self):
        """Steps before the last should not be terminal."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(50)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # 5 bars -> 4 steps. First 3 should not terminate.
        for i in range(3):
            _, _, terminated, _, _ = env.step(1)
            assert not terminated, f"Premature termination at step {i}"
        _, _, terminated, _, _ = env.step(1)
        assert terminated


# ===========================================================================
# Test 10: Flattening penalty
# ===========================================================================


class TestBarLevelEnvFlatteningPenalty:
    """Terminal step with position != 0 should incur flattening penalty."""

    def test_long_at_terminal(self):
        """Long at terminal: penalty = spread/2 * |position|."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.0,
                                            spread=0.50)
        _, bar_mid_close, bar_spread_close = aggregate_bars(obs, mid, spread,
                                                             bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, reward, terminated, _, _ = env.step(2)  # long, terminal (2 bars)
        assert terminated
        # pnl = 1*(bmc[1]-bmc[0]) = 0 (constant mid)
        # flattening = -|1| * bar_spread_close[1]/2 (or spread at last bar)
        # With constant spread=0.5, penalty = -0.5/2 = -0.25
        # Actually per spec: reward -= bar_spread_close[bar_index] / 2 * |position|
        # bar_index is 1 after step
        assert reward < 0, "Should have negative reward from flattening penalty"

    def test_flat_at_terminal_no_penalty(self):
        """Flat at terminal: no flattening penalty."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(20, mid_start=100.0, mid_step=0.0)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        _, reward, terminated, _, _ = env.step(1)  # flat, terminal
        assert terminated
        assert reward == pytest.approx(0.0, abs=1e-7)


# ===========================================================================
# Test 11: Cross-bar temporal features (indices 13-19 in observation)
# ===========================================================================


class TestCrossBarTemporalFeatures:
    """Cross-bar temporal features should be correctly computed."""

    def test_return_lag1_at_bar_0(self):
        """return_lag1 at bar 0 should be 0 (no previous bar)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        # Feature index 13 (13th = return_lag1)
        assert obs_out[13] == pytest.approx(0.0, abs=1e-7)

    def test_return_lag1_at_bar_1(self):
        """return_lag1 at bar 1 should be bar_return[0]."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(1)
        # obs_out[13] = return_lag1 = bar_return[t-1] = bar_return[0]
        assert obs_out[13] == pytest.approx(bar_features[0, 0], rel=1e-5)

    def test_return_lag3_zero_when_t_lt_3(self):
        """return_lag3 should be 0 when t < 3."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()  # t=0
        assert obs_out[14] == pytest.approx(0.0, abs=1e-7)

        obs_out, _, _, _, _ = env.step(1)  # t=1
        assert obs_out[14] == pytest.approx(0.0, abs=1e-7)

        obs_out, _, _, _, _ = env.step(1)  # t=2
        assert obs_out[14] == pytest.approx(0.0, abs=1e-7)

    def test_return_lag5_zero_when_t_lt_5(self):
        """return_lag5 should be 0 when t < 5."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()  # t=0
        assert obs_out[15] == pytest.approx(0.0, abs=1e-7)

    def test_cumulative_return_5(self):
        """cumulative_return_5 = sum(bar_return[t-5:t])."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        # Step through 6 bars to get to t=6
        for _ in range(6):
            obs_out, _, _, _, _ = env.step(1)
        # At t=6, cumulative_return_5 = sum(bar_return[1:6])
        expected = float(np.sum(bar_features[1:6, 0]))
        assert obs_out[16] == pytest.approx(expected, rel=1e-4)

    def test_rolling_vol_5_zero_when_t_lt_2(self):
        """rolling_vol_5 should be 0 when t < 2."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()  # t=0
        assert obs_out[17] == pytest.approx(0.0, abs=1e-7)

        obs_out, _, _, _, _ = env.step(1)  # t=1
        assert obs_out[17] == pytest.approx(0.0, abs=1e-7)

    def test_imb_delta_3_zero_when_t_lt_3(self):
        """imb_delta_3 should be 0 when t < 3."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()  # t=0
        assert obs_out[18] == pytest.approx(0.0, abs=1e-7)

    def test_spread_delta_3_zero_when_t_lt_3(self):
        """spread_delta_3 should be 0 when t < 3."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()  # t=0
        assert obs_out[19] == pytest.approx(0.0, abs=1e-7)

    def test_temporal_features_count(self):
        """There should be exactly 7 temporal features (indices 13-19)."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        # 13 intra-bar + 7 temporal + 1 position = 21
        assert obs_out.shape == (21,)

    def test_imb_delta_3_at_bar_3(self):
        """imb_delta_3 at t=3 = imbalance_close[3] - imbalance_close[0]."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        for _ in range(3):
            obs_out, _, _, _, _ = env.step(1)
        # At t=3: imb_delta_3 = imbalance_close[3] - imbalance_close[0]
        expected = float(bar_features[3, 6] - bar_features[0, 6])
        assert obs_out[18] == pytest.approx(expected, rel=1e-4)


# ===========================================================================
# Test 12: Intra-bar features in observation
# ===========================================================================


class TestIntraBarFeaturesInObs:
    """First 13 dims of observation should be intra-bar features."""

    def test_obs_first_13_match_bar_features(self):
        """obs[0:13] at bar t should match bar_features[t]."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        np.testing.assert_array_almost_equal(
            obs_out[:13], bar_features[0], decimal=5,
            err_msg="Bar 0 intra-bar features mismatch"
        )

    def test_obs_after_step_matches_bar_1(self):
        """After one step, obs[:13] should match bar_features[1]."""
        from lob_rl.bar_level_env import BarLevelEnv
        from lob_rl.bar_aggregation import aggregate_bars
        obs, mid, spread = _make_tick_data(100)
        bar_features, _, _ = aggregate_bars(obs, mid, spread, bar_size=10)

        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        env.reset()
        obs_out, _, _, _, _ = env.step(1)
        np.testing.assert_array_almost_equal(
            obs_out[:13], bar_features[1], decimal=5,
            err_msg="Bar 1 intra-bar features mismatch"
        )


# ===========================================================================
# Test 13: from_cache() classmethod
# ===========================================================================


class TestBarLevelEnvFromCache:
    """BarLevelEnv.from_cache() should load from .npz and construct env."""

    def test_from_cache_exists(self):
        """BarLevelEnv should have a from_cache classmethod."""
        from lob_rl.bar_level_env import BarLevelEnv
        assert hasattr(BarLevelEnv, "from_cache")
        assert callable(BarLevelEnv.from_cache)

    def test_from_cache_returns_bar_level_env(self):
        """from_cache should return a BarLevelEnv instance."""
        from lob_rl.bar_level_env import BarLevelEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            obs, mid, spread = _make_tick_data(100)
            npz_path = os.path.join(tmpdir, "test.npz")
            np.savez(npz_path, obs=obs, mid=mid, spread=spread)
            env = BarLevelEnv.from_cache(npz_path, bar_size=10)
            assert isinstance(env, BarLevelEnv)

    def test_from_cache_obs_shape(self):
        """from_cache env should have obs shape (21,)."""
        from lob_rl.bar_level_env import BarLevelEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            obs, mid, spread = _make_tick_data(100)
            npz_path = os.path.join(tmpdir, "test.npz")
            np.savez(npz_path, obs=obs, mid=mid, spread=spread)
            env = BarLevelEnv.from_cache(npz_path, bar_size=10)
            obs_out, _ = env.reset()
            assert obs_out.shape == (21,)

    def test_from_cache_accepts_all_params(self):
        """from_cache should accept bar_size, reward_mode, execution_cost, etc."""
        from lob_rl.bar_level_env import BarLevelEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            obs, mid, spread = _make_tick_data(100)
            npz_path = os.path.join(tmpdir, "test.npz")
            np.savez(npz_path, obs=obs, mid=mid, spread=spread)
            env = BarLevelEnv.from_cache(
                npz_path, bar_size=10,
                reward_mode="pnl_delta",
                lambda_=0.0,
                execution_cost=True,
                participation_bonus=0.01,
            )
            env.reset()
            _, reward, _, _, _ = env.step(2)
            assert isinstance(reward, (float, np.floating))

    def test_from_cache_missing_key_raises(self):
        """from_cache with missing keys in .npz should raise."""
        from lob_rl.bar_level_env import BarLevelEnv
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "bad.npz")
            np.savez(npz_path, obs=np.zeros((10, 43)))
            with pytest.raises((KeyError, ValueError)):
                BarLevelEnv.from_cache(npz_path, bar_size=5)


# ===========================================================================
# Test 14: from_file() classmethod
# ===========================================================================


class TestBarLevelEnvFromFile:
    """BarLevelEnv.from_file() should construct from .bin file."""

    def test_from_file_exists(self):
        """BarLevelEnv should have a from_file classmethod."""
        from lob_rl.bar_level_env import BarLevelEnv
        assert hasattr(BarLevelEnv, "from_file")
        assert callable(BarLevelEnv.from_file)

    def test_from_file_returns_bar_level_env(self):
        """from_file should return a BarLevelEnv instance."""
        from lob_rl.bar_level_env import BarLevelEnv
        env = BarLevelEnv.from_file(PRECOMPUTE_EPISODE_FILE, bar_size=50)
        assert isinstance(env, BarLevelEnv)

    def test_from_file_obs_shape(self):
        """from_file env should have obs shape (21,)."""
        from lob_rl.bar_level_env import BarLevelEnv
        env = BarLevelEnv.from_file(PRECOMPUTE_EPISODE_FILE, bar_size=50)
        obs_out, _ = env.reset()
        assert obs_out.shape == (21,)

    def test_from_file_can_step(self):
        """from_file env should be steppable."""
        from lob_rl.bar_level_env import BarLevelEnv
        env = BarLevelEnv.from_file(PRECOMPUTE_EPISODE_FILE, bar_size=50)
        env.reset()
        obs_out, reward, terminated, truncated, info = env.step(1)
        assert obs_out.shape == (21,)
        assert isinstance(reward, (float, np.floating))


# ===========================================================================
# Test 15: check_env passes
# ===========================================================================


class TestBarLevelEnvCheckEnv:
    """BarLevelEnv should pass gymnasium's check_env."""

    def test_check_env(self):
        """check_env() should pass on BarLevelEnv."""
        from gymnasium.utils.env_checker import check_env
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        check_env(env, skip_render_check=True)


# ===========================================================================
# Test 16: Determinism
# ===========================================================================


class TestBarLevelEnvDeterminism:
    """Same actions on same data should produce identical results."""

    def test_deterministic_episode(self):
        """Two episodes with same actions should be identical."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(50)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        actions = [2, 0, 1, 2]

        # Episode 1
        env.reset()
        r1 = []
        for a in actions:
            _, reward, _, _, _ = env.step(a)
            r1.append(reward)

        # Episode 2
        env.reset()
        r2 = []
        for a in actions:
            _, reward, _, _, _ = env.step(a)
            r2.append(reward)

        for i, (a, b) in enumerate(zip(r1, r2)):
            assert a == pytest.approx(b), f"Reward differs at step {i}"


# ===========================================================================
# Test 17: Edge case — day with 1 bar
# ===========================================================================


class TestSingleBarDay:
    """Day with only 1 valid bar should terminate immediately on first step."""

    def test_single_bar_terminates_on_first_step(self):
        """1 bar means the episode terminates immediately since B-1=0 steps
        but reset returns bar 0 and first step tries bar 1 which doesn't exist.
        Actually with 1 bar, episode is 0 steps — reset returns obs but no valid step.
        The spec says 'Day with only 1 valid bar -> episode terminates immediately on first step'.
        This means the env should be constructable but the episode has 0 effective steps,
        or it terminates immediately on the first step.
        """
        from lob_rl.bar_level_env import BarLevelEnv
        # 10 ticks / bar_size=10 = 1 bar
        obs, mid, spread = _make_tick_data(10)
        # The env with 1 bar should either:
        # - raise ValueError (like PrecomputedEnv with <2 rows), or
        # - terminate immediately on first step
        # Spec says "1 valid bar → episode terminates immediately on first step"
        # So it should be constructable.
        # Actually the spec doesn't say how to handle this exactly,
        # but "terminated = (bar_index >= num_bars - 1)" means with 1 bar,
        # after reset bar_index=0, 0 >= 1-1=0 is True already.
        # The implementation could either refuse construction or handle it.
        # Let's test the spec's stated behavior:
        # "Day with only 1 valid bar → episode terminates immediately on first step"
        # This implies we can construct but on first step it terminates.
        # But actually at reset, bar_index=0 and num_bars=1, so already at terminal.
        # The env should still be steppable for at least one step.
        # We'll test that the env either raises on construction or terminates on first step.
        try:
            env = BarLevelEnv(obs, mid, spread, bar_size=10)
            env.reset()
            # If construction succeeded, first step should terminate
            _, _, terminated, _, _ = env.step(1)
            assert terminated, "Single-bar episode should terminate on first step"
        except ValueError:
            # Also acceptable: refusing construction with < 2 bars
            pass


# ===========================================================================
# Test 18: No NaN or Inf in observations
# ===========================================================================


class TestNoNanInObs:
    """Observations should never contain NaN or Inf."""

    def test_no_nan_in_full_episode(self):
        """Run full episode and check all obs for NaN."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert not np.any(np.isnan(obs_out))

        terminated = False
        while not terminated:
            obs_out, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isnan(obs_out)), "NaN found in obs"

    def test_no_inf_in_full_episode(self):
        """Run full episode and check all obs for Inf."""
        from lob_rl.bar_level_env import BarLevelEnv
        obs, mid, spread = _make_tick_data(100)
        env = BarLevelEnv(obs, mid, spread, bar_size=10)
        obs_out, _ = env.reset()
        assert not np.any(np.isinf(obs_out))

        terminated = False
        while not terminated:
            obs_out, _, terminated, _, _ = env.step(1)
            assert not np.any(np.isinf(obs_out)), "Inf found in obs"
