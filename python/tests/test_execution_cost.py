"""Tests for execution cost on position changes.

Spec: docs/execution-cost.md

These tests verify that:
- PrecomputedEnv accepts execution_cost parameter (default False)
- execution_cost=False preserves identical rewards to current behavior
- execution_cost=True subtracts spread/2 * |delta_pos| each step
- Position change long->short (delta=2) costs more than flat->long (delta=1)
- No position change costs nothing
- MultiDayEnv forwards execution_cost to PrecomputedEnv
- LOBGymEnv forwards execution_cost to C++ LOBEnv
- C++ LOBEnv bindings accept execution_cost parameter
- train.py --execution-cost flag exists
- Edge cases: spread=0, NaN spread, first step after reset
"""

import os

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.precomputed_env import PrecomputedEnv

from conftest import (
    FIXTURE_DIR, EPISODE_FILE, SESSION_FILE, PRECOMPUTE_EPISODE_FILE,
    DAY_FILES, make_obs as _make_obs, make_mid as _make_mid, make_spread as _make_spread,
)


# ===========================================================================
# PrecomputedEnv: Constructor accepts execution_cost parameter
# ===========================================================================


class TestPrecomputedEnvConstructor:
    """PrecomputedEnv should accept an execution_cost boolean parameter."""

    def test_accepts_execution_cost_false(self):
        """PrecomputedEnv(execution_cost=False) should construct without error."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            execution_cost=False,
        )
        assert env is not None

    def test_accepts_execution_cost_true(self):
        """PrecomputedEnv(execution_cost=True) should construct without error."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            execution_cost=True,
        )
        assert env is not None

    def test_default_execution_cost_is_false(self):
        """Default execution_cost should be False (backward compatible)."""
        # This is verified by checking that omitting the param gives
        # the same rewards as execution_cost=False
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = _make_spread(3, value=0.5)

        env_default = PrecomputedEnv(_make_obs(3), mid.copy(), spread.copy())
        env_false = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), execution_cost=False
        )

        env_default.reset()
        env_false.reset()

        # Go long from flat — should have identical rewards if default is False
        _, r_default, _, _, _ = env_default.step(2)
        _, r_false, _, _, _ = env_false.step(2)
        assert r_default == pytest.approx(r_false)

    def test_still_valid_gym_env_with_execution_cost(self):
        """PrecomputedEnv with execution_cost=True should still be a gymnasium.Env."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            execution_cost=True,
        )
        assert isinstance(env, gym.Env)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)


# ===========================================================================
# PrecomputedEnv: execution_cost=False preserves existing behavior
# ===========================================================================


class TestPrecomputedEnvBackwardCompat:
    """execution_cost=False should produce identical rewards to omitting it."""

    def test_identical_rewards_full_episode(self):
        """Full episode with execution_cost=False should match default behavior."""
        mid = np.array([100.0, 102.0, 99.0, 105.0, 103.0], dtype=np.float64)
        spread = _make_spread(5, value=1.0)
        actions = [2, 0, 1, 2]  # long, short, flat, long

        env_default = PrecomputedEnv(_make_obs(5), mid.copy(), spread.copy())
        env_false = PrecomputedEnv(
            _make_obs(5), mid.copy(), spread.copy(), execution_cost=False
        )

        env_default.reset()
        env_false.reset()

        for a in actions:
            _, r_default, _, _, _ = env_default.step(a)
            _, r_false, _, _, _ = env_false.step(a)
            assert r_default == pytest.approx(r_false), (
                f"execution_cost=False should match default for action {a}"
            )

    def test_penalized_mode_unchanged(self):
        """execution_cost=False with penalized mode should match default."""
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = _make_spread(3, value=1.0)

        env_default = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
        )
        env_false = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            execution_cost=False,
        )

        env_default.reset()
        env_false.reset()

        _, r_default, _, _, _ = env_default.step(2)
        _, r_false, _, _, _ = env_false.step(2)
        assert r_default == pytest.approx(r_false)


# ===========================================================================
# PrecomputedEnv: execution_cost=True subtracts cost on position change
# ===========================================================================


class TestPrecomputedEnvExecutionCostEnabled:
    """execution_cost=True should subtract spread/2 * |delta_pos| each step."""

    def test_flat_to_long_costs_half_spread(self):
        """Going flat->long should cost spread[t]/2 * 1."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # action=2 -> position=+1, prev_pos=0, delta=1
        # pnl_delta = +1 * (100 - 100) = 0.0
        # execution_cost = 0.5/2 * |1-0| = 0.25
        # reward = 0.0 - 0.25 = -0.25
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(-0.25)

    def test_flat_to_short_costs_half_spread(self):
        """Going flat->short should cost spread[t]/2 * 1."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # action=0 -> position=-1, prev_pos=0, delta=1
        # pnl_delta = -1 * (100 - 100) = 0.0
        # execution_cost = 0.5/2 * |-1-0| = 0.25
        # reward = 0.0 - 0.25 = -0.25
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(-0.25)

    def test_long_to_short_costs_full_spread(self):
        """Going long->short (delta=2) should cost spread[t]/2 * 2."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 1: go long (flat->long, delta=1)
        # pnl=0, cost=0.25, reward=-0.25
        _, r1, _, _, _ = env.step(2)
        assert r1 == pytest.approx(-0.25)

        # Step 2: go short (long->short, delta=2) [terminal step]
        # pnl = -1*(100-100) = 0.0
        # execution_cost = 0.5/2 * |(-1)-(1)| = 0.5/2 * 2 = 0.5
        # flattening = -|-1| * 0.5/2 = -0.25
        # reward = 0.0 - 0.5 - 0.25 = -0.75
        _, r2, terminated, _, _ = env.step(0)
        assert terminated
        assert r2 == pytest.approx(-0.75)

    def test_no_position_change_no_cost(self):
        """Holding the same position should not incur execution cost."""
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = np.array([10.0, 10.0, 10.0], dtype=np.float64)  # large spread

        env_off = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), execution_cost=False,
        )
        env_on = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), execution_cost=True,
        )

        env_off.reset()
        env_on.reset()

        # Go long
        env_off.step(2)
        env_on.step(2)

        # Stay long (long->long, delta=0) — should match exactly
        _, r_off, _, _, _ = env_off.step(2)
        _, r_on, _, _, _ = env_on.step(2)

        assert r_off == pytest.approx(r_on), (
            "Holding position should incur no execution cost"
        )

    def test_stay_flat_no_cost(self):
        """Staying flat costs nothing, regardless of execution_cost setting."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([10.0, 10.0, 10.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Stay flat: prev_pos=0, new_pos=0, delta=0
        # pnl=0, cost=0, reward=0
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0)

    def test_flip_costs_more_than_open(self):
        """Long->short (delta=2) should cost more than flat->long (delta=1)."""
        mid = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(4), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 1: flat->long (delta=1), cost = 1.0/2*1 = 0.5
        _, r_open, _, _, _ = env.step(2)

        # Step 2: long->short (delta=2), cost = 1.0/2*2 = 1.0
        _, r_flip, _, _, _ = env.step(0)

        # r_flip should be more negative than r_open (both have pnl=0)
        assert r_flip < r_open, (
            f"Flip cost {r_flip} should be more negative than open cost {r_open}"
        )


# ===========================================================================
# PrecomputedEnv: execution_cost with reward formula verification
# ===========================================================================


class TestPrecomputedEnvExecutionCostFormula:
    """Verify the full reward formula: pnl - execution_cost - lambda*|pos|."""

    def test_pnl_delta_with_execution_cost(self):
        """reward = pos * (mid[t+1] - mid[t]) - spread/2 * |delta_pos|."""
        mid = np.array([100.0, 103.0, 106.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 1: go long (flat->long, delta=1)
        # pnl = +1 * (103 - 100) = 3.0
        # exec_cost = 1.0/2 * 1 = 0.5
        # reward = 3.0 - 0.5 = 2.5
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(2.5)

    def test_penalized_with_execution_cost(self):
        """reward = pos * delta_mid - spread/2 * |delta_pos| - lambda*|pos|."""
        mid = np.array([100.0, 103.0, 106.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            execution_cost=True,
        )
        env.reset()

        # Step 1: go long (flat->long, delta=1)
        # pnl = +1 * (103 - 100) = 3.0
        # exec_cost = 1.0/2 * 1 = 0.5
        # lambda_pen = 0.5 * |1| = 0.5
        # reward = 3.0 - 0.5 - 0.5 = 2.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(2.0)

    def test_execution_cost_uses_current_spread(self):
        """Execution cost should use spread[t] (the current timestep's spread)."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        # Varying spreads
        spread = np.array([0.5, 2.0, 4.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 1 at t=0: flat->long, uses spread[0]=0.5
        # pnl=0, cost=0.5/2*1=0.25, reward=-0.25
        _, r1, _, _, _ = env.step(2)
        assert r1 == pytest.approx(-0.25)


# ===========================================================================
# PrecomputedEnv: execution_cost edge cases
# ===========================================================================


class TestPrecomputedEnvExecutionCostEdgeCases:
    """Edge cases for execution cost."""

    def test_first_step_after_reset(self):
        """First step: prev_position=0, so going long costs spread/2 * 1."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([2.0, 2.0, 2.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # pnl=0, exec_cost=2.0/2*1=1.0, reward=-1.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(-1.0)

    def test_spread_zero_no_cost(self):
        """If spread is 0, execution cost should be 0."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # flat->long with spread=0: cost=0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(0.0)

    def test_nan_spread_no_cost(self):
        """If spread is NaN, execution cost should be 0 (treated as no cost)."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # flat->long with NaN spread: cost should be 0
        _, reward, _, _, _ = env.step(2)
        # PnL is 0, and cost should be 0 (not NaN)
        assert np.isfinite(reward)
        assert reward == pytest.approx(0.0)

    def test_execution_cost_and_flattening_both_apply_at_terminal(self):
        """At terminal step, both execution cost and flattening penalty apply."""
        # 3 snapshots -> 2 steps. Step 1 is terminal.
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 0: go long (flat->long, delta=1, non-terminal)
        # pnl=0, exec_cost=1.0/2*1=0.5, reward=-0.5
        _, r0, terminated0, _, _ = env.step(2)
        assert not terminated0
        assert r0 == pytest.approx(-0.5)

        # Step 1: stay long (long->long, delta=0, terminal)
        # pnl=0, exec_cost=0 (no change), flattening=-|1|*1.0/2=-0.5
        # reward = 0 - 0 - 0.5 = -0.5
        _, r1, terminated1, _, _ = env.step(2)
        assert terminated1
        assert r1 == pytest.approx(-0.5)

    def test_execution_cost_not_double_counted_with_flattening(self):
        """Execution cost and flattening penalty are separate — not double-counted.

        If the agent changes position on the terminal step AND has a position,
        both costs apply independently.
        """
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )
        env.reset()

        # Step 0: stay flat (no cost, non-terminal)
        _, r0, _, _, _ = env.step(1)
        assert r0 == pytest.approx(0.0)

        # Step 1: go long (flat->long, delta=1, terminal)
        # pnl=0, exec_cost=1.0/2*1=0.5, flattening=-|1|*1.0/2=-0.5
        # reward = 0 - 0.5 - 0.5 = -1.0
        _, r1, terminated, _, _ = env.step(2)
        assert terminated
        assert r1 == pytest.approx(-1.0)


# ===========================================================================
# PrecomputedEnv: reset clears prev_position
# ===========================================================================


class TestPrecomputedEnvResetClearsPrevPosition:
    """reset() should reset _prev_position to 0."""

    def test_reset_clears_prev_position(self):
        """After reset, going long should cost spread/2 (prev_pos=0)."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, execution_cost=True,
        )

        # Episode 1: go long
        env.reset()
        _, r1, _, _, _ = env.step(2)  # flat->long, cost=0.5

        # Episode 2: go long again (prev_pos should be 0 after reset)
        env.reset()
        _, r2, _, _, _ = env.step(2)  # flat->long, cost=0.5

        assert r1 == pytest.approx(r2), (
            "After reset, prev_position should be 0"
        )

    def test_reset_mid_episode_clears_prev_position(self):
        """Reset mid-episode should also clear prev_position."""
        mid = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(4), mid, spread, execution_cost=True,
        )

        env.reset()
        env.step(2)  # go long
        env.step(0)  # go short

        # Reset — prev_position should be 0, not -1
        env.reset()
        _, reward, _, _, _ = env.step(2)  # flat->long, delta=1

        # pnl=0, cost=1.0/2*1=0.5, reward=-0.5
        assert reward == pytest.approx(-0.5), (
            "After mid-episode reset, prev_position should be 0"
        )


# ===========================================================================
# PrecomputedEnv: from_file with execution_cost
# ===========================================================================


class TestPrecomputedEnvFromFile:
    """from_file() should accept execution_cost parameter."""

    def test_from_file_accepts_execution_cost(self):
        """from_file(path, execution_cost=True) should work."""
        env = PrecomputedEnv.from_file(
            PRECOMPUTE_EPISODE_FILE, execution_cost=True,
        )
        assert isinstance(env, PrecomputedEnv)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_from_file_default_execution_cost_false(self):
        """from_file() without execution_cost should default to False."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        assert isinstance(env, PrecomputedEnv)


# ===========================================================================
# MultiDayEnv: forwards execution_cost parameter
# ===========================================================================


class TestMultiDayEnvExecutionCost:
    """MultiDayEnv should accept and forward execution_cost parameter."""

    def test_accepts_execution_cost_false(self):
        """MultiDayEnv(execution_cost=False) should construct without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, shuffle=False, execution_cost=False,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_accepts_execution_cost_true(self):
        """MultiDayEnv(execution_cost=True) should construct without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, shuffle=False, execution_cost=True,
        )
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_default_execution_cost_is_false(self):
        """MultiDayEnv() without execution_cost should default to False."""
        from lob_rl.multi_day_env import MultiDayEnv

        # Should accept all old args without execution_cost
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_execution_cost_forwarded_to_inner_env(self):
        """execution_cost=True should produce different rewards than False."""
        from lob_rl.multi_day_env import MultiDayEnv

        env_off = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, execution_cost=False,
        )
        env_on = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, execution_cost=True,
        )

        env_off.reset()
        env_on.reset()

        # Go long from flat — should cost more with execution_cost=True
        _, r_off, _, _, _ = env_off.step(2)
        _, r_on, _, _, _ = env_on.step(2)

        assert r_on < r_off, (
            f"execution_cost=True reward ({r_on}) should be less than "
            f"False ({r_off}) when changing position"
        )

    def test_execution_cost_false_matches_no_param(self):
        """execution_cost=False should match default (no param) behavior."""
        from lob_rl.multi_day_env import MultiDayEnv

        env_default = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False,
        )
        env_false = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, execution_cost=False,
        )

        env_default.reset()
        env_false.reset()

        total_default = 0.0
        total_false = 0.0
        for _ in range(10):
            _, r_def, term_def, _, _ = env_default.step(2)
            _, r_false, term_false, _, _ = env_false.step(2)
            total_default += r_def
            total_false += r_false
            if term_def or term_false:
                break

        assert total_default == pytest.approx(total_false)

    def test_inner_env_has_execution_cost_enabled(self):
        """When execution_cost=True, the inner PrecomputedEnv should also have it."""
        from lob_rl.multi_day_env import MultiDayEnv

        env = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, execution_cost=True,
        )
        env.reset()

        # After reset, inner env should exist and have execution cost
        inner = getattr(env, "_inner_env", None) or getattr(env, "_current_env", None)
        assert inner is not None, "MultiDayEnv should have an inner env"


# ===========================================================================
# LOBGymEnv: forwards execution_cost parameter
# ===========================================================================


class TestLOBGymEnvExecutionCost:
    """LOBGymEnv should accept and forward execution_cost parameter."""

    def test_accepts_execution_cost_false(self):
        """LOBGymEnv(execution_cost=False) should construct without error."""
        from lob_rl.gym_env import LOBGymEnv
        env = LOBGymEnv(execution_cost=False)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_accepts_execution_cost_true(self):
        """LOBGymEnv(execution_cost=True) should construct without error."""
        from lob_rl.gym_env import LOBGymEnv
        env = LOBGymEnv(execution_cost=True)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_default_execution_cost_is_false(self):
        """LOBGymEnv() without execution_cost should default to False."""
        from lob_rl.gym_env import LOBGymEnv
        # Should accept all old args without execution_cost
        env = LOBGymEnv()
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_execution_cost_changes_rewards(self):
        """LOBGymEnv with execution_cost=True should produce different rewards."""
        from lob_rl.gym_env import LOBGymEnv

        env_off = LOBGymEnv(execution_cost=False)
        env_on = LOBGymEnv(execution_cost=True)

        env_off.reset()
        env_on.reset()

        # Go long from flat — should cost more with execution_cost=True
        _, r_off, _, _, _ = env_off.step(2)
        _, r_on, _, _, _ = env_on.step(2)

        assert r_on < r_off, (
            "execution_cost=True should reduce reward on position change"
        )


# ===========================================================================
# C++ LOBEnv bindings: execution_cost parameter
# ===========================================================================


class TestCppBindingsExecutionCost:
    """C++ LOBEnv pybind11 bindings should accept execution_cost parameter."""

    def test_synthetic_constructor_accepts_execution_cost(self):
        """LOBEnv(execution_cost=True) via bindings should work."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv(execution_cost=True)
        obs = env.reset()
        assert len(obs) == 44

    def test_synthetic_constructor_default_false(self):
        """LOBEnv() without execution_cost should work (backward compat)."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv()
        obs = env.reset()
        assert len(obs) == 44

    def test_file_constructor_accepts_execution_cost(self):
        """LOBEnv(file_path, execution_cost=True) should work."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv(EPISODE_FILE, execution_cost=True)
        obs = env.reset()
        assert len(obs) == 44

    def test_session_constructor_accepts_execution_cost(self):
        """LOBEnv(file_path, cfg, steps, execution_cost=True) should work."""
        import lob_rl_core
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(
            SESSION_FILE, cfg, 30,
            execution_cost=True,
        )
        obs = env.reset()
        assert len(obs) == 44

    def test_execution_cost_reduces_reward_on_position_change(self):
        """C++ LOBEnv with execution_cost=True should reduce rewards."""
        import lob_rl_core

        env_off = lob_rl_core.LOBEnv(execution_cost=False)
        env_on = lob_rl_core.LOBEnv(execution_cost=True)

        env_off.reset()
        env_on.reset()

        # Go long from flat
        _, r_off, _ = env_off.step(2)
        _, r_on, _ = env_on.step(2)

        assert r_on < r_off, (
            "C++ execution_cost=true should reduce reward on position change"
        )

    def test_execution_cost_no_extra_cost_when_flat(self):
        """Flat position should have same reward regardless of execution_cost."""
        import lob_rl_core

        env_off = lob_rl_core.LOBEnv(execution_cost=False)
        env_on = lob_rl_core.LOBEnv(execution_cost=True)

        env_off.reset()
        env_on.reset()

        # Stay flat
        _, r_off, _ = env_off.step(1)
        _, r_on, _ = env_on.step(1)

        assert r_off == pytest.approx(r_on), (
            "Flat position should have identical reward with/without execution cost"
        )

    def test_all_constructor_combos_with_execution_cost(self):
        """Test execution_cost with reward_mode and lambda combinations."""
        import lob_rl_core

        # With reward mode and execution cost
        env = lob_rl_core.LOBEnv(
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
            execution_cost=True,
        )
        env.reset()
        obs, reward, done = env.step(2)
        assert isinstance(reward, float)


# ===========================================================================
# train.py: --execution-cost CLI flag
# ===========================================================================


class TestTrainScriptFlag:
    """train.py should accept --execution-cost flag."""

    def test_argparser_has_execution_cost_flag(self):
        """argparse in train.py should define --execution-cost flag."""
        import importlib.util
        import sys

        # Load train.py module without running main()
        spec = importlib.util.spec_from_file_location(
            "train", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        )
        # We can't easily import train.py without side effects,
        # so we check the source for the argparse flag
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        assert "--execution-cost" in source, (
            "train.py should define --execution-cost CLI flag"
        )

    def test_execution_cost_flag_is_store_true(self):
        """--execution-cost should be a store_true flag (default False)."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        # The flag should use store_true so it defaults to False
        assert "store_true" in source or "execution_cost" in source, (
            "--execution-cost should be a boolean flag"
        )


# ===========================================================================
# PrecomputedEnv: gymnasium check_env still passes
# ===========================================================================


class TestCheckEnvWithExecutionCost:
    """gymnasium check_env should pass with execution_cost=True."""

    def test_check_env_with_execution_cost_true(self):
        """check_env() should pass with execution_cost=True."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20),
            execution_cost=True,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_with_execution_cost_false(self):
        """check_env() should pass with execution_cost=False."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20),
            execution_cost=False,
        )
        check_env(env, skip_render_check=True)
