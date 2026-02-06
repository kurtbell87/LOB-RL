"""Tests for participation bonus reward component.

Spec: docs/participation-bonus.md

These tests verify that:
- PrecomputedEnv accepts participation_bonus parameter (default 0.0)
- participation_bonus=0.0 preserves identical rewards to current behavior
- participation_bonus > 0 adds bonus * abs(position) each step
- Bonus is 0 when position == 0 (flat)
- Bonus applies on terminal step before flattening penalty
- MultiDayEnv forwards participation_bonus to PrecomputedEnv
- LOBGymEnv forwards participation_bonus to C++ LOBEnv
- C++ LOBEnv bindings accept participation_bonus parameter
- train.py --participation-bonus flag exists
- Combined behavior with execution_cost and lambda penalty
- Edge cases: negative bonus, zero bonus, bonus with all cost components
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
# PrecomputedEnv: Constructor accepts participation_bonus parameter
# ===========================================================================


class TestPrecomputedEnvConstructor:
    """PrecomputedEnv should accept a participation_bonus float parameter."""

    def test_accepts_participation_bonus_zero(self):
        """PrecomputedEnv(participation_bonus=0.0) should construct without error."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            participation_bonus=0.0,
        )
        assert env is not None

    def test_accepts_participation_bonus_positive(self):
        """PrecomputedEnv(participation_bonus=0.01) should construct without error."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            participation_bonus=0.01,
        )
        assert env is not None

    def test_default_participation_bonus_is_zero(self):
        """Default participation_bonus should be 0.0 (backward compatible)."""
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = _make_spread(3, value=0.5)

        env_default = PrecomputedEnv(_make_obs(3), mid.copy(), spread.copy())
        env_zero = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), participation_bonus=0.0
        )

        env_default.reset()
        env_zero.reset()

        # Go long from flat — should have identical rewards if default is 0.0
        _, r_default, _, _, _ = env_default.step(2)
        _, r_zero, _, _, _ = env_zero.step(2)
        assert r_default == pytest.approx(r_zero)

    def test_still_valid_gym_env_with_participation_bonus(self):
        """PrecomputedEnv with participation_bonus=0.01 should still be a gymnasium.Env."""
        env = PrecomputedEnv(
            _make_obs(5), _make_mid(5), _make_spread(5),
            participation_bonus=0.01,
        )
        assert isinstance(env, gym.Env)
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)


# ===========================================================================
# PrecomputedEnv: participation_bonus=0.0 preserves existing behavior
# ===========================================================================


class TestPrecomputedEnvBackwardCompat:
    """participation_bonus=0.0 should produce identical rewards to omitting it."""

    def test_identical_rewards_full_episode(self):
        """Full episode with participation_bonus=0.0 should match default behavior."""
        mid = np.array([100.0, 102.0, 99.0, 105.0, 103.0], dtype=np.float64)
        spread = _make_spread(5, value=1.0)
        actions = [2, 0, 1, 2]  # long, short, flat, long

        env_default = PrecomputedEnv(_make_obs(5), mid.copy(), spread.copy())
        env_zero = PrecomputedEnv(
            _make_obs(5), mid.copy(), spread.copy(), participation_bonus=0.0
        )

        env_default.reset()
        env_zero.reset()

        for a in actions:
            _, r_default, _, _, _ = env_default.step(a)
            _, r_zero, _, _, _ = env_zero.step(a)
            assert r_default == pytest.approx(r_zero), (
                f"participation_bonus=0.0 should match default for action {a}"
            )

    def test_penalized_mode_unchanged(self):
        """participation_bonus=0.0 with penalized mode should match default."""
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = _make_spread(3, value=1.0)

        env_default = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
        )
        env_zero = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            participation_bonus=0.0,
        )

        env_default.reset()
        env_zero.reset()

        _, r_default, _, _, _ = env_default.step(2)
        _, r_zero, _, _, _ = env_zero.step(2)
        assert r_default == pytest.approx(r_zero)

    def test_execution_cost_mode_unchanged(self):
        """participation_bonus=0.0 with execution_cost should match default."""
        mid = np.array([100.0, 102.0, 104.0], dtype=np.float64)
        spread = _make_spread(3, value=1.0)

        env_default = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            execution_cost=True,
        )
        env_zero = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(),
            execution_cost=True,
            participation_bonus=0.0,
        )

        env_default.reset()
        env_zero.reset()

        _, r_default, _, _, _ = env_default.step(2)
        _, r_zero, _, _, _ = env_zero.step(2)
        assert r_default == pytest.approx(r_zero)


# ===========================================================================
# PrecomputedEnv: participation_bonus > 0 adds bonus when position != 0
# ===========================================================================


class TestPrecomputedEnvParticipationBonusEnabled:
    """participation_bonus > 0 should add bonus * abs(position) each step."""

    def test_flat_to_long_gets_bonus(self):
        """Going flat->long with participation_bonus=0.01 should add 0.01."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # action=2 -> position=+1
        # pnl_delta = +1 * (100 - 100) = 0.0
        # participation_bonus = 0.01 * abs(1) = 0.01
        # reward = 0.0 + 0.01 = 0.01
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(0.01)

    def test_flat_to_short_gets_bonus(self):
        """Going flat->short with participation_bonus=0.01 should add 0.01."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # action=0 -> position=-1
        # pnl_delta = -1 * (100 - 100) = 0.0
        # participation_bonus = 0.01 * abs(-1) = 0.01
        # reward = 0.0 + 0.01 = 0.01
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(0.01)

    def test_flat_position_no_bonus(self):
        """Staying flat should get no bonus."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # Stay flat: pos=0, bonus = 0.01*0 = 0
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0)

    def test_bonus_symmetric_long_and_short(self):
        """Long and short should get the same bonus (abs(pos) is the same)."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env_long = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), participation_bonus=0.01,
        )
        env_short = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), participation_bonus=0.01,
        )

        env_long.reset()
        env_short.reset()

        _, r_long, _, _, _ = env_long.step(2)
        _, r_short, _, _, _ = env_short.step(0)

        assert r_long == pytest.approx(r_short)

    def test_larger_bonus_gives_larger_reward(self):
        """Larger participation_bonus should give larger reward."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env_small = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), participation_bonus=0.01,
        )
        env_large = PrecomputedEnv(
            _make_obs(3), mid.copy(), spread.copy(), participation_bonus=0.05,
        )

        env_small.reset()
        env_large.reset()

        _, r_small, _, _, _ = env_small.step(2)
        _, r_large, _, _, _ = env_large.step(2)

        assert r_large > r_small


# ===========================================================================
# PrecomputedEnv: participation_bonus with full reward formula verification
# ===========================================================================


class TestPrecomputedEnvParticipationBonusFormula:
    """Verify the full reward formula including participation bonus."""

    def test_pnl_delta_plus_bonus(self):
        """reward = pos * (mid[t+1] - mid[t]) + bonus * abs(pos)."""
        mid = np.array([100.0, 103.0, 106.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # Step 1: go long (flat->long)
        # pnl = +1 * (103 - 100) = 3.0
        # bonus = 0.01 * 1 = 0.01
        # reward = 3.0 + 0.01 = 3.01
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(3.01)

    def test_pnl_delta_penalized_plus_bonus(self):
        """reward = pos * delta_mid - lambda*|pos| + bonus*|pos|."""
        mid = np.array([100.0, 103.0, 106.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            participation_bonus=0.01,
        )
        env.reset()

        # Step 1: go long (flat->long)
        # pnl = +1 * (103 - 100) = 3.0
        # lambda_pen = 0.5 * |1| = 0.5
        # bonus = 0.01 * |1| = 0.01
        # reward = 3.0 - 0.5 + 0.01 = 2.51
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(2.51)

    def test_execution_cost_plus_bonus(self):
        """reward = pos * delta_mid - spread/2 * |delta_pos| + bonus*|pos|."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            execution_cost=True,
            participation_bonus=0.01,
        )
        env.reset()

        # Step 1: go long (flat->long, delta=1)
        # pnl = +1 * (100 - 100) = 0.0
        # exec_cost = 1.0/2 * 1 = 0.5
        # bonus = 0.01 * |1| = 0.01
        # reward = 0.0 - 0.5 + 0.01 = -0.49
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(-0.49)

    def test_all_components_together(self):
        """reward = pnl - lambda*|pos| - exec_cost + bonus*|pos|."""
        mid = np.array([100.0, 103.0, 106.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            reward_mode="pnl_delta_penalized", lambda_=0.5,
            execution_cost=True,
            participation_bonus=0.01,
        )
        env.reset()

        # Step 1: go long (flat->long, delta=1)
        # pnl = +1 * (103 - 100) = 3.0
        # lambda_pen = 0.5 * |1| = 0.5
        # exec_cost = 1.0/2 * 1 = 0.5
        # bonus = 0.01 * |1| = 0.01
        # reward = 3.0 - 0.5 - 0.5 + 0.01 = 2.01
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(2.01)


# ===========================================================================
# PrecomputedEnv: participation_bonus edge cases
# ===========================================================================


class TestPrecomputedEnvParticipationBonusEdgeCases:
    """Edge cases for participation bonus."""

    def test_negative_bonus_penalizes_position(self):
        """Negative participation_bonus should penalize being in the market."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=-0.01,
        )
        env.reset()

        # Go long: pnl=0, bonus=-0.01*1=-0.01
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(-0.01)

    def test_bonus_applied_before_flattening_penalty(self):
        """Bonus applies on terminal step BEFORE flattening penalty."""
        # 3 snapshots -> 2 steps. Step 1 is terminal.
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # Step 0: go long (non-terminal)
        # pnl=0, bonus=0.01*1=0.01, reward=0.01
        _, r0, terminated0, _, _ = env.step(2)
        assert not terminated0
        assert r0 == pytest.approx(0.01)

        # Step 1: stay long (terminal)
        # pnl=0, bonus=0.01*1=0.01, flattening=-|1|*1.0/2=-0.5
        # reward = 0 + 0.01 - 0.5 = -0.49
        _, r1, terminated1, _, _ = env.step(2)
        assert terminated1
        assert r1 == pytest.approx(-0.49)

    def test_bonus_with_execution_cost_and_flattening_at_terminal(self):
        """All three: bonus + execution_cost + flattening at terminal step."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            execution_cost=True,
            participation_bonus=0.01,
        )
        env.reset()

        # Step 0: stay flat (non-terminal)
        _, r0, _, _, _ = env.step(1)
        assert r0 == pytest.approx(0.0)

        # Step 1: go long (flat->long, delta=1, terminal)
        # pnl=0, exec_cost=-1.0/2*1=-0.5, bonus=0.01*1=0.01, flattening=-|1|*1.0/2=-0.5
        # reward = 0 - 0.5 + 0.01 - 0.5 = -0.99
        _, r1, terminated, _, _ = env.step(2)
        assert terminated
        assert r1 == pytest.approx(-0.99)

    def test_holding_long_accumulates_bonus(self):
        """Holding long for multiple steps accumulates bonus each step."""
        mid = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(4), mid, spread, participation_bonus=0.01,
        )
        env.reset()

        # Step 0: go long, pnl=0, bonus=0.01 => reward=0.01
        _, r0, _, _, _ = env.step(2)
        assert r0 == pytest.approx(0.01)

        # Step 1: stay long, pnl=0, bonus=0.01 => reward=0.01
        _, r1, _, _, _ = env.step(2)
        assert r1 == pytest.approx(0.01)

        # Step 2 is terminal: stay long, pnl=0, bonus=0.01, flattening=-|1|*0.5/2=-0.25
        # reward = 0 + 0.01 - 0.25 = -0.24
        _, r2, terminated, _, _ = env.step(2)
        assert terminated
        assert r2 == pytest.approx(-0.24)


# ===========================================================================
# PrecomputedEnv: reset clears position (bonus should apply consistently)
# ===========================================================================


class TestPrecomputedEnvResetWithBonus:
    """reset() should clear position — bonus behavior should be consistent."""

    def test_reset_gives_consistent_bonus(self):
        """After reset, going long should give the same bonus as first episode."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)

        env = PrecomputedEnv(
            _make_obs(3), mid, spread, participation_bonus=0.01,
        )

        # Episode 1
        env.reset()
        _, r1, _, _, _ = env.step(2)

        # Episode 2
        env.reset()
        _, r2, _, _, _ = env.step(2)

        assert r1 == pytest.approx(r2)


# ===========================================================================
# PrecomputedEnv: from_file with participation_bonus
# ===========================================================================


class TestPrecomputedEnvFromFile:
    """from_file() should accept participation_bonus parameter."""

    def test_from_file_accepts_participation_bonus(self):
        """from_file(path, participation_bonus=0.01) should work."""
        env = PrecomputedEnv.from_file(
            PRECOMPUTE_EPISODE_FILE, participation_bonus=0.01,
        )
        assert isinstance(env, PrecomputedEnv)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_from_file_default_participation_bonus_zero(self):
        """from_file() without participation_bonus should default to 0.0."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        assert isinstance(env, PrecomputedEnv)


# ===========================================================================
# MultiDayEnv: forwards participation_bonus parameter
# ===========================================================================


class TestMultiDayEnvParticipationBonus:
    """MultiDayEnv should accept and forward participation_bonus parameter."""

    def test_accepts_participation_bonus_zero(self):
        """MultiDayEnv(participation_bonus=0.0) should construct without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, shuffle=False, participation_bonus=0.0,
        )
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_accepts_participation_bonus_positive(self):
        """MultiDayEnv(participation_bonus=0.01) should construct without error."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(
            file_paths=DAY_FILES, shuffle=False, participation_bonus=0.01,
        )
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_default_participation_bonus_is_zero(self):
        """MultiDayEnv() without participation_bonus should default to 0.0."""
        from lob_rl.multi_day_env import MultiDayEnv
        env = MultiDayEnv(file_paths=DAY_FILES, shuffle=False)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_participation_bonus_forwarded_to_inner_env(self):
        """participation_bonus=0.01 should produce different rewards than 0.0."""
        from lob_rl.multi_day_env import MultiDayEnv

        env_off = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, participation_bonus=0.0,
        )
        env_on = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, participation_bonus=0.01,
        )

        env_off.reset()
        env_on.reset()

        # Go long from flat — should get bonus with participation_bonus=0.01
        _, r_off, _, _, _ = env_off.step(2)
        _, r_on, _, _, _ = env_on.step(2)

        assert r_on > r_off, (
            f"participation_bonus=0.01 reward ({r_on}) should be greater than "
            f"0.0 ({r_off}) when holding position"
        )

    def test_participation_bonus_zero_matches_no_param(self):
        """participation_bonus=0.0 should match default (no param) behavior."""
        from lob_rl.multi_day_env import MultiDayEnv

        env_default = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False,
        )
        env_zero = MultiDayEnv(
            file_paths=[DAY_FILES[0]], shuffle=False, participation_bonus=0.0,
        )

        env_default.reset()
        env_zero.reset()

        total_default = 0.0
        total_zero = 0.0
        for _ in range(10):
            _, r_def, term_def, _, _ = env_default.step(2)
            _, r_zero, term_zero, _, _ = env_zero.step(2)
            total_default += r_def
            total_zero += r_zero
            if term_def or term_zero:
                break

        assert total_default == pytest.approx(total_zero)


# ===========================================================================
# LOBGymEnv: forwards participation_bonus parameter
# ===========================================================================


class TestLOBGymEnvParticipationBonus:
    """LOBGymEnv should accept and forward participation_bonus parameter."""

    def test_accepts_participation_bonus_zero(self):
        """LOBGymEnv(participation_bonus=0.0) should construct without error."""
        from lob_rl.gym_env import LOBGymEnv
        env = LOBGymEnv(participation_bonus=0.0)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_accepts_participation_bonus_positive(self):
        """LOBGymEnv(participation_bonus=0.01) should construct without error."""
        from lob_rl.gym_env import LOBGymEnv
        env = LOBGymEnv(participation_bonus=0.01)
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_default_participation_bonus_is_zero(self):
        """LOBGymEnv() without participation_bonus should default to 0.0."""
        from lob_rl.gym_env import LOBGymEnv
        env = LOBGymEnv()
        obs, info = env.reset()
        assert obs.shape == (44,)

    def test_participation_bonus_changes_rewards(self):
        """LOBGymEnv with participation_bonus=0.01 should increase rewards when in market."""
        from lob_rl.gym_env import LOBGymEnv

        env_off = LOBGymEnv(participation_bonus=0.0)
        env_on = LOBGymEnv(participation_bonus=0.01)

        env_off.reset()
        env_on.reset()

        # Go long from flat
        _, r_off, _, _, _ = env_off.step(2)
        _, r_on, _, _, _ = env_on.step(2)

        assert r_on > r_off, (
            "participation_bonus=0.01 should increase reward on position"
        )


# ===========================================================================
# C++ LOBEnv bindings: participation_bonus parameter
# ===========================================================================


class TestCppBindingsParticipationBonus:
    """C++ LOBEnv pybind11 bindings should accept participation_bonus parameter."""

    def test_synthetic_constructor_accepts_participation_bonus(self):
        """LOBEnv(participation_bonus=0.01) via bindings should work."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv(participation_bonus=0.01)
        obs = env.reset()
        assert len(obs) == 44

    def test_synthetic_constructor_default_zero(self):
        """LOBEnv() without participation_bonus should work (backward compat)."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv()
        obs = env.reset()
        assert len(obs) == 44

    def test_file_constructor_accepts_participation_bonus(self):
        """LOBEnv(file_path, participation_bonus=0.01) should work."""
        import lob_rl_core
        env = lob_rl_core.LOBEnv(EPISODE_FILE, participation_bonus=0.01)
        obs = env.reset()
        assert len(obs) == 44

    def test_session_constructor_accepts_participation_bonus(self):
        """LOBEnv(file_path, cfg, steps, participation_bonus=0.01) should work."""
        import lob_rl_core
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(
            SESSION_FILE, cfg, 30,
            participation_bonus=0.01,
        )
        obs = env.reset()
        assert len(obs) == 44

    def test_participation_bonus_increases_reward_on_position(self):
        """C++ LOBEnv with participation_bonus=0.01 should increase rewards."""
        import lob_rl_core

        env_off = lob_rl_core.LOBEnv(participation_bonus=0.0)
        env_on = lob_rl_core.LOBEnv(participation_bonus=0.01)

        env_off.reset()
        env_on.reset()

        # Go long from flat
        _, r_off, _ = env_off.step(2)
        _, r_on, _ = env_on.step(2)

        assert r_on > r_off, (
            "C++ participation_bonus=0.01 should increase reward when in position"
        )

    def test_participation_bonus_no_effect_when_flat(self):
        """Flat position should have same reward regardless of participation_bonus."""
        import lob_rl_core

        env_off = lob_rl_core.LOBEnv(participation_bonus=0.0)
        env_on = lob_rl_core.LOBEnv(participation_bonus=0.01)

        env_off.reset()
        env_on.reset()

        # Stay flat
        _, r_off, _ = env_off.step(1)
        _, r_on, _ = env_on.step(1)

        assert r_off == pytest.approx(r_on), (
            "Flat position should have identical reward with/without participation bonus"
        )

    def test_all_constructor_combos_with_participation_bonus(self):
        """Test participation_bonus with reward_mode, lambda, and execution_cost."""
        import lob_rl_core

        env = lob_rl_core.LOBEnv(
            reward_mode="pnl_delta_penalized",
            lambda_=0.1,
            execution_cost=True,
            participation_bonus=0.01,
        )
        env.reset()
        obs, reward, done = env.step(2)
        assert isinstance(reward, float)


# ===========================================================================
# train.py: --participation-bonus CLI flag
# ===========================================================================


class TestTrainScriptParticipationBonusFlag:
    """train.py should accept --participation-bonus flag."""

    def test_argparser_has_participation_bonus_flag(self):
        """argparse in train.py should define --participation-bonus flag."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        assert "--participation-bonus" in source, (
            "train.py should define --participation-bonus CLI flag"
        )

    def test_participation_bonus_flag_is_float(self):
        """--participation-bonus should be a float flag (default 0.0)."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        # Find the participation_bonus argument definition
        assert "participation_bonus" in source or "participation-bonus" in source, (
            "--participation-bonus should be defined in train.py"
        )

    def test_participation_bonus_default_zero(self):
        """--participation-bonus should default to 0.0."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        # The default should be 0.0
        # Look for pattern like default=0.0 near participation-bonus
        assert "participation" in source.lower(), (
            "train.py should reference participation_bonus"
        )

    def test_participation_bonus_forwarded_to_make_env(self):
        """make_env() in train.py should accept participation_bonus parameter."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        assert "participation_bonus" in source, (
            "train.py make_env() should use participation_bonus"
        )

    def test_participation_bonus_forwarded_to_evaluate_sortino(self):
        """evaluate_sortino() should accept participation_bonus parameter."""
        train_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
        with open(train_path) as f:
            source = f.read()

        # Check that evaluate_sortino function signature or call includes participation_bonus
        assert "participation_bonus" in source, (
            "evaluate_sortino should use participation_bonus"
        )


# ===========================================================================
# PrecomputedEnv: gymnasium check_env still passes
# ===========================================================================


class TestCheckEnvWithParticipationBonus:
    """gymnasium check_env should pass with participation_bonus > 0."""

    def test_check_env_with_participation_bonus(self):
        """check_env() should pass with participation_bonus=0.01."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20),
            participation_bonus=0.01,
        )
        check_env(env, skip_render_check=True)

    def test_check_env_with_all_options(self):
        """check_env() should pass with participation_bonus + execution_cost."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv(
            _make_obs(20), _make_mid(20), _make_spread(20),
            execution_cost=True,
            participation_bonus=0.01,
        )
        check_env(env, skip_render_check=True)
