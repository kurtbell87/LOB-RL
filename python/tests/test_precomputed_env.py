"""Tests for PrecomputedEnv — pure-numpy gymnasium env for pre-computed LOB data.

Spec: docs/precomputed-env.md

These tests verify that:
- PrecomputedEnv is a valid gymnasium.Env with correct spaces
- reset() returns 54-float obs with position=0 at index 53
- step() returns correct 5-tuple (obs, reward, terminated, truncated, info)
- Action mapping: {0->-1, 1->0, 2->+1} reflected in obs[53]
- Reward = position * (mid[t+1] - mid[t]) for PnLDelta
- PnLDeltaPenalized subtracts lambda_ * |position|
- Episode terminates at t == N-1 (N snapshots -> N-1 steps)
- Flattening penalty at terminal step: -|position| * spread[t] / 2
- from_file() classmethod works
- Passes gymnasium.utils.env_checker.check_env()
- Constructor validates obs has >= 2 rows
"""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from lob_rl.precomputed_env import PrecomputedEnv

from conftest import PRECOMPUTE_EPISODE_FILE, make_obs as _make_obs, make_mid as _make_mid, make_spread as _make_spread


# ===========================================================================
# Test 1: Constructor stores arrays correctly and sets spaces
# ===========================================================================


class TestConstructor:
    """PrecomputedEnv constructor should store arrays and define gym spaces."""

    def test_constructor_accepts_arrays(self):
        """PrecomputedEnv(obs, mid, spread) should construct without error."""
        obs = _make_obs(5)
        mid = _make_mid(5)
        spread = _make_spread(5)
        env = PrecomputedEnv(obs, mid, spread)
        assert env is not None

    def test_observation_space_is_box(self):
        """observation_space should be a gymnasium.spaces.Box."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert isinstance(env.observation_space, spaces.Box)

    def test_observation_space_shape_is_44(self):
        """observation_space shape should be (54,) — 43 market + 10 temporal + 1 position."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert env.observation_space.shape == (54,)

    def test_observation_space_dtype_float32(self):
        """observation_space dtype should be float32."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert env.observation_space.dtype == np.float32

    def test_observation_space_bounds_infinite(self):
        """observation_space should have -inf/+inf bounds."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert np.all(env.observation_space.low == -np.inf)
        assert np.all(env.observation_space.high == np.inf)

    def test_action_space_is_discrete_3(self):
        """action_space should be Discrete(3)."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_is_gymnasium_env(self):
        """PrecomputedEnv should be a subclass of gymnasium.Env."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        assert isinstance(env, gym.Env)

    def test_metadata_has_render_modes(self):
        """metadata should include render_modes key."""
        assert "render_modes" in PrecomputedEnv.metadata


# ===========================================================================
# Test 2: reset returns 54-float observation with position=0
# ===========================================================================


class TestReset:
    """reset() should return (obs_54, info) with position=0."""

    def test_reset_returns_tuple_of_two(self):
        """reset() should return a 2-tuple."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape_is_44(self):
        """reset() observation should have shape (54,)."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_reset_obs_dtype_is_float32(self):
        """reset() observation dtype should be float32."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        assert obs.dtype == np.float32

    def test_reset_position_is_zero(self):
        """reset() should set position to 0.0 at index 53."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        assert obs[53] == pytest.approx(0.0)

    def test_reset_obs_first_43_match_obs_row_0(self):
        """reset() obs[:43] should match the first row of the input obs array."""
        input_obs = _make_obs(5, fill=42.0)
        env = PrecomputedEnv(input_obs, _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])

    def test_reset_info_is_empty_dict(self):
        """reset() should return an empty dict as info."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        assert info == {}

    def test_reset_accepts_seed_kwarg(self):
        """reset(seed=42) should not raise."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset(seed=42)
        assert obs.shape == (54,)

    def test_reset_accepts_options_kwarg(self):
        """reset(options={}) should not raise."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset(options={})
        assert obs.shape == (54,)

    def test_reset_obs_in_observation_space(self):
        """reset() observation should be within observation_space."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, info = env.reset()
        assert env.observation_space.contains(obs)


# ===========================================================================
# Test 3: step returns correct 5-tuple format
# ===========================================================================


class TestStepFormat:
    """step() should return (obs, reward, terminated, truncated, info)."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        self.env = PrecomputedEnv(
            _make_obs(10), _make_mid(10), _make_spread(10)
        )
        self.env.reset()

    def test_step_returns_5_tuple(self):
        """step() should return exactly 5 elements."""
        result = self.env.step(1)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_is_ndarray(self):
        """step() obs should be a numpy ndarray."""
        obs, _, _, _, _ = self.env.step(1)
        assert isinstance(obs, np.ndarray)

    def test_step_obs_shape(self):
        """step() obs should have shape (54,)."""
        obs, _, _, _, _ = self.env.step(1)
        assert obs.shape == (54,)

    def test_step_obs_dtype(self):
        """step() obs should have dtype float32."""
        obs, _, _, _, _ = self.env.step(1)
        assert obs.dtype == np.float32

    def test_step_reward_is_float(self):
        """step() reward should be a float."""
        _, reward, _, _, _ = self.env.step(1)
        assert isinstance(reward, (float, np.floating))

    def test_step_terminated_is_bool(self):
        """step() terminated should be a bool."""
        _, _, terminated, _, _ = self.env.step(1)
        assert isinstance(terminated, (bool, np.bool_))

    def test_step_not_terminated_on_first_step(self):
        """First step should not be terminal (env has 10 snapshots)."""
        _, _, terminated, _, _ = self.env.step(1)
        assert terminated is False

    def test_step_truncated_is_false(self):
        """step() truncated should always be False."""
        _, _, _, truncated, _ = self.env.step(1)
        assert truncated is False

    def test_step_info_is_dict(self):
        """step() info should be a dict."""
        _, _, _, _, info = self.env.step(1)
        assert isinstance(info, dict)

    def test_step_obs_in_observation_space(self):
        """step() observation should be within observation_space."""
        obs, _, _, _, _ = self.env.step(1)
        assert self.env.observation_space.contains(obs)


# ===========================================================================
# Test 4: action mapping {0->-1, 1->0, 2->+1}
# ===========================================================================


class TestActionMapping:
    """Actions should map to positions: 0->-1, 1->0, 2->+1."""

    def test_action_0_gives_position_neg1(self):
        """action=0 should set position to -1.0 (short)."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs[53] == pytest.approx(-1.0)

    def test_action_1_gives_position_0(self):
        """action=1 should set position to 0.0 (flat)."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        obs, _, _, _, _ = env.step(1)
        assert obs[53] == pytest.approx(0.0)

    def test_action_2_gives_position_pos1(self):
        """action=2 should set position to +1.0 (long)."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        obs, _, _, _, _ = env.step(2)
        assert obs[53] == pytest.approx(1.0)

    def test_position_changes_across_steps(self):
        """Position should reflect the most recent action, not accumulate."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        obs, _, _, _, _ = env.step(2)  # long
        assert obs[53] == pytest.approx(1.0)
        obs, _, _, _, _ = env.step(0)  # short
        assert obs[53] == pytest.approx(-1.0)
        obs, _, _, _, _ = env.step(1)  # flat
        assert obs[53] == pytest.approx(0.0)


# ===========================================================================
# Test 5: reward = position * (mid[t+1] - mid[t]) for PnLDelta
# ===========================================================================


class TestRewardPnLDelta:
    """Reward for pnl_delta mode: position * (mid[t+1] - mid[t])."""

    def test_long_position_positive_move(self):
        """Long position with mid going up should give positive reward."""
        mid = np.array([100.0, 101.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=2 -> position=+1, reward = +1 * (101 - 100) = 1.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.0)

    def test_short_position_positive_move(self):
        """Short position with mid going up should give negative reward."""
        mid = np.array([100.0, 103.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=0 -> position=-1, reward = -1 * (103 - 100) = -3.0
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(-3.0)

    def test_flat_position_gives_zero_reward(self):
        """Flat position should give zero reward regardless of mid movement."""
        mid = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=1 -> position=0, reward = 0 * (200 - 100) = 0.0
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0)

    def test_long_position_negative_move(self):
        """Long position with mid going down should give negative reward."""
        mid = np.array([100.0, 98.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=2 -> position=+1, reward = +1 * (98 - 100) = -2.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(-2.0)

    def test_short_position_negative_move(self):
        """Short position with mid going down should give positive reward."""
        mid = np.array([100.0, 97.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=0 -> position=-1, reward = -1 * (97 - 100) = 3.0
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(3.0)

    def test_consecutive_rewards_use_correct_indices(self):
        """Each step should use mid[t+1] - mid[t] for the current t."""
        mid = np.array([100.0, 102.0, 105.0, 103.0], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(4), mid, _make_spread(4), reward_mode="pnl_delta"
        )
        env.reset()

        # Step 0: position=+1, reward = +1 * (102 - 100) = 2.0
        _, r0, _, _, _ = env.step(2)
        assert r0 == pytest.approx(2.0)

        # Step 1: position=+1, reward = +1 * (105 - 102) = 3.0
        _, r1, _, _, _ = env.step(2)
        assert r1 == pytest.approx(3.0)

        # Step 2 is terminal — tested separately in flattening tests


# ===========================================================================
# Test 6: PnLDeltaPenalized subtracts lambda * |position|
# ===========================================================================


class TestRewardPnLDeltaPenalized:
    """Penalized mode: reward = position * delta_mid - lambda * |position|."""

    def test_penalized_long_position(self):
        """Penalized reward: pnl_delta - lambda * |position|."""
        mid = np.array([100.0, 102.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(3), mid, _make_spread(3),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
        )
        env.reset()
        # action=2 -> position=+1
        # pnl_delta = +1 * (102 - 100) = 2.0
        # penalty = 0.5 * |+1| = 0.5
        # reward = 2.0 - 0.5 = 1.5
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(1.5)

    def test_penalized_short_position(self):
        """Penalized reward for short position."""
        mid = np.array([100.0, 97.0, 99.0], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(3), mid, _make_spread(3),
            reward_mode="pnl_delta_penalized", lambda_=1.0,
        )
        env.reset()
        # action=0 -> position=-1
        # pnl_delta = -1 * (97 - 100) = 3.0
        # penalty = 1.0 * |-1| = 1.0
        # reward = 3.0 - 1.0 = 2.0
        _, reward, _, _, _ = env.step(0)
        assert reward == pytest.approx(2.0)

    def test_penalized_flat_position_no_penalty(self):
        """Flat position should incur zero penalty even with large lambda."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(3), mid, _make_spread(3),
            reward_mode="pnl_delta_penalized", lambda_=100.0,
        )
        env.reset()
        # action=1 -> position=0
        # pnl_delta = 0 * (100 - 100) = 0.0
        # penalty = 100.0 * |0| = 0.0
        # reward = 0.0
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0)

    def test_penalized_vs_pnl_delta_with_position(self):
        """Penalized mode should give lower reward than pnl_delta for same action."""
        mid = np.array([100.0, 105.0, 110.0], dtype=np.float64)
        obs = _make_obs(3)

        env_plain = PrecomputedEnv(
            obs.copy(), mid.copy(), _make_spread(3), reward_mode="pnl_delta"
        )
        env_penalized = PrecomputedEnv(
            obs.copy(), mid.copy(), _make_spread(3),
            reward_mode="pnl_delta_penalized", lambda_=0.5,
        )

        env_plain.reset()
        env_penalized.reset()

        _, r_plain, _, _, _ = env_plain.step(2)  # long
        _, r_penalized, _, _, _ = env_penalized.step(2)  # long

        assert r_penalized < r_plain, (
            f"Penalized reward {r_penalized} should be less than "
            f"plain PnL delta {r_plain}"
        )

    def test_default_reward_mode_is_pnl_delta(self):
        """Default reward_mode should be 'pnl_delta' (no penalty)."""
        mid = np.array([100.0, 105.0, 110.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, _make_spread(3))
        env.reset()
        # action=2 -> position=+1, reward = +1 * (105 - 100) = 5.0
        _, reward, _, _, _ = env.step(2)
        assert reward == pytest.approx(5.0)


# ===========================================================================
# Test 7: flat action (1) has zero reward when mid doesn't change
# ===========================================================================


class TestFlatActionZeroReward:
    """Flat action with constant mid should give zero reward."""

    def test_flat_constant_mid_zero_reward(self):
        """action=1 with constant mid prices -> reward == 0."""
        mid = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(4), mid, _make_spread(4))
        env.reset()
        _, reward, _, _, _ = env.step(1)
        assert reward == pytest.approx(0.0)

    def test_flat_multiple_steps_all_zero(self):
        """Multiple flat steps with constant mid all give zero reward."""
        mid = np.full(10, 100.0, dtype=np.float64)
        env = PrecomputedEnv(_make_obs(10), mid, _make_spread(10))
        env.reset()
        for _ in range(9):  # N-1 steps
            _, reward, terminated, _, _ = env.step(1)
            if not terminated:
                # Non-terminal flat steps on constant mid should be 0
                assert reward == pytest.approx(0.0)


# ===========================================================================
# Test 8: episode terminates at t == N-1 (N snapshots -> N-1 steps)
# ===========================================================================


class TestEpisodeLength:
    """N snapshots should give exactly N-1 steps before termination."""

    def test_5_snapshots_gives_4_steps(self):
        """5 snapshots -> 4 steps, terminated on the 4th."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        for i in range(3):
            _, _, terminated, _, _ = env.step(1)
            assert terminated is False, (
                f"Premature termination at step {i}"
            )
        _, _, terminated, _, _ = env.step(1)
        assert terminated is True, (
            "Should terminate on step 4 (index 3)"
        )

    def test_2_snapshots_gives_1_step(self):
        """Minimum env: 2 snapshots -> 1 step (immediately terminal)."""
        env = PrecomputedEnv(_make_obs(2), _make_mid(2), _make_spread(2))
        env.reset()
        _, _, terminated, _, _ = env.step(1)
        assert terminated is True

    def test_10_snapshots_gives_9_steps(self):
        """10 snapshots -> exactly 9 steps."""
        env = PrecomputedEnv(_make_obs(10), _make_mid(10), _make_spread(10))
        env.reset()
        step_count = 0
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(1)
            step_count += 1
        assert step_count == 9

    def test_truncated_always_false(self):
        """truncated should always be False throughout the episode."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        terminated = False
        while not terminated:
            _, _, terminated, truncated, _ = env.step(1)
            assert truncated is False


# ===========================================================================
# Test 9: flattening penalty at terminal step
# ===========================================================================


class TestFlatteningPenalty:
    """Terminal step: forced flatten with close cost = -spread[t]/2 * |prev_position|."""

    def test_long_position_flattening_penalty(self):
        """Long position at terminal: forced flatten, reward = -spread/2 * |prev_pos|."""
        # 3 snapshots -> 2 steps. Step 1 is terminal.
        mid = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, spread)
        env.reset()

        # Step 0: go long (non-terminal)
        _, _, terminated, _, _ = env.step(2)
        assert not terminated

        # Step 1: terminal — forced flatten (no PnL, just close cost)
        # close_cost = spread[t]/2 * |prev_position| = 0.5/2 * 1 = 0.25
        # reward = -0.25
        _, reward, terminated, _, info = env.step(2)
        assert terminated
        assert reward == pytest.approx(-0.25)
        assert info["forced_flatten"] is True

    def test_short_position_flattening_penalty(self):
        """Short position at terminal: forced flatten, reward = -spread/2 * |prev_pos|."""
        mid = np.array([100.0, 99.0, 98.0], dtype=np.float64)
        spread = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, spread)
        env.reset()

        # Step 0: go short (non-terminal)
        env.step(0)

        # Step 1: terminal — forced flatten (no PnL, just close cost)
        # close_cost = spread[t]/2 * |prev_position| = 1.0/2 * 1 = 0.5
        # reward = -0.5
        _, reward, terminated, _, info = env.step(0)
        assert terminated
        assert reward == pytest.approx(-0.5)
        assert info["forced_flatten"] is True


# ===========================================================================
# Test 10: no flattening penalty when flat at terminal
# ===========================================================================


class TestNoFlatteningPenaltyWhenFlat:
    """No flattening penalty when position=0 at terminal step."""

    def test_flat_at_terminal_no_penalty(self):
        """Flat position at terminal -> no flattening cost applied."""
        mid = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, spread)
        env.reset()

        # Step 0: stay flat
        env.step(1)

        # Step 1: flat at terminal
        # pnl_delta = 0 * (102 - 101) = 0.0
        # flattening = -|0| * 0.5 / 2 = 0.0
        # total = 0.0
        _, reward, terminated, _, _ = env.step(1)
        assert terminated
        assert reward == pytest.approx(0.0)

    def test_compare_flat_vs_positioned_at_terminal(self):
        """Positioned terminal should have lower reward than flat terminal (same PnL delta)."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([2.0, 2.0, 2.0], dtype=np.float64)

        # Flat env
        env_flat = PrecomputedEnv(_make_obs(3), mid.copy(), spread.copy())
        env_flat.reset()
        env_flat.step(1)  # non-terminal, flat
        _, reward_flat, _, _, _ = env_flat.step(1)  # terminal, flat

        # Long env
        env_long = PrecomputedEnv(_make_obs(3), mid.copy(), spread.copy())
        env_long.reset()
        env_long.step(2)  # non-terminal, long
        _, reward_long, _, _, _ = env_long.step(2)  # terminal, long

        # Flat terminal: reward = 0 (no PnL, no penalty)
        # Long terminal: reward = 0 (no PnL) - 1.0 (flattening: |1|*2/2)
        assert reward_flat > reward_long, (
            f"Flat terminal reward ({reward_flat}) should exceed "
            f"positioned terminal reward ({reward_long})"
        )


# ===========================================================================
# Test 11: from_file() classmethod works
# ===========================================================================


class TestFromFile:
    """PrecomputedEnv.from_file() should construct a valid env from a binary file."""

    def test_from_file_returns_precomputed_env(self):
        """from_file() should return a PrecomputedEnv instance."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        assert isinstance(env, PrecomputedEnv)

    def test_from_file_is_gymnasium_env(self):
        """from_file() result should be a gymnasium.Env."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        assert isinstance(env, gym.Env)

    def test_from_file_can_reset(self):
        """from_file() env should support reset()."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        obs, info = env.reset()
        assert obs.shape == (54,)
        assert obs.dtype == np.float32

    def test_from_file_can_step(self):
        """from_file() env should support step()."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (54,)
        assert isinstance(reward, (float, np.floating))

    def test_from_file_full_episode(self):
        """from_file() env should run a complete episode."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        env.reset()
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, _, _ = env.step(steps % 3)
            steps += 1
            if steps > 100_000:
                pytest.fail("Episode did not terminate within 100K steps")
        assert steps > 0

    def test_from_file_with_reward_mode(self):
        """from_file() should accept reward_mode kwarg."""
        env = PrecomputedEnv.from_file(
            PRECOMPUTE_EPISODE_FILE, reward_mode="pnl_delta_penalized", lambda_=0.1
        )
        env.reset()
        obs, reward, _, _, _ = env.step(2)
        assert isinstance(reward, (float, np.floating))


# ===========================================================================
# Test 12: passes gymnasium check_env()
# ===========================================================================


class TestCheckEnv:
    """PrecomputedEnv should pass gymnasium's env checker."""

    def test_check_env_synthetic(self):
        """check_env() should pass on synthetic-array env."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv(_make_obs(20), _make_mid(20), _make_spread(20))
        check_env(env, skip_render_check=True)

    def test_check_env_from_file(self):
        """check_env() should pass on file-backed env."""
        from gymnasium.utils.env_checker import check_env
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE)
        check_env(env, skip_render_check=True)


# ===========================================================================
# Test 13: reset resets position and time to 0
# ===========================================================================


class TestResetMidEpisode:
    """reset() mid-episode should reset time and position to 0."""

    def test_reset_after_steps_returns_t0_obs(self):
        """After stepping, reset() should return obs from t=0."""
        input_obs = _make_obs(10, fill=7.77)
        env = PrecomputedEnv(input_obs, _make_mid(10), _make_spread(10))
        env.reset()

        # Take a few steps
        env.step(2)
        env.step(0)
        env.step(1)

        # Reset
        obs, info = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])
        assert obs[53] == pytest.approx(0.0)

    def test_reset_after_done_returns_t0_obs(self):
        """After episode ends, reset() should return t=0 obs with position=0."""
        input_obs = _make_obs(3, fill=3.14)
        env = PrecomputedEnv(input_obs, _make_mid(3), _make_spread(3))
        env.reset()

        # Run to completion (2 steps for 3 snapshots)
        env.step(2)
        env.step(1)

        # Reset
        obs, info = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])
        assert obs[53] == pytest.approx(0.0)

    def test_multiple_episodes_same_start(self):
        """Multiple reset/episode cycles should always start from t=0."""
        input_obs = _make_obs(5, fill=2.71)
        env = PrecomputedEnv(input_obs, _make_mid(5), _make_spread(5))

        for _ in range(3):
            obs, info = env.reset()
            np.testing.assert_array_almost_equal(obs[:43], input_obs[0])
            assert obs[53] == pytest.approx(0.0)
            # Take some steps
            terminated = False
            while not terminated:
                _, _, terminated, _, _ = env.step(2)


# ===========================================================================
# Test 14: obs at each step comes from correct time index
# ===========================================================================


class TestObsTimeIndex:
    """obs[:43] at step t should match input obs_array[t]."""

    def test_obs_matches_input_at_each_step(self):
        """After step i, obs[:43] should correspond to obs_array[i+1] (t advances)."""
        # Per spec: step increments t, then obs is built from obs[t]
        # After reset: t=0, obs from obs[0]
        # After step 0: t=1, obs from obs[1]
        # After step 1: t=2, obs from obs[2]
        input_obs = _make_obs(5, fill=10.0)
        env = PrecomputedEnv(input_obs, _make_mid(5), _make_spread(5))

        obs, _ = env.reset()
        np.testing.assert_array_almost_equal(obs[:43], input_obs[0])

        for step_idx in range(4):  # 5 snapshots -> 4 steps
            obs, _, _, _, _ = env.step(1)
            np.testing.assert_array_almost_equal(
                obs[:43], input_obs[step_idx + 1],
                err_msg=f"Obs mismatch at step {step_idx}, expected row {step_idx + 1}"
            )

    def test_each_row_distinguishable(self):
        """Verify that obs rows are actually different (sanity check)."""
        input_obs = _make_obs(5, fill=10.0)
        # Rows should differ: row i = fill + i*0.01
        assert not np.array_equal(input_obs[0], input_obs[1])
        assert not np.array_equal(input_obs[1], input_obs[2])


# ===========================================================================
# Test 15: constructor raises ValueError if obs has < 2 rows
# ===========================================================================


class TestConstructorValidation:
    """Constructor should validate that obs has at least 2 rows."""

    def test_1_row_raises_value_error(self):
        """obs with only 1 row should raise ValueError."""
        with pytest.raises(ValueError):
            PrecomputedEnv(_make_obs(1), _make_mid(1), _make_spread(1))

    def test_0_rows_raises_value_error(self):
        """obs with 0 rows should raise ValueError."""
        obs = np.empty((0, 43), dtype=np.float32)
        mid = np.empty(0, dtype=np.float64)
        spread = np.empty(0, dtype=np.float64)
        with pytest.raises(ValueError):
            PrecomputedEnv(obs, mid, spread)

    def test_2_rows_does_not_raise(self):
        """obs with exactly 2 rows should be accepted (minimum valid)."""
        env = PrecomputedEnv(_make_obs(2), _make_mid(2), _make_spread(2))
        assert env is not None


# ===========================================================================
# Test 16: from_file with custom session_config dict
# ===========================================================================


class TestFromFileCustomConfig:
    """from_file() should accept session_config as a dict."""

    def test_from_file_with_session_config_dict(self):
        """from_file(path, session_config={...}) should work."""
        config = {
            "rth_open_ns": 48_600_000_000_000,  # 13:30 UTC in ns
            "rth_close_ns": 72_000_000_000_000,  # 20:00 UTC in ns
        }
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, session_config=config)
        assert isinstance(env, PrecomputedEnv)
        obs, info = env.reset()
        assert obs.shape == (54,)

    def test_from_file_none_session_config_uses_default(self):
        """from_file(path, session_config=None) should use default_rth()."""
        env = PrecomputedEnv.from_file(PRECOMPUTE_EPISODE_FILE, session_config=None)
        assert isinstance(env, PrecomputedEnv)
        obs, info = env.reset()
        assert obs.shape == (54,)


# ===========================================================================
# Additional: import from package
# ===========================================================================


class TestPackageImport:
    """PrecomputedEnv should be importable from the lob_rl package."""

    def test_import_from_lob_rl(self):
        """from lob_rl import PrecomputedEnv should work."""
        from lob_rl import PrecomputedEnv as Imported
        assert Imported is PrecomputedEnv


# ===========================================================================
# Additional: pure-numpy — no C++ calls during step/reset
# ===========================================================================


class TestResetSeedDeterminism:
    """reset(seed=...) should be deterministic (spec test 13)."""

    def test_same_seed_same_obs(self):
        """reset(seed=42) called twice should return identical observations."""
        env = PrecomputedEnv(_make_obs(10), _make_mid(10), _make_spread(10))
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2, err_msg="Same seed should give same obs")

    def test_different_seeds_still_deterministic(self):
        """Pure-data env should return the same obs regardless of seed."""
        env = PrecomputedEnv(_make_obs(10), _make_mid(10), _make_spread(10))
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=999)
        # For PrecomputedEnv, seed doesn't change the data — obs should be identical
        np.testing.assert_array_equal(
            obs1, obs2,
            err_msg="PrecomputedEnv data is deterministic regardless of seed",
        )


# ===========================================================================
# Additional: flattening penalty spread at terminal timestep
# ===========================================================================


class TestFlatteningSpreadIndex:
    """Flattening penalty should use spread[t] at the terminal timestep."""

    def test_varying_spread_uses_terminal_value(self):
        """With varying spread, flattening penalty should use spread[t] not spread[0]."""
        mid = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        # spread[0]=0.5, spread[1]=1.0, spread[2]=4.0 (terminal)
        spread = np.array([0.50, 1.00, 4.00], dtype=np.float64)
        env = PrecomputedEnv(_make_obs(3), mid, spread)
        env.reset()

        # Step 0: go long (non-terminal)
        _, _, terminated, _, _ = env.step(2)
        assert not terminated

        # Step 1: stay long (terminal)
        # pnl_delta = +1 * (100 - 100) = 0.0
        # flattening = -|1| * 4.0 / 2 = -2.0  (uses spread[2], not spread[0])
        _, reward, terminated, _, _ = env.step(2)
        assert terminated
        assert reward == pytest.approx(-2.0), (
            f"Expected -2.0 (using spread[2]=4.0), got {reward}"
        )

    def test_non_terminal_step_has_no_flattening(self):
        """Non-terminal steps should NOT include flattening penalty even with position."""
        mid = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        spread = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)  # large spread
        env = PrecomputedEnv(_make_obs(4), mid, spread)
        env.reset()

        # Step 0: go long (non-terminal) — no flattening penalty
        _, reward, terminated, _, _ = env.step(2)
        assert not terminated
        # pnl_delta = +1 * (100 - 100) = 0.0, no flattening
        assert reward == pytest.approx(0.0), (
            f"Non-terminal step should have no flattening penalty, got {reward}"
        )


# ===========================================================================
# Additional: pure-numpy — no C++ calls during step/reset
# ===========================================================================


class TestPureNumpy:
    """step() and reset() should be pure numpy — verify array types."""

    def test_step_reward_is_python_float(self):
        """Reward from step() should be a Python float or numpy scalar, not a C++ type."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        _, reward, _, _, _ = env.step(2)
        assert isinstance(reward, (float, np.floating))

    def test_step_terminated_is_python_bool(self):
        """terminated from step() should be a Python bool or numpy bool."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        env.reset()
        _, _, terminated, _, _ = env.step(2)
        assert isinstance(terminated, (bool, np.bool_))

    def test_obs_is_numpy_not_list(self):
        """Observation should be numpy ndarray, not Python list."""
        env = PrecomputedEnv(_make_obs(5), _make_mid(5), _make_spread(5))
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        obs, _, _, _, _ = env.step(1)
        assert isinstance(obs, np.ndarray)


# ===========================================================================
# Additional: deterministic episodes
# ===========================================================================


class TestDeterminism:
    """Two episodes with same actions should produce identical results."""

    def test_same_actions_same_results(self):
        """Replaying the same action sequence should give identical obs/rewards."""
        obs_arr = _make_obs(5, fill=5.0)
        mid = _make_mid(5)
        spread = _make_spread(5)

        env = PrecomputedEnv(obs_arr.copy(), mid.copy(), spread.copy())
        actions = [2, 0, 1, 2]  # 4 steps for 5 snapshots

        # Episode 1
        obs1, _ = env.reset()
        results1 = []
        for a in actions:
            obs, reward, terminated, _, _ = env.step(a)
            results1.append((obs.copy(), reward, terminated))

        # Episode 2 (same env, same actions)
        obs2, _ = env.reset()
        results2 = []
        for a in actions:
            obs, reward, terminated, _, _ = env.step(a)
            results2.append((obs.copy(), reward, terminated))

        np.testing.assert_array_equal(obs1, obs2)
        for i, ((o1, r1, t1), (o2, r2, t2)) in enumerate(zip(results1, results2)):
            np.testing.assert_array_equal(o1, o2, err_msg=f"obs differ at step {i}")
            assert r1 == pytest.approx(r2), f"rewards differ at step {i}"
            assert t1 == t2, f"terminated flags differ at step {i}"


# ===========================================================================
# Additional: flattening penalty with penalized reward mode
# ===========================================================================


class TestFlatteningWithPenalizedMode:
    """Forced flatten on terminal overrides penalized mode — only close cost."""

    def test_terminal_penalized_and_flattening(self):
        """Terminal step: forced flatten ignores penalized mode, reward = -spread/2 * |prev_pos|."""
        mid = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        spread = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        env = PrecomputedEnv(
            _make_obs(3), mid, spread,
            reward_mode="pnl_delta_penalized", lambda_=0.5,
        )
        env.reset()

        # Step 0: go long (non-terminal)
        # pnl = +1 * (101 - 100) = 1.0, penalty = 0.5 * 1 = 0.5 -> 0.5
        _, r0, terminated, _, _ = env.step(2)
        assert not terminated
        assert r0 == pytest.approx(0.5)

        # Step 1: terminal — forced flatten (no PnL, no lambda penalty)
        # close_cost = spread[t]/2 * |prev_position| = 0.5/2 * 1 = 0.25
        # reward = -0.25
        _, r1, terminated, _, info = env.step(2)
        assert terminated
        assert r1 == pytest.approx(-0.25)
        assert info["forced_flatten"] is True
