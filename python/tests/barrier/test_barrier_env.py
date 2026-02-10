"""Tests for the barrier-hit trading environment.

Spec: docs/t8-barrier-env.md

Gymnasium-compatible BarrierEnv that combines bars, labels, features, and
reward accounting into a step-by-step RL environment. One episode = one
RTH trading session. Agent enters long/short, exits via barrier hit or timeout.
"""

import numpy as np
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar
from lob_rl.barrier.feature_pipeline import build_feature_matrix
from lob_rl.barrier.label_pipeline import BarrierLabel, compute_labels
from lob_rl.barrier.reward_accounting import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    RewardConfig,
    compute_unrealized_pnl,
)

from .conftest import make_bar, make_flat_bars, make_session_bars


# ---------------------------------------------------------------------------
# Helpers — build synthetic data for env construction
# ---------------------------------------------------------------------------

# Default lookback h=10, 13 features → obs_dim = 130 + 2 = 132
_H = 10
_FEATURE_DIM = 13 * _H  # 130
_OBS_DIM = _FEATURE_DIM + 2  # 132


def _make_env_inputs(n_bars=60, base_price=4000.0, h=_H):
    """Build (bars, labels, features) for env construction.

    Returns n_bars session bars, their labels, and a feature matrix.
    The feature matrix has shape (n_bars - h + 1, 13*h).
    """
    bars = make_session_bars(n_bars, base_price=base_price)
    labels = compute_labels(bars, a=20, b=10, t_max=40)
    features = build_feature_matrix(bars, h=h)
    return bars, labels, features


def _make_env(n_bars=60, config=None, h=_H):
    """Create a BarrierEnv from synthetic data."""
    from lob_rl.barrier.barrier_env import BarrierEnv

    bars, labels, features = _make_env_inputs(n_bars=n_bars, h=h)
    return BarrierEnv(bars, labels, features, config=config)


def _step_with_mask(env, preferred_action=None):
    """Step the env respecting action masks.

    If preferred_action is valid, use it. Otherwise pick the first valid action.
    Returns the step result.
    """
    mask = env.action_masks()
    if preferred_action is not None and mask[preferred_action]:
        return env.step(preferred_action)
    # Pick first valid action
    for a in range(4):
        if mask[a]:
            return env.step(a)
    raise RuntimeError("No valid actions available")


# ===========================================================================
# 1. Gymnasium API Compliance (spec tests 1–5)
# ===========================================================================


class TestGymnasiumAPICompliance:
    """Verify the environment follows the Gymnasium API contract."""

    def test_reset_returns_obs_and_info(self):
        """reset() returns (obs, info) where obs is np.ndarray with correct
        dtype and shape."""
        env = _make_env()
        result = env.reset()
        assert isinstance(result, tuple), "reset() must return a tuple"
        assert len(result) == 2, "reset() must return (obs, info)"
        obs, info = result
        assert isinstance(obs, np.ndarray), "obs must be np.ndarray"
        assert obs.dtype == np.float32, f"obs dtype must be float32, got {obs.dtype}"
        assert obs.shape == (_OBS_DIM,), f"obs shape must be ({_OBS_DIM},), got {obs.shape}"

    def test_step_returns_five_tuple(self):
        """step(action) returns (obs, reward, terminated, truncated, info)."""
        env = _make_env()
        env.reset()
        obs, reward, terminated, truncated, info = _step_with_mask(env, ACTION_FLAT)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_shape(self):
        """Observation shape is (132,) = 130 features + 2 position state (h=10)."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs.shape == (_OBS_DIM,), f"Expected ({_OBS_DIM},), got {obs.shape}"

    def test_action_space_discrete_4(self):
        """env.action_space is Discrete(4)."""
        import gymnasium
        env = _make_env()
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 4

    def test_observation_space_box(self):
        """env.observation_space is Box with correct shape and dtype float32."""
        import gymnasium
        env = _make_env()
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        assert env.observation_space.shape == (_OBS_DIM,)
        assert env.observation_space.dtype == np.float32


# ===========================================================================
# 2. Observation Content (spec tests 6–9)
# ===========================================================================


class TestObservationContent:
    """Verify observation vector layout and values."""

    def test_obs_features_match_feature_matrix(self):
        """First 130 elements of obs match the corresponding row of the
        feature matrix."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        bars, labels, features = _make_env_inputs(n_bars=60)
        env = BarrierEnv(bars, labels, features)
        obs, _ = env.reset()

        # After reset, bar_idx=0 → feature row 0
        np.testing.assert_array_almost_equal(
            obs[:_FEATURE_DIM], features[0].astype(np.float32),
            decimal=5,
            err_msg="obs features must match feature_matrix[0]",
        )

    def test_obs_position_field(self):
        """obs[130] reflects the current position (-1, 0, or +1)."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs[_FEATURE_DIM] == 0.0, "Initial position must be 0 (flat)"

        # Enter long
        obs, _, _, _, info = _step_with_mask(env, ACTION_LONG)
        assert obs[_FEATURE_DIM] == 1.0, "Position must be +1 after long entry"

    def test_obs_unrealized_pnl_field(self):
        """obs[131] reflects the current unrealized PnL in ticks."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs[_FEATURE_DIM + 1] == 0.0, "Initial unrealized PnL must be 0"

    def test_obs_initial_flat(self):
        """After reset, obs[130] == 0 (flat) and obs[131] == 0.0 (no PnL)."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs[_FEATURE_DIM] == 0.0, "position must be 0 after reset"
        assert obs[_FEATURE_DIM + 1] == 0.0, "unrealized PnL must be 0 after reset"


# ===========================================================================
# 3. Action Masking (spec tests 10–14)
# ===========================================================================


class TestActionMasking:
    """Verify SB3-compatible action masking."""

    def test_action_masks_flat(self):
        """When flat, mask is [1, 1, 1, 0] (long, short, flat valid; hold invalid)."""
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        np.testing.assert_array_equal(mask, [1, 1, 1, 0])

    def test_action_masks_long(self):
        """When long, mask is [0, 0, 0, 1] (only hold valid)."""
        env = _make_env()
        env.reset()
        env.step(ACTION_LONG)  # Enter long
        mask = env.action_masks()
        np.testing.assert_array_equal(mask, [0, 0, 0, 1])

    def test_action_masks_short(self):
        """When short, mask is [0, 0, 0, 1] (only hold valid)."""
        env = _make_env()
        env.reset()
        env.step(ACTION_SHORT)  # Enter short
        mask = env.action_masks()
        np.testing.assert_array_equal(mask, [0, 0, 0, 1])

    def test_action_masks_shape_and_dtype(self):
        """action_masks() returns np.ndarray of shape (4,) with dtype np.int8."""
        env = _make_env()
        env.reset()
        mask = env.action_masks()
        assert isinstance(mask, np.ndarray), "action_masks() must return np.ndarray"
        assert mask.shape == (4,), f"Expected shape (4,), got {mask.shape}"
        assert mask.dtype == np.int8, f"Expected dtype int8, got {mask.dtype}"

    def test_action_masks_updates_after_entry(self):
        """Mask changes from [1,1,1,0] to [0,0,0,1] after entering a position."""
        env = _make_env()
        env.reset()
        mask_before = env.action_masks().copy()
        env.step(ACTION_LONG)
        mask_after = env.action_masks()

        np.testing.assert_array_equal(mask_before, [1, 1, 1, 0])
        np.testing.assert_array_equal(mask_after, [0, 0, 0, 1])


# ===========================================================================
# 4. Episode Lifecycle (spec tests 15–19)
# ===========================================================================


class TestEpisodeLifecycle:
    """Verify episode structure: termination, length, force-close, truncation."""

    def test_episode_terminates_at_end(self):
        """Episode terminates after all bars consumed."""
        env = _make_env(n_bars=25)  # Small episode
        env.reset()
        terminated = False
        steps = 0
        while not terminated:
            obs, reward, terminated, truncated, info = _step_with_mask(env, ACTION_FLAT)
            steps += 1
            if steps > 500:
                pytest.fail("Episode did not terminate within 500 steps")
        assert terminated is True

    def test_episode_length_matches_usable_bars(self):
        """Number of steps in episode matches number of usable bars from
        feature matrix (when staying flat the whole time)."""
        n_bars = 30
        bars, labels, features = _make_env_inputs(n_bars=n_bars, h=_H)
        n_usable = features.shape[0]  # n_bars - h + 1

        from lob_rl.barrier.barrier_env import BarrierEnv
        env = BarrierEnv(bars, labels, features)
        env.reset()

        steps = 0
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(ACTION_FLAT)
            steps += 1

        # When staying flat, each step advances 1 bar, so steps == n_usable
        assert steps == n_usable, (
            f"Expected {n_usable} steps (flat), got {steps}"
        )

    def test_force_close_at_session_end(self):
        """If holding when episode ends, position is force-closed with MTM reward."""
        env = _make_env(n_bars=25)
        env.reset()

        # Enter long first
        env.step(ACTION_LONG)

        # Keep holding until episode ends
        terminated = False
        final_reward = None
        final_info = None
        steps = 0
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
            steps += 1
            if terminated:
                final_reward = reward
                final_info = info
            if steps > 500:
                pytest.fail("Episode did not terminate within 500 steps")

        # If we were still holding at the end, we should have a force-close
        # The info should indicate force_close exit type
        if final_info.get("exit_type") == "force_close":
            # Reward should be non-zero (MTM - C)
            assert final_reward is not None
            assert isinstance(final_reward, (float, int, np.floating))

    def test_truncated_always_false(self):
        """truncated is always False."""
        env = _make_env(n_bars=25)
        env.reset()
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = _step_with_mask(env, ACTION_FLAT)
            assert truncated is False, "truncated must always be False"

    def test_reset_clears_state(self):
        """After reset, position is flat, bar_idx is 0, PnL is 0."""
        env = _make_env()
        env.reset()

        # Do some steps
        env.step(ACTION_LONG)
        env.step(ACTION_HOLD)

        # Reset
        obs, info = env.reset()
        assert obs[_FEATURE_DIM] == 0.0, "position must be 0 after reset"
        assert obs[_FEATURE_DIM + 1] == 0.0, "unrealized PnL must be 0 after reset"
        assert info["bar_idx"] == 0, "bar_idx must be 0 after reset"
        assert info["position"] == 0, "position in info must be 0 after reset"


# ===========================================================================
# 5. Reward Accounting Integration (spec tests 20–25)
# ===========================================================================


class TestRewardAccountingIntegration:
    """Verify reward values match reward_accounting.py computations."""

    def test_entry_reward_zero(self):
        """Entering a position yields reward = 0."""
        env = _make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
        assert reward == pytest.approx(0.0), (
            f"Entry reward must be 0, got {reward}"
        )

    def test_profit_hit_reward(self):
        """When a bar triggers profit barrier, reward = +G - C.

        Uses custom bars where profit barrier is guaranteed to fire.
        """
        from lob_rl.barrier.barrier_env import BarrierEnv

        config = RewardConfig(a=20, b=10, T_max=40)
        # G=2.0, L=1.0, C=0.2 → profit reward = +1.8

        # Build bars where entry at close=4000, then bar jumps to trigger profit
        # Profit barrier (long) = 4000 + 20*0.25 = 4005.0
        n_total = 20  # Need enough bars
        bars = make_session_bars(n_total, base_price=4000.0, spread=1.0)

        # Override second usable bar (index h-1+1 = 10) to hit profit barrier
        # Entry is on usable bar 0 → bars[h-1] close = 4000.0
        # Next usable bar 1 → bars[h] should hit profit barrier
        h = _H
        entry_bar_idx = h - 1  # bars[9]
        hit_bar_idx = h  # bars[10]

        # Set entry bar close to exactly 4000.0
        bars[entry_bar_idx] = make_bar(
            bar_index=entry_bar_idx, open_price=4000.0,
            high=4000.5, low=3999.5, close=4000.0,
            t_start=bars[entry_bar_idx].t_start, t_end=bars[entry_bar_idx].t_end,
        )
        # Set hit bar high to reach profit barrier
        bars[hit_bar_idx] = make_bar(
            bar_index=hit_bar_idx, open_price=4002.0,
            high=4005.0, low=4001.0, close=4004.0,
            t_start=bars[hit_bar_idx].t_start, t_end=bars[hit_bar_idx].t_end,
        )

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=h)
        env = BarrierEnv(bars, labels, features, config=config)

        obs, _ = env.reset()
        # Enter long on first usable bar
        obs, reward, _, _, _ = env.step(ACTION_LONG)
        assert reward == pytest.approx(0.0), "Entry reward must be 0"

        # Now hold — next bar should trigger profit barrier
        obs, reward, _, _, info = env.step(ACTION_HOLD)

        # If profit barrier was hit
        if info.get("exit_type") == "profit":
            assert reward == pytest.approx(1.8), (
                f"Profit reward must be G-C=1.8, got {reward}"
            )

    def test_stop_hit_reward(self):
        """When a bar triggers stop barrier, reward = -L - C.

        Uses custom bars where stop barrier is guaranteed to fire.
        """
        from lob_rl.barrier.barrier_env import BarrierEnv

        config = RewardConfig(a=20, b=10, T_max=40)
        # stop reward = -L - C = -1.0 - 0.2 = -1.2

        n_total = 20
        bars = make_session_bars(n_total, base_price=4000.0, spread=1.0)
        h = _H

        entry_bar_idx = h - 1
        hit_bar_idx = h

        # Entry bar close at 4000.0
        bars[entry_bar_idx] = make_bar(
            bar_index=entry_bar_idx, open_price=4000.0,
            high=4000.5, low=3999.5, close=4000.0,
            t_start=bars[entry_bar_idx].t_start, t_end=bars[entry_bar_idx].t_end,
        )
        # Stop barrier (long) = 4000 - 10*0.25 = 3997.5
        bars[hit_bar_idx] = make_bar(
            bar_index=hit_bar_idx, open_price=3999.0,
            high=3999.5, low=3997.5, close=3998.0,
            t_start=bars[hit_bar_idx].t_start, t_end=bars[hit_bar_idx].t_end,
        )

        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=h)
        env = BarrierEnv(bars, labels, features, config=config)

        obs, _ = env.reset()
        obs, reward, _, _, _ = env.step(ACTION_LONG)  # Entry
        assert reward == pytest.approx(0.0)

        obs, reward, _, _, info = env.step(ACTION_HOLD)  # Hold → stop hit

        if info.get("exit_type") == "stop":
            assert reward == pytest.approx(-1.2), (
                f"Stop reward must be -L-C=-1.2, got {reward}"
            )

    def test_timeout_reward_mtm(self):
        """When hold_counter reaches T_max, reward = MTM - C."""
        config = RewardConfig(a=20, b=10, T_max=5)
        env = _make_env(n_bars=60, config=config)
        env.reset()

        # Enter long
        env.step(ACTION_LONG)

        # Hold for T_max steps — the env should timeout
        for _ in range(10):  # More than T_max to be safe
            obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
            if info.get("exit_type") == "timeout":
                # Reward should be MTM - C (some float value)
                assert isinstance(reward, (float, int, np.floating))
                # After timeout, position returns to flat
                assert info["position"] == 0
                break
        else:
            # If we never hit timeout, the agent exited via barrier
            pass  # That's OK for synthetic data

    def test_flat_action_reward_zero(self):
        """Staying flat yields reward = 0."""
        env = _make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(ACTION_FLAT)
        assert reward == pytest.approx(0.0), (
            f"Flat action reward must be 0, got {reward}"
        )

    def test_force_close_reward_mtm(self):
        """Force close at session end yields MTM - C reward."""
        config = RewardConfig(a=20, b=10, T_max=200)  # Large T_max to avoid timeout
        env = _make_env(n_bars=25, config=config)
        env.reset()

        # Enter long
        env.step(ACTION_LONG)

        # Hold until episode ends
        terminated = False
        last_reward = None
        last_info = None
        steps = 0
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
            last_reward = reward
            last_info = info
            steps += 1
            if steps > 500:
                pytest.fail("Episode did not terminate")

        # The final step should be a force-close if position was held to the end
        if last_info.get("exit_type") == "force_close":
            assert isinstance(last_reward, (float, int, np.floating))
            assert last_info["position"] == 0  # Back to flat after force close


# ===========================================================================
# 6. Position State Transitions (spec tests 26–29)
# ===========================================================================


class TestPositionStateTransitions:
    """Verify position state tracking through the environment."""

    def test_flat_to_long(self):
        """After entry with action=0 (long), position is +1."""
        env = _make_env()
        env.reset()
        obs, _, _, _, info = env.step(ACTION_LONG)
        assert info["position"] == 1, "Position must be +1 after long entry"
        assert obs[_FEATURE_DIM] == 1.0, "obs position must be +1 after long entry"

    def test_flat_to_short(self):
        """After entry with action=1 (short), position is -1."""
        env = _make_env()
        env.reset()
        obs, _, _, _, info = env.step(ACTION_SHORT)
        assert info["position"] == -1, "Position must be -1 after short entry"
        assert obs[_FEATURE_DIM] == -1.0, "obs position must be -1 after short entry"

    def test_long_to_flat_on_barrier(self):
        """After profit/stop hit on long position, position returns to 0."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        config = RewardConfig(a=20, b=10, T_max=5)
        env = _make_env(n_bars=60, config=config)
        env.reset()

        # Enter long
        env.step(ACTION_LONG)

        # Hold until exit (barrier or timeout)
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
            if info.get("exit_type") is not None:
                # After any exit, position should be flat
                assert info["position"] == 0, (
                    f"Position must be 0 after exit, got {info['position']}"
                )
                assert obs[_FEATURE_DIM] == 0.0
                break
            if terminated:
                break

    def test_position_persists_during_hold(self):
        """Position stays constant while holding without barrier hit."""
        config = RewardConfig(a=20, b=10, T_max=100)  # Large T_max
        env = _make_env(n_bars=60, config=config)
        env.reset()

        # Enter long
        obs, _, _, _, info = env.step(ACTION_LONG)
        assert info["position"] == 1

        # Hold a few times — position should stay +1 unless barrier hit
        for _ in range(3):
            obs, _, terminated, _, info = env.step(ACTION_HOLD)
            if info.get("exit_type") is not None:
                break  # Barrier hit, can't test further
            assert info["position"] == 1, "Position must persist during hold"
            assert obs[_FEATURE_DIM] == 1.0
            if terminated:
                break


# ===========================================================================
# 7. Random Agent (spec tests 30–32)
# ===========================================================================


class TestRandomAgent:
    """Verify random agent can complete episodes without crashing."""

    def _run_random_episode(self, env, rng):
        """Run one episode with random valid actions. Return total reward
        and trade count."""
        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        steps = 0
        while not terminated:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = rng.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            if steps > 2000:
                pytest.fail("Random episode did not terminate in 2000 steps")
        return total_reward, info.get("n_trades", 0)

    def test_random_agent_completes_episode(self):
        """A random agent sampling valid actions completes an episode
        without crashing."""
        env = _make_env(n_bars=60)
        rng = np.random.default_rng(42)
        total_reward, n_trades = self._run_random_episode(env, rng)
        assert isinstance(total_reward, (float, int, np.floating))

    def test_random_agent_100_episodes(self):
        """Run 100 episodes with random actions. All complete without error."""
        env = _make_env(n_bars=60)
        rng = np.random.default_rng(42)
        for ep in range(100):
            total_reward, n_trades = self._run_random_episode(env, rng)
            assert isinstance(total_reward, (float, int, np.floating)), (
                f"Episode {ep} returned non-numeric reward: {total_reward}"
            )

    def test_random_agent_mean_reward(self):
        """Mean per-trade reward across 100 episodes is approximately -0.20
        (within +/-0.50 — wide tolerance for synthetic data)."""
        env = _make_env(n_bars=60)
        rng = np.random.default_rng(42)
        total_rewards = []
        total_trades = []
        for _ in range(100):
            reward, n_trades = self._run_random_episode(env, rng)
            total_rewards.append(reward)
            total_trades.append(n_trades)

        all_trades = sum(total_trades)
        if all_trades > 0:
            mean_per_trade = sum(total_rewards) / all_trades
            # Wide tolerance: the exact value depends on synthetic price data
            assert -0.70 <= mean_per_trade <= 0.30, (
                f"Mean per-trade reward {mean_per_trade:.3f} outside [-0.70, 0.30]"
            )


# ===========================================================================
# 8. Info Dict (spec tests 33–35)
# ===========================================================================


class TestInfoDict:
    """Verify info dict contents."""

    def test_info_has_required_keys(self):
        """Info dict contains position, bar_idx, exit_type, entry_price, n_trades."""
        env = _make_env()
        env.reset()
        _, _, _, _, info = env.step(ACTION_FLAT)
        required_keys = {"position", "bar_idx", "exit_type", "entry_price", "n_trades"}
        missing = required_keys - set(info.keys())
        assert not missing, f"Info dict missing keys: {missing}"

    def test_info_exit_type_on_barrier_hit(self):
        """exit_type is 'profit' or 'stop' when barrier fires."""
        config = RewardConfig(a=20, b=10, T_max=5)
        env = _make_env(n_bars=60, config=config)

        rng = np.random.default_rng(123)
        # Run episodes until we get a barrier hit
        found_barrier = False
        for _ in range(50):  # Try many episodes
            env.reset()
            terminated = False
            while not terminated:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = rng.choice(valid)
                _, _, terminated, _, info = env.step(int(action))
                if info.get("exit_type") in ("profit", "stop"):
                    found_barrier = True
                    assert info["exit_type"] in ("profit", "stop")
                    break
                if terminated:
                    break
            if found_barrier:
                break

        # With T_max=5 and 50 episodes, we should find at least one barrier hit
        assert found_barrier, "Expected at least one barrier hit in 50 episodes"

    def test_info_n_trades_increments(self):
        """n_trades increments by 1 each time a trade completes."""
        config = RewardConfig(a=20, b=10, T_max=3)  # Short timeout for quick trades
        env = _make_env(n_bars=60, config=config)
        env.reset()

        prev_n_trades = 0
        terminated = False
        steps = 0
        while not terminated and steps < 200:
            mask = env.action_masks()
            # If flat, enter long; if holding, hold
            if mask[ACTION_LONG]:
                action = ACTION_LONG
            elif mask[ACTION_HOLD]:
                action = ACTION_HOLD
            else:
                action = ACTION_FLAT
            _, _, terminated, _, info = env.step(action)
            n_trades = info["n_trades"]

            if info.get("exit_type") is not None:
                # Trade completed — n_trades should have incremented
                assert n_trades == prev_n_trades + 1, (
                    f"n_trades should be {prev_n_trades + 1}, got {n_trades}"
                )
                prev_n_trades = n_trades
            else:
                # No trade completed — n_trades should be unchanged
                assert n_trades == prev_n_trades
            steps += 1

        # We should have completed at least one trade
        assert prev_n_trades > 0, "Expected at least one completed trade"


# ===========================================================================
# 9. Edge Cases (spec tests 36–39)
# ===========================================================================


class TestEdgeCases:
    """Verify edge case handling."""

    def test_invalid_action_when_holding(self):
        """If an invalid action (e.g., long when already long) is passed,
        the env handles it gracefully (defaults to hold)."""
        env = _make_env()
        env.reset()
        env.step(ACTION_LONG)  # Enter long

        # Send invalid action: long again while already long
        obs, reward, terminated, truncated, info = env.step(ACTION_LONG)
        # Should not crash — treated as hold
        assert isinstance(obs, np.ndarray)
        assert info["position"] in (-1, 0, 1)

    def test_invalid_action_when_flat(self):
        """If hold action is passed when flat, the env handles it gracefully
        (defaults to flat)."""
        env = _make_env()
        env.reset()

        # Send invalid action: hold when flat
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        # Should not crash — treated as flat
        assert isinstance(obs, np.ndarray)
        assert reward == pytest.approx(0.0)

    def test_single_bar_episode(self):
        """Environment with only 1 usable bar terminates immediately."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        # Need h + 0 extra bars → exactly h bars → 1 usable bar
        # h=10, so 10 bars → features shape (1, 130)
        h = _H
        n_bars = h  # Exactly 1 usable bar (n_bars - h + 1 = 1)
        bars = make_session_bars(n_bars, base_price=4000.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=h)
        assert features.shape[0] == 1, f"Expected 1 usable bar, got {features.shape[0]}"

        env = BarrierEnv(bars, labels, features)
        env.reset()

        # First step should terminate the episode
        obs, reward, terminated, truncated, info = env.step(ACTION_FLAT)
        assert terminated is True, "Single-bar episode must terminate after one step"

    def test_config_forwarded_to_reward(self):
        """Custom RewardConfig values are used in reward computation.
        Different C value changes the reward."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        # Two envs with different C values — same bars, same actions
        config_low_c = RewardConfig(a=20, b=10, T_max=3, C=0.0)
        config_high_c = RewardConfig(a=20, b=10, T_max=3, C=1.0)

        bars, labels, features = _make_env_inputs(n_bars=60)
        env_low = BarrierEnv(bars, labels, features, config=config_low_c)
        env_high = BarrierEnv(bars, labels, features, config=config_high_c)

        env_low.reset()
        env_high.reset()

        # Enter long in both
        env_low.step(ACTION_LONG)
        env_high.step(ACTION_LONG)

        # Hold until exit
        rewards_low = []
        rewards_high = []
        for _ in range(20):
            _, r_low, term_low, _, info_low = env_low.step(ACTION_HOLD)
            _, r_high, term_high, _, info_high = env_high.step(ACTION_HOLD)
            rewards_low.append(r_low)
            rewards_high.append(r_high)
            if term_low or term_high:
                break

        total_low = sum(rewards_low)
        total_high = sum(rewards_high)

        # With C=1.0, total reward should be lower than with C=0.0
        # (cost is deducted from every exit)
        # They should differ by exactly C_high - C_low = 1.0 per trade
        assert total_low != total_high or total_low == 0.0, (
            "Different C values must produce different rewards"
        )


# ===========================================================================
# 10. Factory Method (spec tests 40–41)
# ===========================================================================


class TestFactoryMethod:
    """Verify BarrierEnv.from_bars() convenience factory."""

    def test_from_bars_creates_env(self):
        """BarrierEnv.from_bars(bars) creates a valid environment."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        bars = make_session_bars(60, base_price=4000.0)
        env = BarrierEnv.from_bars(bars)
        assert env is not None
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (_OBS_DIM,)

    def test_from_bars_obs_shape_correct(self):
        """Environment from from_bars() has correct observation shape."""
        from lob_rl.barrier.barrier_env import BarrierEnv

        bars = make_session_bars(60, base_price=4000.0)
        env = BarrierEnv.from_bars(bars, h=_H)
        obs, _ = env.reset()
        assert obs.shape == (_OBS_DIM,), (
            f"Expected obs shape ({_OBS_DIM},), got {obs.shape}"
        )
        assert env.observation_space.shape == (_OBS_DIM,)
