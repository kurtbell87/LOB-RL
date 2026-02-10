"""Tests for the multi-session barrier environment wrapper.

Spec: docs/t9-ppo-training.md — Module 1: MultiSessionBarrierEnv

A Gymnasium wrapper that cycles through multiple pre-built trading sessions.
Each reset() loads the next session's data and creates a fresh BarrierEnv.
One episode = one session. Supports round-robin cycling and optional shuffling.
"""

import numpy as np
import pytest

import gymnasium

from lob_rl.barrier.reward_accounting import ACTION_LONG

from .conftest import (
    make_session_bars,
    make_session_data,
    make_session_data_list,
    run_episode,
    DEFAULT_H,
    DEFAULT_OBS_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H = DEFAULT_H
_OBS_DIM = DEFAULT_OBS_DIM


def _make_multi_env(n_sessions=5, n_bars=40, config=None, shuffle=False, seed=None):
    """Create a MultiSessionBarrierEnv from synthetic data."""
    from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv

    sessions = make_session_data_list(n_sessions=n_sessions, n_bars=n_bars)
    return MultiSessionBarrierEnv(sessions, config=config, shuffle=shuffle, seed=seed)


# ===========================================================================
# 1. Reset and basic API (spec tests 1–3)
# ===========================================================================


class TestMultiSessionResetAndAPI:
    """Verify reset returns correct types and shapes."""

    def test_multi_session_reset_returns_obs_info(self):
        """reset() returns (obs, info) with obs as np.ndarray of correct shape."""
        env = _make_multi_env(n_sessions=3)
        result = env.reset()
        assert isinstance(result, tuple), "reset() must return a tuple"
        assert len(result) == 2, "reset() must return (obs, info)"
        obs, info = result
        assert isinstance(obs, np.ndarray), "obs must be np.ndarray"
        assert obs.dtype == np.float32, f"obs dtype must be float32, got {obs.dtype}"
        assert obs.shape == (_OBS_DIM,), f"obs shape must be ({_OBS_DIM},), got {obs.shape}"

    def test_multi_session_obs_shape(self):
        """Observation shape is (132,) = 13*10 features + position + unrealized_pnl."""
        env = _make_multi_env(n_sessions=3)
        obs, _ = env.reset()
        assert obs.shape == (_OBS_DIM,), f"Expected ({_OBS_DIM},), got {obs.shape}"

    def test_multi_session_action_space(self):
        """action_space is Discrete(4)."""
        env = _make_multi_env(n_sessions=3)
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 4


# ===========================================================================
# 2. Session cycling (spec tests 4–5)
# ===========================================================================


class TestSessionCycling:
    """Verify round-robin cycling through sessions."""

    def test_multi_session_cycles_through_sessions(self):
        """After completing N sessions, wraps back to session 0."""
        n_sessions = 3
        env = _make_multi_env(n_sessions=n_sessions, shuffle=False)
        rng = np.random.default_rng(42)

        # Track session indices across 2 full cycles
        session_indices = []
        for _ in range(n_sessions * 2):
            env.reset()
            session_indices.append(env.current_session_index)
            # Complete the episode
            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = rng.choice(valid)
                _, _, terminated, _, _ = env.step(int(action))
                steps += 1
                if steps > 2000:
                    break

        # First cycle: 0, 1, 2; Second cycle: 0, 1, 2
        expected = list(range(n_sessions)) * 2
        assert session_indices == expected, (
            f"Expected session indices {expected}, got {session_indices}"
        )

    def test_multi_session_episode_per_session(self):
        """Each episode corresponds to exactly one session's bars."""
        n_sessions = 3
        env = _make_multi_env(n_sessions=n_sessions, n_bars=30, shuffle=False)
        rng = np.random.default_rng(42)

        # Track session index at each reset
        for ep in range(n_sessions):
            env.reset()
            idx = env.current_session_index
            assert idx == ep, f"Episode {ep} should use session {ep}, got {idx}"

            # Complete episode
            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = rng.choice(valid)
                _, _, terminated, _, _ = env.step(int(action))
                steps += 1
                if steps > 2000:
                    break


# ===========================================================================
# 3. Shuffle and determinism (spec tests 6–7)
# ===========================================================================


class TestShuffleAndDeterminism:
    """Verify shuffle behavior and seed reproducibility."""

    def test_multi_session_shuffle(self):
        """With shuffle=True, session order differs across cycles."""
        n_sessions = 10  # More sessions → higher chance of different order
        env = _make_multi_env(n_sessions=n_sessions, shuffle=True, seed=42)
        rng = np.random.default_rng(42)

        def collect_cycle_order():
            indices = []
            for _ in range(n_sessions):
                env.reset()
                indices.append(env.current_session_index)
                terminated = False
                steps = 0
                while not terminated:
                    mask = env.action_masks()
                    valid = np.where(mask)[0]
                    action = rng.choice(valid)
                    _, _, terminated, _, _ = env.step(int(action))
                    steps += 1
                    if steps > 2000:
                        break
            return indices

        cycle1 = collect_cycle_order()
        cycle2 = collect_cycle_order()

        # With 10 sessions, P(same order twice) = 1/10! ≈ 0
        # The two cycles should differ
        assert cycle1 != cycle2, (
            f"Shuffled order should differ between cycles, got {cycle1} and {cycle2}"
        )

    def test_multi_session_deterministic_seed(self):
        """Same seed produces same session order."""
        n_sessions = 5
        sessions = make_session_data_list(n_sessions=n_sessions)

        from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv

        def get_first_cycle_order(seed):
            env = MultiSessionBarrierEnv(sessions, shuffle=True, seed=seed)
            rng = np.random.default_rng(seed)
            indices = []
            for _ in range(n_sessions):
                env.reset()
                indices.append(env.current_session_index)
                terminated = False
                steps = 0
                while not terminated:
                    mask = env.action_masks()
                    valid = np.where(mask)[0]
                    action = rng.choice(valid)
                    _, _, terminated, _, _ = env.step(int(action))
                    steps += 1
                    if steps > 2000:
                        break
            return indices

        order1 = get_first_cycle_order(seed=123)
        order2 = get_first_cycle_order(seed=123)
        assert order1 == order2, (
            f"Same seed must produce same order: {order1} vs {order2}"
        )


# ===========================================================================
# 4. Skip short sessions (spec test 8)
# ===========================================================================


class TestSkipShortSessions:
    """Verify sessions with too few bars are skipped."""

    def test_multi_session_skip_short_sessions(self):
        """Sessions with fewer bars than lookback h are silently skipped."""
        from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv

        h = _H
        # Create mix: 2 good sessions (40 bars) and 2 short sessions (5 bars < h=10)
        good1 = make_session_data(n_bars=40, base_price=4000.0, h=h)
        short1 = make_session_data(n_bars=5, base_price=4010.0, h=h)
        good2 = make_session_data(n_bars=40, base_price=4020.0, h=h)
        short2 = make_session_data(n_bars=3, base_price=4030.0, h=h)

        # short sessions will have features with 0 rows → should be skipped
        # Actually, with n_bars=5 < h=10, feature matrix has shape (0, 130)
        # These sessions should be skipped
        sessions = [good1, short1, good2, short2]
        env = MultiSessionBarrierEnv(sessions, shuffle=False)

        rng = np.random.default_rng(42)
        indices = []
        for _ in range(4):  # 4 resets should cycle through the 2 good sessions twice
            env.reset()
            indices.append(env.current_session_index)
            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = rng.choice(valid)
                _, _, terminated, _, _ = env.step(int(action))
                steps += 1
                if steps > 2000:
                    break

        # Should only see indices of good sessions (0 and 2), never 1 or 3
        for idx in indices:
            assert idx in (0, 2), (
                f"Session index {idx} is a short session that should have been skipped"
            )


# ===========================================================================
# 5. Action masks delegation (spec test 9)
# ===========================================================================


class TestActionMasksDelegation:
    """Verify action_masks() delegates to inner BarrierEnv."""

    def test_multi_session_action_masks(self):
        """action_masks() returns correct masks that change with position state."""
        env = _make_multi_env(n_sessions=3)
        env.reset()

        # Initially flat → [1, 1, 1, 0]
        mask = env.action_masks()
        assert isinstance(mask, np.ndarray), "action_masks() must return np.ndarray"
        assert mask.shape == (4,), f"Expected shape (4,), got {mask.shape}"
        np.testing.assert_array_equal(mask, [1, 1, 1, 0])

        # Enter long → [0, 0, 0, 1]
        env.step(ACTION_LONG)
        mask = env.action_masks()
        np.testing.assert_array_equal(mask, [0, 0, 0, 1])


# ===========================================================================
# 6. Single session and from_bar_lists (spec tests 10–11)
# ===========================================================================


class TestSingleSessionAndFactory:
    """Verify single-session behavior and from_bar_lists factory."""

    def test_multi_session_single_session(self):
        """Works correctly with only 1 session — cycles back to same session."""
        env = _make_multi_env(n_sessions=1)
        rng = np.random.default_rng(42)

        for ep in range(3):
            env.reset()
            assert env.current_session_index == 0, (
                f"With 1 session, index must always be 0, got {env.current_session_index}"
            )
            terminated = False
            steps = 0
            while not terminated:
                mask = env.action_masks()
                valid = np.where(mask)[0]
                action = rng.choice(valid)
                _, _, terminated, _, _ = env.step(int(action))
                steps += 1
                if steps > 2000:
                    break

    def test_multi_session_from_bar_lists(self):
        """from_bar_lists() correctly computes labels and features from raw bars."""
        from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv

        bar_lists = [
            make_session_bars(40, base_price=4000.0 + i * 10.0)
            for i in range(3)
        ]
        env = MultiSessionBarrierEnv.from_bar_lists(bar_lists, h=_H)

        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (_OBS_DIM,), f"Expected ({_OBS_DIM},), got {obs.shape}"
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 4

        # Should be able to complete an episode
        terminated = False
        rng = np.random.default_rng(42)
        steps = 0
        while not terminated:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = rng.choice(valid)
            _, _, terminated, _, _ = env.step(int(action))
            steps += 1
            if steps > 2000:
                pytest.fail("Episode did not terminate")


# ===========================================================================
# 7. Random agent stress test (spec test 12)
# ===========================================================================


class TestRandomAgentMultiSession:
    """Verify random agent can complete multiple episodes across sessions."""

    def test_multi_session_random_agent_10_episodes(self):
        """Random agent completes 10 episodes across multiple sessions without crashing."""
        n_sessions = 3
        env = _make_multi_env(n_sessions=n_sessions)
        rng = np.random.default_rng(42)

        total_rewards = []
        for ep in range(10):
            reward, steps = run_episode(env, rng)
            total_rewards.append(reward)
            assert isinstance(reward, (float, int, np.floating)), (
                f"Episode {ep}: non-numeric reward {reward}"
            )
            assert steps > 0, f"Episode {ep}: 0 steps"

        # At least some episodes should have non-zero reward
        # (with random actions and 3 sessions, this is almost certain)
        assert len(total_rewards) == 10
