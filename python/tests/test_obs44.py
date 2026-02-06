"""Tests for 44-float observation space from LOBEnv via Python bindings."""

import math
import pytest

import lob_rl_core


# ===========================================================================
# Observation size is 44
# ===========================================================================

def test_reset_returns_44_floats():
    """reset() should return a 44-element observation."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert len(obs) == 44, f"Expected 44 elements, got {len(obs)}"


def test_step_returns_44_floats(env):
    """step() should return a 44-element observation."""
    obs, _, _ = env.step(1)
    assert len(obs) == 44, f"Expected 44 elements, got {len(obs)}"


def test_all_obs_values_are_floats():
    """All 44 observation elements should be float."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i, v in enumerate(obs):
        assert isinstance(v, float), f"obs[{i}] = {v} is not a float"


# ===========================================================================
# All values finite (no NaN/Inf)
# ===========================================================================

def test_all_obs_finite_after_reset():
    """All observation values should be finite after reset."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i, v in enumerate(obs):
        assert math.isfinite(v), f"obs[{i}] = {v} is not finite after reset"


def test_all_obs_finite_through_episode(env):
    """All observation values should be finite throughout an episode."""
    for step in range(20):
        obs, _, done = env.step(step % 3)
        if done:
            break
        for i, v in enumerate(obs):
            assert math.isfinite(v), (
                f"obs[{i}] = {v} not finite at step {step}"
            )


# ===========================================================================
# Position at index 43
# ===========================================================================

def test_position_at_index_43_after_reset():
    """Initial position should be 0.0 at obs[43]."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert obs[43] == pytest.approx(0.0), f"Expected position=0, got {obs[43]}"


def test_action_0_sets_position_neg1(env):
    """Action 0 (short) should set obs[43] = -1."""
    obs, _, _ = env.step(0)
    assert obs[43] == pytest.approx(-1.0), f"Expected -1, got {obs[43]}"


def test_action_1_sets_position_0(env):
    """Action 1 (flat) should set obs[43] = 0."""
    obs, _, _ = env.step(1)
    assert obs[43] == pytest.approx(0.0), f"Expected 0, got {obs[43]}"


def test_action_2_sets_position_1(env):
    """Action 2 (long) should set obs[43] = +1."""
    obs, _, _ = env.step(2)
    assert obs[43] == pytest.approx(1.0), f"Expected 1, got {obs[43]}"


# ===========================================================================
# Time remaining at index 42
# ===========================================================================

def test_time_remaining_at_index_42_no_session():
    """Without session config, time_remaining at obs[42] should be 0.5."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert obs[42] == pytest.approx(0.5), (
        f"Expected time_remaining=0.5 without session, got {obs[42]}"
    )


# ===========================================================================
# Spread at index 40
# ===========================================================================

def test_spread_positive_after_reset():
    """Spread at obs[40] should be positive after reset."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert obs[40] > 0.0, f"Spread should be positive, got {obs[40]}"


def test_spread_normalized_small_value():
    """Normalized spread should be a small fraction (spread/mid)."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    # SyntheticSource has spread ~0.50, mid ~100 → ~0.005
    assert obs[40] < 0.1, f"Normalized spread should be < 0.1, got {obs[40]}"


# ===========================================================================
# Imbalance at index 41
# ===========================================================================

def test_imbalance_in_range(env):
    """Imbalance at obs[41] should be in [-1, 1] throughout episode."""
    for step in range(20):
        obs, _, done = env.step(step % 3)
        if done:
            break
        assert -1.0 <= obs[41] <= 1.0, (
            f"Imbalance {obs[41]} out of range at step {step}"
        )


# ===========================================================================
# Bid prices (indices 0-9) are <= 0 (relative to mid)
# ===========================================================================

def test_bid_prices_non_positive():
    """Bid prices (obs[0:10]) should be <= 0 (below mid)."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i in range(10):
        assert obs[i] <= 0.0, f"Bid price obs[{i}] = {obs[i]} should be <= 0"


# ===========================================================================
# Ask prices (indices 20-29) are >= 0 (relative to mid)
# ===========================================================================

def test_ask_prices_non_negative():
    """Ask prices (obs[20:30]) should be >= 0 (above mid)."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i in range(20, 30):
        assert obs[i] >= 0.0, f"Ask price obs[{i}] = {obs[i]} should be >= 0"


# ===========================================================================
# Sizes (indices 10-19, 30-39) in [0, 1]
# ===========================================================================

def test_sizes_in_0_1():
    """Size features should be in [0, 1]."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i in list(range(10, 20)) + list(range(30, 40)):
        assert 0.0 <= obs[i] <= 1.0, (
            f"Size obs[{i}] = {obs[i]} should be in [0, 1]"
        )


# ===========================================================================
# Deterministic resets produce same obs
# ===========================================================================

def test_deterministic_resets():
    """Two resets should produce identical 44-float observations."""
    env = lob_rl_core.LOBEnv()

    obs1 = env.reset()
    obs2 = env.reset()

    assert len(obs1) == 44
    assert len(obs2) == 44
    for i in range(44):
        assert obs1[i] == pytest.approx(obs2[i]), (
            f"Obs mismatch at index {i}: {obs1[i]} vs {obs2[i]}"
        )


# ===========================================================================
# Full episode with 44-float obs, no crash
# ===========================================================================

def test_full_episode_44_float_obs():
    """Complete a full episode verifying 44-float obs throughout."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert len(obs) == 44

    done = False
    step_count = 0
    while not done and step_count < 500:
        obs, reward, done = env.step(step_count % 3)
        assert len(obs) == 44, f"Step {step_count}: obs has {len(obs)} elements"
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        step_count += 1

    assert done, "Episode should terminate"


# ===========================================================================
# Multiple episodes deterministic with 44-float obs
# ===========================================================================

def test_multiple_episodes_deterministic_44():
    """Two full episodes should produce identical 44-float observations."""
    env = lob_rl_core.LOBEnv()

    # Episode 1
    obs1_list = [env.reset()]
    done = False
    while not done:
        obs, _, done = env.step(1)
        obs1_list.append(obs)

    # Episode 2
    obs2_list = [env.reset()]
    done = False
    while not done:
        obs, _, done = env.step(1)
        obs2_list.append(obs)

    assert len(obs1_list) == len(obs2_list), "Episodes should have same length"
    for i, (o1, o2) in enumerate(zip(obs1_list, obs2_list)):
        assert len(o1) == 44
        assert len(o2) == 44
        for j in range(44):
            assert o1[j] == pytest.approx(o2[j]), (
                f"Obs mismatch at step {i}, index {j}: {o1[j]} vs {o2[j]}"
            )
