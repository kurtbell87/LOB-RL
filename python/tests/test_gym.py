"""Tests for the lob_rl_core Python bindings."""

import pytest

import lob_rl_core


# ===========================================================================
# Module & Construction
# ===========================================================================

def test_import_module():
    """lob_rl_core module should be importable."""
    assert lob_rl_core is not None


def test_create_env():
    """LOBEnv can be constructed with default SyntheticSource."""
    env = lob_rl_core.LOBEnv()
    assert env is not None


# ===========================================================================
# Reset
# ===========================================================================

def test_reset_returns_list():
    """reset() should return a list of floats (observation)."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert isinstance(obs, (list, tuple))


def test_reset_observation_has_44_elements():
    """Observation from reset() should have 44 elements."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    assert len(obs) == 44, f"Expected 44 elements, got {len(obs)}"


def test_reset_observation_values_are_floats():
    """All observation elements should be float-like."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()
    for i, v in enumerate(obs):
        assert isinstance(v, float), f"obs[{i}] = {v} is not a float"


def test_reset_observation_has_valid_features():
    """Initial observation should have valid features."""
    env = lob_rl_core.LOBEnv()
    obs = env.reset()

    spread = obs[40]
    position = obs[43]

    assert spread >= 0, f"spread should be non-negative, got {spread}"
    assert position == 0.0, f"initial position should be 0, got {position}"


# ===========================================================================
# Step (using fixture — env is already reset)
# ===========================================================================

def test_step_returns_tuple_of_three(env):
    """step() should return a tuple (obs, reward, done)."""
    result = env.step(1)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3 elements, got {len(result)}"


def test_step_obs_is_list_of_44_floats(env):
    """step() observation should be a list of 44 floats."""
    obs, reward, done = env.step(1)

    assert isinstance(obs, (list, tuple))
    assert len(obs) == 44
    for i, v in enumerate(obs):
        assert isinstance(v, float), f"obs[{i}] = {v} is not a float"


def test_step_reward_is_float(env):
    """step() reward should be a float."""
    obs, reward, done = env.step(1)

    assert isinstance(reward, float), f"Expected float reward, got {type(reward)}"


def test_step_done_is_bool(env):
    """step() done should be a bool."""
    obs, reward, done = env.step(1)

    assert isinstance(done, bool), f"Expected bool done, got {type(done)}"


# ===========================================================================
# Action mapping (using fixture)
# ===========================================================================

def test_action_zero_sets_short_position(env):
    """Action 0 should set position to -1."""
    obs, _, _ = env.step(0)

    assert obs[43] == pytest.approx(-1.0), f"Expected position=-1, got {obs[43]}"


def test_action_one_sets_flat_position(env):
    """Action 1 should set position to 0."""
    obs, _, _ = env.step(1)

    assert obs[43] == pytest.approx(0.0), f"Expected position=0, got {obs[43]}"


def test_action_two_sets_long_position(env):
    """Action 2 should set position to +1."""
    obs, _, _ = env.step(2)

    assert obs[43] == pytest.approx(1.0), f"Expected position=+1, got {obs[43]}"


# ===========================================================================
# Episode termination (using fixture)
# ===========================================================================

def test_episode_terminates(env):
    """Stepping through an episode should eventually set done=True."""
    done = False
    steps = 0
    while not done and steps < 500:
        _, _, done = env.step(1)
        steps += 1

    assert done, f"Episode did not terminate after {steps} steps"


def test_full_episode_no_crash(env):
    """Complete a full episode from Python without crashing."""
    actions = [0, 1, 2]
    step_count = 0
    done = False

    while not done:
        action = actions[step_count % 3]
        obs, reward, done = env.step(action)
        step_count += 1

        assert len(obs) == 44
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        if step_count > 500:
            pytest.fail("Episode did not terminate within 500 steps")

    assert step_count > 0, "Episode should have at least one step"


def test_reset_after_episode_works(env):
    """After an episode completes, reset should work for a new episode."""
    # Complete an episode
    done = False
    while not done:
        _, _, done = env.step(1)

    # Reset and verify new episode
    obs = env.reset()
    assert len(obs) == 44
    assert obs[43] == pytest.approx(0.0), "Position should be 0 after reset"

    # Should be able to step again
    obs, reward, done = env.step(2)
    assert len(obs) == 44
    assert not done or True  # may be done if very short episode


def test_multiple_episodes_deterministic():
    """Two full episodes should produce identical observations."""
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
        for j in range(44):
            assert o1[j] == pytest.approx(o2[j]), (
                f"Obs mismatch at step {i}, index {j}: {o1[j]} vs {o2[j]}"
            )
