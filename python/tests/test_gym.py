"""Tests for Gym interface compliance and basic functionality."""

import numpy as np
import pytest


def test_core_import():
    """Test that the C++ module can be imported."""
    import lob_rl_core
    assert hasattr(lob_rl_core, "LOBEnv")
    assert hasattr(lob_rl_core, "EnvConfig")


def test_env_creation():
    """Test basic environment creation."""
    from lob_rl_core import LOBEnv, EnvConfig

    config = EnvConfig()
    config.book_depth = 10
    config.trades_per_step = 50

    env = LOBEnv(config, seed=42, num_messages=1000)
    assert env.observation_size() == 44


def test_reset():
    """Test reset returns correct observation."""
    from lob_rl_core import LOBEnv, EnvConfig

    config = EnvConfig()
    env = LOBEnv(config, seed=42, num_messages=1000)
    result = env.reset()

    obs = np.array(result.obs, dtype=np.float32)
    assert obs.shape == (44,)
    assert result.done == False
    assert result.position == 0
    assert result.pnl == 0.0


def test_step():
    """Test step returns correct types."""
    from lob_rl_core import LOBEnv, EnvConfig

    config = EnvConfig()
    env = LOBEnv(config, seed=42, num_messages=1000)
    env.reset()

    result = env.step(2)  # Go long
    obs = np.array(result.obs, dtype=np.float32)

    assert obs.shape == (44,)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert result.position == 1


def test_episode_rollout():
    """Test running a full episode."""
    from lob_rl_core import LOBEnv, EnvConfig

    config = EnvConfig()
    config.trades_per_step = 50

    env = LOBEnv(config, seed=42, num_messages=1000)
    env.reset()

    steps = 0
    done = False
    while not done:
        action = steps % 3  # Cycle through actions
        result = env.step(action)
        done = result.done
        steps += 1

    assert steps > 0
    assert result.position == 0  # Flat at end


def test_numpy_dtypes():
    """Test that observations are proper numpy float32 arrays."""
    from lob_rl_core import LOBEnv, EnvConfig

    config = EnvConfig()
    env = LOBEnv(config, seed=42, num_messages=1000)
    result = env.reset()

    obs = np.array(result.obs, dtype=np.float32)
    assert obs.dtype == np.float32
    assert not np.any(np.isnan(obs))
    assert not np.any(np.isinf(obs))


def test_gym_wrapper():
    """Test Gymnasium wrapper if gymnasium is available."""
    try:
        from lob_rl.wrappers import LOBEnvGym
    except ImportError:
        pytest.skip("gymnasium not installed")

    env = LOBEnvGym(seed=42, num_messages=1000, trades_per_step=50)

    obs, info = env.reset()
    assert obs.shape == (44,)
    assert obs.dtype == np.float32
    assert "position" in info

    obs, reward, terminated, truncated, info = env.step(2)
    assert obs.shape == (44,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
