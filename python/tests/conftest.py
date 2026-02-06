import os

import numpy as np
import pytest

import lob_rl_core

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
EPISODE_FILE = os.path.join(FIXTURE_DIR, "episode_200records.bin")
SESSION_FILE = os.path.join(FIXTURE_DIR, "session_180records.bin")
PRECOMPUTE_EPISODE_FILE = os.path.join(FIXTURE_DIR, "precompute_session.bin")
DAY_FILES = [os.path.join(FIXTURE_DIR, f"day{i}.bin") for i in range(5)]


@pytest.fixture
def env():
    """Create a LOBEnv with default SyntheticSource, already reset."""
    e = lob_rl_core.LOBEnv()
    e.reset()
    return e


# ---------------------------------------------------------------------------
# Shared synthetic array factories for deterministic testing
# ---------------------------------------------------------------------------

def make_obs(n, fill=1.0):
    """Create a (n, 43) float32 obs array with distinguishable rows.

    Row i has all values set to (fill + i * 0.01) so we can verify
    which row is being returned.
    """
    obs = np.empty((n, 43), dtype=np.float32)
    for i in range(n):
        obs[i, :] = fill + i * 0.01
    return obs


def make_mid(n, start=100.0, step=1.0):
    """Create a (n,) float64 mid price array with a linear ramp."""
    return np.arange(start, start + n * step, step, dtype=np.float64)[:n]


def make_spread(n, value=0.5):
    """Create a (n,) float64 spread array with constant value."""
    return np.full(n, value, dtype=np.float64)


def run_episode(env, max_steps=5000):
    """Step through an episode until terminated. Returns step count."""
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(1)
        steps += 1
    return steps
