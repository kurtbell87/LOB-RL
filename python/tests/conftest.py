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


def make_realistic_obs(n, mid_start=100.0, mid_step=0.25, spread=0.50):
    """Build (n, 43) obs array with realistic LOB fields for testing.

    Returns (obs, mid, spread_arr) — obs has proper bid/ask/size/imbalance
    structure and each row is distinguishable via row-dependent values.
    """
    DEPTH = 10
    OBS_COLS = 43
    obs = np.zeros((n, OBS_COLS), dtype=np.float32)

    t = np.arange(n, dtype=np.float64)
    mid = mid_start + t * mid_step
    spread_arr = np.full(n, spread, dtype=np.float64)
    half_spread = spread / 2.0
    lvl = np.arange(DEPTH, dtype=np.float32)

    # Bid prices: bid0 - lvl * 0.25 for each level
    bid0 = (mid - half_spread).astype(np.float32)
    obs[:, :DEPTH] = bid0[:, np.newaxis] - lvl[np.newaxis, :] * 0.25

    # Ask prices: ask0 + lvl * 0.25 for each level
    ask0 = (mid + half_spread).astype(np.float32)
    obs[:, 20:20 + DEPTH] = ask0[:, np.newaxis] + lvl[np.newaxis, :] * 0.25

    # Bid sizes: 10.0 + t * 0.5 + lvl
    obs[:, 10:10 + DEPTH] = (10.0 + t[:, np.newaxis] * 0.5 + lvl[np.newaxis, :]).astype(np.float32)

    # Ask sizes: 8.0 + t * 0.3 + lvl
    obs[:, 30:30 + DEPTH] = (8.0 + t[:, np.newaxis] * 0.3 + lvl[np.newaxis, :]).astype(np.float32)

    # Relative spread
    obs[:, 40] = (spread / mid).astype(np.float32)

    # Imbalance at level 0
    bs0 = obs[:, 10]
    as0 = obs[:, 30]
    denom = bs0 + as0
    obs[:, 41] = np.where(denom > 0, (bs0 - as0) / denom, 0.0)

    # Time left
    obs[:, 42] = (1.0 - t / max(n - 1, 1)).astype(np.float32)

    return obs, mid, spread_arr


def run_episode(env, max_steps=5000):
    """Step through an episode until terminated. Returns step count."""
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(1)
        steps += 1
    return steps
