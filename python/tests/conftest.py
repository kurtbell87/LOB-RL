import os
import re

import numpy as np
import pytest

import lob_rl_core
from lob_rl._obs_layout import (
    BASE_OBS_SIZE,
    BID_PRICES, BID_SIZES, ASK_PRICES, ASK_SIZES,
    REL_SPREAD, IMBALANCE, TIME_LEFT,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# ---------------------------------------------------------------------------
# train.py source helpers — shared across test_shuffle_split, test_frame_stacking,
# test_recurrent_ppo, test_checkpointing, test_training_pipeline_v2, etc.
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
)


def load_train_source():
    """Read train.py source as a string."""
    with open(TRAIN_SCRIPT) as f:
        return f.read()


def extract_main_body(source):
    """Extract the main() function body from source."""
    pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
    match = re.search(pattern, source, re.DOTALL)
    assert match is not None, "main() function not found"
    return match.group(0)


def extract_evaluate_sortino_body(source):
    """Extract the evaluate_sortino() function body from source."""
    pattern = r"def\s+evaluate_sortino\s*\(.*?\n(?=def\s|\Z)"
    match = re.search(pattern, source, re.DOTALL)
    assert match is not None, "evaluate_sortino() function not found"
    return match.group(0)
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
    """Create a (n, BASE_OBS_SIZE) float32 obs array with distinguishable rows.

    Row i has all values set to (fill + i * 0.01) so we can verify
    which row is being returned.
    """
    obs = np.empty((n, BASE_OBS_SIZE), dtype=np.float32)
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
    """Build (n, BASE_OBS_SIZE) obs array with realistic LOB fields for testing.

    Returns (obs, mid, spread_arr) — obs has proper bid/ask/size/imbalance
    structure and each row is distinguishable via row-dependent values.
    """
    DEPTH = 10
    obs = np.zeros((n, BASE_OBS_SIZE), dtype=np.float32)

    t = np.arange(n, dtype=np.float64)
    mid = mid_start + t * mid_step
    spread_arr = np.full(n, spread, dtype=np.float64)
    half_spread = spread / 2.0
    lvl = np.arange(DEPTH, dtype=np.float32)

    # Bid prices: bid0 - lvl * 0.25 for each level
    bid0 = (mid - half_spread).astype(np.float32)
    obs[:, BID_PRICES] = bid0[:, np.newaxis] - lvl[np.newaxis, :] * 0.25

    # Ask prices: ask0 + lvl * 0.25 for each level
    ask0 = (mid + half_spread).astype(np.float32)
    obs[:, ASK_PRICES] = ask0[:, np.newaxis] + lvl[np.newaxis, :] * 0.25

    # Bid sizes: 10.0 + t * 0.5 + lvl
    obs[:, BID_SIZES] = (10.0 + t[:, np.newaxis] * 0.5 + lvl[np.newaxis, :]).astype(np.float32)

    # Ask sizes: 8.0 + t * 0.3 + lvl
    obs[:, ASK_SIZES] = (8.0 + t[:, np.newaxis] * 0.3 + lvl[np.newaxis, :]).astype(np.float32)

    # Relative spread
    obs[:, REL_SPREAD] = (spread / mid).astype(np.float32)

    # Imbalance at level 0
    bs0 = obs[:, BID_SIZES.start]
    as0 = obs[:, ASK_SIZES.start]
    denom = bs0 + as0
    obs[:, IMBALANCE] = np.where(denom > 0, (bs0 - as0) / denom, 0.0)

    # Time left
    obs[:, TIME_LEFT] = (1.0 - t / max(n - 1, 1)).astype(np.float32)

    return obs, mid, spread_arr


def run_episode(env, max_steps=5000):
    """Step through an episode until terminated. Returns step count."""
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        obs, reward, terminated, truncated, info = env.step(1)
        steps += 1
    return steps


def make_tick_data(n, mid_start=100.0, mid_step=0.25, spread=0.50):
    """Create (obs, mid, spread) arrays with n ticks."""
    obs, mid, spread_arr = make_realistic_obs(n, mid_start=mid_start,
                                               mid_step=mid_step, spread=spread)
    return obs, mid, spread_arr


def save_cache_with_instrument_id(tmpdir, filename, obs, mid, spread, instrument_id):
    """Save an .npz cache file with instrument_id included."""
    path = os.path.join(tmpdir, filename)
    np.savez(path, obs=obs, mid=mid, spread=spread,
             instrument_id=np.array([instrument_id], dtype=np.uint32))
    return path


def save_cache_without_instrument_id(tmpdir, filename, obs, mid, spread):
    """Save an .npz cache file without instrument_id (legacy format)."""
    path = os.path.join(tmpdir, filename)
    np.savez(path, obs=obs, mid=mid, spread=spread)
    return path


def create_synthetic_cache_dir(tmpdir, n_days=3, n_rows=50):
    """Create a cache directory with synthetic .npz files.

    Each day gets a unique mid_start offset (100 + i*10) so days are
    distinguishable. Returns (cache_dir, dates).
    """
    cache_dir = os.path.join(tmpdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    dates = [f"2025-01-{i + 10:02d}" for i in range(n_days)]
    for i, date in enumerate(dates):
        obs, mid, spread = make_realistic_obs(n_rows, mid_start=100.0 + i * 10)
        np.savez(os.path.join(cache_dir, f"{date}.npz"),
                 obs=obs, mid=mid, spread=spread)

    return cache_dir, dates
