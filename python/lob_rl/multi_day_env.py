"""Multi-day training environment that cycles through multiple data files.

Precomputes all days at construction time using lob_rl_core.precompute(),
or loads pre-cached .npz files from a cache directory.
step() and reset() use pure numpy via PrecomputedEnv or BarLevelEnv.
"""

import glob
import os
import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lob_rl.precomputed_env import PrecomputedEnv


class MultiDayEnv(gym.Env):
    """Gymnasium env that cycles through multiple day files.

    Each reset() advances to the next day's data file. In sequential mode
    (shuffle=False), files are visited in order and wrap around. In shuffle
    mode (shuffle=True), the order is randomized and re-shuffled at each
    epoch boundary.

    All days are precomputed at construction time — step() and reset()
    use pure numpy via PrecomputedEnv.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        file_paths=None,
        session_config=None,
        steps_per_episode=50,
        reward_mode="pnl_delta",
        lambda_=0.0,
        shuffle=False,
        seed=None,
        execution_cost=False,
        participation_bonus=0.0,
        step_interval=1,
        cache_dir=None,
        bar_size=0,
    ):
        super().__init__()

        # Validate mutual exclusivity
        if file_paths is not None and cache_dir is not None:
            raise ValueError("Provide exactly one of file_paths or cache_dir, not both")
        if file_paths is None and cache_dir is None:
            raise ValueError("Provide exactly one of file_paths or cache_dir")

        self._reward_mode = reward_mode
        self._lambda = lambda_
        self._shuffle = shuffle
        self._execution_cost = execution_cost
        self._participation_bonus = participation_bonus
        self._step_interval = step_interval
        self._bar_size = bar_size

        # Warn if step_interval is set with bar_size > 0
        if bar_size > 0 and step_interval > 1:
            warnings.warn(
                f"step_interval={step_interval} is ignored when bar_size={bar_size} > 0. "
                "Bar-level aggregation replaces tick subsampling."
            )

        # Set observation space based on mode
        if bar_size > 0:
            obs_dim = 21  # 13 intra-bar + 7 temporal + 1 position
        else:
            obs_dim = 54  # 43 base + 10 temporal + 1 position

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._precomputed_days = []  # list of (obs, mid, spread) tuples

        if cache_dir is not None:
            self._load_from_cache_dir(cache_dir)
        else:
            self._load_from_file_paths(file_paths, session_config)

        # If bar_size > 0, filter out days that produce < 2 bars
        if bar_size > 0:
            self._filter_days_for_bar_size()

        if not self._precomputed_days:
            raise ValueError(
                "No valid day files: all files produced < 2 BBO snapshots"
            )

        # RNG for shuffle ordering
        self._rng = np.random.RandomState(seed)
        self._order = list(range(len(self._precomputed_days)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._day_index = 0
        self._inner_env = None

    def _filter_days_for_bar_size(self):
        """Remove days that produce < 2 bars with the given bar_size."""
        from lob_rl.bar_aggregation import aggregate_bars

        filtered = []
        for obs, mid, spread in self._precomputed_days:
            bar_features, _, _ = aggregate_bars(obs, mid, spread, self._bar_size)
            if bar_features.shape[0] >= 2:
                filtered.append((obs, mid, spread))
            else:
                warnings.warn(
                    f"Skipping day with {obs.shape[0]} ticks: "
                    f"only {bar_features.shape[0]} bars with bar_size={self._bar_size}"
                )
        self._precomputed_days = filtered

    def _load_from_file_paths(self, file_paths, session_config):
        """Load days from raw .bin files via C++ precompute."""
        import lob_rl_core
        from lob_rl._config import make_session_config

        if not file_paths:
            raise ValueError("file_paths must be a non-empty list")

        cfg = make_session_config(session_config)
        for path in file_paths:
            obs, mid, spread, num_steps = lob_rl_core.precompute(path, cfg)
            if num_steps < 2:
                warnings.warn(
                    f"Skipping {path}: only {num_steps} BBO snapshots (need >= 2)"
                )
                continue
            self._precomputed_days.append((obs, mid, spread))

    def _load_from_cache_dir(self, cache_dir):
        """Load days from pre-cached .npz files in a directory."""
        npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not npz_files:
            raise ValueError(f"No .npz files found in {cache_dir}")

        for npz_path in npz_files:
            try:
                data = np.load(npz_path)
                if not all(k in data for k in ("obs", "mid", "spread")):
                    warnings.warn(f"Skipping {npz_path}: missing required keys")
                    continue
                obs = data["obs"]
                mid = data["mid"]
                spread = data["spread"]
                if obs.shape[0] < 2:
                    warnings.warn(
                        f"Skipping {npz_path}: only {obs.shape[0]} rows (need >= 2)"
                    )
                    continue
                self._precomputed_days.append((obs, mid, spread))
            except Exception as e:
                warnings.warn(f"Skipping {npz_path}: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # If seed is provided, reseed the RNG
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._order = list(range(len(self._precomputed_days)))
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._day_index = 0

        # Check if we need to start a new epoch (wrap around)
        if self._day_index >= len(self._order):
            self._day_index = 0
            if self._shuffle:
                self._rng.shuffle(self._order)

        # Get the current day's index
        file_idx = self._order[self._day_index]
        self._day_index += 1

        # Create inner env for this day
        obs, mid, spread = self._precomputed_days[file_idx]

        if self._bar_size > 0:
            from lob_rl.bar_level_env import BarLevelEnv
            self._inner_env = BarLevelEnv(
                obs, mid, spread,
                bar_size=self._bar_size,
                reward_mode=self._reward_mode,
                lambda_=self._lambda,
                execution_cost=self._execution_cost,
                participation_bonus=self._participation_bonus,
            )
        else:
            self._inner_env = PrecomputedEnv(
                obs, mid, spread,
                reward_mode=self._reward_mode,
                lambda_=self._lambda,
                execution_cost=self._execution_cost,
                participation_bonus=self._participation_bonus,
                step_interval=self._step_interval,
            )
        obs_out, info = self._inner_env.reset()
        info["day_index"] = file_idx
        return obs_out, info

    def step(self, action):
        return self._inner_env.step(action)
