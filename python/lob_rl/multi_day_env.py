"""Multi-day training environment that cycles through multiple data files.

Precomputes all days at construction time using lob_rl_core.precompute(),
or lazily loads pre-cached .npz files from a cache directory or explicit file list.
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

    Supports three modes:
    - file_paths: Eager load from raw .bin files via C++ precompute (arrays in memory).
    - cache_dir: Lazy load from a directory of .npz files (paths only at init).
    - cache_files: Lazy load from an explicit list of .npz file paths.
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
        cache_files=None,
    ):
        super().__init__()

        # Validate mutual exclusivity: exactly one of file_paths, cache_dir, cache_files
        provided = sum(x is not None for x in (file_paths, cache_dir, cache_files))
        if provided != 1:
            raise ValueError(
                "Provide exactly one of file_paths, cache_dir, or cache_files"
            )

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

        # Lazy mode: store paths only; Eager mode (file_paths): store arrays
        self._lazy = False
        self._npz_paths = []  # list of str (lazy mode only)
        self._precomputed_days = []  # list of (obs, mid, spread) tuples (eager mode only)
        self._contract_ids = []  # list of instrument_id (int or None) per day

        if cache_dir is not None:
            self._lazy = True
            self._init_lazy_from_cache_dir(cache_dir)
        elif cache_files is not None:
            self._lazy = True
            self._init_lazy_from_cache_files(cache_files)
        else:
            self._load_from_file_paths(file_paths, session_config)

        # If bar_size > 0, filter out days that produce < 2 bars
        if bar_size > 0:
            self._filter_days_for_bar_size()

        # Validate we have data
        n_days = len(self._npz_paths) if self._lazy else len(self._precomputed_days)
        if n_days == 0:
            raise ValueError(
                "No valid day files: all files produced < 2 BBO snapshots"
            )

        self._prev_contract_id = None  # for roll detection

        # RNG for shuffle ordering
        self._rng = np.random.RandomState(seed)
        self._order = list(range(n_days))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._day_index = 0
        self._inner_env = None

    @property
    def contract_ids(self):
        """List of instrument_id (int or None) per day, in load order."""
        return list(self._contract_ids)

    def _num_days(self):
        """Number of days available."""
        return len(self._npz_paths) if self._lazy else len(self._precomputed_days)

    def _has_enough_bars(self, obs, mid, spread):
        """Return True if arrays produce >= 2 bars, warn and return False otherwise."""
        from lob_rl.bar_aggregation import aggregate_bars

        bar_features, _, _ = aggregate_bars(obs, mid, spread, self._bar_size)
        if bar_features.shape[0] >= 2:
            return True
        warnings.warn(
            f"Skipping day with {obs.shape[0]} ticks: "
            f"only {bar_features.shape[0]} bars with bar_size={self._bar_size}"
        )
        return False

    def _filter_days_for_bar_size(self):
        """Remove days that produce < 2 bars with the given bar_size."""
        if self._lazy:
            keep = []
            for i, npz_path in enumerate(self._npz_paths):
                data = np.load(npz_path)
                if self._has_enough_bars(data["obs"], data["mid"], data["spread"]):
                    keep.append(i)
            self._npz_paths = [self._npz_paths[i] for i in keep]
            self._contract_ids = [self._contract_ids[i] for i in keep]
        else:
            keep = []
            for i, (obs, mid, spread) in enumerate(self._precomputed_days):
                if self._has_enough_bars(obs, mid, spread):
                    keep.append(i)
            self._precomputed_days = [self._precomputed_days[i] for i in keep]
            self._contract_ids = [self._contract_ids[i] for i in keep]

    def _validate_npz(self, npz_path):
        """Validate and extract metadata from an .npz file.

        Returns (is_valid, contract_id) where contract_id is int or None.
        """
        try:
            data = np.load(npz_path)
            if not all(k in data for k in ("obs", "mid", "spread")):
                warnings.warn(f"Skipping {npz_path}: missing required keys")
                return False, None
            if data["obs"].shape[0] < 2:
                warnings.warn(
                    f"Skipping {npz_path}: only {data['obs'].shape[0]} rows (need >= 2)"
                )
                return False, None
            contract_id = int(data["instrument_id"][0]) if "instrument_id" in data else None
            return True, contract_id
        except Exception as e:
            warnings.warn(f"Skipping {npz_path}: {e}")
            return False, None

    def _add_valid_npz_files(self, npz_files):
        """Validate and append valid .npz files to internal lists."""
        for npz_path in npz_files:
            is_valid, contract_id = self._validate_npz(npz_path)
            if is_valid:
                self._npz_paths.append(npz_path)
                self._contract_ids.append(contract_id)

    def _init_lazy_from_cache_dir(self, cache_dir):
        """Initialize lazy loading from a cache directory."""
        npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not npz_files:
            raise ValueError(f"No .npz files found in {cache_dir}")
        self._add_valid_npz_files(npz_files)

    def _init_lazy_from_cache_files(self, cache_files):
        """Initialize lazy loading from an explicit list of .npz paths."""
        if not cache_files:
            raise ValueError("cache_files must be a non-empty list")
        self._add_valid_npz_files(cache_files)

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
            self._contract_ids.append(None)

    def _load_day_from_npz(self, npz_path):
        """Load arrays from an .npz file. Returns (obs, mid, spread)."""
        data = np.load(npz_path)
        return data["obs"], data["mid"], data["spread"]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        n_days = self._num_days()

        # If seed is provided, reseed the RNG
        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self._order = list(range(n_days))
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

        # Get arrays for this day
        if self._lazy:
            obs, mid, spread = self._load_day_from_npz(self._npz_paths[file_idx])
        else:
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

        # Contract boundary tracking
        current_contract_id = self._contract_ids[file_idx]
        info["instrument_id"] = current_contract_id

        # Detect contract roll: both must be non-None and different
        if (self._prev_contract_id is not None and
                current_contract_id is not None and
                self._prev_contract_id != current_contract_id):
            info["contract_roll"] = True
        else:
            info["contract_roll"] = False

        self._prev_contract_id = current_contract_id

        return obs_out, info

    def step(self, action):
        return self._inner_env.step(action)
