"""Multi-day training environment that cycles through multiple data files.

Precomputes all days at construction time using lob_rl_core.precompute().
step() and reset() use pure numpy via PrecomputedEnv.
"""

import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import lob_rl_core
from lob_rl._config import make_session_config
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
        file_paths,
        session_config=None,
        steps_per_episode=50,
        reward_mode="pnl_delta",
        lambda_=0.0,
        shuffle=False,
        seed=None,
        execution_cost=False,
        participation_bonus=0.0,
    ):
        super().__init__()

        if not file_paths:
            raise ValueError("file_paths must be a non-empty list")

        self._reward_mode = reward_mode
        self._lambda = lambda_
        self._shuffle = shuffle
        self._execution_cost = execution_cost
        self._participation_bonus = participation_bonus

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Precompute all days at construction time
        cfg = make_session_config(session_config)
        self._precomputed_days = []  # list of (obs, mid, spread) tuples
        for path in file_paths:
            obs, mid, spread, num_steps = lob_rl_core.precompute(path, cfg)
            if num_steps < 2:
                warnings.warn(
                    f"Skipping {path}: only {num_steps} BBO snapshots (need >= 2)"
                )
                continue
            self._precomputed_days.append((obs, mid, spread))

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

        # Create inner PrecomputedEnv for this day
        obs, mid, spread = self._precomputed_days[file_idx]
        self._inner_env = PrecomputedEnv(
            obs, mid, spread,
            reward_mode=self._reward_mode,
            lambda_=self._lambda,
            execution_cost=self._execution_cost,
            participation_bonus=self._participation_bonus,
        )
        obs_out, info = self._inner_env.reset()
        info["day_index"] = file_idx
        return obs_out, info

    def step(self, action):
        return self._inner_env.step(action)
