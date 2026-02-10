"""Multi-session barrier environment wrapper.

Gymnasium-compatible wrapper that cycles through multiple pre-built
trading sessions. Each reset() loads the next session's data and
creates a fresh BarrierEnv. One episode = one session.
Supports round-robin cycling and optional shuffling.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from lob_rl.barrier.barrier_env import BarrierEnv
from lob_rl.barrier.feature_pipeline import build_feature_matrix
from lob_rl.barrier.label_pipeline import compute_labels
from lob_rl.barrier.reward_accounting import RewardConfig


class MultiSessionBarrierEnv(gymnasium.Env):
    """Gymnasium wrapper that cycles through multiple trading sessions.

    Parameters
    ----------
    sessions : list[dict]
        Each dict has keys: bars, labels, features.
    config : RewardConfig, optional
        Reward configuration passed to inner BarrierEnv.
    shuffle : bool
        If True, shuffle session order at each cycle boundary.
    seed : int, optional
        RNG seed for reproducible shuffling.
    """

    def __init__(self, sessions, config=None, shuffle=False, seed=None):
        super().__init__()
        self._all_sessions = sessions
        self._config = config or RewardConfig()
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)

        # Identify valid sessions (features with > 0 rows)
        self._valid_indices = []
        for i, s in enumerate(sessions):
            if s["features"].shape[0] > 0:
                self._valid_indices.append(i)

        if len(self._valid_indices) == 0:
            raise ValueError("No valid sessions (all have 0 usable bars)")

        # Build initial ordering
        self._order = list(self._valid_indices)
        if self._shuffle:
            self._rng.shuffle(self._order)

        self._cycle_pos = 0  # position within current cycle
        self._inner_env = None
        self._current_session_index = 0

        # Set up spaces from first valid session
        first = sessions[self._valid_indices[0]]
        feature_dim = first["features"].shape[1]
        obs_dim = feature_dim + 2  # + position + unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

    @property
    def current_session_index(self):
        return self._current_session_index

    @classmethod
    def from_bar_lists(cls, bar_lists, h=10, config=None, **kwargs):
        """Create a MultiSessionBarrierEnv from lists of raw bars.

        Computes labels and features for each session.
        """
        cfg = config or RewardConfig()
        sessions = []
        for bars in bar_lists:
            labels = compute_labels(bars, a=cfg.a, b=cfg.b, t_max=cfg.T_max)
            features = build_feature_matrix(bars, h=h)
            sessions.append({"bars": bars, "labels": labels, "features": features})
        return cls(sessions, config=cfg, **kwargs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick next session from ordering
        if self._cycle_pos >= len(self._order):
            # New cycle — reshuffle if needed
            self._order = list(self._valid_indices)
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._cycle_pos = 0

        session_idx = self._order[self._cycle_pos]
        self._cycle_pos += 1
        self._current_session_index = session_idx

        # Create fresh inner env
        s = self._all_sessions[session_idx]
        self._inner_env = BarrierEnv(
            s["bars"], s["labels"], s["features"], config=self._config,
        )
        return self._inner_env.reset(seed=seed)

    def step(self, action):
        return self._inner_env.step(action)

    def action_masks(self):
        return self._inner_env.action_masks()
