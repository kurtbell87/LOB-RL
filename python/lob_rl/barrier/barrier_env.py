"""Barrier-hit trading environment.

Gymnasium-compatible RL environment that combines bars, labels, features,
and reward accounting into a step-by-step trading environment.
One episode = one RTH trading session.
"""

import numpy as np
import gymnasium
from gymnasium import spaces

from lob_rl.barrier import TICK_SIZE, N_FEATURES
from lob_rl.barrier.feature_pipeline import build_feature_matrix
from lob_rl.barrier.label_pipeline import compute_labels
from lob_rl.barrier.reward_accounting import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    RewardConfig,
    PositionState,
    classify_exit,
    compute_entry,
    compute_hold_reward,
    compute_mtm_reward,
    compute_unrealized_pnl,
    get_action_mask,
)


class BarrierEnv(gymnasium.Env):
    """Gymnasium environment for barrier-hit trading.

    Parameters
    ----------
    bars : list[TradeBar]
        Session bars.
    labels : list[BarrierLabel]
        Barrier labels for each bar.
    features : np.ndarray
        Feature matrix of shape (n_usable, feature_dim).
    config : RewardConfig, optional
        Reward configuration. Defaults to RewardConfig().
    """

    def __init__(self, bars, labels, features, config=None):
        super().__init__()
        self._bars = bars
        self._labels = labels
        self._config = config or RewardConfig()

        # Infer lookback h from feature dimensions (feature_dim = 13 * h)
        self._feature_dim = features.shape[1] if features.ndim == 2 else 130
        self._h = self._feature_dim // N_FEATURES

        # Bar offset: usable bar i corresponds to bars[i + h - 1]
        self._bar_offset = self._h - 1

        if features.shape[0] > 0:
            self._features = features.astype(np.float32)
            self._n_usable = features.shape[0]
        else:
            self._n_usable = max(len(bars) - self._bar_offset, 0)
            self._features = np.zeros(
                (max(self._n_usable, 1), self._feature_dim), dtype=np.float32
            )

        self._obs_dim = self._feature_dim + 2  # + position + unrealized_pnl

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )

        # State
        self._bar_idx = 0  # index into feature matrix (0..n_usable-1)
        self._position_state = PositionState()
        self._n_trades = 0

    @classmethod
    def from_bars(cls, bars, h=10, config=None, **kwargs):
        """Create a BarrierEnv from bars, computing labels and features.

        Parameters
        ----------
        bars : list[TradeBar]
        h : int
            Lookback horizon for features.
        config : RewardConfig, optional
        """
        cfg = config or RewardConfig()
        labels = compute_labels(bars, a=cfg.a, b=cfg.b, t_max=cfg.T_max)
        features = build_feature_matrix(bars, h=h)
        return cls(bars, labels, features, config=cfg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._bar_idx = 0
        self._position_state = PositionState()
        self._n_trades = 0
        obs = self._build_obs()
        info = self._build_info(exit_type=None)
        return obs, info

    def step(self, action):
        action = self._sanitize_action(action)
        reward = 0.0
        exit_type = None

        pos = self._position_state.position

        if pos == 0:
            # Flat — can enter long, short, or stay flat
            if action == ACTION_LONG or action == ACTION_SHORT:
                # Enter position on current bar
                actual_bar_idx = self._bar_offset + self._bar_idx
                bar = self._bars[actual_bar_idx]
                self._position_state = compute_entry(bar, action, self._config)
                reward = 0.0
            else:
                # ACTION_FLAT (or sanitized hold → flat)
                reward = 0.0
        else:
            # Holding — process hold
            actual_bar_idx = self._bar_offset + self._bar_idx
            bar = self._bars[actual_bar_idx]
            reward, new_state = compute_hold_reward(
                bar, self._position_state, self._config
            )
            if new_state.position == 0:
                exit_type = classify_exit(
                    reward, self._position_state.hold_counter + 1, self._config
                )
                self._n_trades += 1
            self._position_state = new_state

        # Advance bar index
        self._bar_idx += 1

        # Check if episode is done
        terminated = self._bar_idx >= self._n_usable

        # Force-close if holding at end of episode
        if terminated and self._position_state.position != 0:
            actual_bar_idx = self._bar_offset + self._bar_idx - 1  # last bar
            bar = self._bars[actual_bar_idx]
            reward = compute_mtm_reward(self._position_state, bar.close, self._config)
            exit_type = "force_close"
            self._position_state = PositionState()
            self._n_trades += 1

        obs = self._build_obs()
        info = self._build_info(exit_type=exit_type)
        truncated = False

        return obs, float(reward), terminated, truncated, info

    def action_masks(self):
        mask = get_action_mask(self._position_state.position)
        return np.array(mask, dtype=np.int8)

    def _sanitize_action(self, action):
        """Handle invalid actions gracefully."""
        mask = get_action_mask(self._position_state.position)
        if mask[action]:
            return action
        # Invalid action — default to hold if in position, flat if flat
        if self._position_state.position != 0:
            return ACTION_HOLD
        return ACTION_FLAT

    def _build_obs(self):
        """Build observation vector: [features | position | unrealized_pnl]."""
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # Feature row — clamp to valid range
        feat_idx = min(self._bar_idx, self._n_usable - 1)
        obs[:self._feature_dim] = self._features[feat_idx]

        # Position state
        obs[self._feature_dim] = float(self._position_state.position)

        # Unrealized PnL
        if self._position_state.position != 0:
            actual_bar_idx = self._bar_offset + min(self._bar_idx, self._n_usable - 1)
            bar = self._bars[actual_bar_idx]
            obs[self._feature_dim + 1] = float(
                compute_unrealized_pnl(self._position_state, bar.close)
            )
        else:
            obs[self._feature_dim + 1] = 0.0

        return obs

    def _build_info(self, exit_type=None):
        """Build info dict."""
        return {
            "position": self._position_state.position,
            "bar_idx": self._bar_idx,
            "exit_type": exit_type,
            "entry_price": self._position_state.entry_price,
            "n_trades": self._n_trades,
        }
