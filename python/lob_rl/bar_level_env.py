"""BarLevelEnv — bar-level gymnasium environment for LOB RL."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lob_rl.bar_aggregation import aggregate_bars
from lob_rl._reward import compute_forced_flatten, compute_step_reward

# Observation layout: 13 intra-bar + 7 cross-bar temporal + 1 position = 21
_NUM_BAR_FEATURES = 13
_NUM_TEMPORAL = 7
_OBS_SIZE = 21
_POSITION_IDX = 20


class BarLevelEnv(gym.Env):
    """Gymnasium env that steps at bar-level granularity.

    Each step corresponds to one completed bar (bar_size ticks).
    Observation is 21-dim: 13 intra-bar features + 7 temporal + 1 position.
    """

    metadata = {"render_modes": []}
    _ACTION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}

    def __init__(self, obs, mid, spread, bar_size=500,
                 reward_mode="pnl_delta", lambda_=0.0,
                 execution_cost=False, participation_bonus=0.0):
        super().__init__()

        self._reward_mode = reward_mode
        self._lambda = lambda_
        self._execution_cost = execution_cost
        self._participation_bonus = participation_bonus

        # Aggregate ticks into bars
        self._bar_features, self._bar_mid_close, self._bar_spread_close = \
            aggregate_bars(obs, mid, spread, bar_size)

        self._num_bars = self._bar_features.shape[0]

        if self._num_bars < 2:
            raise ValueError(
                f"Need at least 2 bars, got {self._num_bars} "
                f"(try a smaller bar_size)"
            )

        # Precompute cross-bar temporal features
        self._precompute_temporal()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._bar_index = 0
        self._position = 0.0
        self._prev_position = 0.0

    def _precompute_temporal(self):
        """Compute 7 cross-bar temporal features for each bar index."""
        num_bars = self._num_bars
        self._temporal = np.zeros((num_bars, _NUM_TEMPORAL), dtype=np.float32)

        if num_bars == 0:
            return

        bar_return = self._bar_features[:, 0]
        imbalance_close = self._bar_features[:, 6]
        spread_close = self._bar_features[:, 4]

        # 0: return_lag1 — bar_return shifted by 1
        if num_bars > 1:
            self._temporal[1:, 0] = bar_return[:-1]

        # 1: return_lag3 — bar_return shifted by 3
        if num_bars > 3:
            self._temporal[3:, 1] = bar_return[:-3]

        # 2: return_lag5 — bar_return shifted by 5
        if num_bars > 5:
            self._temporal[5:, 2] = bar_return[:-5]

        # 3: cumulative_return_5 — rolling sum of bar_return[max(0,t-5):t]
        cumsum_ret = np.concatenate(([0.0], np.cumsum(bar_return)))
        for t in range(1, min(num_bars, 6)):
            self._temporal[t, 3] = cumsum_ret[t] - cumsum_ret[0]
        if num_bars > 5:
            self._temporal[5:, 3] = (cumsum_ret[5:num_bars] - cumsum_ret[:num_bars - 5]).astype(np.float32)

        # 4: rolling_vol_5 — std of bar_return[max(0,t-5):t] for t >= 2
        # Use cumulative sums for O(num_bars) computation of rolling variance
        cumsum_ret_sq = np.concatenate(([0.0], np.cumsum(bar_return.astype(np.float64) ** 2)))
        cumsum_ret_f64 = np.concatenate(([0.0], np.cumsum(bar_return.astype(np.float64))))
        for t in range(2, min(num_bars, 6)):
            w = t  # window size: bar_return[0:t]
            roll_sum = cumsum_ret_f64[t]
            roll_sum2 = cumsum_ret_sq[t]
            roll_mean = roll_sum / w
            roll_var = roll_sum2 / w - roll_mean ** 2
            self._temporal[t, 4] = np.sqrt(max(roll_var, 0.0))
        if num_bars > 5:
            w = 5
            roll_sum = cumsum_ret_f64[5:num_bars] - cumsum_ret_f64[:num_bars - 5]
            roll_sum2 = cumsum_ret_sq[5:num_bars] - cumsum_ret_sq[:num_bars - 5]
            roll_mean = roll_sum / w
            roll_var = roll_sum2 / w - roll_mean ** 2
            np.maximum(roll_var, 0.0, out=roll_var)
            self._temporal[5:, 4] = np.sqrt(roll_var).astype(np.float32)

        # 5: imb_delta_3 — imbalance_close[t] - imbalance_close[t-3]
        if num_bars > 3:
            self._temporal[3:, 5] = imbalance_close[3:] - imbalance_close[:-3]

        # 6: spread_delta_3 — spread_close[t] - spread_close[t-3]
        if num_bars > 3:
            self._temporal[3:, 6] = spread_close[3:] - spread_close[:-3]

    def _build_obs(self):
        """Build 21-dim observation for current bar index."""
        t = self._bar_index
        obs = np.empty(_OBS_SIZE, dtype=np.float32)
        obs[:_NUM_BAR_FEATURES] = self._bar_features[t]
        obs[_NUM_BAR_FEATURES:_POSITION_IDX] = self._temporal[t]
        obs[_POSITION_IDX] = np.float32(self._position)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._bar_index = 0
        self._position = 0.0
        self._prev_position = 0.0
        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        self._position = self._ACTION_MAP[action]

        self._bar_index += 1
        terminated = self._bar_index >= self._num_bars - 1

        info = {}

        if terminated:
            reward, info = compute_forced_flatten(
                self._bar_spread_close[self._bar_index],
                self._prev_position, action)
            self._position = 0.0
        else:
            reward = compute_step_reward(
                self._position, self._prev_position,
                self._bar_mid_close[self._bar_index],
                self._bar_mid_close[self._bar_index - 1],
                self._bar_spread_close[self._bar_index - 1],
                self._reward_mode, self._lambda,
                self._execution_cost, self._participation_bonus)

        self._prev_position = self._position

        obs = self._build_obs()
        return obs, float(reward), bool(terminated), False, info

    @classmethod
    def from_cache(cls, npz_path, bar_size=500, reward_mode="pnl_delta",
                   lambda_=0.0, execution_cost=False, participation_bonus=0.0):
        """Create BarLevelEnv from a cached .npz file."""
        data = np.load(npz_path)
        for key in ("obs", "mid", "spread"):
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in {npz_path}")
        return cls(data["obs"], data["mid"], data["spread"],
                   bar_size=bar_size, reward_mode=reward_mode,
                   lambda_=lambda_, execution_cost=execution_cost,
                   participation_bonus=participation_bonus)

    @classmethod
    def from_file(cls, path, session_config=None, bar_size=500,
                  reward_mode="pnl_delta", lambda_=0.0,
                  execution_cost=False, participation_bonus=0.0):
        """Create BarLevelEnv from a raw .bin file via C++ precompute."""
        import lob_rl_core
        from lob_rl._config import make_session_config

        cfg = make_session_config(session_config)
        obs, mid, spread, num_steps = lob_rl_core.precompute(path, cfg)
        return cls(obs, mid, spread, bar_size=bar_size,
                   reward_mode=reward_mode, lambda_=lambda_,
                   execution_cost=execution_cost,
                   participation_bonus=participation_bonus)
