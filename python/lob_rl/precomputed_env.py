"""PrecomputedEnv — pure-numpy gymnasium env for pre-computed LOB data."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lob_rl._reward import compute_forced_flatten, compute_step_reward


def _lagged_diff(arr, lag):
    """Compute arr[t] - arr[t-lag], zero-padded for t < lag."""
    result = np.zeros(len(arr), dtype=np.float32)
    if len(arr) > lag:
        result[lag:] = arr[lag:] - arr[:-lag]
    return result


# C++ observation layout (43 features per timestep)
_BID_PRICES = slice(0, 10)
_BID_SIZES = slice(10, 20)
_ASK_PRICES = slice(20, 30)
_ASK_SIZES = slice(30, 40)
_REL_SPREAD = 40
_IMBALANCE = 41
_TIME_LEFT = 42
_BASE_OBS_SIZE = 43
_FULL_OBS_SIZE = 54
_POSITION_IDX = 53


class PrecomputedEnv(gym.Env):
    metadata = {"render_modes": []}
    _ACTION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}

    def __init__(self, obs, mid, spread, reward_mode="pnl_delta", lambda_=0.0,
                 execution_cost=False, participation_bonus=0.0, step_interval=1):
        super().__init__()
        if not isinstance(step_interval, int) or step_interval < 1:
            raise ValueError("step_interval must be a positive integer")

        self._obs = np.asarray(obs, dtype=np.float32)
        self._mid = np.asarray(mid, dtype=np.float64)
        self._spread = np.asarray(spread, dtype=np.float64)

        if step_interval > 1:
            self._obs = self._obs[::step_interval]
            self._mid = self._mid[::step_interval]
            self._spread = self._spread[::step_interval]

        if self._obs.shape[0] < 2:
            raise ValueError("obs must have at least 2 rows")
        self._reward_mode = reward_mode
        self._lambda = lambda_
        self._execution_cost = execution_cost
        self._participation_bonus = participation_bonus

        self._precompute_temporal_features()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(_FULL_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._t = 0
        self._position = 0.0
        self._prev_position = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._t = 0
        self._position = 0.0
        self._prev_position = 0.0
        obs = self._build_obs()
        return obs, {}

    def step(self, action):
        self._position = self._ACTION_MAP[action]

        self._t += 1
        terminated = self._t >= self._obs.shape[0] - 1

        info = {}

        if terminated:
            reward, info = compute_forced_flatten(
                self._spread[self._t], self._prev_position, action)
            self._position = 0.0
        else:
            reward = compute_step_reward(
                self._position, self._prev_position,
                self._mid[self._t], self._mid[self._t - 1],
                self._spread[self._t - 1],
                self._reward_mode, self._lambda,
                self._execution_cost, self._participation_bonus)

        self._prev_position = self._position

        obs = self._build_obs()
        return obs, float(reward), bool(terminated), False, info

    def _precompute_temporal_features(self):
        """Precompute 10 temporal features and pack into (N, 10) array."""
        N = self._obs.shape[0]
        mid = self._mid

        mid_returns = self._compute_mid_returns(mid, N)
        vol_20 = self._compute_rolling_volatility(mid, N)
        microprice_offset = self._compute_microprice_offset(mid, N)
        total_vol_imb = self._compute_volume_imbalance(N)

        imbalance = self._obs[:, _IMBALANCE]
        rel_spread = self._obs[:, _REL_SPREAD]

        # Pack all 10 temporal features for fast slice copy in _build_obs.
        self._temporal = np.column_stack([
            mid_returns[1], mid_returns[5], mid_returns[20], mid_returns[50],
            vol_20,
            _lagged_diff(imbalance, 5),
            _lagged_diff(imbalance, 20),
            microprice_offset.astype(np.float32),
            total_vol_imb.astype(np.float32),
            _lagged_diff(rel_spread, 5),
        ])  # (N, 10) float32
        self._temporal[0] = 0.0  # t=0 convention

    def _compute_mid_returns(self, mid, N):
        """Mid-price returns at various lookbacks."""
        mid_returns = {}
        for lag in (1, 5, 20, 50):
            arr = np.zeros(N, dtype=np.float32)
            if lag < N:
                denom = np.where(mid[:-lag] != 0, mid[:-lag], 1.0)
                arr[lag:] = ((mid[lag:] - mid[:-lag]) / denom).astype(np.float32)
            mid_returns[lag] = arr
        return mid_returns

    def _compute_rolling_volatility(self, mid, N):
        """Rolling std of 1-step returns over 20 steps."""
        ret1 = np.zeros(N, dtype=np.float64)
        if N > 1:
            denom = np.where(mid[:-1] != 0, mid[:-1], 1.0)
            ret1[1:] = (mid[1:] - mid[:-1]) / denom
        vol_20 = np.zeros(N, dtype=np.float32)
        if N > 20:
            window = 20
            # Cumulative sums for O(N) rolling variance
            cumsum_ret = np.concatenate(([0.0], np.cumsum(ret1)))
            cumsum_ret_sq = np.concatenate(([0.0], np.cumsum(ret1 ** 2)))
            roll_sum = cumsum_ret[window:N] - cumsum_ret[:N - window]
            roll_sum2 = cumsum_ret_sq[window:N] - cumsum_ret_sq[:N - window]
            roll_mean = roll_sum / window
            roll_var = roll_sum2 / window - roll_mean ** 2
            np.maximum(roll_var, 0.0, out=roll_var)
            vol_20[window:N] = np.sqrt(roll_var).astype(np.float32)
        return vol_20

    def _compute_microprice_offset(self, mid, N):
        """Microprice offset: (microprice / mid) - 1."""
        bid0 = self._obs[:, _BID_PRICES.start]
        bidsize0 = self._obs[:, _BID_SIZES.start]
        ask0 = self._obs[:, _ASK_PRICES.start]
        asksize0 = self._obs[:, _ASK_SIZES.start]
        denom = bidsize0 + asksize0
        safe_denom = np.where(denom > 0, denom, np.float32(1.0))
        microprice = np.where(denom > 0,
            (ask0 * bidsize0 + bid0 * asksize0) / safe_denom,
            (bid0 + ask0) / np.float32(2.0))
        safe_mid = np.where(mid != 0, mid, 1.0)
        return np.where(mid != 0,
            microprice.astype(np.float64) / safe_mid - 1.0, 0.0)

    def _compute_volume_imbalance(self, N):
        """Total volume imbalance across all 10 levels."""
        bid_sizes_sum = self._obs[:, _BID_SIZES].astype(np.float64).sum(axis=1)
        ask_sizes_sum = self._obs[:, _ASK_SIZES].astype(np.float64).sum(axis=1)
        total = bid_sizes_sum + ask_sizes_sum
        safe_total = np.where(total > 0, total, 1.0)
        return np.where(total > 0,
            (bid_sizes_sum - ask_sizes_sum) / safe_total, 0.0)

    def _build_obs(self):
        t = self._t
        obs = np.empty(_FULL_OBS_SIZE, dtype=np.float32)
        obs[:_BASE_OBS_SIZE] = self._obs[t]
        obs[_BASE_OBS_SIZE:_POSITION_IDX] = self._temporal[t]
        obs[_POSITION_IDX] = np.float32(self._position)
        return obs

    @classmethod
    def from_file(cls, path, session_config=None, reward_mode="pnl_delta", lambda_=0.0,
                  execution_cost=False, participation_bonus=0.0, step_interval=1):
        import lob_rl_core
        from lob_rl._config import make_session_config

        cfg = make_session_config(session_config)
        obs, mid, spread, num_steps = lob_rl_core.precompute(path, cfg)
        return cls(obs, mid, spread, reward_mode=reward_mode, lambda_=lambda_,
                   execution_cost=execution_cost, participation_bonus=participation_bonus,
                   step_interval=step_interval)

    @classmethod
    def from_cache(cls, npz_path, reward_mode="pnl_delta", lambda_=0.0,
                   execution_cost=False, participation_bonus=0.0, step_interval=1):
        data = np.load(npz_path)
        for key in ("obs", "mid", "spread"):
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in {npz_path}")
        obs = data["obs"]
        mid = data["mid"]
        spread = data["spread"]
        return cls(obs, mid, spread, reward_mode=reward_mode, lambda_=lambda_,
                   execution_cost=execution_cost, participation_bonus=participation_bonus,
                   step_interval=step_interval)
