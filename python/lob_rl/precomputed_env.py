"""PrecomputedEnv — pure-numpy gymnasium env for pre-computed LOB data."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PrecomputedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, obs, mid, spread, reward_mode="pnl_delta", lambda_=0.0,
                 execution_cost=False, participation_bonus=0.0):
        super().__init__()
        if obs.shape[0] < 2:
            raise ValueError("obs must have at least 2 rows")

        self._obs = np.asarray(obs, dtype=np.float32)
        self._mid = np.asarray(mid, dtype=np.float64)
        self._spread = np.asarray(spread, dtype=np.float64)
        self._reward_mode = reward_mode
        self._lambda = lambda_
        self._execution_cost = execution_cost
        self._participation_bonus = participation_bonus

        self._precompute_temporal_features()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32
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
        action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        self._position = action_map[action]

        reward = self._position * (self._mid[self._t + 1] - self._mid[self._t])

        if self._reward_mode == "pnl_delta_penalized":
            reward -= self._lambda * abs(self._position)

        # Execution cost: spread/2 * |delta_pos|
        if self._execution_cost:
            spread = self._spread[self._t]
            if np.isfinite(spread):
                reward -= spread / 2.0 * abs(self._position - self._prev_position)
        self._prev_position = self._position

        # Participation bonus: bonus * |position|
        if self._participation_bonus != 0.0:
            reward += self._participation_bonus * abs(self._position)

        self._t += 1
        terminated = self._t >= self._obs.shape[0] - 1

        if terminated:
            reward -= abs(self._position) * self._spread[self._t] / 2.0

        obs = self._build_obs()
        return obs, float(reward), bool(terminated), False, {}

    def _precompute_temporal_features(self):
        N = self._obs.shape[0]
        mid = self._mid

        # Mid-price returns at various lookbacks
        mid_returns = {}
        for lag in (1, 5, 20, 50):
            arr = np.zeros(N, dtype=np.float32)
            if lag < N:
                denom = np.where(mid[:-lag] != 0, mid[:-lag], 1.0)
                arr[lag:] = ((mid[lag:] - mid[:-lag]) / denom).astype(np.float32)
            mid_returns[lag] = arr

        # Volatility: rolling std of 1-step returns over 20 steps
        ret1 = np.zeros(N, dtype=np.float64)
        if N > 1:
            denom = np.where(mid[:-1] != 0, mid[:-1], 1.0)
            ret1[1:] = (mid[1:] - mid[:-1]) / denom
        vol_20 = np.zeros(N, dtype=np.float32)
        if N > 20:
            window = 20
            # Use cumulative sums for O(N) rolling variance instead of O(N*W) loop.
            # For t in [window, N): vol_20[t] = std(ret1[t-window:t])
            cs = np.concatenate(([0.0], np.cumsum(ret1)))
            cs2 = np.concatenate(([0.0], np.cumsum(ret1 ** 2)))
            roll_sum = cs[window:N] - cs[:N - window]
            roll_sum2 = cs2[window:N] - cs2[:N - window]
            roll_mean = roll_sum / window
            roll_var = roll_sum2 / window - roll_mean ** 2
            np.maximum(roll_var, 0.0, out=roll_var)
            vol_20[window:N] = np.sqrt(roll_var).astype(np.float32)

        # Imbalance deltas
        imbalance = self._obs[:, 41]
        imb_delta_5 = np.zeros(N, dtype=np.float32)
        imb_delta_20 = np.zeros(N, dtype=np.float32)
        if N > 5:
            imb_delta_5[5:] = imbalance[5:] - imbalance[:-5]
        if N > 20:
            imb_delta_20[20:] = imbalance[20:] - imbalance[:-20]

        # Microprice offset — compute microprice in float32 (matching obs dtype),
        # then divide by float64 mid for the final offset
        bid0 = self._obs[:, 0]
        bidsize0 = self._obs[:, 10]
        ask0 = self._obs[:, 20]
        asksize0 = self._obs[:, 30]
        denom = bidsize0 + asksize0
        microprice = np.where(denom > 0,
            (ask0 * bidsize0 + bid0 * asksize0) / np.where(denom > 0, denom, np.float32(1.0)),
            (bid0 + ask0) / np.float32(2.0))
        safe_mid = np.where(mid != 0, mid, 1.0)
        microprice_offset = np.where(mid != 0,
            microprice.astype(np.float64) / safe_mid - 1.0, 0.0)

        # Total volume imbalance across all 10 levels (compute in float64)
        bid_sizes_sum = self._obs[:, 10:20].astype(np.float64).sum(axis=1)
        ask_sizes_sum = self._obs[:, 30:40].astype(np.float64).sum(axis=1)
        total = bid_sizes_sum + ask_sizes_sum
        total_vol_imb = np.where(total > 0,
            (bid_sizes_sum - ask_sizes_sum) / np.where(total > 0, total, 1.0), 0.0)

        # Spread change over 5 steps
        rel_spread = self._obs[:, 40]
        spread_change_5 = np.zeros(N, dtype=np.float32)
        if N > 5:
            spread_change_5[5:] = rel_spread[5:] - rel_spread[:-5]

        # Pack all 10 temporal features into a single (N, 10) array for fast
        # slice copy in _build_obs.  Row 0 is zeroed (t=0 convention).
        self._temporal = np.column_stack([
            mid_returns[1], mid_returns[5], mid_returns[20], mid_returns[50],
            vol_20,
            imb_delta_5, imb_delta_20,
            microprice_offset.astype(np.float32),
            total_vol_imb.astype(np.float32),
            spread_change_5,
        ])  # (N, 10) float32
        self._temporal[0] = 0.0

    def _build_obs(self):
        t = self._t
        obs = np.empty(54, dtype=np.float32)
        obs[:43] = self._obs[t]
        obs[43:53] = self._temporal[t]
        obs[53] = np.float32(self._position)
        return obs

    @classmethod
    def from_file(cls, path, session_config=None, reward_mode="pnl_delta", lambda_=0.0,
                  execution_cost=False, participation_bonus=0.0):
        import lob_rl_core
        from lob_rl._config import make_session_config

        cfg = make_session_config(session_config)
        obs, mid, spread, num_steps = lob_rl_core.precompute(path, cfg)
        return cls(obs, mid, spread, reward_mode=reward_mode, lambda_=lambda_,
                   execution_cost=execution_cost, participation_bonus=participation_bonus)
