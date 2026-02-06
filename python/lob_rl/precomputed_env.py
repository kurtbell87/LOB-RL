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

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
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

    def _build_obs(self):
        obs = np.empty(44, dtype=np.float32)
        obs[:43] = self._obs[self._t]
        obs[43] = np.float32(self._position)
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
