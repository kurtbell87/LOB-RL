"""Gymnasium wrapper for LOBEnv."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import lob_rl_core
from lob_rl._config import make_session_config


class LOBGymEnv(gym.Env):
    """Gymnasium-compatible wrapper around the C++ LOBEnv."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        file_path=None,
        session_config=None,
        steps_per_episode=50,
        reward_mode="pnl_delta",
        lambda_=0.0,
        execution_cost=False,
        participation_bonus=0.0,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Build the C++ LOBEnv with appropriate constructor
        if file_path is not None and session_config is not None:
            cfg = make_session_config(session_config)
            self._env = lob_rl_core.LOBEnv(
                file_path, cfg, steps_per_episode,
                reward_mode, lambda_, execution_cost,
                participation_bonus,
            )
        elif file_path is not None:
            self._env = lob_rl_core.LOBEnv(
                file_path, steps_per_episode,
                reward_mode, lambda_, execution_cost,
                participation_bonus,
            )
        else:
            self._env = lob_rl_core.LOBEnv(
                steps_per_episode,
                reward_mode, lambda_, execution_cost,
                participation_bonus,
            )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        obs_list = self._env.reset()
        obs = np.array(obs_list, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action = int(action)
        obs_list, reward, done = self._env.step(action)
        obs = np.array(obs_list, dtype=np.float32)
        terminated = bool(done)
        truncated = False
        info = {}
        return obs, float(reward), terminated, truncated, info
