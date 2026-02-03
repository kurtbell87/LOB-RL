"""Gymnasium wrapper for LOBEnv."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lob_rl_core import LOBEnv as _LOBEnv
from lob_rl_core import EnvConfig, RewardType


class LOBEnvGym(gym.Env):
    """Gymnasium-compatible wrapper for the C++ LOBEnv.

    Actions:
        0: Target short (-1)
        1: Target flat (0)
        2: Target long (+1)

    Observation: 44-float vector (10-level book + features)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int = 42,
        num_messages: int = 10000,
        book_depth: int = 10,
        trades_per_step: int = 100,
        reward_type: str = "pnl_delta",
        inventory_penalty: float = 0.0,
    ):
        super().__init__()

        config = EnvConfig()
        config.book_depth = book_depth
        config.trades_per_step = trades_per_step
        config.inventory_penalty = inventory_penalty

        if reward_type == "pnl_delta":
            config.reward_type = RewardType.PnLDelta
        elif reward_type == "pnl_delta_penalized":
            config.reward_type = RewardType.PnLDeltaPenalized
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        self._env = _LOBEnv(config, seed, num_messages)
        self._obs_size = self._env.observation_size()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        result = self._env.reset()
        obs = np.array(result.obs, dtype=np.float32)
        info = {
            "position": result.position,
            "pnl": result.pnl,
            "timestamp_ns": result.timestamp_ns,
        }
        return obs, info

    def step(self, action: int):
        result = self._env.step(int(action))
        obs = np.array(result.obs, dtype=np.float32)
        info = {
            "position": result.position,
            "pnl": result.pnl,
            "timestamp_ns": result.timestamp_ns,
        }
        return obs, result.reward, result.done, False, info
