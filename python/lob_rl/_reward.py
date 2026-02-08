"""Shared reward and forced-flatten logic for LOB environments."""

import numpy as np


def compute_forced_flatten(spread, prev_position, action):
    """Compute forced-flatten reward on the terminal step.

    Returns (reward, info_dict). Position should be set to 0.0 after calling.
    """
    close_cost = spread / 2.0 * abs(prev_position)
    info = {
        "forced_flatten": True,
        "forced_flatten_cost": float(close_cost),
        "intended_action": action,
    }
    return -close_cost, info


def compute_step_reward(position, prev_position, mid_now, mid_prev,
                        spread_prev, reward_mode, lambda_,
                        execution_cost, participation_bonus):
    """Compute reward for a non-terminal step.

    Returns reward (float).
    """
    reward = position * (mid_now - mid_prev)

    if reward_mode == "pnl_delta_penalized":
        reward -= lambda_ * abs(position)

    if execution_cost:
        if np.isfinite(spread_prev):
            reward -= spread_prev / 2.0 * abs(position - prev_position)

    if participation_bonus != 0.0:
        reward += participation_bonus * abs(position)

    return reward
