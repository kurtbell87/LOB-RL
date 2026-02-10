"""Barrier training diagnostics callback.

Monitors training health: entropy, value loss, flat action rate,
episode reward, trade win rate, NaN detection, and red flag alerts.
Also provides linear_schedule() utility for learning rate scheduling.
"""

import csv
import math
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Red-flag thresholds (Section 5.3 of PRD)
_ENTROPY_COLLAPSE_THRESHOLD = 0.3
_ENTROPY_COLLAPSE_WINDOW = 100        # first N rollouts to check
_FLAT_RATE_LOW = 0.10                 # flat action rate floor
_FLAT_RATE_HIGH = 0.90                # flat action rate ceiling


def linear_schedule(initial_value):
    """Return a callable that linearly decays from initial_value to 0.

    Parameters
    ----------
    initial_value : float
        The value at progress_remaining=1.0.

    Returns
    -------
    callable
        f(progress_remaining) -> current_value
    """
    def _schedule(progress_remaining):
        return initial_value * progress_remaining
    return _schedule


class BarrierDiagnosticCallback(BaseCallback):
    """SB3 callback that tracks training diagnostics for barrier envs.

    Parameters
    ----------
    check_freq : int
        Check diagnostics every N rollouts (default 1).
    output_dir : str, optional
        Directory to write CSV diagnostics.
    verbose : int
        Verbosity level.
    """

    def __init__(self, check_freq=1, output_dir=None, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.output_dir = output_dir
        self.diagnostics = []
        self._rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.check_freq != 0:
            return

        snapshot = self._collect_snapshot()
        self.diagnostics.append(snapshot)

        if self.output_dir is not None:
            self._write_csv()

    def _read_logger_metric(self, key):
        """Read a metric from the SB3 logger, returning (value, is_nan).

        Returns (None, False) if the metric is unavailable.
        """
        try:
            val = self.model.logger.name_to_value.get(key)
            if val is not None:
                val = float(val)
                return val, math.isnan(val)
        except (AttributeError, TypeError, ValueError):
            pass
        return None, False

    def _read_episode_metrics(self):
        """Read episode reward mean, trade win rate, and trade count.

        Returns (episode_reward_mean, trade_win_rate, n_trades).
        """
        episode_reward_mean = None
        trade_win_rate = None
        n_trades = 0
        try:
            ep_info = self.model.ep_info_buffer
            if ep_info and len(ep_info) > 0:
                rewards = [info["r"] for info in ep_info]
                episode_reward_mean = float(np.mean(rewards))
                n_trades = len(ep_info)
                wins = sum(1 for info in ep_info if info.get("r", 0) > 0)
                trade_win_rate = float(wins / n_trades)
        except (AttributeError, KeyError, TypeError):
            pass
        return episode_reward_mean, trade_win_rate, n_trades

    def _read_flat_action_rate(self):
        """Read flat action rate from the rollout buffer."""
        try:
            buf = self.model.rollout_buffer
            if hasattr(buf, "actions"):
                actions = buf.actions.flatten()
                if len(actions) > 0:
                    flat_count = np.sum(actions == 2)  # ACTION_FLAT=2
                    return float(flat_count / len(actions))
        except (AttributeError, TypeError):
            pass
        return None

    def _collect_snapshot(self):
        """Collect diagnostic metrics from the current training state."""
        has_nan = False

        entropy_flat, nan = self._read_logger_metric("train/entropy_loss")
        has_nan = has_nan or nan

        value_loss, nan = self._read_logger_metric("train/value_loss")
        has_nan = has_nan or nan

        policy_loss, nan = self._read_logger_metric("train/policy_gradient_loss")
        has_nan = has_nan or nan

        episode_reward_mean, trade_win_rate, n_trades = self._read_episode_metrics()
        flat_action_rate = self._read_flat_action_rate()

        return {
            "entropy_flat": entropy_flat,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "episode_reward_mean": episode_reward_mean,
            "flat_action_rate": flat_action_rate,
            "trade_win_rate": trade_win_rate,
            "n_trades": n_trades,
            "has_nan": has_nan,
        }

    def check_red_flags(self):
        """Check diagnostics for red flags.

        Returns
        -------
        list[str]
            List of red flag descriptions. Empty if all OK.
        """
        flags = []

        if len(self.diagnostics) == 0:
            return flags

        # Check for NaN in any snapshot
        for snap in self.diagnostics:
            if snap.get("has_nan", False):
                flags.append("NaN detected in training metrics")
                break

        # Entropy collapse: entropy below threshold in early rollouts
        early_snapshots = self.diagnostics[:_ENTROPY_COLLAPSE_WINDOW]
        for snap in early_snapshots:
            ent = snap.get("entropy_flat")
            if ent is not None and not math.isnan(ent) and ent < _ENTROPY_COLLAPSE_THRESHOLD:
                flags.append(
                    f"Entropy collapse: flat-state entropy {ent:.3f} < "
                    f"{_ENTROPY_COLLAPSE_THRESHOLD} "
                    f"in first {_ENTROPY_COLLAPSE_WINDOW} updates"
                )
                break

        # Flat action rate outside acceptable range
        for snap in self.diagnostics:
            rate = snap.get("flat_action_rate")
            if rate is not None:
                if rate < _FLAT_RATE_LOW:
                    flags.append(
                        f"Flat action rate {rate:.3f} < {_FLAT_RATE_LOW:.0%}"
                    )
                    break
                if rate > _FLAT_RATE_HIGH:
                    flags.append(
                        f"Flat action rate {rate:.3f} > {_FLAT_RATE_HIGH:.0%}"
                    )
                    break

        return flags

    def _write_csv(self):
        """Write diagnostics to CSV file."""
        if not self.output_dir:
            return

        csv_path = os.path.join(self.output_dir, "diagnostics.csv")
        fieldnames = [
            "entropy_flat", "value_loss", "policy_loss",
            "episode_reward_mean", "flat_action_rate",
            "trade_win_rate", "n_trades", "has_nan",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for snap in self.diagnostics:
                writer.writerow(snap)
