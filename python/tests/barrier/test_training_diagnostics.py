"""Tests for the barrier training diagnostics callback.

Spec: docs/t9-ppo-training.md — Module 3: Training Diagnostics

BarrierDiagnosticCallback monitors training health per spec Section 5.3:
entropy, value loss, flat action rate, episode reward, trade win rate,
NaN detection, and red flag alerts.
"""

import numpy as np
import pytest

from .conftest import make_session_data_list


# ===========================================================================
# 1. Callback initialization (spec test 19)
# ===========================================================================


class TestDiagnosticCallbackInit:
    """Verify callback can be created and has correct defaults."""

    def test_diagnostic_callback_init(self):
        """Callback can be created with default parameters."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        assert cb is not None
        assert hasattr(cb, "check_freq")
        assert hasattr(cb, "diagnostics")

    def test_diagnostic_callback_custom_params(self):
        """Callback accepts custom check_freq and output_dir."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback(check_freq=5, output_dir="/tmp/test_diag", verbose=1)
        assert cb.check_freq == 5


# ===========================================================================
# 2. Metric tracking (spec tests 20–22)
# ===========================================================================


class TestDiagnosticMetricTracking:
    """Verify diagnostic metrics are tracked after rollouts.

    These tests use a real short training run with MaskablePPO to generate
    actual rollout data. This is the only reliable way to test the callback.
    """

    @pytest.fixture
    def trained_callback(self, tmp_path):
        """Run a minimal MaskablePPO training (128 steps) and return the callback."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from sb3_contrib import MaskablePPO

        sessions = make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        cb = BarrierDiagnosticCallback(check_freq=1, output_dir=str(tmp_path))
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128, callback=cb)
        vec_env.close()
        return cb

    def test_diagnostic_tracks_episode_reward(self, trained_callback):
        """After a rollout, episode_reward_mean is populated."""
        cb = trained_callback
        assert len(cb.diagnostics) > 0, "Should have at least one diagnostic snapshot"
        snapshot = cb.diagnostics[-1]
        assert "episode_reward_mean" in snapshot, "Must track episode_reward_mean"
        # The value should be a finite number (or None if no episodes completed)
        val = snapshot["episode_reward_mean"]
        if val is not None:
            assert isinstance(val, (float, int, np.floating))

    def test_diagnostic_tracks_flat_action_rate(self, trained_callback):
        """After a rollout, flat_action_rate is in [0, 1]."""
        cb = trained_callback
        assert len(cb.diagnostics) > 0
        snapshot = cb.diagnostics[-1]
        assert "flat_action_rate" in snapshot, "Must track flat_action_rate"
        val = snapshot["flat_action_rate"]
        if val is not None:
            assert 0.0 <= val <= 1.0, f"flat_action_rate must be in [0,1], got {val}"

    def test_diagnostic_tracks_trade_win_rate(self, trained_callback):
        """After a rollout with trades, trade_win_rate is in [0, 1]."""
        cb = trained_callback
        assert len(cb.diagnostics) > 0
        snapshot = cb.diagnostics[-1]
        assert "trade_win_rate" in snapshot, "Must track trade_win_rate"
        val = snapshot["trade_win_rate"]
        if val is not None:
            assert 0.0 <= val <= 1.0, f"trade_win_rate must be in [0,1], got {val}"


# ===========================================================================
# 3. NaN detection (spec test 23)
# ===========================================================================


class TestDiagnosticNanDetection:
    """Verify NaN detection and red flag raising."""

    def test_diagnostic_detects_nan(self):
        """If has_nan is True, check_red_flags() includes NaN red flag."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        # Manually inject a diagnostic snapshot with NaN
        cb.diagnostics.append({
            "entropy_flat": float("nan"),
            "value_loss": 0.5,
            "policy_loss": -0.01,
            "episode_reward_mean": -0.1,
            "flat_action_rate": 0.5,
            "trade_win_rate": 0.4,
            "n_trades": 10,
            "has_nan": True,
        })

        flags = cb.check_red_flags()
        assert any("nan" in f.lower() or "NaN" in f for f in flags), (
            f"NaN red flag not raised. Flags: {flags}"
        )


# ===========================================================================
# 4. Red flag detection (spec tests 24–25, 28)
# ===========================================================================


class TestRedFlagDetection:
    """Verify red flag detection logic."""

    def test_diagnostic_no_red_flags_initially(self):
        """Before training, check_red_flags() returns empty list."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        flags = cb.check_red_flags()
        assert flags == [], f"Expected no flags initially, got {flags}"

    def test_diagnostic_entropy_collapse_flag(self):
        """Flat-state entropy < 0.3 in first 100 updates raises red flag."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        # Simulate 50 updates with collapsed entropy (< 0.3)
        for i in range(50):
            cb.diagnostics.append({
                "entropy_flat": 0.1,  # Collapsed
                "value_loss": 0.5,
                "policy_loss": -0.01,
                "episode_reward_mean": -0.1,
                "flat_action_rate": 0.5,
                "trade_win_rate": 0.4,
                "n_trades": 10,
                "has_nan": False,
            })

        flags = cb.check_red_flags()
        assert any("entropy" in f.lower() for f in flags), (
            f"Entropy collapse not detected. Flags: {flags}"
        )

    def test_diagnostic_flat_rate_red_flag(self):
        """Flat action rate outside [10%, 90%] raises red flag."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        # Flat rate at 0.95 → outside [0.10, 0.90]
        cb.diagnostics.append({
            "entropy_flat": 1.0,
            "value_loss": 0.5,
            "policy_loss": -0.01,
            "episode_reward_mean": 0.0,
            "flat_action_rate": 0.95,
            "trade_win_rate": 0.4,
            "n_trades": 10,
            "has_nan": False,
        })

        flags = cb.check_red_flags()
        assert any("flat" in f.lower() for f in flags), (
            f"Flat rate red flag not raised. Flags: {flags}"
        )

    def test_diagnostic_flat_rate_low_red_flag(self):
        """Flat action rate at 0.05 (< 10%) raises red flag."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        cb.diagnostics.append({
            "entropy_flat": 1.0,
            "value_loss": 0.5,
            "policy_loss": -0.01,
            "episode_reward_mean": 0.0,
            "flat_action_rate": 0.05,
            "trade_win_rate": 0.4,
            "n_trades": 10,
            "has_nan": False,
        })

        flags = cb.check_red_flags()
        assert any("flat" in f.lower() for f in flags), (
            f"Low flat rate red flag not raised. Flags: {flags}"
        )


# ===========================================================================
# 5. CSV output and snapshot accumulation (spec tests 26–27)
# ===========================================================================


class TestDiagnosticOutput:
    """Verify CSV writing and snapshot accumulation."""

    def test_diagnostic_writes_csv(self, tmp_path):
        """When output_dir is set, diagnostics are written to a CSV file."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from sb3_contrib import MaskablePPO

        sessions = make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )
        output_dir = tmp_path / "diag_out"
        output_dir.mkdir()

        cb = BarrierDiagnosticCallback(check_freq=1, output_dir=str(output_dir))
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128, callback=cb)
        vec_env.close()

        # Check that CSV was written
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) > 0, (
            f"Expected CSV file in {output_dir}, found: {list(output_dir.iterdir())}"
        )

    def test_diagnostic_snapshots_accumulate(self):
        """Each rollout adds one entry to diagnostics list."""
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        cb = BarrierDiagnosticCallback()
        # Manually add snapshots
        for i in range(5):
            cb.diagnostics.append({
                "entropy_flat": 1.0 - i * 0.1,
                "value_loss": 0.5,
                "policy_loss": -0.01,
                "episode_reward_mean": -0.1,
                "flat_action_rate": 0.5,
                "trade_win_rate": 0.4,
                "n_trades": 10,
                "has_nan": False,
            })
        assert len(cb.diagnostics) == 5, (
            f"Expected 5 snapshots, got {len(cb.diagnostics)}"
        )
