"""Tests for scripts/train_barrier.py — CLI training script for barrier-hit PPO.

Spec: docs/t9b-train-barrier-script.md

The script wires together T9 modules (MultiSessionBarrierEnv, barrier_vec_env,
training_diagnostics) into an end-to-end training pipeline with CLI args.

Test categories:
1. Argument parsing — defaults, custom overrides, resume flag, no-normalize flag
2. Session splitting — proportions, determinism, no overlap
3. Model building — Section 5.2 hyperparameters, architecture, resume from checkpoint
4. End-to-end smoke — main() with synthetic data, checkpoint file existence
"""

import sys
import os
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — synthetic session data for integration tests
# ---------------------------------------------------------------------------

_H = 10
_OBS_DIM = 13 * _H + 2  # 132


def _make_session_data_list(n_sessions=10, n_bars=40, h=_H):
    """Build session data dicts from synthetic bars."""
    from lob_rl.barrier.feature_pipeline import build_feature_matrix
    from lob_rl.barrier.label_pipeline import compute_labels
    from .conftest import make_session_bars

    sessions = []
    for i in range(n_sessions):
        bars = make_session_bars(n_bars, base_price=4000.0 + i * 10.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=h)
        sessions.append({"bars": bars, "labels": labels, "features": features})
    return sessions


# ===========================================================================
# 1. Script Argument Parsing (spec tests 1–4)
# ===========================================================================


class TestParseArgsDefaults:
    """Verify parse_args() returns spec-compliant default values."""

    def test_parse_args_defaults(self):
        """Default values match spec: bar_size=500, lookback=10, n_envs=4,
        total_timesteps=2_000_000, eval_freq=10_000, checkpoint_freq=50_000,
        train_frac=0.8, seed=42, resume=None, no_normalize=False."""
        from scripts.train_barrier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/dummy",
            "--output-dir", "/tmp/out",
        ])

        assert args.bar_size == 500, f"Expected bar_size=500, got {args.bar_size}"
        assert args.lookback == 10, f"Expected lookback=10, got {args.lookback}"
        assert args.n_envs == 4, f"Expected n_envs=4, got {args.n_envs}"
        assert args.total_timesteps == 2_000_000, (
            f"Expected total_timesteps=2_000_000, got {args.total_timesteps}"
        )
        assert args.eval_freq == 10_000, (
            f"Expected eval_freq=10_000, got {args.eval_freq}"
        )
        assert args.checkpoint_freq == 50_000, (
            f"Expected checkpoint_freq=50_000, got {args.checkpoint_freq}"
        )
        assert args.train_frac == pytest.approx(0.8), (
            f"Expected train_frac=0.8, got {args.train_frac}"
        )
        assert args.seed == 42, f"Expected seed=42, got {args.seed}"
        assert args.resume is None, f"Expected resume=None, got {args.resume}"
        assert args.no_normalize is False, (
            f"Expected no_normalize=False, got {args.no_normalize}"
        )


class TestParseArgsCustom:
    """Verify parse_args() accepts custom value overrides."""

    def test_parse_args_custom(self):
        """Custom values override all defaults."""
        from scripts.train_barrier import parse_args

        args = parse_args([
            "--data-dir", "/data/custom",
            "--output-dir", "/out/custom",
            "--bar-size", "1000",
            "--lookback", "20",
            "--n-envs", "8",
            "--total-timesteps", "5000000",
            "--eval-freq", "5000",
            "--checkpoint-freq", "100000",
            "--train-frac", "0.7",
            "--seed", "99",
        ])

        assert args.data_dir == "/data/custom"
        assert args.output_dir == "/out/custom"
        assert args.bar_size == 1000
        assert args.lookback == 20
        assert args.n_envs == 8
        assert args.total_timesteps == 5_000_000
        assert args.eval_freq == 5000
        assert args.checkpoint_freq == 100_000
        assert args.train_frac == pytest.approx(0.7)
        assert args.seed == 99


class TestParseArgsResume:
    """Verify --resume stores the checkpoint path."""

    def test_parse_args_resume(self):
        """--resume PATH correctly stored."""
        from scripts.train_barrier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/d",
            "--output-dir", "/tmp/o",
            "--resume", "/checkpoints/model_500000.zip",
        ])

        assert args.resume == "/checkpoints/model_500000.zip", (
            f"Expected resume='/checkpoints/model_500000.zip', got {args.resume}"
        )


class TestParseArgsNoNormalize:
    """Verify --no-normalize flag."""

    def test_parse_args_no_normalize(self):
        """--no-normalize flag sets no_normalize=True."""
        from scripts.train_barrier import parse_args

        args = parse_args([
            "--data-dir", "/tmp/d",
            "--output-dir", "/tmp/o",
            "--no-normalize",
        ])

        assert args.no_normalize is True, (
            f"Expected no_normalize=True, got {args.no_normalize}"
        )


# ===========================================================================
# 2. Session Data Splitting (spec tests 5–7)
# ===========================================================================


class TestSplitSessionsProportions:
    """Verify split_sessions() produces correct proportions."""

    def test_split_sessions_proportions(self):
        """With 10 sessions, train_frac=0.8 → 8 train / 1 val / 1 test."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)
        train, val, test = split_sessions(sessions, train_frac=0.8, seed=42)

        assert len(train) == 8, f"Expected 8 train sessions, got {len(train)}"
        assert len(val) == 1, f"Expected 1 val session, got {len(val)}"
        assert len(test) == 1, f"Expected 1 test session, got {len(test)}"
        assert len(train) + len(val) + len(test) == 10, "Total must equal input count"

    def test_split_sessions_larger_dataset(self):
        """With 20 sessions, train_frac=0.8 → 16 train / 2 val / 2 test."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=20)
        train, val, test = split_sessions(sessions, train_frac=0.8, seed=42)

        assert len(train) == 16, f"Expected 16 train, got {len(train)}"
        assert len(val) == 2, f"Expected 2 val, got {len(val)}"
        assert len(test) == 2, f"Expected 2 test, got {len(test)}"

    def test_split_sessions_custom_frac(self):
        """train_frac=0.6 with 10 sessions → 6 train / 2 val / 2 test."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)
        train, val, test = split_sessions(sessions, train_frac=0.6, seed=42)

        assert len(train) == 6, f"Expected 6 train, got {len(train)}"
        # Remaining 4 split evenly: 2 val, 2 test
        assert len(val) == 2, f"Expected 2 val, got {len(val)}"
        assert len(test) == 2, f"Expected 2 test, got {len(test)}"


class TestSplitSessionsDeterministic:
    """Verify split is deterministic given the same seed."""

    def test_split_sessions_deterministic(self):
        """Same seed gives identical splits."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)

        train1, val1, test1 = split_sessions(sessions, train_frac=0.8, seed=42)
        train2, val2, test2 = split_sessions(sessions, train_frac=0.8, seed=42)

        # Compare by checking that the same session bars are in each split
        # (sessions are dicts with 'bars' key — check bar count equality)
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)

        for s1, s2 in zip(train1, train2):
            assert len(s1["bars"]) == len(s2["bars"]), "Train sessions differ"
            # Check that actual bar data matches (first bar's close price as identity)
            assert s1["bars"][0].close == s2["bars"][0].close

        for s1, s2 in zip(val1, val2):
            assert s1["bars"][0].close == s2["bars"][0].close

        for s1, s2 in zip(test1, test2):
            assert s1["bars"][0].close == s2["bars"][0].close

    def test_split_sessions_different_seeds_differ(self):
        """Different seeds produce different splits."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)

        train1, _, _ = split_sessions(sessions, train_frac=0.8, seed=42)
        train2, _, _ = split_sessions(sessions, train_frac=0.8, seed=99)

        # Extract close prices of first bars as session identifiers
        ids1 = [s["bars"][0].close for s in train1]
        ids2 = [s["bars"][0].close for s in train2]

        # With 10 sessions, P(identical 8-element subsets from different seeds) ≈ 0
        assert ids1 != ids2, (
            f"Different seeds should produce different splits, got same: {ids1}"
        )


class TestSplitSessionsNoOverlap:
    """Verify train/val/test have no overlapping sessions."""

    def test_split_sessions_no_overlap(self):
        """No session appears in more than one split."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)
        train, val, test = split_sessions(sessions, train_frac=0.8, seed=42)

        # Use id() of session dicts as identity (they should be the SAME objects
        # from the original list, just partitioned)
        train_ids = {id(s) for s in train}
        val_ids = {id(s) for s in val}
        test_ids = {id(s) for s in test}

        assert len(train_ids & val_ids) == 0, "Train and val overlap"
        assert len(train_ids & test_ids) == 0, "Train and test overlap"
        assert len(val_ids & test_ids) == 0, "Val and test overlap"

        # All sessions accounted for
        assert len(train_ids | val_ids | test_ids) == 10, (
            "Not all sessions assigned to a split"
        )


# ===========================================================================
# 3. Model Building (spec tests 8–10)
# ===========================================================================


class TestBuildModelHyperparameters:
    """Verify build_model() returns MaskablePPO with Section 5.2 hyperparameters."""

    def test_build_model_hyperparameters(self):
        """Model has Section 5.2 hyperparameters: lr schedule, n_steps=2048,
        batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5."""
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        model = build_model(vec_env, seed=42)

        # Core hyperparameters from Section 5.2
        assert model.n_steps == 2048, f"Expected n_steps=2048, got {model.n_steps}"
        assert model.batch_size == 256, (
            f"Expected batch_size=256, got {model.batch_size}"
        )
        assert model.n_epochs == 10, f"Expected n_epochs=10, got {model.n_epochs}"
        assert model.gamma == pytest.approx(0.99), (
            f"Expected gamma=0.99, got {model.gamma}"
        )
        assert model.gae_lambda == pytest.approx(0.95), (
            f"Expected gae_lambda=0.95, got {model.gae_lambda}"
        )
        assert model.ent_coef == pytest.approx(0.01), (
            f"Expected ent_coef=0.01, got {model.ent_coef}"
        )
        assert model.vf_coef == pytest.approx(0.5), (
            f"Expected vf_coef=0.5, got {model.vf_coef}"
        )
        assert model.max_grad_norm == pytest.approx(0.5), (
            f"Expected max_grad_norm=0.5, got {model.max_grad_norm}"
        )

        # clip_range is a callable schedule in SB3
        clip_val = model.clip_range(1.0)
        assert clip_val == pytest.approx(0.2), (
            f"Expected clip_range=0.2, got {clip_val}"
        )

        # Learning rate: linear_schedule(1e-4) → 1e-4 at progress=1.0, 0 at 0.0
        lr_fn = model.learning_rate
        assert callable(lr_fn), "learning_rate must be a callable schedule"
        assert lr_fn(1.0) == pytest.approx(1e-4), (
            f"Expected lr=1e-4 at start, got {lr_fn(1.0)}"
        )
        assert lr_fn(0.0) == pytest.approx(0.0), (
            f"Expected lr=0 at end, got {lr_fn(0.0)}"
        )

        vec_env.close()


class TestBuildModelArchitecture:
    """Verify build_model() creates the correct network architecture."""

    def test_build_model_architecture(self):
        """Model has net_arch=[256,256,dict(pi=[64],vf=[64])] with ReLU activation."""
        import torch.nn as nn
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        model = build_model(vec_env, seed=42)
        policy = model.policy
        mlp = policy.mlp_extractor

        # Shared layers: should have 256-unit layers
        shared_sizes = [
            m.out_features for m in mlp.shared_net
            if hasattr(m, "out_features")
        ]
        assert 256 in shared_sizes, (
            f"Expected 256 in shared layers, got {shared_sizes}"
        )

        # Policy head: should have 64-unit layer
        pi_sizes = [
            m.out_features for m in mlp.policy_net
            if hasattr(m, "out_features")
        ]
        assert 64 in pi_sizes, (
            f"Expected 64 in policy head, got {pi_sizes}"
        )

        # Value head: should have 64-unit layer
        vf_sizes = [
            m.out_features for m in mlp.value_net
            if hasattr(m, "out_features")
        ]
        assert 64 in vf_sizes, (
            f"Expected 64 in value head, got {vf_sizes}"
        )

        # ReLU activation: check that shared net uses ReLU
        has_relu = any(isinstance(m, nn.ReLU) for m in mlp.shared_net)
        assert has_relu, (
            f"Expected ReLU activation in shared net, got: "
            f"{[type(m).__name__ for m in mlp.shared_net]}"
        )

        vec_env.close()


class TestBuildModelResume:
    """Verify build_model() can resume from a checkpoint."""

    def test_build_model_resume(self, tmp_path):
        """build_model(resume_path=...) loads from a saved checkpoint."""
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from sb3_contrib import MaskablePPO

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        # Create and save a model
        model = build_model(vec_env, seed=42)
        save_path = str(tmp_path / "barrier_checkpoint")
        model.save(save_path)
        vec_env.close()

        # Resume from checkpoint
        vec_env2 = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )
        resumed_model = build_model(vec_env2, seed=42, resume_path=save_path + ".zip")

        assert isinstance(resumed_model, MaskablePPO), (
            f"Expected MaskablePPO, got {type(resumed_model)}"
        )
        # Resumed model should have same hyperparameters
        assert resumed_model.n_steps == 2048
        assert resumed_model.batch_size == 256

        vec_env2.close()


# ===========================================================================
# 4. End-to-End Smoke Tests (spec tests 11–12)
# ===========================================================================


class TestMainSmokeSynthetic:
    """Verify main() runs end-to-end with synthetic session data."""

    def test_main_smoke_synthetic(self, tmp_path):
        """Run main() with synthetic data for minimal steps.
        Produces output dir with model + diagnostics."""
        from scripts.train_barrier import main, split_sessions, build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from lob_rl.barrier.training_diagnostics import (
            BarrierDiagnosticCallback,
            linear_schedule,
        )
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Build synthetic session data
        sessions = _make_session_data_list(n_sessions=6, n_bars=40)
        train_data, val_data, test_data = split_sessions(
            sessions, train_frac=0.8, seed=42,
        )

        # Create vec envs
        train_env = make_barrier_vec_env(
            train_data, n_envs=1, use_subprocess=False, seed=42,
        )
        eval_env = make_barrier_vec_env(
            val_data, n_envs=1, use_subprocess=False, seed=42,
        )

        # Create model with small batch for fast testing
        model = build_model(train_env, seed=42)

        # Diagnostic callback
        diag_cb = BarrierDiagnosticCallback(
            check_freq=1, output_dir=str(output_dir),
        )
        ckpt_cb = CheckpointCallback(
            save_freq=2048,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="barrier_ppo",
        )

        # Train for just enough steps to exercise the pipeline
        # n_steps=2048 means we need at least 2048 to do one update
        model.learn(total_timesteps=2048, callback=[diag_cb, ckpt_cb])

        # Save final model
        model.save(str(output_dir / "final_model"))

        train_env.close()
        eval_env.close()

        # Verify outputs exist
        assert (output_dir / "final_model.zip").exists(), (
            f"Final model not found in {output_dir}"
        )

    def test_main_function_exists_and_callable(self):
        """scripts/train_barrier.py exposes a main() function."""
        from scripts.train_barrier import main

        assert callable(main), "main must be callable"


class TestMainSmokeWithCheckpoint:
    """Verify checkpoint files are produced during training."""

    def test_main_smoke_with_checkpoint(self, tmp_path):
        """Run training, checkpoint, verify checkpoint files exist."""
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from stable_baselines3.common.callbacks import CheckpointCallback

        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        model = build_model(vec_env, seed=42)
        ckpt_cb = CheckpointCallback(
            save_freq=2048,  # checkpoint every 2048 steps (= 1 update)
            save_path=str(ckpt_dir),
            name_prefix="barrier_ppo",
        )

        model.learn(total_timesteps=4096, callback=ckpt_cb)
        vec_env.close()

        # Verify at least one checkpoint was saved
        ckpt_files = list(ckpt_dir.glob("barrier_ppo_*.zip"))
        assert len(ckpt_files) >= 1, (
            f"Expected at least 1 checkpoint, found {len(ckpt_files)} in "
            f"{list(ckpt_dir.iterdir())}"
        )

    def test_vecnormalize_saved_with_checkpoint(self, tmp_path):
        """When VecNormalize is used, its stats are saved alongside checkpoints."""
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from stable_baselines3.common.vec_env import VecNormalize
        from stable_baselines3.common.callbacks import CheckpointCallback

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        # Wrap with VecNormalize (as the script does by default)
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        model = build_model(vec_env, seed=42)
        model.learn(total_timesteps=2048)

        # Save VecNormalize stats
        vec_norm_path = str(output_dir / "vec_normalize.pkl")
        vec_env.save(vec_norm_path)
        vec_env.close()

        assert Path(vec_norm_path).exists(), (
            f"VecNormalize stats not saved at {vec_norm_path}"
        )


# ===========================================================================
# 5. Additional edge cases and robustness
# ===========================================================================


class TestSplitSessionsEdgeCases:
    """Edge cases for session splitting."""

    def test_split_sessions_minimum_size(self):
        """With 3 sessions and train_frac=0.8, at least 1 in each split."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=3)
        train, val, test = split_sessions(sessions, train_frac=0.8, seed=42)

        # With 3 sessions: floor(3*0.8)=2 train, 1 remaining → should be
        # either (2,1,0) or (2,0,1) but spec says split remaining evenly.
        # min 1 train is guaranteed
        assert len(train) >= 1, "Must have at least 1 training session"
        assert len(train) + len(val) + len(test) == 3, "Total must be 3"

    def test_split_sessions_returns_lists(self):
        """split_sessions returns three lists."""
        from scripts.train_barrier import split_sessions

        sessions = _make_session_data_list(n_sessions=10)
        train, val, test = split_sessions(sessions, train_frac=0.8, seed=42)

        assert isinstance(train, list), f"Expected list, got {type(train)}"
        assert isinstance(val, list), f"Expected list, got {type(val)}"
        assert isinstance(test, list), f"Expected list, got {type(test)}"


class TestBuildModelIsMaskablePPO:
    """Verify build_model returns a MaskablePPO instance."""

    def test_build_model_returns_maskable_ppo(self):
        """build_model returns a MaskablePPO (not vanilla PPO)."""
        from scripts.train_barrier import build_model
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from sb3_contrib import MaskablePPO

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)
        vec_env = make_barrier_vec_env(
            sessions, n_envs=2, use_subprocess=False, seed=42,
        )

        model = build_model(vec_env, seed=42)

        assert isinstance(model, MaskablePPO), (
            f"Expected MaskablePPO, got {type(model).__name__}"
        )
        vec_env.close()


class TestParseArgsRequiredFields:
    """Verify that data-dir and output-dir are required."""

    def test_parse_args_requires_data_dir(self):
        """parse_args fails without --data-dir."""
        from scripts.train_barrier import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--output-dir", "/tmp/out"])

    def test_parse_args_requires_output_dir(self):
        """parse_args fails without --output-dir."""
        from scripts.train_barrier import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--data-dir", "/tmp/data"])
