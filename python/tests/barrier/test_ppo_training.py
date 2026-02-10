"""Tests for PPO training infrastructure: linear schedule, model creation,
training smoke tests, checkpointing, and integration.

Spec: docs/t9-ppo-training.md — Modules 2 & 4, plus integration tests.

Tests the training script's constituent components:
- linear_schedule() utility
- MaskablePPO model creation with correct hyperparameters
- Short training run completion
- Checkpoint save/resume
- End-to-end integration with synthetic data
- Action masking enforcement
- Eval callback
- Policy architecture verification
"""

import numpy as np
import pytest
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers — build synthetic session data for training tests
# ---------------------------------------------------------------------------

_H = 10
_OBS_DIM = 13 * _H + 2  # 132


def _make_session_data_list(n_sessions=5, n_bars=40, h=_H):
    """Build session data for training tests."""
    from lob_rl.barrier.feature_pipeline import build_feature_matrix
    from lob_rl.barrier.label_pipeline import compute_labels
    from ..barrier.conftest import make_session_bars

    sessions = []
    for i in range(n_sessions):
        bars = make_session_bars(n_bars, base_price=4000.0 + i * 10.0)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=h)
        sessions.append({"bars": bars, "labels": labels, "features": features})
    return sessions


def _make_vec_env(n_sessions=5, n_bars=40, n_envs=2):
    """Create a DummyVecEnv for training tests."""
    from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

    sessions = _make_session_data_list(n_sessions=n_sessions, n_bars=n_bars)
    return make_barrier_vec_env(
        sessions, n_envs=n_envs, use_subprocess=False, seed=42,
    )


# ===========================================================================
# 1. Linear schedule (spec test 29)
# ===========================================================================


class TestLinearSchedule:
    """Verify linear_schedule utility function."""

    def test_linear_schedule(self):
        """linear_schedule(1e-4) returns 1e-4 at progress=1.0, 0 at 0.0, 5e-5 at 0.5."""
        from lob_rl.barrier.training_diagnostics import linear_schedule

        schedule_fn = linear_schedule(1e-4)
        assert callable(schedule_fn), "linear_schedule must return a callable"

        # At start of training (progress_remaining=1.0)
        assert schedule_fn(1.0) == pytest.approx(1e-4), (
            f"Expected 1e-4 at progress=1.0, got {schedule_fn(1.0)}"
        )
        # At end of training (progress_remaining=0.0)
        assert schedule_fn(0.0) == pytest.approx(0.0), (
            f"Expected 0.0 at progress=0.0, got {schedule_fn(0.0)}"
        )
        # At midpoint (progress_remaining=0.5)
        assert schedule_fn(0.5) == pytest.approx(5e-5), (
            f"Expected 5e-5 at progress=0.5, got {schedule_fn(0.5)}"
        )

    def test_linear_schedule_different_initial(self):
        """linear_schedule works with different initial values."""
        from lob_rl.barrier.training_diagnostics import linear_schedule

        schedule_fn = linear_schedule(3e-3)
        assert schedule_fn(1.0) == pytest.approx(3e-3)
        assert schedule_fn(0.0) == pytest.approx(0.0)
        assert schedule_fn(0.25) == pytest.approx(7.5e-4)


# ===========================================================================
# 2. Model creation (spec test 30)
# ===========================================================================


class TestModelCreation:
    """Verify MaskablePPO model creation with correct hyperparameters."""

    def test_train_barrier_creates_model(self):
        """Training script can create a MaskablePPO model with correct hyperparameters."""
        from sb3_contrib import MaskablePPO
        from lob_rl.barrier.training_diagnostics import linear_schedule

        vec_env = _make_vec_env(n_sessions=3, n_bars=30, n_envs=2)

        policy_kwargs = dict(
            net_arch=[256, 256, dict(pi=[64], vf=[64])],
            activation_fn=nn.ReLU,
        )
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=linear_schedule(1e-4),
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=42,
            verbose=0,
            policy_kwargs=policy_kwargs,
        )
        assert model is not None
        assert model.n_steps == 2048
        assert model.batch_size == 256
        assert model.n_epochs == 10
        assert model.gamma == pytest.approx(0.99)
        assert model.gae_lambda == pytest.approx(0.95)
        assert model.clip_range(1.0) == pytest.approx(0.2)
        assert model.ent_coef == pytest.approx(0.01)
        vec_env.close()


# ===========================================================================
# 3. Model prediction with masks (spec test 31)
# ===========================================================================


class TestModelPrediction:
    """Verify model can predict with action masks."""

    def test_train_barrier_model_predicts_with_masks(self):
        """Model can predict actions given observations and action masks."""
        from sb3_contrib import MaskablePPO

        vec_env = _make_vec_env(n_sessions=3, n_bars=30, n_envs=2)
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )

        obs = vec_env.reset()
        # Flat position mask: [1, 1, 1, 0] for each env
        action_masks = np.array([[1, 1, 1, 0]] * vec_env.num_envs)
        actions, _states = model.predict(obs, action_masks=action_masks)

        assert actions.shape == (vec_env.num_envs,), (
            f"Expected actions shape ({vec_env.num_envs},), got {actions.shape}"
        )
        # All predicted actions should be valid (not ACTION_HOLD=3)
        for a in actions:
            assert a in (0, 1, 2), f"Action {a} is invalid when flat (hold masked)"
        vec_env.close()


# ===========================================================================
# 4. Short training run (spec test 32)
# ===========================================================================


class TestShortTrainingRun:
    """Verify training completes without error."""

    def test_train_barrier_short_training_run(self):
        """Training for 128 steps (small for speed) completes without error."""
        from sb3_contrib import MaskablePPO

        vec_env = _make_vec_env(n_sessions=3, n_bars=40, n_envs=2)
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        # Should not raise
        model.learn(total_timesteps=128)
        vec_env.close()


# ===========================================================================
# 5. Checkpoint save/resume (spec tests 33–34)
# ===========================================================================


class TestCheckpointing:
    """Verify checkpoint save and resume."""

    def test_train_barrier_checkpoint_saved(self, tmp_path):
        """After training with checkpoint callback, checkpoint files exist."""
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import CheckpointCallback

        vec_env = _make_vec_env(n_sessions=3, n_bars=40, n_envs=2)
        checkpoint_cb = CheckpointCallback(
            save_freq=64,  # Save every 64 steps
            save_path=str(tmp_path),
            name_prefix="barrier_ppo",
        )
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128, callback=checkpoint_cb)
        vec_env.close()

        # Check checkpoint files exist
        from pathlib import Path
        ckpt_files = list(Path(tmp_path).glob("barrier_ppo_*.zip"))
        assert len(ckpt_files) > 0, (
            f"No checkpoint files found in {tmp_path}: {list(Path(tmp_path).iterdir())}"
        )

    def test_train_barrier_resume_training(self, tmp_path):
        """Training can be resumed from a saved checkpoint."""
        from sb3_contrib import MaskablePPO

        vec_env = _make_vec_env(n_sessions=3, n_bars=40, n_envs=2)
        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128)

        # Save
        save_path = str(tmp_path / "barrier_ppo_checkpoint")
        model.save(save_path)
        vec_env.close()

        # Resume
        vec_env2 = _make_vec_env(n_sessions=3, n_bars=40, n_envs=2)
        loaded_model = MaskablePPO.load(save_path, env=vec_env2)
        # Continue training for another 128 steps — should not crash
        loaded_model.learn(total_timesteps=128, reset_num_timesteps=False)
        vec_env2.close()


# ===========================================================================
# 6. Integration: end-to-end synthetic training (spec test 35)
# ===========================================================================


class TestEndToEndIntegration:
    """Full pipeline: synthetic bars → multi-session env → train → evaluate."""

    def test_end_to_end_synthetic_training(self, tmp_path):
        """Full pipeline: generate synthetic bars → build multi-session env →
        train MaskablePPO for 256 steps → evaluate on held-out sessions → no errors."""
        from sb3_contrib import MaskablePPO
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
        from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback

        # Split sessions into train and eval
        all_sessions = _make_session_data_list(n_sessions=6, n_bars=40)
        train_sessions = all_sessions[:4]
        eval_sessions = all_sessions[4:]

        # Create train and eval vec envs
        train_env = make_barrier_vec_env(
            train_sessions, n_envs=2, use_subprocess=False, seed=42,
        )
        eval_env = make_barrier_vec_env(
            eval_sessions, n_envs=1, use_subprocess=False, seed=42,
        )

        # Train
        diag_cb = BarrierDiagnosticCallback(
            check_freq=1, output_dir=str(tmp_path),
        )
        model = MaskablePPO(
            "MlpPolicy", train_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128, callback=diag_cb)

        # Evaluate on held-out
        obs = eval_env.reset()
        total_reward = 0.0
        steps = 0
        while steps < 500:
            action_masks = np.array([[1, 1, 1, 0]])  # Flat position default
            actions, _ = model.predict(obs, action_masks=action_masks)
            obs, rewards, dones, infos = eval_env.step(actions)
            total_reward += rewards.sum()
            steps += 1
            if dones.any():
                break

        train_env.close()
        eval_env.close()

        # No crash is the primary assertion
        assert isinstance(total_reward, (float, np.floating))


# ===========================================================================
# 7. Action masking enforcement (spec test 36)
# ===========================================================================


class TestActionMaskingEnforcement:
    """Verify MaskablePPO respects action masks during training."""

    def test_action_masking_respected(self):
        """During training, MaskablePPO never selects masked-out actions.

        Uses a logging wrapper to track actions and verify against masks.
        """
        from sb3_contrib import MaskablePPO
        from lob_rl.barrier.multi_session_env import MultiSessionBarrierEnv

        import gymnasium

        sessions = _make_session_data_list(n_sessions=3, n_bars=40)

        # Wrapper that logs actions and their validity
        class MaskCheckWrapper(gymnasium.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.violations = []

            def step(self, action):
                mask = self.env.action_masks()
                if not mask[action]:
                    self.violations.append((action, mask.tolist()))
                return self.env.step(action)

            def action_masks(self):
                return self.env.action_masks()

        def make_checked_env():
            inner = MultiSessionBarrierEnv(sessions, seed=42)
            return MaskCheckWrapper(inner)

        from stable_baselines3.common.vec_env import DummyVecEnv
        env_fns = [make_checked_env for _ in range(2)]
        vec_env = DummyVecEnv(env_fns)

        model = MaskablePPO(
            "MlpPolicy", vec_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128)

        # Check for mask violations across all sub-envs
        for i, sub_env in enumerate(vec_env.envs):
            assert len(sub_env.violations) == 0, (
                f"Env {i} had {len(sub_env.violations)} mask violations: "
                f"{sub_env.violations[:5]}"
            )
        vec_env.close()


# ===========================================================================
# 8. Eval callback (spec test 37)
# ===========================================================================


class TestEvalCallback:
    """Verify eval callback works with barrier envs."""

    def test_eval_callback_runs(self, tmp_path):
        """Eval callback evaluates on held-out sessions and logs mean reward."""
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env

        all_sessions = _make_session_data_list(n_sessions=6, n_bars=40)
        train_sessions = all_sessions[:4]
        eval_sessions = all_sessions[4:]

        train_env = make_barrier_vec_env(
            train_sessions, n_envs=2, use_subprocess=False, seed=42,
        )
        eval_env = make_barrier_vec_env(
            eval_sessions, n_envs=1, use_subprocess=False, seed=42,
        )

        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(tmp_path),
            log_path=str(tmp_path),
            eval_freq=64,
            n_eval_episodes=2,
            verbose=0,
        )

        model = MaskablePPO(
            "MlpPolicy", train_env, n_steps=64, batch_size=32,
            n_epochs=2, seed=42, verbose=0,
        )
        model.learn(total_timesteps=128, callback=eval_cb)

        train_env.close()
        eval_env.close()

        # Check that eval logged results
        from pathlib import Path
        eval_log = Path(tmp_path) / "evaluations.npz"
        assert eval_log.exists(), f"Eval log not found at {eval_log}"


# ===========================================================================
# 9. Policy architecture (spec test 38)
# ===========================================================================


class TestPolicyArchitecture:
    """Verify model has the correct network architecture."""

    def test_policy_kwargs_architecture(self):
        """Model's policy network has shared [256,256], pi head [64], vf head [64]."""
        from sb3_contrib import MaskablePPO

        vec_env = _make_vec_env(n_sessions=3, n_bars=30, n_envs=2)
        policy_kwargs = dict(
            net_arch=[256, 256, dict(pi=[64], vf=[64])],
            activation_fn=nn.ReLU,
        )
        model = MaskablePPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
            n_steps=64, batch_size=32, seed=42, verbose=0,
        )

        # Check the policy network architecture
        policy = model.policy

        # The MLP extractor should have shared layers [256, 256]
        # and separate policy/value heads [64]
        mlp = policy.mlp_extractor

        # Shared network: should have 2 layers of size 256
        shared_layers = []
        for module in mlp.shared_net:
            if hasattr(module, "out_features"):
                shared_layers.append(module.out_features)
        assert 256 in shared_layers, (
            f"Expected 256 in shared layers, got {shared_layers}"
        )

        # Policy net: should have a layer of size 64
        pi_layers = []
        for module in mlp.policy_net:
            if hasattr(module, "out_features"):
                pi_layers.append(module.out_features)
        assert 64 in pi_layers, (
            f"Expected 64 in policy head, got {pi_layers}"
        )

        # Value net: should have a layer of size 64
        vf_layers = []
        for module in mlp.value_net:
            if hasattr(module, "out_features"):
                vf_layers.append(module.out_features)
        assert 64 in vf_layers, (
            f"Expected 64 in value head, got {vf_layers}"
        )

        vec_env.close()
