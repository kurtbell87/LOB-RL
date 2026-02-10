#!/usr/bin/env python3
"""Train barrier-hit PPO agent using MaskablePPO.

Wires together T9 modules (MultiSessionBarrierEnv, barrier_vec_env,
training_diagnostics) into an end-to-end CLI training pipeline.
"""

import argparse
import sys

import numpy as np
import torch.nn as nn

from sb3_contrib import MaskablePPO

from lob_rl.barrier.training_diagnostics import linear_schedule


def parse_args(argv=None):
    """Parse CLI arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train barrier-hit PPO agent using MaskablePPO.",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with .dbn.zst or precomputed session data",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to save model, logs, metrics",
    )
    parser.add_argument(
        "--bar-size", type=int, default=500,
        help="Trade bar size (default: 500)",
    )
    parser.add_argument(
        "--lookback", type=int, default=10,
        help="Feature lookback h (default: 10)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Parallel environments (default: 4)",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=2_000_000,
        help="Total training steps (default: 2_000_000)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=10_000,
        help="Steps between eval runs (default: 10_000)",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=50_000,
        help="Steps between checkpoints (default: 50_000)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of sessions for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--no-normalize", action="store_true", default=False,
        help="Skip VecNormalize wrapper",
    )
    return parser.parse_args(argv)


def split_sessions(sessions, train_frac, seed):
    """Split session data into train/val/test.

    Shuffles with seed, then splits: train_frac for train,
    remainder split evenly between val and test.

    Parameters
    ----------
    sessions : list[dict]
        Session data dicts with keys: bars, labels, features.
    train_frac : float
        Fraction of sessions for training (e.g. 0.8).
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    tuple[list, list, list]
        (train_data, val_data, test_data)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sessions))
    rng.shuffle(indices)

    n = len(sessions)
    n_train = int(n * train_frac)
    n_remaining = n - n_train
    n_val = n_remaining // 2
    n_test = n_remaining - n_val

    # Ensure at least 1 in each split when possible
    if n_train == 0 and n > 0:
        n_train = 1

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [sessions[i] for i in train_idx]
    val = [sessions[i] for i in val_idx]
    test = [sessions[i] for i in test_idx]

    return train, val, test


def build_model(env, seed, resume_path=None):
    """Create or load MaskablePPO model with Section 5.2 hyperparameters.

    Parameters
    ----------
    env : VecEnv
        SB3-compatible vectorized environment.
    seed : int
        Random seed.
    resume_path : str, optional
        Path to a saved checkpoint to resume from.

    Returns
    -------
    MaskablePPO
    """
    if resume_path is not None:
        return MaskablePPO.load(resume_path, env=env)

    policy_kwargs = dict(
        net_arch=[256, 256, dict(pi=[64], vf=[64])],
        activation_fn=nn.ReLU,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
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
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=0,
    )
    return model


def main():
    """Main training loop."""
    from pathlib import Path
    from lob_rl.barrier.barrier_vec_env import make_barrier_vec_env
    from lob_rl.barrier.training_diagnostics import BarrierDiagnosticCallback
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import VecNormalize

    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: load_session_data from data_dir
    # For now main() is a placeholder that will be filled when
    # .dbn.zst data loading is wired up
    raise NotImplementedError(
        "main() requires --data-dir with .dbn.zst files. "
        "Use build_model() and split_sessions() directly for programmatic use."
    )


if __name__ == "__main__":
    main()
