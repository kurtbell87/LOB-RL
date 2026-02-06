"""Train a PPO agent on LOB data using Stable-Baselines3."""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from lob_rl.gym_env import LOBGymEnv
from lob_rl.multi_day_env import MultiDayEnv

DEFAULT_SESSION_CONFIG = {
    'rth_open_ns': 48_600_000_000_000,   # 13:30 UTC
    'rth_close_ns': 72_000_000_000_000,  # 20:00 UTC
    'warmup_messages': -1,                # Use all pre-market data
}


def load_manifest(data_dir):
    """Load manifest and return sorted list of (date, file_path, record_count)."""
    manifest_path = os.path.join(data_dir, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)

    files = []
    for entry in manifest['files']:
        path = os.path.join(data_dir, f"{entry['date']}.bin")
        if os.path.exists(path):
            files.append((entry['date'], path, entry['record_count']))

    return sorted(files, key=lambda x: x[0])


def make_env(file_path, reward_mode='pnl_delta', lambda_=0.0, execution_cost=False,
             participation_bonus=0.0):
    """Create a LOBGymEnv for a single day's data."""
    return LOBGymEnv(
        file_path=file_path,
        session_config=DEFAULT_SESSION_CONFIG,
        steps_per_episode=0,  # Run until session close
        reward_mode=reward_mode,
        lambda_=lambda_,
        execution_cost=execution_cost,
        participation_bonus=participation_bonus,
    )


def make_train_env(file_paths, session_config, reward_mode, lambda_, execution_cost,
                   participation_bonus=0.0):
    """Factory that returns a closure for SubprocVecEnv (avoids lambda late-binding)."""
    def _init():
        return MultiDayEnv(
            file_paths=file_paths,
            session_config=session_config,
            steps_per_episode=0,
            reward_mode=reward_mode,
            lambda_=lambda_,
            shuffle=True,
            execution_cost=execution_cost,
            participation_bonus=participation_bonus,
        )
    return _init


def evaluate_sortino(model, eval_files, n_eval_episodes=10, execution_cost=False,
                     vec_normalize_path=None, participation_bonus=0.0):
    """Evaluate model on held-out data and compute Sortino ratio."""
    all_returns = []

    for date, path, _ in eval_files[:n_eval_episodes]:
        env = make_env(path, execution_cost=execution_cost,
                       participation_bonus=participation_bonus)
        venv = DummyVecEnv([lambda: env])

        if vec_normalize_path is not None:
            venv = VecNormalize.load(vec_normalize_path, venv)
            venv.training = False
            venv.norm_reward = False

        obs = venv.reset()

        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            episode_reward += reward[0]
            done = dones[0]

        all_returns.append(episode_reward)

    returns = np.array(all_returns)
    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        sortino = mean_return / downside_std if downside_std > 0 else np.inf
    else:
        sortino = np.inf if mean_return >= 0 else -np.inf

    return {
        'mean_return': float(mean_return),
        'std_return': float(np.std(returns)),
        'sortino_ratio': float(sortino),
        'n_episodes': len(returns),
        'positive_episodes': int(np.sum(returns > 0)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train PPO on LOB data')
    parser.add_argument('--data-dir', default='data/mes', help='Directory with .bin files')
    parser.add_argument('--train-days', type=int, default=20, help='Number of training days')
    parser.add_argument('--total-timesteps', type=int, default=500_000, help='Total training timesteps')
    parser.add_argument('--reward-mode', default='pnl_delta', choices=['pnl_delta', 'pnl_delta_penalized'])
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.0, help='Inventory penalty lambda')
    parser.add_argument('--execution-cost', action='store_true', default=False,
                        help='Enable execution cost (spread/2 per position change)')
    parser.add_argument('--output-dir', default='runs', help='Output directory for model and logs')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient for PPO')
    parser.add_argument('--n-envs', type=int, default=8, help='Number of parallel training environments')
    parser.add_argument('--batch-size', type=int, default=256, help='PPO minibatch size')
    parser.add_argument('--n-epochs', type=int, default=5, help='PPO epochs per rollout')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--no-norm', action='store_true', default=False,
                        help='Disable VecNormalize')
    parser.add_argument('--participation-bonus', type=float, default=0.0,
                        help='Per-step bonus for holding a position: bonus * |pos|')
    args = parser.parse_args()

    # Load data
    all_files = load_manifest(args.data_dir)
    print(f"Loaded {len(all_files)} days of data")

    # Split: first N days for training, rest for validation/test
    train_files = all_files[:args.train_days]
    val_files = all_files[args.train_days:args.train_days + 5]
    test_files = all_files[args.train_days + 5:]

    print(f"Train: {len(train_files)} days, Val: {len(val_files)} days, Test: {len(test_files)} days")

    if len(train_files) == 0:
        print("ERROR: No training files!")
        return 1

    # Create multi-day training environment cycling through all training days
    train_paths = [f[1] for f in train_files]
    print(f"Creating multi-day env with {len(train_paths)} training days (shuffle=True)")

    env = SubprocVecEnv([
        make_train_env(train_paths, DEFAULT_SESSION_CONFIG, args.reward_mode, args.lambda_, args.execution_cost,
                       args.participation_bonus)
        for _ in range(args.n_envs)
    ])

    if not args.no_norm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        tensorboard_log=os.path.join(args.output_dir, 'tb_logs'),
    )

    # Train
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'ppo_lob')
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save VecNormalize stats
    vec_normalize_path = None
    if not args.no_norm:
        vec_normalize_path = os.path.join(args.output_dir, 'vec_normalize.pkl')
        env.save(vec_normalize_path)
        print(f"VecNormalize stats saved to: {vec_normalize_path}")

    # Evaluate on validation set
    if val_files:
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_sortino(model, val_files, execution_cost=args.execution_cost, vec_normalize_path=vec_normalize_path, participation_bonus=args.participation_bonus)
        print(f"Validation metrics: {val_metrics}")

    # Evaluate on test set
    if test_files:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_sortino(model, test_files, execution_cost=args.execution_cost, vec_normalize_path=vec_normalize_path, participation_bonus=args.participation_bonus)
        print(f"Test metrics: {test_metrics}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
