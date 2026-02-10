"""Train a PPO agent on LOB data using Stable-Baselines3."""

import argparse
import glob
import json
import os
import random
import sys
import warnings

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize

from lob_rl.config import load_json_config
from lob_rl.multi_day_env import MultiDayEnv
from lob_rl.orchestration import build_run_manifest, write_run_manifest
from lob_rl.precomputed_env import PrecomputedEnv
from lob_rl.reporting import METRICS_SCHEMA_VERSION, validate_metrics_payload

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
             participation_bonus=0.0, step_interval=1):
    """Create a PrecomputedEnv for a single day's data."""
    return PrecomputedEnv.from_file(
        file_path,
        session_config=DEFAULT_SESSION_CONFIG,
        reward_mode=reward_mode,
        lambda_=lambda_,
        execution_cost=execution_cost,
        participation_bonus=participation_bonus,
        step_interval=step_interval,
    )


def make_train_env(file_paths=None, session_config=None, reward_mode='pnl_delta',
                   lambda_=0.0, execution_cost=False, participation_bonus=0.0,
                   step_interval=1, cache_dir=None, bar_size=0, cache_files=None):
    """Factory that returns a closure for SubprocVecEnv (avoids lambda late-binding)."""
    # Build the data-source kwarg (exactly one of these will be non-None)
    if cache_files is not None:
        source_kwargs = dict(cache_files=cache_files)
    elif cache_dir is not None:
        source_kwargs = dict(cache_dir=cache_dir)
    else:
        source_kwargs = dict(file_paths=file_paths, session_config=session_config)

    def _init():
        return MultiDayEnv(
            **source_kwargs,
            steps_per_episode=0,
            reward_mode=reward_mode,
            lambda_=lambda_,
            shuffle=True,
            execution_cost=execution_cost,
            participation_bonus=participation_bonus,
            step_interval=step_interval,
            bar_size=bar_size,
        )
    return _init


def evaluate_sortino(model, eval_files, n_eval_episodes=10, execution_cost=False,
                     vec_normalize_path=None, participation_bonus=0.0, step_interval=1,
                     cache_path=None, bar_size=0, frame_stack=1, is_recurrent=False):
    """Evaluate model on held-out data and compute Sortino ratio."""
    all_returns = []

    for date, path, _ in eval_files[:n_eval_episodes]:
        try:
            if bar_size > 0:
                from lob_rl.bar_level_env import BarLevelEnv
                if cache_path is not None:
                    npz_file = os.path.join(cache_path, f"{date}.npz") if os.path.isdir(cache_path) else cache_path
                    env = BarLevelEnv.from_cache(
                        npz_file,
                        bar_size=bar_size,
                        execution_cost=execution_cost,
                        participation_bonus=participation_bonus,
                    )
                else:
                    env = BarLevelEnv.from_file(
                        path,
                        bar_size=bar_size,
                        execution_cost=execution_cost,
                        participation_bonus=participation_bonus,
                    )
            elif cache_path is not None:
                # Use from_cache for the specific .npz file
                npz_file = os.path.join(cache_path, f"{date}.npz") if os.path.isdir(cache_path) else cache_path
                env = PrecomputedEnv.from_cache(
                    npz_file,
                    execution_cost=execution_cost,
                    participation_bonus=participation_bonus,
                    step_interval=step_interval,
                )
            else:
                env = make_env(path, execution_cost=execution_cost,
                               participation_bonus=participation_bonus,
                               step_interval=step_interval)
        except ValueError as e:
            print(f"  Skipping {date}: {e}")
            continue
        venv = DummyVecEnv([lambda: env])

        if frame_stack > 1:
            venv = VecFrameStack(venv, n_stack=frame_stack)

        if vec_normalize_path is not None:
            venv = VecNormalize.load(vec_normalize_path, venv)
            venv.training = False
            venv.norm_reward = False

        obs = venv.reset()

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        episode_reward = 0
        done = False
        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(obs, deterministic=True,
                                                    state=lstm_states,
                                                    episode_start=episode_start)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            episode_reward += reward[0]
            done = dones[0]
            episode_start = dones

        all_returns.append(episode_reward)

    if not all_returns:
        return {'mean_return': 0.0, 'std_return': 0.0, 'sortino': 0.0,
                'downside_std': 0.0, 'n_episodes': 0, 'positive_episodes': 0}

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
        'sortino': float(sortino),
        'downside_std': float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0,
        'n_episodes': len(returns),
        'positive_episodes': int(np.sum(returns > 0)),
    }


def _build_parser(config):
    def cfg(name, default):
        return config.get(name, default)

    parser = argparse.ArgumentParser(description='Train PPO on LOB data')
    parser.add_argument('--config', default=None, help='Path to JSON run config')
    parser.add_argument('--data-dir', default=cfg('data_dir', None), help='Directory with .bin files')
    parser.add_argument('--cache-dir', default=cfg('cache_dir', None), help='Directory with cached .npz files')
    parser.add_argument('--train-days', type=int, default=cfg('train_days', 20), help='Number of training days')
    parser.add_argument('--val-days', type=int, default=cfg('val_days', 5), help='Number of validation days')
    parser.add_argument('--n-eval-episodes', type=int, default=cfg('n_eval_episodes', 10),
                        help='Max episodes to evaluate per split')
    parser.add_argument('--total-timesteps', type=int, default=cfg('total_timesteps', 500_000), help='Total training timesteps')
    parser.add_argument('--reward-mode', default=cfg('reward_mode', 'pnl_delta'), choices=['pnl_delta', 'pnl_delta_penalized'])
    parser.add_argument('--lambda', dest='lambda_', type=float, default=cfg('lambda_', 0.0), help='Inventory penalty lambda')
    parser.add_argument('--execution-cost', action='store_true', default=cfg('execution_cost', False),
                        help='Enable execution cost (spread/2 per position change)')
    parser.add_argument('--output-dir', default=cfg('output_dir', 'runs'), help='Output directory for model and logs')
    parser.add_argument('--ent-coef', type=float, default=cfg('ent_coef', 0.01), help='Entropy coefficient for PPO')
    parser.add_argument('--n-envs', type=int, default=cfg('n_envs', 8), help='Number of parallel training environments')
    parser.add_argument('--batch-size', type=int, default=cfg('batch_size', 256), help='PPO minibatch size')
    parser.add_argument('--n-epochs', type=int, default=cfg('n_epochs', 5), help='PPO epochs per rollout')
    parser.add_argument('--learning-rate', type=float, default=cfg('learning_rate', 3e-4), help='Learning rate')
    parser.add_argument('--no-norm', action='store_true', default=cfg('no_norm', False),
                        help='Disable VecNormalize')
    parser.add_argument('--participation-bonus', type=float, default=cfg('participation_bonus', 0.0),
                        help='Per-step bonus for holding a position: bonus * |pos|')
    parser.add_argument('--step-interval', type=int, default=cfg('step_interval', 1),
                        help='Subsample every Nth BBO snapshot (default: 1, no subsampling)')
    parser.add_argument('--bar-size', type=int, default=cfg('bar_size', 0),
                        help='Number of ticks per bar (0 = tick-level, >0 = bar-level)')
    parser.add_argument('--policy-arch', type=str, default=cfg('policy_arch', '64,64'),
                        help='Comma-separated hidden layer sizes (default: 64,64)')
    parser.add_argument('--activation', type=str, default=cfg('activation', 'tanh'), choices=['tanh', 'relu'],
                        help='Activation function for policy/value networks (default: tanh)')
    parser.add_argument('--shuffle-split', action='store_true', default=cfg('shuffle_split', False),
                        help='Shuffle files before train/val/test split (random instead of chronological)')
    parser.add_argument('--seed', type=int, default=cfg('seed', 42), help='Random seed for shuffle-split')
    parser.add_argument('--frame-stack', type=int, default=cfg('frame_stack', 1),
                        help='Number of frames to stack (1 = no stacking)')
    parser.add_argument('--recurrent', action='store_true', default=cfg('recurrent', False),
                        help='Use LSTM-based recurrent policy instead of MLP')
    parser.add_argument('--checkpoint-freq', type=int, default=cfg('checkpoint_freq', 0),
                        help='Save a checkpoint every N timesteps (0 = disabled)')
    parser.add_argument('--resume', type=str, default=cfg('resume', None),
                        help='Path to a checkpoint .zip file to resume training from')
    return parser


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config = load_json_config(pre_args.config)

    parser = _build_parser(config)
    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.recurrent and args.frame_stack > 1:
        parser.error("--recurrent and --frame-stack > 1 are mutually exclusive")
    if args.cache_dir and args.data_dir:
        parser.error("--cache-dir and --data-dir are mutually exclusive")
    if not args.cache_dir and not args.data_dir:
        parser.error("One of --cache-dir or --data-dir is required")

    # Warn if bar_size and step_interval are both set
    if args.bar_size > 0 and args.step_interval > 1:
        warnings.warn(
            f"step_interval={args.step_interval} is ignored when bar_size={args.bar_size} > 0"
        )

    # Parse policy architecture
    import torch
    net_arch = [int(x) for x in args.policy_arch.split(',')]
    activation_fn = torch.nn.Tanh if args.activation == 'tanh' else torch.nn.ReLU
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=activation_fn,
    )

    if args.cache_dir:
        # Load from cached .npz files
        npz_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
        all_files = [(os.path.splitext(os.path.basename(f))[0], f, 0) for f in npz_files]
        print(f"Loaded {len(all_files)} cached days from {args.cache_dir}")

        if args.shuffle_split:
            random.Random(args.seed).shuffle(all_files)

        train_files = all_files[:args.train_days]
        val_files = all_files[args.train_days:args.train_days + args.val_days]
        test_files = all_files[args.train_days + args.val_days:]

        print(f"Train: {len(train_files)} days, Val: {len(val_files)} days, Test: {len(test_files)} days")
        print(f"Train dates: {[f[0] for f in train_files]}")
        print(f"Val dates: {[f[0] for f in val_files]}")
        print(f"Test dates: {[f[0] for f in test_files]}")

        if len(train_files) == 0:
            print("ERROR: No training files!")
            return 1

        # Pass only train-split .npz paths to workers via cache_files
        train_npz_paths = [f[1] for f in train_files]

        env = SubprocVecEnv([
            make_train_env(
                cache_files=train_npz_paths,
                reward_mode=args.reward_mode,
                lambda_=args.lambda_,
                execution_cost=args.execution_cost,
                participation_bonus=args.participation_bonus,
                step_interval=args.step_interval,
                bar_size=args.bar_size,
            )
            for _ in range(args.n_envs)
        ])
    else:
        # Load data from manifest
        all_files = load_manifest(args.data_dir)
        print(f"Loaded {len(all_files)} days of data")

        if args.shuffle_split:
            random.Random(args.seed).shuffle(all_files)

        train_files = all_files[:args.train_days]
        val_files = all_files[args.train_days:args.train_days + args.val_days]
        test_files = all_files[args.train_days + args.val_days:]

        print(f"Train: {len(train_files)} days, Val: {len(val_files)} days, Test: {len(test_files)} days")
        print(f"Train dates: {[f[0] for f in train_files]}")
        print(f"Val dates: {[f[0] for f in val_files]}")
        print(f"Test dates: {[f[0] for f in test_files]}")

        if len(train_files) == 0:
            print("ERROR: No training files!")
            return 1

        train_paths = [f[1] for f in train_files]
        print(f"Creating multi-day env with {len(train_paths)} training days (shuffle=True)")

        env = SubprocVecEnv([
            make_train_env(train_paths, DEFAULT_SESSION_CONFIG, args.reward_mode, args.lambda_, args.execution_cost,
                           args.participation_bonus, step_interval=args.step_interval, bar_size=args.bar_size)
            for _ in range(args.n_envs)
        ])

    if args.frame_stack > 1:
        env = VecFrameStack(env, n_stack=args.frame_stack)

    if not args.no_norm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Resume: auto-load VecNormalize stats if available
    if args.resume and not args.no_norm:
        vecnormalize_path = args.resume.replace('.zip', '_vecnormalize.pkl')
        if os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)
            print(f"Loaded VecNormalize stats from: {vecnormalize_path}")
        else:
            print(f"Warning: VecNormalize file not found: {vecnormalize_path}, skipping")

    # Create or resume model
    if args.resume:
        from sb3_contrib import RecurrentPPO
        if args.recurrent:
            model = RecurrentPPO.load(args.resume, env=env)
        else:
            model = PPO.load(args.resume, env=env)
        print(f"Resumed model from: {args.resume}")
    elif args.recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            'MlpLstmPolicy',
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(args.output_dir, 'tb_logs'),
        )
    else:
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(args.output_dir, 'tb_logs'),
        )

    # Set up checkpoint callbacks
    callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        checkpoint_cb = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=checkpoint_dir,
        )
        callbacks = [checkpoint_cb]
        if not args.no_norm:
            class VecNormalizeSaveCallback(BaseCallback):
                def __init__(self, save_freq, save_path, env, verbose=0):
                    super().__init__(verbose)
                    self.save_freq = save_freq
                    self.save_path = save_path
                    self.env = env

                def _on_step(self):
                    if self.n_calls % self.save_freq == 0:
                        path = os.path.join(
                            self.save_path,
                            f"rl_model_{self.num_timesteps}_steps_vecnormalize.pkl",
                        )
                        self.env.save(path)
                    return True

            vec_norm_cb = VecNormalizeSaveCallback(
                save_freq=args.checkpoint_freq,
                save_path=checkpoint_dir,
                env=env,
            )
            callbacks.append(vec_norm_cb)
        callback = CallbackList(callbacks)

    # Train
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=False if args.resume else True,
    )

    # Save model + run manifest
    os.makedirs(args.output_dir, exist_ok=True)
    run_manifest = build_run_manifest(
        args=vars(args),
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        artifact_schema_version=METRICS_SCHEMA_VERSION,
    )
    manifest_path = write_run_manifest(run_manifest, args.output_dir)
    print(f"Run manifest written to: {manifest_path}")

    model_path = os.path.join(args.output_dir, 'ppo_lob')
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save VecNormalize stats
    vec_normalize_path = None
    if not args.no_norm:
        vec_normalize_path = os.path.join(args.output_dir, 'vec_normalize.pkl')
        env.save(vec_normalize_path)
        print(f"VecNormalize stats saved to: {vec_normalize_path}")

    val_metrics = None
    test_metrics = None

    # Evaluate on validation set
    cache_path = args.cache_dir if args.cache_dir else None
    if val_files:
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_sortino(model, val_files, n_eval_episodes=args.n_eval_episodes, execution_cost=args.execution_cost,
                                       vec_normalize_path=vec_normalize_path,
                                       participation_bonus=args.participation_bonus,
                                       step_interval=args.step_interval,
                                       cache_path=cache_path,
                                       bar_size=args.bar_size,
                                       frame_stack=args.frame_stack,
                                       is_recurrent=args.recurrent)
        print(f"Validation metrics: {val_metrics}")

    # Evaluate on test set
    if test_files:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_sortino(model, test_files, n_eval_episodes=args.n_eval_episodes, execution_cost=args.execution_cost,
                                        vec_normalize_path=vec_normalize_path,
                                        participation_bonus=args.participation_bonus,
                                        step_interval=args.step_interval,
                                        cache_path=cache_path,
                                        bar_size=args.bar_size,
                                        frame_stack=args.frame_stack,
                                        is_recurrent=args.recurrent)
        print(f"Test metrics: {test_metrics}")


    metrics_payload = {
        'schema_version': METRICS_SCHEMA_VERSION,
        'evaluation': {'n_eval_episodes': args.n_eval_episodes},
        'metrics': {},
    }
    if val_metrics is not None:
        metrics_payload['metrics']['validation'] = val_metrics
    if test_metrics is not None:
        metrics_payload['metrics']['test'] = test_metrics

    validation_errors = validate_metrics_payload(metrics_payload)
    if validation_errors:
        raise ValueError(f"Metrics payload validation failed: {validation_errors}")

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)
    print(f"Metrics written to: {metrics_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
