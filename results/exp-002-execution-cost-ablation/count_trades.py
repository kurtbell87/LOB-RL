"""Count mean trades per episode for each run."""
import os
import sys
import glob
import random
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build-release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from lob_rl.bar_level_env import BarLevelEnv

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'mes')
SEED = 42
BAR_SIZE = 1000
ACTION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


def load_split(train_days):
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    all_files = [(os.path.splitext(os.path.basename(f))[0], f, 0) for f in npz_files]
    random.Random(SEED).shuffle(all_files)
    train_files = all_files[:train_days]
    return train_files


def count_trades_per_episode(model, files, vec_norm_path, n_episodes=10):
    """Evaluate model and count position changes per episode."""
    trade_counts = []
    ep_lengths = []

    for date, path, _ in files[:n_episodes]:
        npz_file = os.path.join(CACHE_DIR, f"{date}.npz")
        env = BarLevelEnv.from_cache(npz_file, bar_size=BAR_SIZE, execution_cost=False)
        venv = DummyVecEnv([lambda: env])
        if vec_norm_path:
            venv = VecNormalize.load(vec_norm_path, venv)
            venv.training = False
            venv.norm_reward = False

        obs = venv.reset()
        done = False
        prev_position = 0.0
        trades = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            done = dones[0]
            steps += 1

            # Get the actual position from the action
            new_position = ACTION_MAP[int(action[0])]
            if new_position != prev_position:
                trades += 1
            prev_position = new_position

        trade_counts.append(trades)
        ep_lengths.append(steps)

    return {
        'mean_trades': float(np.mean(trade_counts)),
        'std_trades': float(np.std(trade_counts)),
        'mean_ep_length': float(np.mean(ep_lengths)),
        'trade_counts': trade_counts,
        'ep_lengths': ep_lengths,
    }


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    results = {}

    for run_name, train_days, label in [
        ('run-a', 20, 'Run A (20d exec-cost)'),
        ('run-b', 20, 'Run B (20d no-exec-cost)'),
        ('run-c', 199, 'Run C (199d no-exec-cost)'),
    ]:
        print(f"\n=== {label} ===")
        model = PPO.load(os.path.join(base, run_name, 'ppo_lob.zip'))
        train_files = load_split(train_days)
        vec_norm = os.path.join(base, run_name, 'vec_normalize.pkl')

        tc = count_trades_per_episode(model, train_files, vec_norm, n_episodes=10)
        print(f"  Mean trades/episode: {tc['mean_trades']:.1f} (std: {tc['std_trades']:.1f})")
        print(f"  Mean ep length: {tc['mean_ep_length']:.1f}")
        print(f"  Trade counts: {tc['trade_counts']}")
        results[run_name] = tc

    with open(os.path.join(base, 'trade_counts.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nTrade counts saved to trade_counts.json")
