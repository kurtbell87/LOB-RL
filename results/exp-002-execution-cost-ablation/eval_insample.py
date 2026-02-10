"""Evaluate models on training data (in-sample) to get in-sample returns."""
import os
import sys
import glob
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build-release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from train import evaluate_sortino

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'mes')
SEED = 42
BAR_SIZE = 1000


def load_split(train_days):
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    all_files = [(os.path.splitext(os.path.basename(f))[0], f, 0) for f in npz_files]
    random.Random(SEED).shuffle(all_files)
    train_files = all_files[:train_days]
    return train_files


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    results = {}

    # Run A: 20d, exec cost
    print("=== Run A (20d exec-cost) in-sample eval ===")
    model_a = PPO.load(os.path.join(base, 'run-a', 'ppo_lob.zip'))
    train_a = load_split(20)
    insample_a = evaluate_sortino(
        model_a, train_a, n_eval_episodes=10,
        execution_cost=True,
        vec_normalize_path=os.path.join(base, 'run-a', 'vec_normalize.pkl'),
        cache_path=CACHE_DIR, bar_size=BAR_SIZE,
    )
    print(f"  In-sample (exec cost): {insample_a}")
    results['run_a'] = insample_a

    # Run B: 20d, no exec cost
    print("=== Run B (20d no-exec-cost) in-sample eval ===")
    model_b = PPO.load(os.path.join(base, 'run-b', 'ppo_lob.zip'))
    train_b = load_split(20)
    insample_b = evaluate_sortino(
        model_b, train_b, n_eval_episodes=10,
        execution_cost=False,
        vec_normalize_path=os.path.join(base, 'run-b', 'vec_normalize.pkl'),
        cache_path=CACHE_DIR, bar_size=BAR_SIZE,
    )
    print(f"  In-sample (no exec cost): {insample_b}")
    results['run_b'] = insample_b

    # Run C: 199d, no exec cost — evaluate first 10 training days
    print("=== Run C (199d no-exec-cost) in-sample eval ===")
    model_c = PPO.load(os.path.join(base, 'run-c', 'ppo_lob.zip'))
    train_c = load_split(199)
    insample_c = evaluate_sortino(
        model_c, train_c, n_eval_episodes=10,
        execution_cost=False,
        vec_normalize_path=os.path.join(base, 'run-c', 'vec_normalize.pkl'),
        cache_path=CACHE_DIR, bar_size=BAR_SIZE,
    )
    print(f"  In-sample (no exec cost): {insample_c}")
    results['run_c'] = insample_c

    with open(os.path.join(base, 'insample_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nIn-sample results saved to insample_results.json")
