"""Cross-evaluate no-exec-cost models with execution_cost=True."""
import sys
import os
import glob
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build-release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from stable_baselines3 import PPO

# Re-use evaluate_sortino from train.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from train import evaluate_sortino

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'mes')
SEED = 42
BAR_SIZE = 1000


def load_split(train_days):
    """Load and split files exactly as train.py does."""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    all_files = [(os.path.splitext(os.path.basename(f))[0], f, 0) for f in npz_files]
    random.Random(SEED).shuffle(all_files)
    train_files = all_files[:train_days]
    val_files = all_files[train_days:train_days + 5]
    test_files = all_files[train_days + 5:]
    return train_files, val_files, test_files


def cross_eval(run_dir, train_days, label):
    """Evaluate a model with execution_cost=True (cross-eval)."""
    model_path = os.path.join(run_dir, 'ppo_lob.zip')
    vec_norm_path = os.path.join(run_dir, 'vec_normalize.pkl')

    model = PPO.load(model_path)
    _, val_files, test_files = load_split(train_days)

    print(f"\n=== Cross-eval {label} (with exec cost) ===")
    print(f"Model: {model_path}")
    print(f"Val files: {[f[0] for f in val_files]}")

    val_metrics = evaluate_sortino(
        model, val_files,
        execution_cost=True,
        vec_normalize_path=vec_norm_path,
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Val (with exec cost): {val_metrics}")

    test_metrics = evaluate_sortino(
        model, test_files,
        execution_cost=True,
        vec_normalize_path=vec_norm_path,
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Test (with exec cost): {test_metrics}")

    # Also re-eval without exec cost to confirm original results
    val_no_cost = evaluate_sortino(
        model, val_files,
        execution_cost=False,
        vec_normalize_path=vec_norm_path,
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Val (no exec cost, confirm): {val_no_cost}")

    test_no_cost = evaluate_sortino(
        model, test_files,
        execution_cost=False,
        vec_normalize_path=vec_norm_path,
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Test (no exec cost, confirm): {test_no_cost}")

    return {
        'val_with_exec_cost': val_metrics,
        'test_with_exec_cost': test_metrics,
        'val_no_exec_cost': val_no_cost,
        'test_no_exec_cost': test_no_cost,
    }


if __name__ == '__main__':
    base = os.path.dirname(__file__)

    results = {}

    # Cross-eval Run B (20 days, no exec cost)
    run_b_dir = os.path.join(base, 'run-b')
    results['run_b'] = cross_eval(run_b_dir, 20, 'Run B (20d no-exec-cost)')

    # Cross-eval Run C (199 days, no exec cost)
    run_c_dir = os.path.join(base, 'run-c')
    results['run_c'] = cross_eval(run_c_dir, 199, 'Run C (199d no-exec-cost)')

    # Also cross-eval Run A for completeness (eval without exec cost)
    run_a_dir = os.path.join(base, 'run-a')
    model_a = PPO.load(os.path.join(run_a_dir, 'ppo_lob.zip'))
    _, val_files_a, test_files_a = load_split(20)

    print(f"\n=== Cross-eval Run A (without exec cost) ===")
    val_a_no_cost = evaluate_sortino(
        model_a, val_files_a,
        execution_cost=False,
        vec_normalize_path=os.path.join(run_a_dir, 'vec_normalize.pkl'),
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Run A Val (no exec cost): {val_a_no_cost}")

    test_a_no_cost = evaluate_sortino(
        model_a, test_files_a,
        execution_cost=False,
        vec_normalize_path=os.path.join(run_a_dir, 'vec_normalize.pkl'),
        cache_path=CACHE_DIR,
        bar_size=BAR_SIZE,
    )
    print(f"Run A Test (no exec cost): {test_a_no_cost}")

    results['run_a_cross'] = {
        'val_no_exec_cost': val_a_no_cost,
        'test_no_exec_cost': test_a_no_cost,
    }

    # Save cross-eval results
    with open(os.path.join(base, 'cross_eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nCross-eval results saved to cross_eval_results.json")
