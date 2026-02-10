"""Retroactive OOS evaluation for exp-001 checkpoints.

Evaluates all 4 completed runs on val and test sets using evaluate_sortino().
Models are undertrained (199d runs have explained_variance ~0.30) but this
recovers the primary metrics missing from the RUN phase.
"""
import os
import sys
import glob
import random
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build-release'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from train import evaluate_sortino

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'mes')
SEED = 42
BAR_SIZE = 1000

RUNS = [
    {'name': 'run-a-mlp-20d',    'train_days': 20,  'recurrent': False, 'label': 'MLP 20d (control)'},
    {'name': 'run-b-lstm-20d',   'train_days': 20,  'recurrent': True,  'label': 'LSTM 20d (control)'},
    {'name': 'run-c-mlp-199d',   'train_days': 199, 'recurrent': False, 'label': 'MLP 199d (treatment)'},
    {'name': 'run-d-lstm-199d',  'train_days': 199, 'recurrent': True,  'label': 'LSTM 199d (treatment)'},
]


def load_split(train_days):
    """Reproduce the shuffle-split from train.py with seed 42."""
    npz_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    all_files = [(os.path.splitext(os.path.basename(f))[0], f, 0) for f in npz_files]
    random.Random(SEED).shuffle(all_files)
    train_files = all_files[:train_days]
    val_files = all_files[train_days:train_days + 5]
    test_files = all_files[train_days + 5:]
    return train_files, val_files, test_files


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    results = {}

    for run in RUNS:
        run_dir = os.path.join(base, run['name'])
        model_path = os.path.join(run_dir, 'ppo_lob.zip')
        vec_norm_path = os.path.join(run_dir, 'vec_normalize.pkl')

        if not os.path.exists(model_path):
            print(f"SKIP {run['name']}: no model checkpoint found")
            continue

        print(f"\n{'='*60}")
        print(f"  {run['label']}  ({run['name']})")
        print(f"{'='*60}")

        # Load model
        if run['recurrent']:
            model = RecurrentPPO.load(model_path)
        else:
            model = PPO.load(model_path)

        # Get val/test split
        _, val_files, test_files = load_split(run['train_days'])

        print(f"  Val files: {len(val_files)}, Test files: {len(test_files)}")

        # Eval on val set
        print(f"  Evaluating val set...")
        val_metrics = evaluate_sortino(
            model, val_files,
            n_eval_episodes=len(val_files),
            execution_cost=True,
            vec_normalize_path=vec_norm_path,
            cache_path=CACHE_DIR,
            bar_size=BAR_SIZE,
            is_recurrent=run['recurrent'],
        )
        print(f"  Val: mean={val_metrics['mean_return']:.2f}, std={val_metrics['std_return']:.2f}, "
              f"sortino={val_metrics['sortino_ratio']:.3f}, pos={val_metrics['positive_episodes']}/{val_metrics['n_episodes']}")

        # Eval on test set
        print(f"  Evaluating test set...")
        test_metrics = evaluate_sortino(
            model, test_files,
            n_eval_episodes=len(test_files),
            execution_cost=True,
            vec_normalize_path=vec_norm_path,
            cache_path=CACHE_DIR,
            bar_size=BAR_SIZE,
            is_recurrent=run['recurrent'],
        )
        print(f"  Test: mean={test_metrics['mean_return']:.2f}, std={test_metrics['std_return']:.2f}, "
              f"sortino={test_metrics['sortino_ratio']:.3f}, pos={test_metrics['positive_episodes']}/{test_metrics['n_episodes']}")

        results[run['name']] = {
            'label': run['label'],
            'train_days': run['train_days'],
            'recurrent': run['recurrent'],
            'val': val_metrics,
            'test': test_metrics,
        }

    # Save results
    out_path = os.path.join(base, 'oos_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<25} {'Val Return':>12} {'Test Return':>12} {'Val Sortino':>12} {'Test Sortino':>12} {'Val Pos':>8} {'Test Pos':>8}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
    for run in RUNS:
        if run['name'] not in results:
            continue
        r = results[run['name']]
        v, t = r['val'], r['test']
        print(f"{r['label']:<25} {v['mean_return']:>12.2f} {t['mean_return']:>12.2f} "
              f"{v['sortino_ratio']:>12.3f} {t['sortino_ratio']:>12.3f} "
              f"{v['positive_episodes']}/{v['n_episodes']:>5} {t['positive_episodes']}/{t['n_episodes']:>5}")
