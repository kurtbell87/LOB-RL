"""Supervised diagnostic: can an MLP predict optimal actions from LOB state?

Tests whether the policy network has enough capacity to represent the
optimal policy, independent of PPO training dynamics.

Usage:
    cd build-release && PYTHONPATH=.:../python uv run python \
        ../scripts/supervised_diagnostic.py --data-dir ../data/mes

What it does:
    1. Loads precomputed data, builds 53-dim features (43 C++ + 10 temporal)
    2. Computes oracle labels: best single-step action from flat position
    3. Trains MLPs of various sizes (2x64 Tanh, 2x64 ReLU, 2x256, 2x512)
    4. Reports: overfit accuracy (small batch), train acc, test acc
    5. Prints clear verdict on architecture viability
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lob_rl_core
from lob_rl.precomputed_env import PrecomputedEnv


def load_day_features(path, step_interval=1, use_cache=False):
    """Load features and compute oracle labels for one day.

    Returns (features, oracle_cost, oracle_nocost, mid_delta_stats) or None.
    """
    if use_cache:
        env = PrecomputedEnv.from_cache(path, step_interval=step_interval)
    else:
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs_raw, mid_raw, spread_raw, num_steps = lob_rl_core.precompute(path, cfg)
        if num_steps < 2:
            return None
        env = PrecomputedEnv(obs_raw, mid_raw, spread_raw, step_interval=step_interval)

    # Access internal arrays (53-dim features, no position)
    obs = env._obs           # (M, 43) float32
    temporal = env._temporal  # (M, 10) float32
    mid = env._mid           # (M,) float64
    spread = env._spread     # (M,) float64

    if len(mid) < 2:
        return None

    features = np.concatenate([obs, temporal], axis=1)  # (M, 53)

    # Oracle: best action from flat, one step ahead
    mid_delta = np.diff(mid)       # (M-1,)
    half_spread = spread[:-1] / 2  # cost of entering a position

    # With execution cost
    r_short = -mid_delta - half_spread
    r_flat = np.zeros_like(mid_delta)
    r_long = mid_delta - half_spread
    oracle_cost = np.argmax(
        np.column_stack([r_short, r_flat, r_long]), axis=1
    ).astype(np.int64)

    # Without execution cost
    oracle_nocost = np.argmax(
        np.column_stack([-mid_delta, np.zeros_like(mid_delta), mid_delta]), axis=1
    ).astype(np.int64)

    return features[:-1], oracle_cost, oracle_nocost


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=3, activation='relu'):
        super().__init__()
        act_cls = nn.ReLU if activation == 'relu' else nn.Tanh
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def overfit_small_batch(X, y, hidden_dim, activation='relu',
                        batch_size=64, epochs=500, lr=1e-3, device='cpu'):
    """Can the model memorize a small batch? Returns accuracy."""
    idx = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
    xb = torch.as_tensor(X[idx], dtype=torch.float32, device=device)
    yb = torch.as_tensor(y[idx], dtype=torch.long, device=device)

    model = MLP(X.shape[1], hidden_dim, activation=activation).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xb).argmax(1).cpu().numpy()
    return (pred == y[idx]).mean()


def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_dim,
                       activation='relu', epochs=100, batch_size=512,
                       lr=1e-3, device='cpu'):
    """Train MLP, return (train_acc, test_acc, final_loss)."""
    X_tr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.long, device=device)

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLP(X_train.shape[1], hidden_dim, activation=activation).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    model.eval()
    with torch.no_grad():
        train_pred = model(X_tr).argmax(1).cpu().numpy()
        X_te = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        test_pred = model(X_te).argmax(1).cpu().numpy()

    train_acc = (train_pred == y_train).mean()
    test_acc = (test_pred == y_test).mean()
    return train_acc, test_acc, final_loss


def main():
    parser = argparse.ArgumentParser(description='Supervised MLP diagnostic')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--step-interval', type=int, default=10)
    parser.add_argument('--train-days', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    # Resolve device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Load file paths
    if args.cache_dir:
        paths = sorted(glob.glob(os.path.join(args.cache_dir, '*.npz')))
        use_cache = True
    elif args.data_dir:
        manifest_path = os.path.join(args.data_dir, 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        paths = []
        for entry in sorted(manifest['files'], key=lambda x: x['date']):
            p = os.path.join(args.data_dir, f"{entry['date']}.bin")
            if os.path.exists(p):
                paths.append(p)
        use_cache = False
    else:
        parser.error("One of --data-dir or --cache-dir required")

    print(f"Loading {len(paths)} days (step_interval={args.step_interval})...")
    t0 = time.time()

    all_features, all_labels_cost, all_labels_nocost = [], [], []
    day_sizes = []

    for path in paths:
        result = load_day_features(
            path, step_interval=args.step_interval, use_cache=use_cache
        )
        if result is None:
            continue
        features, oracle_cost, oracle_nocost = result
        all_features.append(features)
        all_labels_cost.append(oracle_cost)
        all_labels_nocost.append(oracle_nocost)
        day_sizes.append(len(features))

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y_cost = np.concatenate(all_labels_cost, axis=0)
    y_nocost = np.concatenate(all_labels_nocost, axis=0)

    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Total samples: {len(X):,}  |  Feature dims: {X.shape[1]}")
    print(f"Days loaded: {len(day_sizes)}  |  Samples/day: "
          f"min={min(day_sizes):,} max={max(day_sizes):,} "
          f"mean={np.mean(day_sizes):,.0f}")
    print()

    # Label distributions
    print("Label distribution:")
    for name, y in [("  With exec cost", y_cost), ("  No exec cost ", y_nocost)]:
        counts = np.bincount(y, minlength=3)
        pcts = counts / len(y) * 100
        print(f"{name}:  short={pcts[0]:5.1f}%  flat={pcts[1]:5.1f}%  long={pcts[2]:5.1f}%")
    print()

    # Train/test split by day count
    split_idx = sum(day_sizes[:args.train_days])
    if split_idx == 0 or split_idx >= len(X):
        split_idx = len(X) * 3 // 4

    X_train, X_test = X[:split_idx], X[split_idx:]

    # Normalize (fit on train)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print()

    configs = [
        ("2x64  Tanh (SB3 dflt)", 64, 'tanh'),
        ("2x64  ReLU",            64, 'relu'),
        ("2x256 ReLU",           256, 'relu'),
        ("2x512 ReLU",           512, 'relu'),
    ]

    for label_name, y_all in [("WITH execution cost", y_cost),
                               ("NO execution cost", y_nocost)]:
        y_tr, y_te = y_all[:split_idx], y_all[split_idx:]
        majority = np.bincount(y_tr, minlength=3).argmax()
        baseline_train = (y_tr == majority).mean()
        baseline_test = (y_te == majority).mean()

        print("=" * 70)
        print(f"  {label_name}")
        label_names = ['short', 'flat', 'long']
        print(f"  Majority class: {label_names[majority]} "
              f"(train={baseline_train:.1%}, test={baseline_test:.1%})")
        print("=" * 70)
        print()
        print(f"  {'Config':<24} {'Overfit64':>9} {'Train':>9} {'Test':>9} "
              f"{'vs Base':>9} {'Loss':>9}")
        print(f"  {'-'*24} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

        for name, hidden, activation in configs:
            overfit = overfit_small_batch(
                X_train, y_tr, hidden, activation=activation, device=device
            )
            train_acc, test_acc, loss = train_and_evaluate(
                X_train, y_tr, X_test, y_te, hidden,
                activation=activation, epochs=args.epochs, device=device
            )
            delta = test_acc - baseline_test
            print(f"  {name:<24} {overfit:>9.1%} {train_acc:>9.1%} "
                  f"{test_acc:>9.1%} {delta:>+9.1%} {loss:>9.4f}")

        print()

    # Verdict
    print("=" * 70)
    print("  INTERPRETATION GUIDE")
    print("=" * 70)
    print()
    print("  Overfit < 90%  -> Architecture too small to represent the mapping")
    print("  Overfit ~100%, Train < 50%  -> Need more epochs or lower LR")
    print("  Train >> Test  -> Overfitting (features lack generalizable signal)")
    print("  Train ~ Test ~ Baseline  -> Features carry no predictive signal")
    print("  Test > Baseline  -> Signal exists; architecture can learn it")
    print()
    print("  If 'WITH exec cost' is mostly flat: that's expected — most moves")
    print("  are smaller than half-spread. The real question is whether the")
    print("  model can identify the RARE profitable moves (Test > Baseline).")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
