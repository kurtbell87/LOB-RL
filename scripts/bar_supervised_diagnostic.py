"""Bar-level supervised diagnostic: does the 20-dim obs space contain signal?

Extends the tick-level supervised_diagnostic.py to support:
- Bar-level feature extraction at multiple bar sizes (200, 500, 1000)
- Multiple oracle definitions (no-exec-cost, with-exec-cost)
- Multiple seeds for model initialization
- Per-class precision/recall/F1
- Balanced accuracy
- Confusion matrices
- Per-feature permutation importance
- Cross-entropy loss on test set
- Overfit-64 capacity check

Usage:
    cd build-release && PYTHONPATH=.:../python uv run python \
        ../scripts/bar_supervised_diagnostic.py --cache-dir ../cache/mes/ \
        --train-days 199 --epochs 100 --bar-size 1000 --seeds 42,43,44

    # MVE mode (quick validation):
    cd build-release && PYTHONPATH=.:../python uv run python \
        ../scripts/bar_supervised_diagnostic.py --cache-dir ../cache/mes/ \
        --train-days 8 --test-days 2 --epochs 50 --bar-size 1000 --seeds 42
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


def load_bar_features(npz_path, bar_size):
    """Load cached .npz, aggregate into bars, compute temporal features.

    Returns (features_20dim, bar_mid_close, bar_spread_close) or None.
    features_20dim is (B, 20) = 13 intra-bar + 7 temporal (no position).
    """
    from lob_rl.bar_aggregation import aggregate_bars

    data = np.load(npz_path)
    obs = data["obs"]     # (N, 43)
    mid = data["mid"]     # (N,)
    spread = data["spread"]  # (N,)

    bar_features, bar_mid_close, bar_spread_close = aggregate_bars(
        obs, mid, spread, bar_size
    )

    num_bars = bar_features.shape[0]
    if num_bars < 2:
        return None

    # Compute 7 cross-bar temporal features (same as BarLevelEnv._precompute_temporal)
    temporal = np.zeros((num_bars, 7), dtype=np.float32)

    bar_return = bar_features[:, 0]
    imbalance_close = bar_features[:, 6]
    spread_close_feat = bar_features[:, 4]

    # 0: return_lag1
    if num_bars > 1:
        temporal[1:, 0] = bar_return[:-1]

    # 1: return_lag3
    if num_bars > 3:
        temporal[3:, 1] = bar_return[:-3]

    # 2: return_lag5
    if num_bars > 5:
        temporal[5:, 2] = bar_return[:-5]

    # 3: cumulative_return_5
    cumsum_ret = np.concatenate(([0.0], np.cumsum(bar_return)))
    for t in range(1, min(num_bars, 6)):
        temporal[t, 3] = cumsum_ret[t] - cumsum_ret[0]
    if num_bars > 5:
        temporal[5:, 3] = (cumsum_ret[5:num_bars] - cumsum_ret[:num_bars - 5]).astype(np.float32)

    # 4: rolling_vol_5
    cumsum_ret_sq = np.concatenate(([0.0], np.cumsum(bar_return.astype(np.float64) ** 2)))
    cumsum_ret_f64 = np.concatenate(([0.0], np.cumsum(bar_return.astype(np.float64))))
    for t in range(2, min(num_bars, 6)):
        w = t
        roll_sum = cumsum_ret_f64[t]
        roll_sum2 = cumsum_ret_sq[t]
        roll_mean = roll_sum / w
        roll_var = roll_sum2 / w - roll_mean ** 2
        temporal[t, 4] = np.sqrt(max(roll_var, 0.0))
    if num_bars > 5:
        w = 5
        roll_sum = cumsum_ret_f64[5:num_bars] - cumsum_ret_f64[:num_bars - 5]
        roll_sum2 = cumsum_ret_sq[5:num_bars] - cumsum_ret_sq[:num_bars - 5]
        roll_mean = roll_sum / w
        roll_var = roll_sum2 / w - roll_mean ** 2
        np.maximum(roll_var, 0.0, out=roll_var)
        temporal[5:, 4] = np.sqrt(roll_var).astype(np.float32)

    # 5: imb_delta_3
    if num_bars > 3:
        temporal[3:, 5] = imbalance_close[3:] - imbalance_close[:-3]

    # 6: spread_delta_3
    if num_bars > 3:
        temporal[3:, 6] = spread_close_feat[3:] - spread_close_feat[:-3]

    # Concatenate: 13 intra-bar + 7 temporal = 20 dims (no position)
    features = np.concatenate([bar_features, temporal], axis=1)  # (B, 20)

    return features, bar_mid_close, bar_spread_close


def make_oracle_labels(bar_mid_close, bar_spread_close, oracle_type):
    """Generate oracle labels from bar data.

    oracle_type:
        'no_exec_cost': short if mid_delta < 0, flat if == 0, long if > 0
        'with_exec_cost': short if -mid_delta > half_spread,
                          long if mid_delta > half_spread,
                          flat otherwise

    Returns labels (B-1,) for bars [0, B-2] predicting direction at bar [1, B-1].
    """
    mid_delta = np.diff(bar_mid_close)  # (B-1,)

    if oracle_type == 'no_exec_cost':
        labels = np.ones(len(mid_delta), dtype=np.int64)  # flat=1
        labels[mid_delta < 0] = 0   # short
        labels[mid_delta > 0] = 2   # long
    elif oracle_type == 'with_exec_cost':
        half_spread = bar_spread_close[:-1] / 2
        labels = np.ones(len(mid_delta), dtype=np.int64)  # flat=1
        labels[(-mid_delta) > half_spread] = 0   # short (profitable after cost)
        labels[mid_delta > half_spread] = 2      # long (profitable after cost)
    else:
        raise ValueError(f"Unknown oracle_type: {oracle_type}")

    return labels


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
    return float((pred == y[idx]).mean())


def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_dim,
                       activation='relu', epochs=100, batch_size=512,
                       lr=1e-3, device='cpu', seed=42):
    """Train MLP, return detailed metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.as_tensor(y_train, dtype=torch.long, device=device)

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = MLP(input_dim, hidden_dim, activation=activation).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        # Train predictions
        train_logits = model(X_tr)
        train_pred = train_logits.argmax(1).cpu().numpy()

        # Test predictions and loss
        X_te = torch.as_tensor(X_test, dtype=torch.float32, device=device)
        y_te_tensor = torch.as_tensor(y_test, dtype=torch.long, device=device)
        test_logits = model(X_te)
        test_pred = test_logits.argmax(1).cpu().numpy()
        test_loss = loss_fn(test_logits, y_te_tensor).item()

    train_acc = float((train_pred == y_train).mean())
    test_acc = float((test_pred == y_test).mean())

    # Per-class metrics
    class_names = ['short', 'flat', 'long']
    per_class = {}
    for c in range(3):
        tp = int(((test_pred == c) & (y_test == c)).sum())
        fp = int(((test_pred == c) & (y_test != c)).sum())
        fn = int(((test_pred != c) & (y_test == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[class_names[c]] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }

    # Balanced accuracy (macro-averaged recall)
    recalls = []
    for c in range(3):
        tp = int(((test_pred == c) & (y_test == c)).sum())
        fn = int(((test_pred != c) & (y_test == c)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    balanced_acc = float(np.mean(recalls))

    # Confusion matrix (rows=true, cols=pred)
    confusion = np.zeros((3, 3), dtype=int)
    for true_c in range(3):
        for pred_c in range(3):
            confusion[true_c, pred_c] = int(((y_test == true_c) & (test_pred == pred_c)).sum())

    # All 3 classes appear in predictions?
    classes_predicted = sorted(set(test_pred.tolist()))

    return {
        'train_acc': round(train_acc, 4),
        'test_acc': round(test_acc, 4),
        'balanced_acc': round(balanced_acc, 4),
        'test_loss': round(test_loss, 4),
        'per_class': per_class,
        'confusion_matrix': confusion.tolist(),
        'classes_predicted': classes_predicted,
        'model': model,  # return model for permutation importance
    }


def permutation_importance(model, X_test, y_test, n_repeats=5, device='cpu'):
    """Compute per-feature permutation importance on test set.

    Returns dict of feature_index -> mean_accuracy_drop.
    """
    model.eval()
    X_te = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_te = torch.as_tensor(y_test, dtype=torch.long, device=device)

    with torch.no_grad():
        base_pred = model(X_te).argmax(1).cpu().numpy()
    base_acc = float((base_pred == y_test).mean())

    importances = {}
    n_features = X_test.shape[1]

    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, feat_idx])
            X_perm_t = torch.as_tensor(X_perm, dtype=torch.float32, device=device)
            with torch.no_grad():
                perm_pred = model(X_perm_t).argmax(1).cpu().numpy()
            perm_acc = float((perm_pred == y_test).mean())
            drops.append(base_acc - perm_acc)
        importances[feat_idx] = round(float(np.mean(drops)), 6)

    return importances


def load_tick_features(npz_path, step_interval=1):
    """Load tick-level 53-dim features for upper bound test.

    Returns (features_53dim, mid, spread) or None.
    """
    from lob_rl.precomputed_env import PrecomputedEnv

    env = PrecomputedEnv.from_cache(npz_path, step_interval=step_interval)
    obs = env._obs           # (M, 43)
    temporal = env._temporal  # (M, 10)
    mid = env._mid
    spread = env._spread

    if len(mid) < 2:
        return None

    features = np.concatenate([obs, temporal], axis=1)  # (M, 53)
    return features, mid, spread


def main():
    parser = argparse.ArgumentParser(description='Bar-level supervised diagnostic')
    parser.add_argument('--cache-dir', required=True, help='Directory with cached .npz files')
    parser.add_argument('--train-days', type=int, default=199, help='Number of training days')
    parser.add_argument('--test-days', type=int, default=0,
                        help='Number of test days (0 = all remaining)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bar-size', type=int, default=0,
                        help='Bar size (0 = tick-level)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated seeds for model init')
    parser.add_argument('--oracle', type=str, default='both',
                        choices=['no_exec_cost', 'with_exec_cost', 'both'])
    parser.add_argument('--permutation-importance', action='store_true', default=False,
                        help='Compute per-feature permutation importance')
    parser.add_argument('--step-interval', type=int, default=10,
                        help='Step interval for tick-level (ignored for bar-level)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to write JSON results')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    oracles = (['no_exec_cost', 'with_exec_cost'] if args.oracle == 'both'
               else [args.oracle])

    device = args.device
    print(f"Device: {device}")
    print(f"Bar size: {args.bar_size} ({'tick-level' if args.bar_size == 0 else 'bar-level'})")
    print(f"Seeds: {seeds}")
    print(f"Oracles: {oracles}")
    print(f"Epochs: {args.epochs}")
    print()

    # Load file paths
    paths = sorted(glob.glob(os.path.join(args.cache_dir, '*.npz')))
    print(f"Found {len(paths)} cached days")

    # Split: first train_days for training, remaining for test
    train_paths = paths[:args.train_days]
    if args.test_days > 0:
        test_paths = paths[args.train_days:args.train_days + args.test_days]
    else:
        test_paths = paths[args.train_days:]
    print(f"Train days: {len(train_paths)}, Test days: {len(test_paths)}")

    t0 = time.time()

    if args.bar_size > 0:
        # BAR-LEVEL mode
        train_features_list = []
        train_mid_list = []
        train_spread_list = []
        train_day_sizes = []

        print(f"Loading train data (bar_size={args.bar_size})...")
        for p in train_paths:
            result = load_bar_features(p, args.bar_size)
            if result is None:
                continue
            features, bar_mid_close, bar_spread_close = result
            train_features_list.append(features)
            train_mid_list.append(bar_mid_close)
            train_spread_list.append(bar_spread_close)
            train_day_sizes.append(len(features))

        test_features_list = []
        test_mid_list = []
        test_spread_list = []
        test_day_sizes = []

        print(f"Loading test data (bar_size={args.bar_size})...")
        for p in test_paths:
            result = load_bar_features(p, args.bar_size)
            if result is None:
                continue
            features, bar_mid_close, bar_spread_close = result
            test_features_list.append(features)
            test_mid_list.append(bar_mid_close)
            test_spread_list.append(bar_spread_close)
            test_day_sizes.append(len(features))

        input_dim = 20
        feature_names = [
            'bar_return', 'bar_range', 'bar_volatility', 'spread_mean',
            'spread_close', 'imbalance_mean', 'imbalance_close',
            'bid_volume_mean', 'ask_volume_mean', 'volume_imbalance',
            'microprice_offset', 'time_remaining', 'n_ticks_norm',
            'return_lag1', 'return_lag3', 'return_lag5',
            'cumulative_return_5', 'rolling_vol_5', 'imb_delta_3', 'spread_delta_3',
        ]

    else:
        # TICK-LEVEL mode
        train_features_list = []
        train_mid_list = []
        train_spread_list = []
        train_day_sizes = []

        print(f"Loading train data (tick-level, step_interval={args.step_interval})...")
        for p in train_paths:
            result = load_tick_features(p, step_interval=args.step_interval)
            if result is None:
                continue
            features, mid, spread = result
            train_features_list.append(features)
            train_mid_list.append(mid)
            train_spread_list.append(spread)
            train_day_sizes.append(len(features))

        test_features_list = []
        test_mid_list = []
        test_spread_list = []
        test_day_sizes = []

        print(f"Loading test data (tick-level, step_interval={args.step_interval})...")
        for p in test_paths:
            result = load_tick_features(p, step_interval=args.step_interval)
            if result is None:
                continue
            features, mid, spread = result
            test_features_list.append(features)
            test_mid_list.append(mid)
            test_spread_list.append(spread)
            test_day_sizes.append(len(features))

        input_dim = 53
        feature_names = [f"cpp_feat_{i}" for i in range(43)] + [
            'mid_return_1', 'mid_return_5', 'mid_return_20', 'mid_return_50',
            'volatility_20', 'imb_delta_5', 'imb_delta_20',
            'microprice_offset', 'total_vol_imb', 'spread_change_5',
        ]

    print(f"Data loaded in {time.time() - t0:.1f}s")
    print(f"Train days loaded: {len(train_day_sizes)}, Test days loaded: {len(test_day_sizes)}")
    if train_day_sizes:
        print(f"Train bars/day: min={min(train_day_sizes)}, max={max(train_day_sizes)}, mean={np.mean(train_day_sizes):.0f}")
    if test_day_sizes:
        print(f"Test bars/day: min={min(test_day_sizes)}, max={max(test_day_sizes)}, mean={np.mean(test_day_sizes):.0f}")
    print()

    # Build per-oracle train/test datasets
    # For each oracle, we need to generate labels per-day and concatenate
    results = {}

    configs = [
        ("2x64_tanh", 64, 'tanh'),
        ("2x64_relu", 64, 'relu'),
        ("2x256_relu", 256, 'relu'),
        ("2x512_relu", 512, 'relu'),
    ]

    for oracle_type in oracles:
        print(f"\n{'='*70}")
        print(f"  Oracle: {oracle_type}")
        print(f"{'='*70}")

        # Build training data with labels
        X_train_parts = []
        y_train_parts = []
        for i, (features, mid_close, spread_close) in enumerate(
            zip(train_features_list, train_mid_list, train_spread_list)
        ):
            if args.bar_size > 0:
                labels = make_oracle_labels(mid_close, spread_close, oracle_type)
                # features[:-1] corresponds to labels (predicting next bar)
                X_train_parts.append(features[:-1])
                y_train_parts.append(labels)
            else:
                # Tick-level: use same logic as original supervised_diagnostic
                mid_delta = np.diff(mid_close)
                half_spread = spread_close[:-1] / 2
                if oracle_type == 'no_exec_cost':
                    labels = np.ones(len(mid_delta), dtype=np.int64)
                    labels[mid_delta < 0] = 0
                    labels[mid_delta > 0] = 2
                else:
                    labels = np.ones(len(mid_delta), dtype=np.int64)
                    labels[(-mid_delta) > half_spread] = 0
                    labels[mid_delta > half_spread] = 2
                X_train_parts.append(features[:-1])
                y_train_parts.append(labels)

        X_test_parts = []
        y_test_parts = []
        for i, (features, mid_close, spread_close) in enumerate(
            zip(test_features_list, test_mid_list, test_spread_list)
        ):
            if args.bar_size > 0:
                labels = make_oracle_labels(mid_close, spread_close, oracle_type)
                X_test_parts.append(features[:-1])
                y_test_parts.append(labels)
            else:
                mid_delta = np.diff(mid_close)
                half_spread = spread_close[:-1] / 2
                if oracle_type == 'no_exec_cost':
                    labels = np.ones(len(mid_delta), dtype=np.int64)
                    labels[mid_delta < 0] = 0
                    labels[mid_delta > 0] = 2
                else:
                    labels = np.ones(len(mid_delta), dtype=np.int64)
                    labels[(-mid_delta) > half_spread] = 0
                    labels[mid_delta > half_spread] = 2
                X_test_parts.append(features[:-1])
                y_test_parts.append(labels)

        if not X_train_parts or not X_test_parts:
            print("  ERROR: No data loaded!")
            continue

        X_train = np.concatenate(X_train_parts, axis=0).astype(np.float32)
        y_train = np.concatenate(y_train_parts, axis=0)
        X_test = np.concatenate(X_test_parts, axis=0).astype(np.float32)
        y_test = np.concatenate(y_test_parts, axis=0)

        print(f"  Train samples: {len(X_train):,}  Test samples: {len(X_test):,}")
        print(f"  Feature dims: {X_train.shape[1]}")

        # Check for NaN
        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
            print("  ABORT: NaN detected in features!")
            results[oracle_type] = {"error": "NaN in features"}
            continue

        # Label distributions
        train_counts = np.bincount(y_train, minlength=3)
        test_counts = np.bincount(y_test, minlength=3)
        train_pcts = train_counts / len(y_train) * 100
        test_pcts = test_counts / len(y_test) * 100

        print(f"  Train labels: short={train_pcts[0]:.1f}% flat={train_pcts[1]:.1f}% long={train_pcts[2]:.1f}%")
        print(f"  Test labels:  short={test_pcts[0]:.1f}% flat={test_pcts[1]:.1f}% long={test_pcts[2]:.1f}%")

        # Check abort: 100% one class
        if max(train_pcts) >= 99.9 or max(test_pcts) >= 99.9:
            print("  ABORT: Label distribution is degenerate (100% one class)!")
            results[oracle_type] = {"error": "degenerate label distribution"}
            continue

        # Majority baseline
        majority_class = train_counts.argmax()
        majority_train = float(train_counts[majority_class] / len(y_train))
        majority_test = float((y_test == majority_class).mean())
        class_names = ['short', 'flat', 'long']
        print(f"  Majority class: {class_names[majority_class]} (train={majority_train:.1%}, test={majority_test:.1%})")

        # Z-score normalize
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std < 1e-8] = 1.0
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # Small move stats (for no_exec_cost oracle)
        if args.bar_size > 0 and oracle_type == 'no_exec_cost':
            # Count flat labels as fraction
            flat_frac = float(test_counts[1] / len(y_test))
            print(f"  Flat fraction (test): {flat_frac:.1%}")

        oracle_results = {
            'label_distribution': {
                'train': {'short': round(float(train_pcts[0]), 2),
                          'flat': round(float(train_pcts[1]), 2),
                          'long': round(float(train_pcts[2]), 2)},
                'test': {'short': round(float(test_pcts[0]), 2),
                         'flat': round(float(test_pcts[1]), 2),
                         'long': round(float(test_pcts[2]), 2)},
            },
            'majority_class': class_names[majority_class],
            'majority_baseline_train': round(majority_train, 4),
            'majority_baseline_test': round(majority_test, 4),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
        }

        # Per seed, per architecture
        seed_results = {}
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            np.random.seed(seed)

            arch_results = {}
            for arch_name, hidden_dim, activation in configs:
                t1 = time.time()

                # Overfit check (only for 2x256 relu)
                overfit_acc = None
                if hidden_dim == 256 and activation == 'relu':
                    overfit_acc = overfit_small_batch(
                        X_train_norm, y_train, hidden_dim,
                        activation=activation, device=device
                    )
                    print(f"    {arch_name} overfit-64: {overfit_acc:.1%}")

                # Train and evaluate
                metrics = train_and_evaluate(
                    X_train_norm, y_train, X_test_norm, y_test,
                    hidden_dim, activation=activation,
                    epochs=args.epochs, device=device, seed=seed,
                )

                elapsed = time.time() - t1
                if elapsed > 600:
                    print(f"    ABORT: {arch_name} took {elapsed:.0f}s (>10 min)!")

                delta = metrics['test_acc'] - majority_test
                print(f"    {arch_name}: train={metrics['train_acc']:.1%} "
                      f"test={metrics['test_acc']:.1%} bal={metrics['balanced_acc']:.1%} "
                      f"delta={delta:+.1%} loss={metrics['test_loss']:.4f} "
                      f"({elapsed:.1f}s)")

                # Extract model and remove from stored results
                model_obj = metrics.pop('model')

                entry = {
                    **metrics,
                    'delta_over_baseline': round(delta, 4),
                }
                if overfit_acc is not None:
                    entry['overfit_64_acc'] = round(overfit_acc, 4)

                # Permutation importance for 2x256 relu + no_exec_cost only
                if (args.permutation_importance and
                    hidden_dim == 256 and activation == 'relu' and
                    oracle_type == 'no_exec_cost'):
                    print(f"    Computing permutation importance...")
                    t2 = time.time()
                    perm_imp = permutation_importance(
                        model_obj, X_test_norm, y_test,
                        n_repeats=5, device=device,
                    )
                    print(f"    Permutation importance done ({time.time() - t2:.1f}s)")
                    # Store with feature names
                    entry['permutation_importance'] = {
                        feature_names[k]: v for k, v in perm_imp.items()
                    }

                arch_results[arch_name] = entry

            seed_results[f"seed_{seed}"] = arch_results

        oracle_results['per_seed'] = seed_results

        # Compute cross-seed statistics for 2x256 relu
        test_accs_256 = []
        deltas_256 = []
        for seed_key, arch_res in seed_results.items():
            if '2x256_relu' in arch_res:
                test_accs_256.append(arch_res['2x256_relu']['test_acc'])
                deltas_256.append(arch_res['2x256_relu']['delta_over_baseline'])

        if len(test_accs_256) >= 2:
            mean_acc = float(np.mean(test_accs_256))
            std_acc = float(np.std(test_accs_256, ddof=1))
            mean_delta = float(np.mean(deltas_256))
            std_delta = float(np.std(deltas_256, ddof=1))

            # 95% CI: mean ± t(0.025, df=n-1) * std / sqrt(n)
            from scipy import stats as scipy_stats
            n = len(test_accs_256)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            ci_half = t_crit * std_delta / np.sqrt(n) if std_delta > 0 else 0
            ci_lower = mean_delta - ci_half
            ci_upper = mean_delta + ci_half
            ci_excludes_zero = ci_lower > 0

            oracle_results['cross_seed_2x256_relu'] = {
                'mean_test_acc': round(mean_acc, 4),
                'std_test_acc': round(std_acc, 4),
                'mean_delta': round(mean_delta, 4),
                'std_delta': round(std_delta, 4),
                'ci_95_lower': round(ci_lower, 4),
                'ci_95_upper': round(ci_upper, 4),
                'ci_excludes_zero': bool(ci_excludes_zero),
                'n_seeds': n,
            }
            print(f"\n  2x256 ReLU cross-seed: mean_acc={mean_acc:.1%} delta={mean_delta:+.4f} "
                  f"95%CI=[{ci_lower:.4f}, {ci_upper:.4f}] excludes_zero={ci_excludes_zero}")

        results[oracle_type] = oracle_results

    # Final output
    output = {
        'bar_size': args.bar_size,
        'mode': 'tick_level' if args.bar_size == 0 else 'bar_level',
        'input_dim': input_dim,
        'train_days': len(train_day_sizes),
        'test_days': len(test_day_sizes),
        'epochs': args.epochs,
        'seeds': seeds,
        'feature_names': feature_names,
        'results': results,
        'wall_clock_seconds': round(time.time() - t0, 1),
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {args.output}")

    # Print JSON summary to stdout
    print(f"\n{'='*70}")
    print("JSON_RESULTS_START")
    print(json.dumps(output, indent=2))
    print("JSON_RESULTS_END")

    return 0


if __name__ == '__main__':
    sys.exit(main())
