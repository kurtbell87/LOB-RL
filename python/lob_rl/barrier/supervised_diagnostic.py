"""Supervised diagnostic module for barrier label prediction.

Trains an MLP classifier and random forest baseline to predict barrier labels
H_k in {+1, -1, 0} from bar-level features. Validates that the observation
space contains learnable signal about barrier outcomes.
"""

import numpy as np
import torch
import torch.nn as nn

from lob_rl.barrier.feature_pipeline import (
    compute_bar_features,
    normalize_features,
    assemble_lookback,
)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_labeled_dataset(bars, labels, h=10):
    """Build feature matrix X and mapped label vector y from bars and labels.

    Uses the feature pipeline to compute 13 bar-level features, normalize,
    and assemble lookback windows. Aligns labels so y[i] corresponds to
    labels[i + h - 1].

    Label mapping: -1 -> 0, 0 -> 1, +1 -> 2.

    Parameters
    ----------
    bars : list[TradeBar]
    labels : list[BarrierLabel]
    h : int
        Lookback horizon (default 10).

    Returns
    -------
    X : np.ndarray of shape (n_usable, 13 * h), dtype float32
    y : np.ndarray of shape (n_usable,), dtype int64
    """
    n_labels = len(labels)

    if n_labels < h:
        return (
            np.empty((0, 13 * h), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # Compute raw features for all bars
    raw = compute_bar_features(bars)

    # We need features for the first n_labels bars (labels are 1:1 with bars)
    # Take only the first n_labels rows of raw features
    raw = raw[:n_labels]

    # Replace NaN with 0 before normalization (realized vol warmup)
    raw_filled = np.where(np.isnan(raw), 0.0, raw)

    # Normalize
    normed = normalize_features(raw_filled)

    # Assemble lookback: row i = normed[i:i+h].flatten()
    # This gives (n_labels - h + 1) rows
    X = assemble_lookback(normed, h=h)

    # Map labels: -1 -> 0, 0 -> 1, +1 -> 2
    # y[i] corresponds to labels[i + h - 1]
    label_map = {-1: 0, 0: 1, 1: 2}
    n_usable = n_labels - h + 1
    y = np.array(
        [label_map[labels[i + h - 1].label] for i in range(n_usable)],
        dtype=np.int64,
    )

    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class BarrierMLP(nn.Module):
    """Two-hidden-layer MLP for barrier label classification.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(hidden) -> ReLU -> Linear(n_classes)

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Hidden layer width (default 256).
    n_classes : int
        Number of output classes (default 3).
    """

    def __init__(self, input_dim, hidden_dim=256, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------

def _train_loop(X, y, epochs, batch_size=256, hidden_dim=256, lr=1e-3, seed=42):
    """Shared training loop for BarrierMLP.

    Seeds RNG, builds model, runs mini-batch SGD, evaluates on training data.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float32
    y : np.ndarray of shape (N,), dtype int64
    epochs : int
    batch_size : int
    hidden_dim : int
    lr : float
    seed : int

    Returns
    -------
    model : BarrierMLP
    accuracy : float
        Training accuracy after final epoch.
    final_loss : float
        Cross-entropy loss on full training set after final epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = X.shape[0]
    input_dim = X.shape[1]
    effective_batch = min(batch_size, n)

    model = BarrierMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, effective_batch):
            end = min(start + effective_batch, n)
            idx = perm[start:end]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(dim=1)
        accuracy = float((preds == y_t).float().mean().item())
        final_loss = float(criterion(logits, y_t).item())

    return model, accuracy, final_loss


# ---------------------------------------------------------------------------
# Overfit test
# ---------------------------------------------------------------------------

def overfit_test(X, y, epochs=500, batch_size=256, seed=42):
    """Train MLP on full dataset and check if it can memorize (>95% accuracy).

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float32
    y : np.ndarray of shape (N,), dtype int64
    epochs : int
    batch_size : int
    seed : int

    Returns
    -------
    dict with keys 'train_accuracy' (float) and 'passed' (bool).
    """
    _, accuracy, _ = _train_loop(
        X, y, epochs=epochs, batch_size=batch_size, seed=seed,
    )
    return {
        "train_accuracy": accuracy,
        "passed": accuracy > 0.95,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(X, y, epochs=100, batch_size=256, hidden_dim=256, lr=1e-3, seed=42):
    """Train a BarrierMLP and return the model and training metrics.

    Parameters
    ----------
    X : np.ndarray of shape (N, D), dtype float32
    y : np.ndarray of shape (N,), dtype int64
    epochs : int
    batch_size : int
    hidden_dim : int
    lr : float
    seed : int

    Returns
    -------
    model : BarrierMLP
    metrics : dict with keys 'train_accuracy' (float), 'train_loss' (float).
    """
    model, accuracy, final_loss = _train_loop(
        X, y, epochs=epochs, batch_size=batch_size,
        hidden_dim=hidden_dim, lr=lr, seed=seed,
    )
    return model, {
        "train_accuracy": accuracy,
        "train_loss": final_loss,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _predict(model, X):
    """Get predictions from either a PyTorch model or sklearn-style model."""
    if hasattr(model, 'predict'):
        # sklearn-style model
        return model.predict(X)
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X.astype(np.float32))
            logits = model(X_t)
            return logits.argmax(dim=1).numpy()


def evaluate_classifier(model, X_test, y_test):
    """Evaluate a classifier on test data.

    Works with both PyTorch (BarrierMLP) and sklearn models (duck typing via
    .predict()).

    Parameters
    ----------
    model : BarrierMLP or sklearn classifier
    X_test : np.ndarray of shape (N, D)
    y_test : np.ndarray of shape (N,)

    Returns
    -------
    dict with keys: accuracy, balanced_accuracy, majority_class,
        majority_baseline, beats_baseline, confusion_matrix, per_class.
    """
    preds = _predict(model, X_test)
    y_true = np.asarray(y_test)
    n = len(y_true)

    # Accuracy
    accuracy = float(np.mean(preds == y_true))

    # Majority class and baseline
    classes, counts = np.unique(y_true, return_counts=True)
    majority_idx = np.argmax(counts)
    majority_class = int(classes[majority_idx])
    majority_baseline = float(counts[majority_idx] / n)

    # Confusion matrix (3x3, always for classes 0, 1, 2)
    n_classes = 3
    cm = [[0] * n_classes for _ in range(n_classes)]
    for true_val, pred_val in zip(y_true, preds):
        t = int(true_val)
        p = int(pred_val)
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1

    # Per-class metrics
    per_class = {}
    recalls = []
    for cls in range(n_classes):
        tp = cm[cls][cls]
        # Precision: tp / sum of column cls
        col_sum = sum(cm[r][cls] for r in range(n_classes))
        precision = tp / col_sum if col_sum > 0 else 0.0

        # Recall: tp / sum of row cls
        row_sum = sum(cm[cls])
        recall = tp / row_sum if row_sum > 0 else 0.0

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        recalls.append(recall)

    balanced_accuracy = float(np.mean(recalls))
    beats_baseline = accuracy > majority_baseline

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "majority_class": majority_class,
        "majority_baseline": majority_baseline,
        "beats_baseline": beats_baseline,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Random forest baseline
# ---------------------------------------------------------------------------

def train_random_forest(X_train, y_train, X_test, y_test, seed=42, n_estimators=100):
    """Train a random forest classifier and evaluate on test data.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    seed : int
    n_estimators : int

    Returns
    -------
    dict with same keys as evaluate_classifier.
    """
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_train, y_train)

    return evaluate_classifier(rf, X_test, y_test)


# ---------------------------------------------------------------------------
# Full diagnostic pipeline
# ---------------------------------------------------------------------------

def run_diagnostic(bars, labels, h=10, train_frac=0.8, epochs=100, seed=42):
    """Run the full supervised diagnostic pipeline.

    1. Build labeled dataset from bars + labels.
    2. Split into train/test.
    3. Run overfit test on a small subset.
    4. Train MLP and evaluate.
    5. Train random forest and evaluate.

    Parameters
    ----------
    bars : list[TradeBar]
    labels : list[BarrierLabel]
    h : int
    train_frac : float
    epochs : int
    seed : int

    Returns
    -------
    dict with keys: n_samples, n_train, n_test, label_distribution,
        overfit_test, mlp, random_forest, passed.
    """
    X, y = build_labeled_dataset(bars, labels, h=h)
    n_samples = X.shape[0]

    # Label distribution (from original labels, mapped)
    label_counts = np.bincount(y, minlength=3)
    total = label_counts.sum()
    label_distribution = {
        "p_lower": float(label_counts[0] / total),   # class 0 = orig -1 = lower
        "p_timeout": float(label_counts[1] / total),  # class 1 = orig 0 = timeout
        "p_upper": float(label_counts[2] / total),    # class 2 = orig +1 = upper
    }

    # Train/test split
    n_train = int(n_samples * train_frac)
    n_test = n_samples - n_train

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Overfit test: use up to 256 samples from training set
    overfit_n = min(256, n_train)
    X_overfit = X_train[:overfit_n]
    y_overfit = y_train[:overfit_n]
    overfit_result = overfit_test(X_overfit, y_overfit, epochs=500, seed=seed)

    # Train MLP
    mlp_model, mlp_train_metrics = train_mlp(
        X_train, y_train, epochs=epochs, seed=seed,
    )
    mlp_eval = evaluate_classifier(mlp_model, X_test, y_test)

    # Random forest
    rf_eval = train_random_forest(X_train, y_train, X_test, y_test, seed=seed)

    # Passed flag
    passed = overfit_result["passed"] and mlp_eval["beats_baseline"]

    return {
        "n_samples": n_samples,
        "n_train": n_train,
        "n_test": n_test,
        "label_distribution": label_distribution,
        "overfit_test": overfit_result,
        "mlp": mlp_eval,
        "random_forest": rf_eval,
        "passed": passed,
    }
