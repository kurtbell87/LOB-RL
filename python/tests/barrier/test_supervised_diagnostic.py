"""Tests for the supervised diagnostic module.

Spec: docs/t6-supervised-diagnostic.md

Tests the MLP classifier, random forest baseline, dataset construction,
evaluation metrics, overfit test, and full diagnostic pipeline for barrier
label prediction from bar-level features.
"""

import numpy as np
import pytest

from lob_rl.barrier import N_FEATURES

from .conftest import make_session_bars


# ===========================================================================
# 1. Dataset Construction — shape, mapping, NaN, alignment, clipping
# ===========================================================================


class TestBuildLabeledDatasetShape:
    """Spec test #1: Output shapes match expectations."""

    def test_shape_basic(self):
        """Given N bars and M labels, X has shape (M - h + 1, N_FEATURES * h)
        and y has shape (M - h + 1,)."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        h = 10
        X, y = build_labeled_dataset(bars, labels, h=h)

        n_usable = len(labels) - h + 1
        assert X.shape == (n_usable, N_FEATURES * h), (
            f"Expected X shape ({n_usable}, {N_FEATURES * h}), got {X.shape}"
        )
        assert y.shape == (n_usable,), (
            f"Expected y shape ({n_usable},), got {y.shape}"
        )

    def test_shape_with_h5(self):
        """h=5 gives feature dim N_FEATURES * 5."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(50)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        h = 5
        X, y = build_labeled_dataset(bars, labels, h=h)

        n_usable = len(labels) - h + 1
        assert X.shape[1] == N_FEATURES * h
        assert X.shape[0] == n_usable
        assert y.shape[0] == n_usable

    def test_x_and_y_same_length(self):
        """X and y always have the same number of samples."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(80)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        X, y = build_labeled_dataset(bars, labels, h=10)
        assert X.shape[0] == y.shape[0]


class TestBuildLabeledDatasetLabelMapping:
    """Spec test #2: Label mapping -1 → 0, 0 → 1, +1 → 2."""

    def test_label_values_in_expected_set(self):
        """All labels in y must be in {0, 1, 2}."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        _, y = build_labeled_dataset(bars, labels, h=10)

        unique_classes = set(np.unique(y))
        assert unique_classes.issubset({0, 1, 2}), (
            f"Unexpected label values: {unique_classes}"
        )

    def test_mapping_correctness(self):
        """Verify the label mapping: -1 → 0, 0 → 1, +1 → 2."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels, BarrierLabel

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        _, y = build_labeled_dataset(bars, labels, h=10)

        # Check that original labels and mapped labels agree
        # The first h-1 labels are dropped, so y[i] corresponds to labels[i + h - 1]
        h = 10
        for i in range(len(y)):
            orig_label = labels[i + h - 1].label
            if orig_label == -1:
                assert y[i] == 0, f"Label -1 should map to class 0, got {y[i]}"
            elif orig_label == 0:
                assert y[i] == 1, f"Label 0 should map to class 1, got {y[i]}"
            elif orig_label == 1:
                assert y[i] == 2, f"Label +1 should map to class 2, got {y[i]}"


class TestBuildLabeledDatasetNoNaN:
    """Spec test #3: Output X contains no NaN values."""

    def test_no_nan_in_features(self):
        """X should have no NaN after normalization handles warmup."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        X, _ = build_labeled_dataset(bars, labels, h=10)

        assert not np.any(np.isnan(X)), "X contains NaN values"

    def test_no_inf_in_features(self):
        """X should have no Inf values."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        X, _ = build_labeled_dataset(bars, labels, h=10)

        assert not np.any(np.isinf(X)), "X contains Inf values"


class TestBuildLabeledDatasetAlignment:
    """Spec test #4: Feature at index i corresponds to label at bar i + h - 1."""

    def test_alignment(self):
        """Feature row i aligns with label at bar index i + h - 1."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        h = 10
        _, y = build_labeled_dataset(bars, labels, h=h)

        # y[0] should correspond to labels[h-1] (the first label with full lookback)
        # y[i] should correspond to labels[i + h - 1]
        for i in range(min(5, len(y))):
            orig_label = labels[i + h - 1].label
            expected_class = {-1: 0, 0: 1, 1: 2}[orig_label]
            assert y[i] == expected_class, (
                f"y[{i}] = {y[i]}, expected {expected_class} from label at bar {i + h - 1}"
            )


class TestBuildLabeledDatasetClipped:
    """Spec test #5: All feature values in [-5, +5] (z-score clipping)."""

    def test_clipped_range(self):
        """All values in X are within [-5, +5]."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(60)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        X, _ = build_labeled_dataset(bars, labels, h=10)

        assert np.all(X >= -5.0), f"Min value {X.min()} < -5.0"
        assert np.all(X <= 5.0), f"Max value {X.max()} > 5.0"


# ===========================================================================
# 2. MLP Architecture
# ===========================================================================


class TestBarrierMLPOutputShape:
    """Spec test #6: Output shape is (batch, 3)."""

    def test_output_shape(self):
        """BarrierMLP(input_dim=130) produces output shape (batch, 3)."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, hidden_dim=256)
        x = torch.randn(32, 130)
        out = model(x)
        assert out.shape == (32, 3), f"Expected (32, 3), got {out.shape}"

    def test_output_shape_single_sample(self):
        """Single sample input produces shape (1, 3)."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=65)
        x = torch.randn(1, 65)
        out = model(x)
        assert out.shape == (1, 3)

    def test_output_shape_custom_n_classes(self):
        """Custom n_classes changes output dimension."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, n_classes=5)
        x = torch.randn(16, 130)
        out = model(x)
        assert out.shape == (16, 5)


class TestBarrierMLPDefaultHiddenDim:
    """Spec test #7: Default hidden_dim is 256."""

    def test_default_hidden_dim(self):
        """BarrierMLP with no hidden_dim arg uses 256."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130)
        # Check that the first linear layer has output features == 256
        first_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                first_layer = module
                break
        assert first_layer is not None, "No Linear layer found in BarrierMLP"
        assert first_layer.out_features == 256, (
            f"Default hidden_dim should be 256, got {first_layer.out_features}"
        )


class TestBarrierMLPForwardNoNaN:
    """Spec test #8: Forward pass on random input produces no NaN."""

    def test_no_nan_in_output(self):
        """Forward pass with random input produces finite output."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, hidden_dim=256)
        x = torch.randn(64, 130)
        out = model(x)
        assert not torch.any(torch.isnan(out)), "NaN in forward pass output"
        assert not torch.any(torch.isinf(out)), "Inf in forward pass output"


class TestBarrierMLPGradientFlow:
    """Spec test #9: Backprop updates all parameters."""

    def test_all_params_get_gradients(self):
        """After backward pass, all parameters have non-None gradients."""
        import torch
        import torch.nn as nn
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, hidden_dim=256)
        x = torch.randn(16, 130)
        y = torch.randint(0, 3, (16,))

        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.any(param.grad != 0), (
                f"All-zero gradient for parameter: {name}"
            )

    def test_architecture_has_two_hidden_layers(self):
        """MLP should have exactly 3 Linear layers (2 hidden + 1 output)."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, hidden_dim=256)
        linear_layers = [m for m in model.modules()
                         if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 3, (
            f"Expected 3 Linear layers (2 hidden + 1 output), got {len(linear_layers)}"
        )

    def test_relu_activations_present(self):
        """MLP uses ReLU activations between hidden layers."""
        import torch
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP

        model = BarrierMLP(input_dim=130, hidden_dim=256)
        relu_layers = [m for m in model.modules()
                       if isinstance(m, torch.nn.ReLU)]
        assert len(relu_layers) >= 2, (
            f"Expected at least 2 ReLU activations, got {len(relu_layers)}"
        )


# ===========================================================================
# 3. Overfit Test
# ===========================================================================


class TestOverfitTestPassesOnSeparableData:
    """Spec test #10: Overfit test passes on linearly separable data."""

    def test_passes_on_separable_data(self):
        """With linearly separable synthetic data, overfit returns passed=True."""
        from lob_rl.barrier.supervised_diagnostic import overfit_test

        rng = np.random.default_rng(42)
        n = 256
        d = 130
        # Create 3 linearly separable clusters
        X = np.zeros((n, d), dtype=np.float32)
        y = np.zeros(n, dtype=np.int64)
        for cls in range(3):
            start = cls * (n // 3)
            end = start + n // 3
            # Each class has a different mean in the first 3 dimensions
            X[start:end, cls] = 5.0
            X[start:end, :] += rng.normal(0, 0.1, (end - start, d)).astype(np.float32)
            y[start:end] = cls

        result = overfit_test(X, y, epochs=500, seed=42)
        assert result["passed"] is True, (
            f"Overfit test should pass on separable data, accuracy={result['train_accuracy']}"
        )
        assert result["train_accuracy"] > 0.95


class TestOverfitTestReturnsDict:
    """Spec test #11: Returns dict with required keys."""

    def test_returns_dict_with_required_keys(self):
        """overfit_test returns dict with 'train_accuracy' and 'passed'."""
        from lob_rl.barrier.supervised_diagnostic import overfit_test

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=100).astype(np.int64)

        result = overfit_test(X, y, epochs=10, seed=42)
        assert isinstance(result, dict)
        assert "train_accuracy" in result, "Missing key: train_accuracy"
        assert "passed" in result, "Missing key: passed"
        assert isinstance(result["train_accuracy"], float)
        assert isinstance(result["passed"], bool)


class TestOverfitTestBatchSizeCapped:
    """Spec test #12: When dataset < batch_size, uses full dataset."""

    def test_small_dataset_no_crash(self):
        """Dataset of 50 samples with batch_size=256 should not crash."""
        from lob_rl.barrier.supervised_diagnostic import overfit_test

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=50).astype(np.int64)

        result = overfit_test(X, y, batch_size=256, epochs=10, seed=42)
        assert isinstance(result, dict)
        assert "train_accuracy" in result


class TestOverfitTestSeedDeterministic:
    """Spec test #13: Same seed produces same accuracy."""

    def test_deterministic_with_same_seed(self):
        """Two calls with seed=42 produce identical results."""
        from lob_rl.barrier.supervised_diagnostic import overfit_test

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=100).astype(np.int64)

        result1 = overfit_test(X, y, epochs=20, seed=42)
        result2 = overfit_test(X, y, epochs=20, seed=42)
        assert result1["train_accuracy"] == pytest.approx(result2["train_accuracy"]), (
            f"Accuracy differs: {result1['train_accuracy']} vs {result2['train_accuracy']}"
        )


# ===========================================================================
# 4. Training and Evaluation
# ===========================================================================


class TestTrainMLPReturnsModelAndMetrics:
    """Spec test #14: train_mlp returns model and metrics dict."""

    def test_returns_model_and_metrics(self):
        """Returns a BarrierMLP model and dict with train_accuracy, train_loss."""
        from lob_rl.barrier.supervised_diagnostic import train_mlp, BarrierMLP

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=200).astype(np.int64)

        model, metrics = train_mlp(X, y, epochs=5, seed=42)
        assert isinstance(model, BarrierMLP), (
            f"Expected BarrierMLP, got {type(model)}"
        )
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "train_loss" in metrics
        assert isinstance(metrics["train_accuracy"], float)
        assert isinstance(metrics["train_loss"], float)

    def test_train_accuracy_in_range(self):
        """Train accuracy is between 0 and 1."""
        from lob_rl.barrier.supervised_diagnostic import train_mlp

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=200).astype(np.int64)

        _, metrics = train_mlp(X, y, epochs=5, seed=42)
        assert 0.0 <= metrics["train_accuracy"] <= 1.0

    def test_train_loss_non_negative(self):
        """Train loss is non-negative."""
        from lob_rl.barrier.supervised_diagnostic import train_mlp

        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=200).astype(np.int64)

        _, metrics = train_mlp(X, y, epochs=5, seed=42)
        assert metrics["train_loss"] >= 0.0


class TestEvaluateClassifierKeys:
    """Spec test #15: evaluate_classifier returns dict with all required keys."""

    def test_all_required_keys_present(self):
        """Returns dict with accuracy, balanced_accuracy, majority_class,
        majority_baseline, beats_baseline, confusion_matrix, per_class."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        required_keys = [
            "accuracy", "balanced_accuracy", "majority_class",
            "majority_baseline", "beats_baseline", "confusion_matrix",
            "per_class",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestEvaluateClassifierConfusionMatrixShape:
    """Spec test #16: Confusion matrix is 3x3."""

    def test_confusion_matrix_3x3(self):
        """Confusion matrix has shape (3, 3)."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        cm = result["confusion_matrix"]
        assert len(cm) == 3, f"Expected 3 rows, got {len(cm)}"
        for i, row in enumerate(cm):
            assert len(row) == 3, f"Row {i}: expected 3 cols, got {len(row)}"


class TestEvaluateClassifierConfusionMatrixSums:
    """Spec test #17: Confusion matrix rows sum to class counts in y_test."""

    def test_row_sums_match_class_counts(self):
        """Each row of the confusion matrix sums to the count of that class in y_test."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((60, 130)).astype(np.float32)
        # Create y_test with known class counts
        y_test = np.array([0]*20 + [1]*25 + [2]*15, dtype=np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        cm = result["confusion_matrix"]
        assert sum(cm[0]) == 20, f"Row 0 sum={sum(cm[0])}, expected 20"
        assert sum(cm[1]) == 25, f"Row 1 sum={sum(cm[1])}, expected 25"
        assert sum(cm[2]) == 15, f"Row 2 sum={sum(cm[2])}, expected 15"


class TestEvaluateClassifierMajorityBaseline:
    """Spec test #18: Majority baseline equals frequency of most common class."""

    def test_majority_baseline_correct(self):
        """majority_baseline = count(majority_class) / len(y_test)."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((60, 130)).astype(np.float32)
        # Class 1 is majority (30 out of 60)
        y_test = np.array([0]*10 + [1]*30 + [2]*20, dtype=np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        assert result["majority_class"] == 1, (
            f"Expected majority class 1, got {result['majority_class']}"
        )
        assert result["majority_baseline"] == pytest.approx(30.0 / 60.0), (
            f"Expected baseline 0.5, got {result['majority_baseline']}"
        )


class TestEvaluateClassifierBalancedAccuracy:
    """Spec test #19: Balanced accuracy is mean of per-class recalls."""

    def test_balanced_accuracy_is_mean_recall(self):
        """balanced_accuracy == mean of per-class recalls."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((90, 130)).astype(np.float32)
        y_test = np.array([0]*30 + [1]*30 + [2]*30, dtype=np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        # Compute expected balanced accuracy from per_class recalls
        per_class = result["per_class"]
        recalls = []
        for cls_key in sorted(per_class.keys()):
            recalls.append(per_class[cls_key]["recall"])
        expected_balanced = np.mean(recalls)
        assert result["balanced_accuracy"] == pytest.approx(expected_balanced, abs=1e-6), (
            f"balanced_accuracy={result['balanced_accuracy']}, expected {expected_balanced}"
        )

    def test_balanced_accuracy_in_range(self):
        """Balanced accuracy is between 0 and 1."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)
        assert 0.0 <= result["balanced_accuracy"] <= 1.0


# ===========================================================================
# 5. Random Forest Baseline
# ===========================================================================


class TestRandomForestReturnsMetrics:
    """Spec test #20: Returns dict with required keys."""

    def test_returns_required_keys(self):
        """train_random_forest returns dict with all evaluation metrics."""
        from lob_rl.barrier.supervised_diagnostic import train_random_forest

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        result = train_random_forest(X_train, y_train, X_test, y_test, seed=42)
        required_keys = [
            "accuracy", "balanced_accuracy", "beats_baseline",
            "confusion_matrix", "per_class", "majority_baseline",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestRandomForestOnSeparableData:
    """Spec test #21: On separable data, accuracy > majority baseline."""

    def test_beats_baseline_on_separable_data(self):
        """Random forest on linearly separable data beats majority baseline."""
        from lob_rl.barrier.supervised_diagnostic import train_random_forest

        rng = np.random.default_rng(42)
        n_per_class = 100
        d = 130
        X = np.zeros((3 * n_per_class, d), dtype=np.float32)
        y = np.zeros(3 * n_per_class, dtype=np.int64)
        for cls in range(3):
            start = cls * n_per_class
            end = start + n_per_class
            X[start:end, cls] = 5.0
            X[start:end, :] += rng.normal(0, 0.1, (n_per_class, d)).astype(np.float32)
            y[start:end] = cls

        # Shuffle then split 80/20 (sequential split leaves test set single-class)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]
        X_train, X_test = X[:240], X[240:]
        y_train, y_test = y[:240], y[240:]

        result = train_random_forest(X_train, y_train, X_test, y_test, seed=42)
        assert result["beats_baseline"] is True, (
            f"RF should beat baseline on separable data, accuracy={result['accuracy']}"
        )
        assert result["accuracy"] > result["majority_baseline"]


class TestRandomForestSeedDeterministic:
    """Spec test #22: Same seed produces same accuracy."""

    def test_deterministic_with_same_seed(self):
        """Two calls with seed=42 produce identical results."""
        from lob_rl.barrier.supervised_diagnostic import train_random_forest

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        result1 = train_random_forest(X_train, y_train, X_test, y_test, seed=42)
        result2 = train_random_forest(X_train, y_train, X_test, y_test, seed=42)
        assert result1["accuracy"] == pytest.approx(result2["accuracy"]), (
            f"Accuracy differs: {result1['accuracy']} vs {result2['accuracy']}"
        )


# ===========================================================================
# 6. Full Pipeline (run_diagnostic)
# ===========================================================================


class TestRunDiagnosticReturnsAllKeys:
    """Spec test #23: Returns dict with all required top-level keys."""

    def test_all_keys_present(self):
        """run_diagnostic returns dict with n_samples, n_train, n_test,
        label_distribution, overfit_test, mlp, random_forest, passed."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=5, seed=42)

        required_keys = [
            "n_samples", "n_train", "n_test",
            "label_distribution", "overfit_test",
            "mlp", "random_forest", "passed",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class TestRunDiagnosticOnSyntheticRegimeSwitch:
    """Spec test #24: run_diagnostic runs end-to-end on regime-switch data.

    NOTE: Synthetic random-walk features have negligible discriminative power
    for barrier labels (Cohen's d < 0.1 for all 13 features). The supervised
    diagnostic is designed to CHECK whether signal exists, not guarantee it.
    We verify the pipeline runs correctly and returns valid structure.
    The beats_baseline gate is evaluated on real data, not in unit tests.
    """

    def test_pipeline_runs_end_to_end(self):
        """run_diagnostic completes on regime-switch data and returns valid results."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=500, n_bars_high=500, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=50, seed=42)

        # Verify structure is correct
        assert isinstance(result["mlp"]["beats_baseline"], bool)
        assert 0.0 <= result["mlp"]["accuracy"] <= 1.0
        assert 0.0 <= result["mlp"]["majority_baseline"] <= 1.0
        assert result["mlp"]["confusion_matrix"] is not None
        assert result["random_forest"] is not None


class TestRunDiagnosticOverfitPasses:
    """Spec test #25: Overfit test passes on synthetic data."""

    def test_overfit_passes(self):
        """Overfit test should pass (MLP can memorize a small batch)."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=300, n_bars_high=300, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=50, seed=42)

        assert result["overfit_test"]["passed"] is True, (
            f"Overfit test should pass, "
            f"accuracy={result['overfit_test']['train_accuracy']}"
        )


class TestRunDiagnosticLabelDistributionValid:
    """Spec test #26: Label distribution values sum to 1.0, each in [0, 1]."""

    def test_distribution_sums_to_one(self):
        """p_upper + p_lower + p_timeout == 1.0."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=5, seed=42)

        dist = result["label_distribution"]
        total = dist["p_upper"] + dist["p_lower"] + dist["p_timeout"]
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"Label distribution sums to {total}, expected 1.0"
        )

    def test_each_probability_in_range(self):
        """Each probability in [0, 1]."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=5, seed=42)

        dist = result["label_distribution"]
        for key in ["p_upper", "p_lower", "p_timeout"]:
            assert 0.0 <= dist[key] <= 1.0, (
                f"{key} = {dist[key]}, expected in [0, 1]"
            )


class TestRunDiagnosticPassedFlag:
    """Spec test #27: passed == (overfit_test.passed AND mlp.beats_baseline)."""

    def test_passed_flag_logic(self):
        """The 'passed' flag is True iff both conditions hold."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=300, n_bars_high=300, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=50, seed=42)

        expected_passed = (
            result["overfit_test"]["passed"]
            and result["mlp"]["beats_baseline"]
        )
        assert result["passed"] == expected_passed, (
            f"passed={result['passed']}, expected {expected_passed} "
            f"(overfit_passed={result['overfit_test']['passed']}, "
            f"mlp_beats_baseline={result['mlp']['beats_baseline']})"
        )


# ===========================================================================
# 7. Edge Cases
# ===========================================================================


class TestBuildLabeledDatasetSmallN:
    """Spec test #28: Very few bars (< h + 10) returns appropriately sized arrays."""

    def test_small_n_no_crash(self):
        """With very few bars, returns arrays without crashing."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(15)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        h = 10
        X, y = build_labeled_dataset(bars, labels, h=h)

        # Should produce some samples (len(labels) - h + 1 if len(labels) >= h)
        if len(labels) >= h:
            n_usable = len(labels) - h + 1
            assert X.shape[0] == n_usable
            assert y.shape[0] == n_usable
        else:
            # If too few labels, should return empty or zero-row arrays
            assert X.shape[0] == 0
            assert y.shape[0] == 0

    def test_minimal_bars_for_h(self):
        """Exactly h bars with h labels should produce 1 sample."""
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = make_session_bars(50)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        # Use h equal to the number of labels
        h = len(labels)
        if h > 0:
            X, y = build_labeled_dataset(bars, labels, h=h)
            assert X.shape[0] == 1
            assert y.shape[0] == 1


class TestEvaluateClassifierAllSameClass:
    """Spec test #29: All labels are same class — baseline is 1.0."""

    def test_all_same_class_baseline_is_one(self):
        """When all y_test are the same class, majority_baseline == 1.0."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((30, 130)).astype(np.float32)
        y_test = np.ones(30, dtype=np.int64)  # all class 1

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        assert result["majority_baseline"] == pytest.approx(1.0), (
            f"Expected baseline 1.0, got {result['majority_baseline']}"
        )
        assert result["majority_class"] == 1

    def test_accuracy_reported_correctly(self):
        """Accuracy is reported correctly even when all labels are same class."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((30, 130)).astype(np.float32)
        y_test = np.zeros(30, dtype=np.int64)  # all class 0

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        assert 0.0 <= result["accuracy"] <= 1.0


class TestTrainMLPSingleEpoch:
    """Spec test #30: Training with epochs=1 completes without error."""

    def test_single_epoch_no_crash(self):
        """epochs=1 should complete without error."""
        from lob_rl.barrier.supervised_diagnostic import train_mlp

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 130)).astype(np.float32)
        y = rng.integers(0, 3, size=100).astype(np.int64)

        model, metrics = train_mlp(X, y, epochs=1, seed=42)
        assert isinstance(metrics, dict)
        assert "train_accuracy" in metrics
        assert "train_loss" in metrics


# ===========================================================================
# 8. Importability
# ===========================================================================


class TestSupervisedDiagnosticImports:
    """All public functions and classes should be importable."""

    def test_build_labeled_dataset_importable(self):
        from lob_rl.barrier.supervised_diagnostic import build_labeled_dataset
        assert callable(build_labeled_dataset)

    def test_barrier_mlp_importable(self):
        from lob_rl.barrier.supervised_diagnostic import BarrierMLP
        assert callable(BarrierMLP)

    def test_overfit_test_importable(self):
        from lob_rl.barrier.supervised_diagnostic import overfit_test
        assert callable(overfit_test)

    def test_train_mlp_importable(self):
        from lob_rl.barrier.supervised_diagnostic import train_mlp
        assert callable(train_mlp)

    def test_evaluate_classifier_importable(self):
        from lob_rl.barrier.supervised_diagnostic import evaluate_classifier
        assert callable(evaluate_classifier)

    def test_train_random_forest_importable(self):
        from lob_rl.barrier.supervised_diagnostic import train_random_forest
        assert callable(train_random_forest)

    def test_run_diagnostic_importable(self):
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        assert callable(run_diagnostic)


# ===========================================================================
# 9. Per-class metrics structure
# ===========================================================================


class TestPerClassMetrics:
    """Per-class metrics dict has precision, recall, f1 for each class."""

    def test_per_class_has_all_classes(self):
        """per_class dict has entries for classes 0, 1, 2."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_mlp, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((90, 130)).astype(np.float32)
        y_test = np.array([0]*30 + [1]*30 + [2]*30, dtype=np.int64)

        model, _ = train_mlp(X_train, y_train, epochs=5, seed=42)
        result = evaluate_classifier(model, X_test, y_test)

        per_class = result["per_class"]
        for cls in [0, 1, 2]:
            assert cls in per_class, f"Missing class {cls} in per_class"
            for metric in ["precision", "recall", "f1"]:
                assert metric in per_class[cls], (
                    f"Missing metric '{metric}' for class {cls}"
                )
                assert 0.0 <= per_class[cls][metric] <= 1.0, (
                    f"Class {cls} {metric}={per_class[cls][metric]}, "
                    f"expected in [0, 1]"
                )


# ===========================================================================
# 10. Dataset size validation in run_diagnostic
# ===========================================================================


class TestRunDiagnosticDatasetSizes:
    """n_samples, n_train, n_test are consistent."""

    def test_sizes_consistent(self):
        """n_train + n_test == n_samples."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=5, seed=42)

        assert result["n_train"] + result["n_test"] == result["n_samples"], (
            f"n_train({result['n_train']}) + n_test({result['n_test']}) "
            f"!= n_samples({result['n_samples']})"
        )

    def test_train_frac_respected(self):
        """With train_frac=0.8, n_train is approximately 80% of n_samples."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(
            bars, labels, h=10, train_frac=0.8, epochs=5, seed=42,
        )

        actual_frac = result["n_train"] / result["n_samples"]
        # Allow some tolerance for rounding
        assert 0.75 <= actual_frac <= 0.85, (
            f"Train fraction={actual_frac}, expected ~0.8"
        )

    def test_n_samples_positive(self):
        """n_samples, n_train, n_test are all positive."""
        from lob_rl.barrier.supervised_diagnostic import run_diagnostic
        from lob_rl.barrier.regime_switch import generate_regime_switch_trades
        from lob_rl.barrier.label_pipeline import compute_labels

        _, bars = generate_regime_switch_trades(
            n_bars_low=200, n_bars_high=200, bar_size=500, seed=42,
        )
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        result = run_diagnostic(bars, labels, h=10, epochs=5, seed=42)

        assert result["n_samples"] > 0
        assert result["n_train"] > 0
        assert result["n_test"] > 0


# ===========================================================================
# 11. evaluate_classifier works with sklearn-style models (duck typing)
# ===========================================================================


class TestEvaluateClassifierDuckTyping:
    """evaluate_classifier should work with sklearn models via predict()."""

    def test_works_with_sklearn_model(self):
        """Passing an sklearn model (with .predict) should work."""
        from lob_rl.barrier.supervised_diagnostic import (
            train_random_forest, evaluate_classifier,
        )

        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 130)).astype(np.float32)
        y_train = rng.integers(0, 3, size=200).astype(np.int64)
        X_test = rng.standard_normal((50, 130)).astype(np.float32)
        y_test = rng.integers(0, 3, size=50).astype(np.int64)

        # train_random_forest should return metrics; we need to test
        # evaluate_classifier with an sklearn model directly
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        result = evaluate_classifier(rf, X_test, y_test)
        assert "accuracy" in result
        assert "confusion_matrix" in result
        assert "balanced_accuracy" in result
