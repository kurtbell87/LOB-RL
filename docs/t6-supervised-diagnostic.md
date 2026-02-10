# T6: Supervised Diagnostic

## What to Build

A new module `python/lob_rl/barrier/supervised_diagnostic.py` that trains a supervised classifier to predict barrier labels from bar-level features. This validates that the observation space contains learnable signal about barrier outcomes before proceeding to RL training.

**Spec reference:** Section 6.2 of `ppo_barrier_hit_agent_mes_spec_v1.md`.

## Dependencies

- `label_pipeline.py` — `compute_labels()` produces `BarrierLabel` objects with `.label ∈ {+1, -1, 0}`
- `feature_pipeline.py` — `compute_bar_features()` produces 13-column feature arrays, `normalize_features()` z-score normalizes, `assemble_lookback()` stacks h consecutive feature vectors
- `bar_pipeline.py` — `build_bars_from_trades()` produces `TradeBar` objects
- `gamblers_ruin.py` — `generate_random_walk()` for synthetic test data
- `regime_switch.py` — `generate_regime_switch_trades()` for synthetic test data with known signal
- PyTorch — `torch.nn`, `torch.optim` for MLP training
- scikit-learn — `RandomForestClassifier` for baseline comparison

## Module API

### `build_labeled_dataset(bars, labels, h=10, window=2000)`

Constructs the supervised learning dataset `{(Z_k, H_k)}` from bars and labels.

**Parameters:**
- `bars: list[TradeBar]` — Trade bars from `build_bars_from_trades()`.
- `labels: list[BarrierLabel]` — Barrier labels from `compute_labels()`.
- `h: int` — Lookback window for feature stacking. Default 10.
- `window: int` — Trailing normalization window. Default 2000.

**Returns:**
- `X: np.ndarray` — Feature matrix, shape `(N_usable, 13 * h)`. Z-score normalized, lookback-assembled.
- `y: np.ndarray` — Label array, shape `(N_usable,)`. Values in `{0, 1, 2}` mapped from `{-1, 0, +1}`:
  - `-1` (lower hit) → class `0`
  - `0` (timeout) → class `1`
  - `+1` (upper hit) → class `2`

**Notes:**
- The first `h - 1` bars are dropped (insufficient lookback).
- Labels and features must be aligned by bar index. Labels cover `bars[0:len(labels)]`, features cover `bars[0:len(bars)]`.
- NaN features (from trailing vol warmup) are handled by `normalize_features()` and `assemble_lookback()`.

### `class BarrierMLP(nn.Module)`

Two-hidden-layer MLP for 3-class barrier label prediction.

**Constructor:**
```python
BarrierMLP(input_dim: int, hidden_dim: int = 256, n_classes: int = 3)
```

**Architecture:** `Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim) → ReLU → Linear(hidden_dim, n_classes)`

### `overfit_test(X, y, hidden_dim=256, batch_size=256, epochs=500, lr=1e-3, seed=42)`

Tests that the MLP can memorize a small batch.

**Parameters:**
- `X: np.ndarray` — Feature matrix.
- `y: np.ndarray` — Labels.
- `hidden_dim: int` — Hidden layer size. Default 256.
- `batch_size: int` — Number of samples to try overfitting. Default 256.
- `epochs: int` — Training epochs. Default 500.
- `lr: float` — Learning rate. Default 1e-3.
- `seed: int` — Random seed. Default 42.

**Returns:**
- `dict` with keys: `train_accuracy: float`, `passed: bool` (accuracy > 0.95).

### `train_mlp(X_train, y_train, hidden_dim=256, epochs=100, batch_size=512, lr=1e-3, seed=42)`

Trains the MLP classifier on the full training set.

**Parameters:**
- `X_train, y_train` — Training data.
- `hidden_dim, epochs, batch_size, lr, seed` — Training hyperparameters.

**Returns:**
- `model: BarrierMLP` — Trained model.
- `train_metrics: dict` — Keys: `train_accuracy`, `train_loss`.

### `evaluate_classifier(model, X_test, y_test)`

Evaluates a trained classifier on a test set.

**Parameters:**
- `model` — Trained PyTorch model (or any object with a `predict(X) -> np.ndarray` interface for sklearn).
- `X_test, y_test` — Test data.

**Returns:**
- `dict` with keys:
  - `accuracy: float` — Raw accuracy.
  - `balanced_accuracy: float` — Macro-averaged recall across classes.
  - `majority_class: int` — Most frequent class in `y_test`.
  - `majority_baseline: float` — Accuracy of always predicting majority class.
  - `beats_baseline: bool` — `accuracy > majority_baseline`.
  - `confusion_matrix: list[list[int]]` — 3x3 confusion matrix (rows=true, cols=pred).
  - `per_class: dict` — Per-class precision, recall, F1.

### `train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, seed=42)`

Trains a random forest baseline on the same features.

**Parameters:**
- `X_train, y_train, X_test, y_test` — Train/test data.
- `n_estimators: int` — Number of trees. Default 100.
- `seed: int` — Random seed. Default 42.

**Returns:**
- `dict` with keys: `accuracy`, `balanced_accuracy`, `beats_baseline`, `confusion_matrix`, `per_class`, `majority_baseline`.

### `run_diagnostic(bars, labels, h=10, window=2000, train_frac=0.8, hidden_dim=256, epochs=100, seed=42)`

Runs the full supervised diagnostic pipeline.

**Parameters:**
- `bars, labels` — From bar and label pipelines.
- `h, window` — Feature construction params.
- `train_frac: float` — Fraction of data for training. Default 0.8.
- `hidden_dim, epochs, seed` — MLP hyperparameters.

**Returns:**
- `dict` with keys:
  - `n_samples`, `n_train`, `n_test` — Dataset sizes.
  - `label_distribution: dict` — `{p_upper, p_lower, p_timeout}`.
  - `overfit_test: dict` — Overfit test results.
  - `mlp: dict` — MLP evaluation metrics.
  - `random_forest: dict` — Random forest evaluation metrics.
  - `passed: bool` — `overfit_test["passed"] and mlp["beats_baseline"]`.

## Test Cases

### Dataset Construction (5 tests)

1. **`test_build_labeled_dataset_shape`**: Given N bars and M labels, output X has shape `(M - h + 1, 13 * h)` and y has shape `(M - h + 1,)`.
2. **`test_build_labeled_dataset_label_mapping`**: Labels `-1, 0, +1` are mapped to classes `0, 1, 2`.
3. **`test_build_labeled_dataset_no_nan`**: Output X contains no NaN values (normalization handles warmup).
4. **`test_build_labeled_dataset_alignment`**: Feature at index i corresponds to label at bar `i + h - 1`.
5. **`test_build_labeled_dataset_clipped`**: All feature values are in `[-5, +5]` (z-score clipping).

### MLP Architecture (4 tests)

6. **`test_barrier_mlp_output_shape`**: `BarrierMLP(input_dim=130, hidden_dim=256)` produces output shape `(batch, 3)`.
7. **`test_barrier_mlp_default_hidden_dim`**: Default hidden_dim is 256.
8. **`test_barrier_mlp_forward_no_nan`**: Forward pass on random input produces no NaN.
9. **`test_barrier_mlp_gradient_flow`**: Backprop through the model updates all parameters.

### Overfit Test (4 tests)

10. **`test_overfit_test_passes_on_separable_data`**: With linearly separable synthetic data, overfit test returns `passed=True` and `train_accuracy > 0.95`.
11. **`test_overfit_test_returns_dict`**: Returns dict with keys `train_accuracy` and `passed`.
12. **`test_overfit_test_batch_size_capped`**: When dataset is smaller than batch_size, uses full dataset without error.
13. **`test_overfit_test_seed_deterministic`**: Same seed produces same accuracy.

### Training and Evaluation (6 tests)

14. **`test_train_mlp_returns_model_and_metrics`**: Returns a `BarrierMLP` model and dict with `train_accuracy`, `train_loss`.
15. **`test_evaluate_classifier_keys`**: Returns dict with all required keys (accuracy, balanced_accuracy, majority_class, majority_baseline, beats_baseline, confusion_matrix, per_class).
16. **`test_evaluate_classifier_confusion_matrix_shape`**: Confusion matrix is 3x3.
17. **`test_evaluate_classifier_confusion_matrix_sums`**: Confusion matrix rows sum to class counts in y_test.
18. **`test_evaluate_classifier_majority_baseline_correct`**: Majority baseline equals the frequency of the most common class.
19. **`test_evaluate_classifier_balanced_accuracy`**: Balanced accuracy is the mean of per-class recalls.

### Random Forest Baseline (3 tests)

20. **`test_random_forest_returns_metrics`**: Returns dict with required keys.
21. **`test_random_forest_on_separable_data`**: On linearly separable data, accuracy > majority baseline.
22. **`test_random_forest_seed_deterministic`**: Same seed produces same accuracy.

### Full Pipeline (5 tests)

23. **`test_run_diagnostic_returns_all_keys`**: Returns dict with n_samples, n_train, n_test, label_distribution, overfit_test, mlp, random_forest, passed.
24. **`test_run_diagnostic_on_synthetic_regime_switch`**: On regime-switch synthetic data (which has signal in features), MLP beats majority baseline.
25. **`test_run_diagnostic_overfit_passes`**: Overfit test passes on the synthetic data.
26. **`test_run_diagnostic_label_distribution_valid`**: Label distribution values sum to 1.0 and each is in [0, 1].
27. **`test_run_diagnostic_passed_flag`**: `passed` is True iff both overfit passes and MLP beats baseline.

### Edge Cases (3 tests)

28. **`test_build_labeled_dataset_small_n`**: With very few bars (< h + 10), returns appropriately sized arrays without crashing.
29. **`test_evaluate_classifier_all_same_class`**: When all labels are the same class, baseline is 1.0, accuracy is reported correctly.
30. **`test_train_mlp_single_epoch`**: Training with epochs=1 completes without error.

## Implementation Notes

- Use CPU for all test-time training (synthetic data is small). No GPU dependency in tests.
- The module should work with both synthetic data (from `generate_random_walk()` / `generate_regime_switch_trades()`) and real cached data.
- The existing `scripts/bar_supervised_diagnostic.py` uses oracle direction labels (next-bar mid-price). T6 uses barrier labels from `compute_labels()`. These are different oracles — do not reuse the oracle label logic.
- Use `build_feature_matrix()` from `feature_pipeline.py` for feature construction (it handles normalization + lookback assembly).
- For the overfit test, generate linearly separable synthetic data rather than relying on real data being separable.
- The `evaluate_classifier` function should work with both PyTorch models (via `torch.no_grad()`) and sklearn models (via `.predict()`). Use duck typing or a simple wrapper.
- Random forest requires `scikit-learn` (`sklearn.ensemble.RandomForestClassifier`).
- The regime-switch test (test #24) should use `generate_regime_switch_trades()` with enough bars to produce meaningful labels, then verify that the MLP can learn to distinguish the regimes from features alone. Use a small dataset (e.g., 1000 bars per regime) to keep tests fast.
