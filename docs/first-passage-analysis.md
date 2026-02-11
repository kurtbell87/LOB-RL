# First-Passage Analysis Module

## What to Build

A Python analysis module (`python/lob_rl/barrier/first_passage_analysis.py`) providing all the tools needed for Phase 1 (null calibration) and Phase 2 (signal detection) of the Asymmetric First-Passage Trading plan. This is shared infrastructure — reusable functions for Brier scores, temporal splits, bootstrap tests, calibration curves, and model fitting.

## New File

`python/lob_rl/barrier/first_passage_analysis.py`

## Dependencies

- numpy, scipy (already installed)
- scikit-learn (already installed)
- lightgbm: `uv pip install lightgbm`

## API

### Label Loading

```python
def load_binary_labels(cache_dir: str, lookback: int = 10) -> dict:
    """Load all barrier cache .npz files and extract binary labels + features.

    Returns dict with:
        X: np.ndarray, shape (N, n_features * lookback), float32 — feature matrix
        Y_long: np.ndarray, shape (N,), bool — long profit label
        Y_short: np.ndarray, shape (N,), bool — short profit label
        timeout_long: np.ndarray, shape (N,), bool — long race timed out
        timeout_short: np.ndarray, shape (N,), bool — short race timed out
        tau_long: np.ndarray, shape (N,), int32 — long race duration (bars)
        tau_short: np.ndarray, shape (N,), int32 — short race duration (bars)
        session_boundaries: np.ndarray, shape (n_sessions + 1,), int64 — cumulative row counts
        dates: list[str] — sorted date strings (YYYYMMDD)

    N = total usable rows across all sessions.
    Y_long[i] = (label_values[i] == +1) for the long race.
    Y_short[i] = (short_label_values[i] == -1) for the short race.
    timeout_long[i] = (label_values[i] == 0) — original label before bias.
    timeout_short[i] = (short_label_values[i] == 0) — original label before bias.

    Note: After the bias_timeout_labels step in C++, timeout labels may have been
    converted to +1/-1 based on barrier proximity. The timeout arrays here detect
    timeouts from the label_tau == t_max condition instead.

    Implementation: timeout detection uses tau >= t_max_bars where t_max_bars is
    the t_max parameter stored in the cache (default 40). If t_max is not stored
    in the cache, fall back to checking label == 0 (pre-bias caches).

    Only loads sessions that have both features and short_label_values keys.
    Sessions are sorted by date. session_boundaries[k] is the start index
    of session k; session_boundaries[-1] is N.
    """
```

### Temporal Splits

```python
def temporal_split(n_sessions: int, train_frac: float = 0.6,
                   val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split session indices temporally (chronological order).

    Returns (train_idx, val_idx, test_idx) — arrays of session indices.
    train_frac + val_frac must be < 1.0. Test gets the remainder.
    Sessions are NOT shuffled — split preserves temporal order.
    """

def temporal_cv_folds(n_sessions: int, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window temporal CV folds.

    Fold k: train on sessions [0, ..., split_k-1], validate on [split_k, ..., split_{k+1}-1].
    Each fold's validation set is disjoint and contiguous. Train set always starts from 0.
    Returns list of (train_session_idx, val_session_idx) tuples.
    """
```

### Scoring

```python
def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score: mean((y_true - y_pred)^2).

    y_true: binary {0,1} or boolean array.
    y_pred: predicted probabilities in [0,1].
    Returns float in [0, 1]. Lower is better.
    """

def constant_brier(y_true: np.ndarray) -> float:
    """Brier score of the constant predictor ybar = mean(y_true).

    Returns ybar * (1 - ybar). This is the baseline to beat.
    """

def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """BSS = 1 - Brier(model) / Brier(constant).

    BSS > 0 means model beats constant baseline.
    BSS = 1 means perfect predictions.
    BSS < 0 means model is worse than constant.
    """

def calibration_curve(y_true: np.ndarray, y_pred: np.ndarray,
                      n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration curve: bin predictions, compare mean predicted vs observed.

    Returns (mean_predicted, fraction_positive) — both shape (n_bins,).
    A well-calibrated model has mean_predicted ≈ fraction_positive for each bin.
    Uses equal-width bins over [0, 1]. Bins with 0 samples are excluded.
    """
```

### Bootstrap

```python
def paired_bootstrap_brier(y_true: np.ndarray, pred_model: np.ndarray,
                           pred_baseline: np.ndarray, n_boot: int = 10000,
                           block_size: int = 50,
                           seed: int = 42) -> dict:
    """Block bootstrap test for Brier score difference.

    Tests H0: Brier(model) >= Brier(baseline) vs H1: Brier(model) < Brier(baseline).

    Uses circular block bootstrap to preserve temporal autocorrelation.
    block_size should be ~sqrt(N) or set to cover autocorrelation length.

    Returns dict:
        delta: float — observed Brier(baseline) - Brier(model). Positive = model better.
        ci_lower: float — 2.5th percentile of bootstrap delta distribution
        ci_upper: float — 97.5th percentile of bootstrap delta distribution
        p_value: float — fraction of bootstrap samples where delta <= 0
    """
```

### Phase 1: Null Calibration

```python
def null_calibration_report(Y_long: np.ndarray, Y_short: np.ndarray,
                            tau_long: np.ndarray, tau_short: np.ndarray,
                            timeout_long: np.ndarray, timeout_short: np.ndarray,
                            session_boundaries: np.ndarray) -> dict:
    """Compute Phase 1 null calibration statistics.

    Returns dict:
        ybar_long: float — mean(Y_long)
        ybar_short: float — mean(Y_short)
        se_long: float — standard error of ybar_long (block bootstrap)
        se_short: float — standard error of ybar_short
        sum_ybar: float — ybar_long + ybar_short
        timeout_rate_long: float — fraction of long timeouts
        timeout_rate_short: float — fraction of short timeouts
        mean_tau_long: float — mean race duration (bars) for long
        mean_tau_short: float — mean race duration (bars) for short
        joint_distribution: dict — {(0,0): count, (0,1): count, (1,0): count, (1,1): count}
        rolling_ybar_long: np.ndarray — per-session ybar_long
        rolling_ybar_short: np.ndarray — per-session ybar_short
        gate_passed: bool — True if ybar in [0.28, 0.38] for both, sum in [0.58, 0.72],
                            timeout rate < 5% for both, no session ybar outside [0.20, 0.46]
    """
```

### Phase 2: Signal Detection

```python
def fit_logistic(X_train: np.ndarray, y_train: np.ndarray,
                 max_iter: int = 1000) -> object:
    """Fit L2-regularized logistic regression.

    Returns fitted sklearn.linear_model.LogisticRegression.
    Uses solver='lbfgs', C=1.0 (default regularization).
    """

def fit_gbt(X_train: np.ndarray, y_train: np.ndarray,
            seed: int = 42) -> object:
    """Fit gradient-boosted tree classifier.

    Uses LightGBM with: max_depth=6, n_estimators=200, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=50.
    Falls back to sklearn GradientBoostingClassifier if lightgbm unavailable.

    Returns fitted model with predict_proba() method.
    """

def signal_detection_report(X: np.ndarray, Y_long: np.ndarray, Y_short: np.ndarray,
                            session_boundaries: np.ndarray,
                            seed: int = 42) -> dict:
    """Run Phase 2 signal detection analysis.

    1. Temporal 60/20/20 split by session
    2. 5-fold temporal CV within training set
    3. Fit logistic + GBT on training, predict on val
    4. Report Brier scores, BSS, bootstrap CIs, calibration curves

    Returns dict:
        For each label in ['long', 'short']:
            brier_constant_{label}: float — constant Brier baseline on val
            brier_logistic_{label}: float — logistic Brier on val
            brier_gbt_{label}: float — GBT Brier on val
            bss_logistic_{label}: float — BSS for logistic
            bss_gbt_{label}: float — BSS for GBT
            delta_logistic_{label}: dict — paired bootstrap result
            delta_gbt_{label}: dict — paired bootstrap result
            calibration_logistic_{label}: tuple — (mean_pred, frac_pos)
            calibration_gbt_{label}: tuple — (mean_pred, frac_pos)
            max_pred_logistic_{label}: float — max predicted probability
            max_pred_gbt_{label}: float — max predicted probability
            cv_brier_logistic_{label}: list[float] — per-fold Brier scores
            cv_brier_gbt_{label}: list[float] — per-fold Brier scores
        profitability_bound: dict — {threshold: 0.40, note: "p > 0.40 needed for C=2, R=10"}
        signal_found: bool — True if any (model, label) has delta > 0 with CI excluding 0
    """
```

### Lattice Verification

```python
def verify_lattice(tick_size: float = 0.25, a: int = 20, b: int = 10) -> dict:
    """Verify barrier distances are on the tick lattice (required for T2).

    Returns dict:
        R_ticks: int — profit barrier in ticks (= a for long, = b for short)
        lattice_ok: bool — True if a * tick_size and b * tick_size are exact multiples
    """
```

## Tests

File: `python/tests/barrier/test_first_passage_analysis.py`

### Brier Score Tests (~8)

1. **Perfect predictions get Brier = 0** — `brier_score([0,1,1,0], [0,1,1,0]) == 0.0`
2. **Worst predictions get Brier = 1** — `brier_score([0,0,0], [1,1,1]) == 1.0`
3. **Constant predictor matches formula** — `brier_score(y, [ybar]*N) ≈ ybar*(1-ybar)`
4. **Known hand-computed example** — `brier_score([1,0,1], [0.8,0.3,0.6]) = mean([0.04, 0.09, 0.16]) = 0.0967`
5. **constant_brier matches ybar*(1-ybar)** — synthetic y with known mean
6. **BSS = 0 for constant predictor** — `brier_skill_score(y, [ybar]*N) == 0.0`
7. **BSS > 0 for better-than-constant** — synthetic data with planted signal
8. **BSS < 0 for worse-than-constant** — predictions far from truth

### Temporal Split Tests (~6)

9. **Non-overlapping splits** — train, val, test have no common indices
10. **Union covers all sessions** — `set(train) | set(val) | set(test) == set(range(n))`
11. **Correct fractions** — train ≈ 60%, val ≈ 20%, test ≈ 20% (±1 session)
12. **Temporal order preserved** — `max(train) < min(val) < min(test)`
13. **CV folds are expanding-window** — each fold's train starts at 0, train grows
14. **CV folds cover all sessions** — union of all val folds covers sessions after first split

### Calibration Curve Tests (~4)

15. **Perfect calibration** — predictions exactly match frequencies → diagonal
16. **All same prediction** — single bin, mean_pred = frac_pos = ybar
17. **Returns correct shape** — n_bins elements (or fewer if bins empty)
18. **Empty bins excluded** — predictions all in [0, 0.5] → only ~5 bins returned

### Bootstrap Tests (~4)

19. **Clearly better model has positive delta** — model Brier much lower → delta > 0, p < 0.05
20. **Identical predictions give delta ≈ 0** — same predictions → CI includes 0
21. **CI is ordered** — ci_lower <= delta <= ci_upper
22. **Deterministic with seed** — same seed → same results

### Null Calibration Tests (~8)

23. **Synthetic random walk → ybar ≈ 1/3** — Generate labels from Bernoulli(1/3) → ybar_long in [0.28, 0.38]
24. **Non-complementarity** — sum_ybar ≈ 2/3 (not 1.0)
25. **Joint distribution sums to N** — all 4 entries sum to total count
26. **Rolling ybar has correct length** — one entry per session
27. **Timeout rate computed correctly** — known fraction of timeouts → correct rate
28. **Gate passes for well-behaved data** — synthetic data matching all criteria → gate_passed = True
29. **Gate fails for extreme ybar** — ybar = 0.10 → gate_passed = False
30. **Gate fails for high timeout rate** — 20% timeouts → gate_passed = False

### Signal Detection Tests (~8)

31. **Planted signal → Brier < constant** — X correlated with Y → brier_gbt < brier_constant
32. **Pure noise → Brier ≈ constant** — random X, random Y → BSS ≈ 0
33. **Returns all expected keys** — all dict keys present for both long and short
34. **CV fold Brier scores are list of floats** — correct type and length
35. **Calibration curves are tuples of arrays** — correct structure
36. **Max predicted probability is float in [0,1]** — valid range
37. **signal_found is True for planted signal** — delta > 0, CI excludes 0
38. **signal_found is False for pure noise** — no signal detected

### Model Fitting Tests (~4)

39. **fit_logistic returns sklearn model** — has predict_proba method
40. **fit_gbt returns model with predict_proba** — works with lightgbm or sklearn fallback
41. **Logistic predictions in [0,1]** — valid probabilities
42. **GBT predictions in [0,1]** — valid probabilities

### Label Loading Tests (~6)

43. **load_binary_labels returns correct keys** — X, Y_long, Y_short, etc.
44. **X shape matches (N, features * lookback)** — correct dimensions
45. **Y_long and Y_short are boolean** — dtype is bool
46. **session_boundaries is monotonically increasing** — boundaries valid
47. **dates are sorted** — chronological order
48. **session_boundaries[-1] == N** — last boundary equals total rows

### Lattice Verification Tests (~2)

49. **R=10 ticks with tick_size=0.25 → lattice_ok=True** — 10*0.25 = 2.5, integer ticks
50. **Non-integer tick multiples detected** — if tick_size doesn't divide evenly

## Edge Cases

- Empty cache directory → ValueError with clear message
- Single session → temporal_split still works (all in train)
- All labels identical → brier_score defined, BSS undefined (division by zero) → handle gracefully
- LightGBM not installed → fall back to sklearn GradientBoostingClassifier

## Acceptance Criteria

- All existing tests still pass
- ~50 new tests pass
- Module importable: `from lob_rl.barrier.first_passage_analysis import ...`
- Works with the real barrier cache at `cache/barrier/`
- No dependencies on C++ bindings (pure Python/numpy/sklearn)
