# T5: Regime-Switch Validation

## What to Build

A Python module `python/lob_rl/barrier/regime_switch.py` that validates the label + normalization pipeline against a synthetic volatility regime switch. This ensures the pipeline preserves regime information and that trailing normalization does not smooth over regime transitions pathologically.

## Context

The label and feature pipelines (T2, T3) use trailing z-score normalization with a 2000-bar window. A known risk is that this normalization lag creates a "dead zone" around regime transitions where normalized features reflect stale statistics rather than the new regime. We need to verify that:

1. The pipeline correctly distinguishes high-vol vs low-vol regimes in labels (timeout rates, resolution times).
2. The trailing normalization adapts within a reasonable number of bars (not 2000).
3. Feature distributions show a visible shift at the regime boundary.

## Dependencies

- `python/lob_rl/barrier/bar_pipeline.py` — T1 (PASSED). Provides `TradeBar` and `build_bars_from_trades`.
- `python/lob_rl/barrier/label_pipeline.py` — T2 (PASSED). Provides `compute_labels`, `BarrierLabel`, `compute_label_distribution`.
- `python/lob_rl/barrier/feature_pipeline.py` — T3 (PASSED). Provides `compute_bar_features`, `normalize_features`, `build_feature_matrix`.

## Interface

### `generate_regime_switch_trades(n_bars_low=5000, n_bars_high=5000, bar_size=500, p=0.5, start_price=4000.0, tick_size=0.25, seed=None)` -> `tuple[np.ndarray, list[TradeBar]]`

Generate synthetic trade data with a known volatility regime switch.

**Low-vol regime** (first `n_bars_low * bar_size` trades): tick increments drawn from `{-1, +1}` with probability `p` for uptick (standard random walk, 1-tick moves only).

**High-vol regime** (next `n_bars_high * bar_size` trades): tick increments drawn from `{-3, -2, -1, +1, +2, +3}` uniform (each with prob 1/6). Zero increments are excluded to match the spec.

**Parameters:**
- `n_bars_low`: Number of bars in the low-vol regime. Default 5000.
- `n_bars_high`: Number of bars in the high-vol regime. Default 5000.
- `bar_size`: Trades per bar. Default 500.
- `p`: Uptick probability for low-vol regime. Default 0.5.
- `start_price`: Starting price. Default 4000.0.
- `tick_size`: Tick size. Default 0.25.
- `seed`: Random seed for reproducibility.

**Returns:** Tuple of:
1. Structured numpy array of trades with fields `(price, size, side, ts_event)` — full trade sequence.
2. `list[TradeBar]` — bars built from the trades (using `build_bars_from_trades`).

### `validate_regime_switch(n_bars_low=5000, n_bars_high=5000, bar_size=500, a=20, b=10, t_max=40, seed=42)` -> `dict`

Run the full regime-switch validation. Generates synthetic regime-switch data, computes labels and features for both segments, and runs all statistical tests.

**Parameters:**
- `n_bars_low`, `n_bars_high`, `bar_size`: Forwarded to `generate_regime_switch_trades`.
- `a`, `b`, `t_max`: Forwarded to `compute_labels`.
- `seed`: Random seed.

**Returns:** Dictionary with:
```python
{
    "n_bars_total": int,               # Total bars
    "n_bars_low": int,                 # Bars in low-vol segment
    "n_bars_high": int,                # Bars in high-vol segment
    "boundary_bar": int,               # Bar index at regime boundary

    # Label statistics per segment
    "low_vol": {
        "p_plus": float,               # Fraction label=+1
        "p_minus": float,              # Fraction label=-1
        "p_zero": float,               # Fraction label=0 (timeout)
        "mean_tau": float,             # Mean time-to-resolution
        "median_tau": float,           # Median time-to-resolution
    },
    "high_vol": {
        "p_plus": float,
        "p_minus": float,
        "p_zero": float,
        "mean_tau": float,
        "median_tau": float,
    },

    # Statistical tests
    "timeout_ratio": float,            # low_timeout_rate / high_timeout_rate
    "chi2_p_value": float,             # Chi-squared test on label distributions
    "ks_bar_range_p": float,           # KS test on bar range feature
    "ks_realized_vol_p": float,        # KS test on realized vol feature

    # Normalization adaptation
    "norm_adaptation_bars": int,       # Bars after boundary for normalized realized vol
                                       # to reflect new regime (within 1 stddev of post-boundary mean)
    "pass": bool,                      # All tests pass
}
```

### `compute_segment_stats(labels, start, end)` -> `dict`

Helper: compute label statistics for a contiguous segment of labels.

**Parameters:**
- `labels`: Full list of `BarrierLabel`.
- `start`: Start bar index (inclusive).
- `end`: End bar index (exclusive).

**Returns:** Dict with `p_plus`, `p_minus`, `p_zero`, `mean_tau`, `median_tau`.

### `ks_test_features(features, boundary, window=500)` -> `dict`

Run Kolmogorov-Smirnov tests on feature distributions before vs after the regime boundary.

**Parameters:**
- `features`: Raw (unnormalized) feature array, shape (N, 13).
- `boundary`: Bar index of regime boundary.
- `window`: Number of bars before/after boundary to compare. Default 500.

**Returns:** Dict with KS test p-values for each feature column. Keys: `ks_p_col_0` through `ks_p_col_12`.

### `measure_normalization_adaptation(normed_features, boundary, col=8, threshold_sigma=1.0)` -> `int`

Measure how many bars after the regime boundary it takes for normalized features to reflect the new regime.

**Parameters:**
- `normed_features`: Normalized feature array, shape (N, 13).
- `boundary`: Bar index of regime boundary.
- `col`: Feature column to check. Default 8 (realized vol).
- `threshold_sigma`: How close to the post-boundary mean (in post-boundary stddevs) the feature must be.

**Returns:** Number of bars after boundary until the normalized feature is within `threshold_sigma` standard deviations of the post-boundary steady-state mean (computed from bars boundary+500 onward). Returns -1 if never adapts within available bars.

## Tests to Write

### Synthetic data generation

1. **Output shape:** Total trades = `(n_bars_low + n_bars_high) * bar_size`.
2. **Bar count:** Number of bars = `n_bars_low + n_bars_high`.
3. **Low-vol tick increments:** All trade-to-trade price changes in the low-vol segment are exactly `+/- tick_size`.
4. **High-vol tick increments:** Trade-to-trade price changes in the high-vol segment include values of magnitude 2*tick_size and 3*tick_size (not just 1*tick_size).
5. **Regime boundary is at correct bar index:** Bar index `n_bars_low` is the first high-vol bar.
6. **Reproducibility:** Same seed → same trades and bars.
7. **Price continuity at boundary:** No gap between last low-vol trade and first high-vol trade (the price series is continuous).

### Label distribution differences

8. **Timeout rate higher in low-vol:** `low_vol["p_zero"] > high_vol["p_zero"]`.
9. **Timeout ratio > 2x:** `low_vol["p_zero"] / max(high_vol["p_zero"], 1e-10) > 2.0`.
10. **Mean tau longer in low-vol:** `low_vol["mean_tau"] > high_vol["mean_tau"]`.
11. **Chi-squared test rejects H0:** `chi2_p_value < 0.01` (label distributions differ between segments).
12. **Low-vol has more timeouts than barrier hits:** `low_vol["p_zero"] > low_vol["p_plus"]` (with default a=20, b=10 and 1-tick moves, low vol should timeout frequently).
13. **High-vol resolves quickly:** `high_vol["mean_tau"] < 5` (with 3-tick moves, barriers are hit fast).

### Feature distribution shifts

14. **KS test on bar range:** `ks_bar_range_p < 0.01` (bar range distribution clearly differs between regimes).
15. **KS test on realized vol:** `ks_realized_vol_p < 0.01` (realized vol distribution clearly differs).
16. **Bar range higher in high-vol:** Mean bar range in high-vol > mean bar range in low-vol.
17. **Realized vol higher in high-vol:** Mean realized vol in high-vol > mean realized vol in low-vol (after warmup).

### Normalization adaptation

18. **No 500-bar dead zone:** `norm_adaptation_bars < 500` (normalization reflects new regime within 500 bars, not 2000).
19. **Adaptation within 200 bars:** `norm_adaptation_bars < 200` (tighter bound — with 2000-bar window, exponential weighting of recent data should adapt faster).
20. **Normalized realized vol shows visible shift:** The mean normalized realized vol in bars [boundary+200, boundary+500] differs from the mean in bars [boundary-500, boundary-200] by at least 1.0 standard units.

### Full pipeline validation

21. **All tests pass:** `validate_regime_switch(seed=42)["pass"] == True`.
22. **No NaN in features:** Raw features have no NaN after warmup (bar 19+).
23. **No Inf in features:** No Inf anywhere.
24. **Labels are valid:** All labels in {-1, 0, +1}, all tau >= 1, all tau <= t_max.

### Edge cases

25. **Small regime (100 bars each):** Doesn't crash, produces reasonable results.
26. **Deterministic seed:** Two calls with same seed produce identical results.

## Acceptance Criteria

- Timeout rate is >2x higher in low-vol than high-vol segment.
- Mean time-to-resolution is longer in low-vol than high-vol.
- KS tests reject H0 (p < 0.01) for bar range and realized vol features at the regime boundary.
- Chi-squared test rejects H0 (p < 0.01) for label distribution difference between segments.
- Trailing normalization adapts within 500 bars (not 2000) at the regime boundary.
- No NaN or Inf in outputs.

## File Location

- Module: `python/lob_rl/barrier/regime_switch.py`
- Tests: `python/tests/barrier/test_regime_switch.py`
