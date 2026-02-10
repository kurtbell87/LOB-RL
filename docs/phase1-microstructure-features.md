# Phase 1 Microstructure Features (Cols 13-16)

## Summary

Expand the barrier feature set from 13 to 17 features by adding 4 new LOB microstructure features that leverage the `OrderBook` and MBO data infrastructure built in Cycle 1. Bump `N_FEATURES` from 13 to 17 in `__init__.py`.

## New Features

| Col | Feature | Range | Default (no MBO) | Computation |
|-----|---------|-------|-------------------|-------------|
| 13 | Order Flow Imbalance (OFI) | [-1, +1] | 0.0 | Net signed volume from Add messages at BBO within bar. |
| 14 | Multi-level depth ratio | [0, 1] | 0.5 | Liquidity concentration near BBO. |
| 15 | Weighted mid-price displacement | signed | 0.0 | Intra-bar movement of imbalance-weighted mid in ticks. |
| 16 | Spread dynamics (std) | >= 0 | 0.0 | Std of spread samples within bar, in ticks. |

New observation dim: 17 * 10 + 2 = 172.

## Feature Computation Details

### Col 13: Order Flow Imbalance (OFI)

Based on Cont, Kukanov & Stoikov (2014). Captures directional information from order additions at BBO.

```
For each Add ('A') message in the bar:
  - If side == 'B' and price >= best_bid:  ofi += size
  - If side == 'A' and price <= best_ask:  ofi -= size

ofi_normalized = ofi / (total_add_volume + eps)
# Clamp to [-1, +1]
```

Where `total_add_volume` is the sum of sizes of ALL Add messages in the bar. `eps = 1e-10` to avoid division by zero. Use the book state BEFORE applying the message to determine BBO proximity.

When `mbo_data is None`: default 0.0 (neutral).

### Col 14: Multi-level Depth Ratio

Measures how concentrated liquidity is near the BBO. High concentration = fragile book, more susceptible to large moves.

```
At bar close snapshot:
  top3_bid = book.total_bid_depth(3)
  top3_ask = book.total_ask_depth(3)
  top10_bid = book.total_bid_depth(10)
  top10_ask = book.total_ask_depth(10)

  total_3 = top3_bid + top3_ask
  total_10 = top10_bid + top10_ask

  depth_ratio = total_3 / (total_10 + eps)  # range [0, 1]
```

When `mbo_data is None`: default 0.5 (neutral).

### Col 15: Weighted Mid-Price Displacement

Captures intra-bar movement of the imbalance-weighted mid-price. A positive displacement indicates the book structure shifted upward during the bar.

```
wmid_start = book.weighted_mid_price() at start of bar (before any bar messages)
wmid_end = book.weighted_mid_price() at end of bar (after all bar messages)

displacement = (wmid_end - wmid_start) / TICK_SIZE
```

If book is empty at either end (weighted_mid returns 0.0), default to 0.0.

When `mbo_data is None`: default 0.0.

### Col 16: Spread Dynamics (Std)

Standard deviation of spread samples within the bar. High spread volatility indicates uncertain liquidity conditions and predicts price volatility.

```
For each MBO event in the bar, sample spread_ticks.
If >= 2 samples with spread > 0:
  spread_std = np.std(spread_samples)
Else:
  spread_std = 0.0
```

When `mbo_data is None`: default 0.0.

## Changes to `__init__.py`

```python
N_FEATURES = 17  # was 13
```

## Changes to `feature_pipeline.py`

### `compute_bar_features()`

1. Update the feature array allocation: `np.zeros((n, 17), ...)` — use `N_FEATURES` imported from `__init__`.
2. Update docstring to show 17 features with new cols 13-16.
3. When `book_features` is available, assign cols 13-16 from `book_features[:, 4:8]`.
4. When `mbo_data is None`, set defaults: col 13=0.0, col 14=0.5, col 15=0.0, col 16=0.0.

### `_compute_book_features()`

Expand from shape `(n, 4)` to `(n, 8)`:
- Cols 0-3: existing (BBO imbalance, depth imbalance, cancel asymmetry, mean spread)
- Col 4: OFI
- Col 5: Multi-level depth ratio
- Col 6: Weighted mid-price displacement
- Col 7: Spread dynamics (std)

**New defaults for cols 4-7:** `(0.0, 0.5, 0.0, 0.0)`. Append to `_BOOK_DEFAULTS`.

**Implementation within bar loop:**
- Before processing bar messages: record `wmid_start = book.weighted_mid_price()`.
- Track OFI: for each Add message, check if price is at/near BBO and accumulate signed volume.
- After processing bar messages: record `wmid_end = book.weighted_mid_price()`.
- At bar close: compute depth ratio, displacement, spread std.

### `normalize_features()` and `assemble_lookback()`

No changes needed — they already use `raw.shape[1]` dynamically.

### `build_feature_matrix()` docstring

Update: `(M, 17*h)` instead of `(M, 13*h)`.

## Changes to `_BOOK_DEFAULTS`

Expand from 4 to 8 values:
```python
_BOOK_DEFAULTS = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0)
#                 BBO  Depth Cancel Spread OFI  DepthR WMid SpreadStd
```

## Downstream Impact

All downstream code that uses `N_FEATURES` or feature matrix shapes will automatically adapt:

- `barrier_env.py`: `self._h = self._feature_dim // N_FEATURES` — already uses `N_FEATURES`.
- `conftest.py`: `DEFAULT_FEATURE_DIM = N_FEATURES * DEFAULT_H` — already uses `N_FEATURES`.
- `supervised_diagnostic.py`: `build_labeled_dataset()` uses `compute_bar_features()` output shape dynamically.
- `precompute_barrier_cache.py`: stores `n_features` metadata. Old caches will fail version check (user must re-precompute).

No code changes needed in these files beyond what `N_FEATURES` propagates.

## Edge Cases

1. **No Add messages in bar**: OFI = 0.0.
2. **Book empty at bar start/end**: wmid displacement = 0.0.
3. **Only 1 spread sample**: spread std = 0.0 (need >= 2).
4. **All depth at BBO**: depth ratio = 1.0 (fully concentrated).
5. **No depth at all**: depth ratio = 0.5 (default).
6. **Add at price far from BBO**: Does NOT count toward OFI (only BBO-level adds matter).

## Acceptance Criteria

1. `N_FEATURES == 17` and all files that imported it reflect the new value.
2. `compute_bar_features()` returns shape `(N, 17)`.
3. Without MBO data, new cols 13-16 have correct neutral defaults.
4. With synthetic MBO data, OFI/depth ratio/displacement/spread std compute correctly.
5. All existing barrier tests pass (shape changes propagate cleanly).
6. `build_feature_matrix()` returns shape `(M, 17*h)`.
7. `BarrierEnv` observation dim = 17*10 + 2 = 172 (with h=10).

## Test Strategy

### OFI tests (~6)
- `test_ofi_positive_for_bid_adds`: Bid adds at BBO -> positive OFI.
- `test_ofi_negative_for_ask_adds`: Ask adds at BBO -> negative OFI.
- `test_ofi_zero_no_adds`: No Add messages -> OFI = 0.0.
- `test_ofi_ignores_non_bbo_adds`: Adds far from BBO don't count.
- `test_ofi_range`: Output clamped to [-1, +1].
- `test_ofi_default_no_mbo`: Without mbo_data, col 13 = 0.0.

### Depth ratio tests (~5)
- `test_depth_ratio_concentrated`: All depth at L1 -> ratio near 1.0.
- `test_depth_ratio_dispersed`: Depth evenly spread -> ratio < 1.0.
- `test_depth_ratio_empty_book`: Empty book -> default 0.5.
- `test_depth_ratio_range`: Output in [0, 1].
- `test_depth_ratio_default_no_mbo`: Without mbo_data, col 14 = 0.5.

### Weighted mid displacement tests (~5)
- `test_wmid_displacement_positive`: Book shifts up -> positive.
- `test_wmid_displacement_negative`: Book shifts down -> negative.
- `test_wmid_displacement_no_change`: Book unchanged -> 0.0.
- `test_wmid_displacement_empty_book`: Empty book -> 0.0.
- `test_wmid_displacement_default_no_mbo`: Without mbo_data, col 15 = 0.0.

### Spread std tests (~4)
- `test_spread_std_constant`: Constant spread -> std ≈ 0.
- `test_spread_std_variable`: Variable spread -> std > 0.
- `test_spread_std_single_sample`: Only 1 sample -> 0.0.
- `test_spread_std_default_no_mbo`: Without mbo_data, col 16 = 0.0.

### Integration tests (~5)
- `test_n_features_now_17`: `N_FEATURES == 17`.
- `test_compute_bar_features_shape_17`: Output shape (N, 17).
- `test_build_feature_matrix_shape_17h`: Output shape (M, 17*h).
- `test_barrier_env_obs_dim_172`: With h=10, obs = 172.
- `test_all_existing_barrier_tests_pass`: Existing tests pass (they use `N_FEATURES` dynamically).

Total estimated: ~25 tests.
