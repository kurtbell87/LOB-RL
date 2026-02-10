# T3: Feature Extraction Pipeline

## What to Build

A Python module `python/lob_rl/barrier/feature_pipeline.py` that computes the 13 bar-level features specified in Section 3 of the spec, with z-score normalization and lookback window assembly.

## Context

The bar pipeline (T1) produces `TradeBar` objects with OHLCV data. The feature pipeline computes features from bar data and raw MBO data (for LOB snapshots and cancel events). Features are normalized and assembled into observation windows for the RL agent.

## Dependencies

- `python/lob_rl/barrier/bar_pipeline.py` — T1 (PASSED). Provides `TradeBar` and `extract_trades_from_mbo`.

## Interface

### `compute_bar_features(bars, mbo_data=None)` -> `np.ndarray`

Compute the 13 features for each bar.

**Parameters:**
- `bars`: List of `TradeBar` objects from a single session
- `mbo_data`: Optional raw MBO DataFrame (from `databento.DBNStore.to_df()`) for LOB features. If None, LOB-dependent features (BBO imbalance, depth imbalance, cancel rate asymmetry, mean spread) are set to their neutral values (0.5, 0.5, 0.0, 1.0 respectively).

**Returns:** ndarray of shape `(len(bars), 13)`, one row per bar, 13 features per row.

### Feature Definitions (columns 0-12)

| Col | Feature              | Range | Definition |
|-----|----------------------|-------|------------|
| 0   | Trade flow imbalance | [-1, +1] | `(buy_vol - sell_vol) / total_vol` from TradeBar trade sides |
| 1   | BBO imbalance        | [0, 1] | `bid_L1 / (bid_L1 + ask_L1)` at bar close |
| 2   | Depth imbalance      | [0, 1] | `sum(bid_L1:L5) / (sum(bid_L1:L5) + sum(ask_L1:L5))` at bar close |
| 3   | Bar range            | [0, inf) | `(H_k - L_k)` in ticks (divided by 0.25) |
| 4   | Bar body             | (-inf, inf) | `(C_k - O_k)` in ticks |
| 5   | Body/range ratio     | [-1, +1] | `(C_k - O_k) / (H_k - L_k)` if range > 0, else 0 |
| 6   | VWAP displacement    | [-1, +1] | `(C_k - VWAP_k) / (H_k - L_k)` if range > 0, else 0 |
| 7   | Volume (log)         | (-inf, inf) | `log(V_k)` |
| 8   | Trailing realized vol| [0, inf) | `std(log(C_j / C_{j-1}))` for j in [k-19, k], NaN for first 19 bars |
| 9   | Normalized session time | [0, 1] | `(t_end_k - RTH_open) / (RTH_close - RTH_open)` |
| 10  | Cancel rate asymmetry | [-1, +1] | `(bid_cancels - ask_cancels) / total_cancels` over bar k |
| 11  | Mean spread          | (0, inf) | Average `(best_ask - best_bid)` in ticks within bar k |
| 12  | Session age          | [0, 1] | `min(bar_index / 20, 1.0)` |

### `normalize_features(raw_features, window=2000)` -> `np.ndarray`

Z-score normalize features using a trailing window of means and standard deviations.

**Parameters:**
- `raw_features`: ndarray of shape `(N, 13)` — raw feature matrix
- `window`: Trailing window size for normalization statistics. Default 2000.

**Returns:** ndarray of shape `(N, 13)` — normalized features, clipped to [-5, +5].

### `assemble_lookback(normalized_features, h=10)` -> `np.ndarray`

Stack h consecutive feature vectors into lookback windows.

**Parameters:**
- `normalized_features`: ndarray of shape `(N, 13)` — normalized features
- `h`: Lookback window size. Default 10.

**Returns:** ndarray of shape `(N - h + 1, 13 * h)` — each row is a flattened stack of h consecutive feature vectors. First h-1 bars are dropped (insufficient history).

### `build_feature_matrix(bars, mbo_data=None, window=2000, h=10)` -> `np.ndarray`

End-to-end: compute features, normalize, assemble lookback.

**Returns:** ndarray of shape `(N', 130)` where `N' = len(bars) - max(h, warmup) + 1`.

## Tests to Write

### Feature bounds

1. **Trade flow imbalance in [-1, +1]:** All bars.
2. **BBO imbalance in [0, 1]:** All bars (when mbo_data provided or using neutral default).
3. **Depth imbalance in [0, 1]:** All bars.
4. **Bar range non-negative:** All bars.
5. **Body/range ratio in [-1, +1]:** All bars (0 when range=0).
6. **VWAP displacement in [-1, +1]:** All bars (0 when range=0).
7. **Volume finite and positive:** All bars.
8. **Normalized session time in [0, 1]:** Monotonically non-decreasing within session.
9. **Session age starts at 0, saturates at 1.0 after 20 bars.**
10. **Cancel rate asymmetry in [-1, +1].**
11. **Mean spread positive.**

### Feature computation correctness

12. **Trailing realized vol uses exactly 20 bars:** NaN for first 19 bars. Hand-compute for a known sequence.
13. **Trade flow imbalance hand-computed:** Known buy/sell volumes → expected ratio.
14. **Bar range hand-computed:** Known H, L → expected range in ticks.
15. **Body/range ratio with zero range:** Returns 0 when H==L.
16. **VWAP displacement with zero range:** Returns 0 when H==L.
17. **Session age capping:** Bar 0 → 0.0, bar 10 → 0.5, bar 20+ → 1.0.
18. **Volume log transform:** V_k=100 → log(100) ≈ 4.605.

### Normalization

19. **Z-score normalization:** After normalization, mean ≈ 0 and std ≈ 1 over trailing window.
20. **Clipping to [-5, +5]:** No values outside this range after normalization.
21. **No NaN or Inf in final feature matrix:** After handling warm-up period.
22. **Trailing window behavior:** Features at the start use growing window (fewer than 2000 bars available).

### Lookback assembly

23. **Lookback shape:** Output shape is (N-h+1, 13*h).
24. **Lookback correctness:** Z_k correctly stacks h consecutive feature vectors. Verify row i == concatenation of features[i:i+h].
25. **Default h=10 gives 130-dim output.**

### Edge cases

26. **Single bar session:** Features computed but lookback impossible with h>1.
27. **All same price bar:** Range=0, body=0, ratios=0. No division errors.
28. **Features shape:** Always (N, 13) regardless of input.

## Acceptance Criteria

- All tests pass.
- Feature distributions plotted and visually inspected (no pathological spikes, no constant features).
- No NaN/Inf in final matrix.

## File Location

- Module: `python/lob_rl/barrier/feature_pipeline.py`
- Tests: `python/tests/barrier/test_feature_pipeline.py`
