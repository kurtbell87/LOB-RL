# Barrier Feature Computation, Normalization, and Lookback Assembly

## Context

Cycle 2 built `BarBuilder` which produces `TradeBar` and `BarBookAccum` vectors from a single-pass MBO stream. This cycle implements the C++ equivalent of Python's `feature_pipeline.py` — computing 22 raw bar-level features, z-score normalizing with a trailing window, and assembling lookback matrices.

The C++ version is simpler than Python because book-derived data is already captured in `BarBookAccum` by `BarBuilder`. Python replays MBO events through `OrderBook` per bar; C++ just reads from the accum struct.

### Existing Infrastructure

- `TradeBar` — OHLCV + trade_prices/trade_sizes + timestamps (`include/lob/barrier/trade_bar.h`)
- `BarBookAccum` — BBO qty, depth(3/5/10), cancel counts, OFI, wmid, spread samples, VAMP, aggressor vols, trade/cancel counts (`include/lob/barrier/trade_bar.h`)
- `BarBuilder` — produces `bars()` and `accums()` vectors (`include/lob/barrier/bar_builder.h`)

## Requirements

### 1. Constants

**File:** `include/lob/barrier/feature_compute.h`

```cpp
constexpr int N_FEATURES = 22;
constexpr double TICK_SIZE = 0.25;
constexpr int REALIZED_VOL_WARMUP = 19;
constexpr double SESSION_AGE_PERIOD = 20.0;
```

These must match Python's `lob_rl.barrier.__init__` constants exactly.

### 2. compute_bar_features()

```cpp
std::vector<double> compute_bar_features(
    const std::vector<TradeBar>& bars,
    const std::vector<BarBookAccum>& accums,
    uint64_t rth_open_ns,
    uint64_t rth_close_ns);
```

Returns a flat row-major vector of size `n_bars * N_FEATURES`. Element `[i * N_FEATURES + col]` is feature `col` for bar `i`.

**Precondition:** `bars.size() == accums.size()`. Asserts or throws if violated.

#### 22 Feature Columns

| Col | Name | Formula | Range |
|-----|------|---------|-------|
| 0 | Trade flow imbalance | Tick rule on trade_prices/sizes (see below) | [-1, +1] |
| 1 | BBO imbalance | `bid_qty / (bid_qty + ask_qty)`, 0.5 if both zero | [0, 1] |
| 2 | Depth imbalance | `total_bid_5 / (total_bid_5 + total_ask_5)`, 0.5 if both zero | [0, 1] |
| 3 | Bar range (ticks) | `(high - low) / TICK_SIZE` | >= 0 |
| 4 | Bar body (ticks) | `(close - open) / TICK_SIZE` | signed |
| 5 | Body/range ratio | `(close - open) / (high - low)`, 0.0 if range==0 | [-1, +1] |
| 6 | VWAP displacement | `(close - vwap) / (high - low)`, 0.0 if range==0 | [-1, +1] |
| 7 | Volume (log) | `log(max(volume, 1))` | finite |
| 8 | Trailing realized vol | `std(19 log-returns)`, NaN for bars `i < 19` | >= 0 or NaN |
| 9 | Normalized session time | `clamp((t_end - rth_open) / (rth_close - rth_open), 0, 1)` | [0, 1] |
| 10 | Cancel asymmetry | `(bid_cancels - ask_cancels) / (total_cancels + 1e-10)` | [-1, +1] |
| 11 | Mean spread | `mean(spread_samples)`, 1.0 if empty | > 0 |
| 12 | Session age | `min(bar_index / 20.0, 1.0)` | [0, 1] |
| 13 | OFI | `clamp(ofi_signed_volume / (total_add_volume + 1e-10), -1, 1)`, 0.0 if total_add_volume==0 | [-1, +1] |
| 14 | Depth ratio | `(total_bid_3 + total_ask_3) / (total_bid_10 + total_ask_10 + 1e-10)`, 0.5 if total_10==0 | [0, 1] |
| 15 | WMid displacement | `(wmid_end - wmid_first) / TICK_SIZE`, 0.0 if either is NaN | signed |
| 16 | Spread std | `std(spread_samples, ddof=0)`, 0.0 if < 2 samples | >= 0 |
| 17 | VAMP displacement | `(vamp_at_end - vamp_at_mid) / TICK_SIZE`, 0.0 if either is NaN | signed |
| 18 | Aggressor imbalance | `(buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)`, 0.0 if total==0 | [-1, +1] |
| 19 | Trade arrival rate | `log(1 + n_trades)` | >= 0 |
| 20 | Cancel-to-trade ratio | `log(1 + n_cancels / max(n_trades, 1))` | >= 0 |
| 21 | Price impact per trade | `(close - open) / (max(n_trades, 1) * TICK_SIZE)` | signed |

#### Trade Flow Imbalance (Col 0) — Tick Rule

For a bar with trade_prices `[p0, p1, ..., pN]` and trade_sizes `[s0, s1, ..., sN]`:

1. Compute diffs: `d[i] = sign(p[i] - p[i-1])` for i >= 1
2. Build sides array: `sides[0] = 0` (neutral), `sides[i] = d[i]` for i >= 1
3. Forward-fill zeros: if `sides[i] == 0`, `sides[i] = sides[i-1]`
4. `buy_vol = sum(sizes where sides > 0)`
5. `sell_vol = sum(sizes where sides < 0)`
6. Result: `(buy_vol - sell_vol) / (buy_vol + sell_vol)`, 0.0 if total==0

If trade_prices has <= 1 element, return 0.0.

#### Trailing Realized Vol (Col 8)

For bar i where `i >= REALIZED_VOL_WARMUP` (i.e., i >= 19):
1. Take close prices `[close[i-19], close[i-18], ..., close[i]]` — 20 prices
2. Compute 19 log-returns: `r[j] = log(close[j+1] / close[j])`
3. Compute population std (ddof=0): `std(r)`

For `i < 19`, set to NaN.

### 3. normalize_features()

```cpp
std::vector<double> normalize_features(
    const std::vector<double>& raw,
    int n_rows,
    int n_cols,
    int window = 2000);
```

Z-score normalize each row using a trailing window. Input is flat row-major `(n_rows, n_cols)`.

**Algorithm:**
1. Replace NaN with 0.0 in a working copy
2. For each row `i`:
   a. `start = max(0, i - window + 1)`
   b. Compute column-wise mean and std over rows `[start, i]`
   c. `z = (row[i] - mean) / std`, where `std == 0` → `z = 0`
3. Clip all values to `[-5.0, +5.0]`

Returns flat row-major vector of size `n_rows * n_cols`.

### 4. assemble_lookback()

```cpp
std::vector<float> assemble_lookback(
    const std::vector<double>& normed,
    int n_rows,
    int n_cols,
    int h = 10);
```

Sliding window reshape. Output row `i` = concatenation of normalized rows `[i, i+1, ..., i+h-1]`.

**Output shape:** `(n_rows - h + 1, n_cols * h)`. Returns as flat row-major `float` (float32).

If `n_rows < h`, returns empty vector (size 0).

### 5. CMakeLists.txt Changes

Add `src/barrier/feature_compute.cpp` to `lob_core` sources.

## Files to Change

| File | Change |
|------|--------|
| `include/lob/barrier/feature_compute.h` | **NEW** — Constants + function declarations |
| `src/barrier/feature_compute.cpp` | **NEW** — Implementation |
| `CMakeLists.txt` | Add src/barrier/feature_compute.cpp and test file |
| `tests/test_barrier_features.cpp` | **NEW** — Tests |

## Test Plan

### Constants tests (~3):
1. N_FEATURES equals 22
2. TICK_SIZE equals 0.25
3. REALIZED_VOL_WARMUP equals 19, SESSION_AGE_PERIOD equals 20.0

### Trade flow imbalance (Col 0) tests (~5):
4. Single trade → 0.0
5. Two trades uptick → positive imbalance
6. Two trades downtick → negative imbalance
7. Forward-fill: unchanged price continues previous direction
8. All same price → 0.0

### BBO/Depth imbalance (Cols 1-2) tests (~4):
9. BBO imbalance: bid_qty > ask_qty → > 0.5
10. BBO imbalance: both zero → 0.5
11. Depth imbalance: total_bid_5 > total_ask_5 → > 0.5
12. Depth imbalance: both zero → 0.5

### Bar range/body/ratio (Cols 3-6) tests (~5):
13. Bar range = (high-low)/tick_size
14. Bar body = (close-open)/tick_size, signed
15. Body/range ratio: flat bar (range==0) → 0.0
16. VWAP displacement: close > vwap → positive
17. VWAP displacement: flat bar → 0.0

### Volume and realized vol (Cols 7-8) tests (~4):
18. Volume log: log(volume) for volume > 0
19. Volume log: volume 0 → log(1) = 0.0
20. Realized vol: NaN for bars 0..18
21. Realized vol: correct std for bars >= 19 (hand-computed)

### Session time and age (Cols 9, 12) tests (~3):
22. Session time: midpoint → ~0.5
23. Session time: before open → 0.0 (clamped)
24. Session age: bar 0 → 0.0, bar 20+ → 1.0

### Cancel/spread features (Cols 10-11) tests (~3):
25. Cancel asymmetry: more bid cancels → positive
26. Cancel asymmetry: zero cancels → 0.0
27. Mean spread: non-empty samples → mean, empty → 1.0

### OFI and depth ratio (Cols 13-14) tests (~4):
28. OFI: positive signed volume → positive, clamped to [-1,1]
29. OFI: zero total_add_volume → 0.0
30. Depth ratio: (3+3)/(10+10) computes correctly
31. Depth ratio: total_10 == 0 → 0.5

### WMid/Spread std (Cols 15-16) tests (~3):
32. WMid displacement: (wmid_end - wmid_first) / TICK_SIZE
33. WMid displacement: either NaN → 0.0
34. Spread std: < 2 samples → 0.0, >= 2 → correct

### VAMP/Aggressor/Trade features (Cols 17-21) tests (~6):
35. VAMP displacement: (vamp_end - vamp_mid) / TICK_SIZE
36. VAMP displacement: either NaN → 0.0
37. Aggressor imbalance: buy > sell → positive
38. Trade arrival: log(1 + n_trades)
39. Cancel-to-trade: log(1 + n_cancels/max(n_trades,1))
40. Price impact: (close - open) / (max(n_trades,1) * TICK_SIZE)

### Full feature vector tests (~3):
41. Output size = n_bars * 22
42. All 22 columns populated for a multi-bar sequence
43. Empty bars vector → empty output

### normalize_features() tests (~8):
44. Single row → all zeros (z-score of single point is 0, or 0/0→0)
45. Two identical rows → all zeros
46. Known z-scores for a 3-row sequence
47. NaN replaced with 0 before normalization
48. Clip to [-5, +5]
49. Window=1 → all zeros (single-element windows)
50. Large window acts as expanding window
51. Zero-std columns → 0

### assemble_lookback() tests (~6):
52. h=1 → same as input (but float32)
53. h=2, 3 rows → 2 output rows
54. Output row is concatenation of h consecutive input rows
55. n_rows < h → empty output
56. Output is float32
57. Shape matches (n_rows - h + 1, n_cols * h)

### Integration tests (~3):
58. compute → normalize → assemble pipeline produces correct output shape
59. Precondition: bars.size() != accums.size() → assert/throw
60. Bars with accums from BarBuilder → features consistent with Python

## Acceptance Criteria

- All existing C++ tests still pass (500 + 15 skipped)
- New tests pass (~60 cases)
- Feature column order matches Python `feature_pipeline.py` exactly
- Constants match Python `lob_rl.barrier.__init__`
- normalize_features handles NaN and zero-std correctly
- assemble_lookback outputs float32
