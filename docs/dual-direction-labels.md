# Dual-Direction Binary Labels

## What to Build

Add **short-direction labels** to the C++ barrier precompute pipeline. Currently, `barrier_precompute()` computes one set of labels: `compute_labels(bars, a=20, b=10, t_max)` — the "long race" where upper barrier (+20 ticks) = profit, lower barrier (-10 ticks) = stop.

The short race uses **swapped barriers**: `compute_labels(bars, a=b, b=a, t_max)` = `compute_labels(bars, a=10, b=20, t_max)` — upper barrier (+10 ticks) = stop, lower barrier (-20 ticks) = profit.

Binary labels are derived as:
- `Y_long = (label_values == +1)` — long race upper (profit) hit first
- `Y_short = (short_label_values == -1)` — short race lower (profit) hit first

These are **independent** binary predictions, NOT disjoint. Both can be 0, and under certain conditions both could be 1.

## Changes

### 1. `BarrierPrecomputedDay` struct (`include/lob/barrier/barrier_precompute.h`)

Add three new vectors after the existing label vectors:

```cpp
// Short-direction labels (all size n_bars)
std::vector<int> short_label_values;   // +1, -1, 0  (short race)
std::vector<int> short_label_tau;
std::vector<int> short_label_resolution_bar;
```

### 2. `barrier_precompute_impl()` (`src/barrier/barrier_precompute.cpp`)

After the existing `compute_labels(bars, a, b, t_max)` call, add a second call with swapped barriers:

```cpp
// Compute short-direction labels (swap a and b)
auto short_labels = compute_labels(bars, b, a, t_max);
day.short_label_values.resize(n_bars);
day.short_label_tau.resize(n_bars);
day.short_label_resolution_bar.resize(n_bars);
for (int i = 0; i < n_bars; ++i) {
    day.short_label_values[i] = short_labels[i].label;
    day.short_label_tau[i] = short_labels[i].tau;
    day.short_label_resolution_bar[i] = short_labels[i].resolution_bar;
}
```

### 3. Pybind11 bindings (`src/bindings/bindings.cpp`)

Expose three new numpy arrays in the returned dict:

```python
result["short_label_values"]        # int8, shape (n_bars,)
result["short_label_tau"]           # int32, shape (n_bars,)
result["short_label_resolution_bar"] # int32, shape (n_bars,)
```

Same dtype conventions as the existing label arrays.

### 4. `precompute_barrier_cache.py` — `process_session()`

Add short summary statistics:

```python
short_label_values = result["short_label_values"]
result["short_p_plus"] = np.array(np.sum(short_label_values == 1) / n, dtype=np.float64)
result["short_p_minus"] = np.array(np.sum(short_label_values == -1) / n, dtype=np.float64)
result["short_p_zero"] = np.array(np.sum(short_label_values == 0) / n, dtype=np.float64)
```

### 5. `precompute_barrier_cache.py` — `load_session_from_cache()`

Add validation and reconstruction:

- If `"short_label_values"` key is missing from the `.npz`, raise `ValueError` with message: `"Cache missing short labels. Re-precompute with precompute_barrier_cache.py."`
- Reconstruct short `BarrierLabel` objects in the returned dict under key `"short_labels"`.

## Label Semantics (CRITICAL)

| Race | `compute_labels()` call | Profit hit | Stop hit | Y binary |
|------|------------------------|-----------|---------|----------|
| Long | `compute_labels(bars, a=20, b=10)` | +1 (upper, +20 ticks) | -1 (lower, -10 ticks) | `Y_long = (label == +1)` |
| Short | `compute_labels(bars, a=10, b=20)` | -1 (lower, -20 ticks) | +1 (upper, +10 ticks) | `Y_short = (short_label == -1)` |

Under the martingale null with 2:1 reward:risk, E[Y_long] ≈ E[Y_short] ≈ 1/3.

## Tests

### C++ tests (new section in `tests/test_barrier_precompute.cpp`, ~10 tests)

1. **Short label arrays populated with correct size** — After `barrier_precompute()`, `day.short_label_values.size() == day.n_bars`.
2. **Short label tau/resolution_bar arrays populated** — Same size as n_bars.
3. **Short label values in {-1, 0, +1}** — All values are valid.
4. **Short label tau >= 1** — No zero-tau labels.
5. **Short label resolution_bar >= bar_index** — Resolution is at or after the bar.
6. **Known synthetic path: verify short label** — Construct a price path that clearly goes DOWN (falls past short profit barrier before hitting short stop). The short label should be -1 (lower hit = profit for short). The long label for the same path should be -1 (lower hit = stop for long).
7. **Known synthetic path: upward** — Price goes UP past long profit barrier. Long label = +1. Short label = +1 (upper hit = stop for short).
8. **Symmetric case (a==b)** — When `a==b`, the long and short races have identical barrier geometry. `label_values` and `short_label_values` should be identical.
9. **Short labels differ from long labels** — With default a=20, b=10 (asymmetric), short and long labels should NOT be identical arrays.
10. **Empty stream produces empty short labels** — `day.short_label_values` is empty when n_bars == 0.

### Python binding tests (in existing or new test file, ~8 tests)

1. **Short keys exist in returned dict** — `"short_label_values"`, `"short_label_tau"`, `"short_label_resolution_bar"` all present.
2. **Short label dtypes correct** — `short_label_values` is int8, `short_label_tau` is int32, `short_label_resolution_bar` is int32.
3. **Short label shapes correct** — All have shape `(n_bars,)`.
4. **Binary derivation: Y_long** — `Y_long = (result["label_values"] == 1)` produces a boolean array.
5. **Binary derivation: Y_short** — `Y_short = (result["short_label_values"] == -1)` produces a boolean array.
6. **Non-complementarity** — `Y_long.mean() + Y_short.mean() != 1.0` (they are independent, not complementary).

### Python cache tests (in test for `precompute_barrier_cache.py`, ~5 tests)

1. **process_session returns short summary stats** — `short_p_plus`, `short_p_minus`, `short_p_zero` keys present with valid float values.
2. **Short summary stats sum to ~1.0** — `short_p_plus + short_p_minus + short_p_zero ≈ 1.0`.
3. **load_session_from_cache returns short_labels** — The `"short_labels"` key exists with correct length.
4. **load_session_from_cache raises ValueError on missing short keys** — Old cache without `short_label_values` raises `ValueError` with descriptive message.
5. **Saved+loaded short labels roundtrip** — Save with `np.savez_compressed`, reload, verify short label values match.

## Edge Cases

- **n_bars == 0**: Both label arrays should be empty.
- **n_bars < lookback + REALIZED_VOL_WARMUP**: Labels still computed (size n_bars), even though features may be empty (n_usable == 0).
- **Timeout labels (label == 0)**: The short race can also timeout. Short timeout means neither +10 (stop) nor -20 (profit) was hit.
- **Both Y_long == 0 and Y_short == 0**: Valid — both races stopped out.

## Acceptance Criteria

- All existing C++ tests (635) still pass.
- All existing Python tests (2070) still pass.
- New C++ tests (~10) pass.
- New Python tests (~13) pass.
- `barrier_precompute()` returns 6 label arrays (3 long + 3 short).
- Cache `.npz` files include all 6 label arrays after rebuild.
