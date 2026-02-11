# Barrier Precompute Pipeline Integration + pybind11 Binding

## Context

Cycles 1-4 built all C++ components: Book extensions, BarBuilder, feature computation, and label computation. This final cycle wires them together into a single `barrier_precompute()` function, exposes it to Python via pybind11, and updates the precompute script to use C++ instead of Python.

### Existing Infrastructure

- `BarBuilder(bar_size, cfg)` — single-pass MBO stream → `TradeBar` + `BarBookAccum` vectors (`include/lob/barrier/bar_builder.h`)
- `compute_bar_features(bars, accums, rth_open, rth_close)` — 22 raw features (`include/lob/barrier/feature_compute.h`)
- `normalize_features(raw, n_rows, n_cols, window)` — z-score normalization (`include/lob/barrier/feature_compute.h`)
- `assemble_lookback(normed, n_rows, n_cols, h)` — float32 lookback windows (`include/lob/barrier/feature_compute.h`)
- `compute_labels(bars, a, b, t_max, tick_size)` — triple-barrier labels (`include/lob/barrier/barrier_label.h`)
- `DbnFileSource(path, instrument_id)` — reads `.dbn.zst` files (`src/data/dbn_file_source.h`)
- `SessionConfig::default_rth()` — default RTH config (`include/lob/session.h`)
- Existing pybind11 bindings in `src/bindings/bindings.cpp`

## Requirements

### 1. BarrierPrecomputedDay Struct

**File:** `include/lob/barrier/barrier_precompute.h`

```cpp
struct BarrierPrecomputedDay {
    int n_bars;
    int n_usable;           // rows in features after warmup + lookback
    int bar_size;
    int lookback;

    // Bars: OHLCV arrays
    std::vector<double> bar_open, bar_high, bar_low, bar_close, bar_vwap;
    std::vector<int> bar_volume;
    std::vector<uint64_t> bar_t_start, bar_t_end;

    // Trade sequences (flat + offsets)
    std::vector<double> trade_prices;
    std::vector<int> trade_sizes;
    std::vector<int64_t> bar_trade_offsets;  // size n_bars + 1

    // Labels
    std::vector<int> label_values;   // +1, -1, 0
    std::vector<int> label_tau;
    std::vector<int> label_resolution_bar;

    // Features (final output)
    std::vector<float> features;     // (n_usable, N_FEATURES * lookback) row-major

    // Metadata
    int n_features;                  // N_FEATURES constant
};
```

### 2. barrier_precompute() Function

```cpp
BarrierPrecomputedDay barrier_precompute(
    const std::string& path,
    uint32_t instrument_id,
    int bar_size = 500,
    int lookback = 10,
    int a = 20,
    int b = 10,
    int t_max = 40);
```

**Algorithm:**
1. Create `DbnFileSource(path, instrument_id)` and `BarBuilder(bar_size, SessionConfig::default_rth())`
2. Stream all messages: `while (source.next(msg)) builder.process(msg);`
3. `builder.flush()` for any partial bar
4. Get `bars = builder.bars()`, `accums = builder.accums()`
5. If `bars.size() < lookback + 1`: return minimal struct with `n_bars = bars.size()`, `n_usable = 0`, empty features
6. Compute raw features: `compute_bar_features(bars, accums, builder.rth_open_ns(), builder.rth_close_ns())`
7. Drop realized-vol warmup rows (first `REALIZED_VOL_WARMUP` bars where col 8 is NaN)
8. Normalize: `normalize_features(raw_trimmed, n_trimmed, N_FEATURES, 2000)`
9. Assemble lookback: `assemble_lookback(normed, n_trimmed, N_FEATURES, lookback)`
10. Compute labels: `compute_labels(bars, a, b, t_max)`
11. Pack all bar data + labels + features into `BarrierPrecomputedDay`

**Realized vol warmup handling:** Find the first bar index where col 8 is not NaN (should be `REALIZED_VOL_WARMUP = 19`). Trim raw features from that index. The `n_usable` count reflects the final output after both warmup trimming and lookback assembly.

### 3. pybind11 Binding

Add to `src/bindings/bindings.cpp`:

```cpp
m.def("barrier_precompute", [...] -> py::dict {
    // Returns dict with same keys as Python process_session()
}, py::arg("path"), py::arg("instrument_id"),
   py::arg("bar_size") = 500, py::arg("lookback") = 10,
   py::arg("a") = 20, py::arg("b") = 10, py::arg("t_max") = 40);
```

The returned `py::dict` must contain exactly these keys (matching Python `process_session()` output):
- `features` — np.float32 array of shape (n_usable, N_FEATURES * lookback)
- `bar_open`, `bar_high`, `bar_low`, `bar_close` — np.float64 arrays (n_bars,)
- `bar_volume` — np.int32 array (n_bars,)
- `bar_vwap` — np.float64 array (n_bars,)
- `bar_t_start`, `bar_t_end` — np.int64 arrays (n_bars,)
- `trade_prices` — np.float64 flat array
- `trade_sizes` — np.int32 flat array
- `bar_trade_offsets` — np.int64 array (n_bars + 1,)
- `label_values` — np.int8 array (n_bars,)
- `label_tau` — np.int32 array (n_bars,)
- `label_resolution_bar` — np.int32 array (n_bars,)
- `bar_size`, `lookback`, `a`, `b`, `t_max`, `n_bars`, `n_usable`, `n_features` — scalar np arrays

Returns `None` if insufficient data (`n_bars < lookback + 1`).

### 4. Update precompute_barrier_cache.py

Replace the Python `process_session()` function body with a call to `lob_rl_core.barrier_precompute()`. Keep the Python wrapper for:
- CLI argument parsing
- Roll calendar lookup
- Parallel execution via ProcessPoolExecutor
- np.savez_compressed() for cache saving
- Summary statistics (p_plus, p_minus, p_zero, tiebreak_freq)

The updated `process_session()`:
```python
def process_session(filepath, instrument_id, bar_size, lookback, a, b, t_max):
    import lob_rl_core
    result = lob_rl_core.barrier_precompute(
        str(filepath), instrument_id,
        bar_size=bar_size, lookback=lookback, a=a, b=b, t_max=t_max)
    if result is None:
        return None
    # Add summary stats computed from labels
    label_values = result["label_values"]
    n = len(label_values)
    result["p_plus"] = np.array(np.sum(label_values == 1) / n, dtype=np.float64)
    result["p_minus"] = np.array(np.sum(label_values == -1) / n, dtype=np.float64)
    result["p_zero"] = np.array(np.sum(label_values == 0) / n, dtype=np.float64)
    result["tiebreak_freq"] = np.array(0.0, dtype=np.float64)  # C++ doesn't track resolution_type
    return result
```

### 5. CMakeLists.txt Changes

Add `src/barrier/barrier_precompute.cpp` to `lob_core` sources.

## Files to Change

| File | Change |
|------|--------|
| `include/lob/barrier/barrier_precompute.h` | **NEW** — BarrierPrecomputedDay struct + barrier_precompute() declaration |
| `src/barrier/barrier_precompute.cpp` | **NEW** — Implementation |
| `src/bindings/bindings.cpp` | Add barrier_precompute() binding |
| `scripts/precompute_barrier_cache.py` | Replace Python process_session() with C++ call |
| `CMakeLists.txt` | Add src/barrier/barrier_precompute.cpp and test file |
| `tests/test_barrier_precompute.cpp` | **NEW** — Tests |

## Test Plan

### C++ unit tests (~10):
1. Empty stream → n_bars=0, n_usable=0
2. Insufficient bars (< lookback+1) → n_usable=0, empty features
3. Bar arrays populated correctly (open, high, low, close, volume, vwap)
4. Timestamps populated correctly (t_start, t_end)
5. Trade prices/sizes flat arrays + offsets correct
6. Labels populated: label_values, label_tau, label_resolution_bar all n_bars length
7. Features shape: n_usable rows, N_FEATURES * lookback columns
8. n_features equals N_FEATURES constant
9. Default parameters: bar_size=500, lookback=10, a=20, b=10, t_max=40
10. All features values are finite (no NaN after normalization)

### Integration tests with synthetic data (~5):
11. Known MBO sequence → known bar count
12. Feature dimensions match expected
13. Labels are valid (+1, -1, or 0) with tau >= 1
14. Trade offsets monotonically increasing, last == total trades
15. Bar close prices form valid price series (no NaN)

## Acceptance Criteria

- All existing C++ tests still pass (612 + 15 skipped)
- New C++ tests pass (~15 cases)
- All 2062 Python tests still pass
- `barrier_precompute()` produces dict compatible with `np.savez_compressed()`
- `precompute_barrier_cache.py` updated to use C++ backend
- Rebuild + re-precompute pipeline works end-to-end
