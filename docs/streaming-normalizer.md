# Phase 9: Streaming Feature Normalization

## Problem

LOB-RL's precomputed path normalizes features offline using `normalize_features()` in `src/barrier/feature_compute.cpp`, which requires all bars to be available. The reactive path (OrderSimulationEnv, Phase 8) needs online normalization that produces identical z-scores as bars arrive one at a time. This is a prerequisite for reactive feature computation.

## Reference Implementation

From `src/barrier/feature_compute.cpp` (`normalize_features()`):
- For each bar (row), compute the mean and population variance (ddof=0) over a rolling window of up to `window` bars
- Window: rows `[max(0, row - window + 1), row]` inclusive
- NaN values are replaced with 0.0 before normalization
- Z-score: `(x - mean) / std_dev`, clipped to `[-5, 5]`
- If `std_dev == 0`, output is 0

## What to Build

### StreamingNormalizer (C++)

**File:** `include/lob/barrier/streaming_normalizer.h` (NEW)
**File:** `src/barrier/streaming_normalizer.cpp` (NEW)

```cpp
class StreamingNormalizer {
public:
    /// @param n_features Number of feature columns
    /// @param window Rolling window size (0 = expanding window, default 2000)
    explicit StreamingNormalizer(int n_features, int window = 2000);

    /// Normalize a single bar's raw feature vector (in-place-safe variant below).
    /// Returns the z-scored features clipped to [-5, 5].
    /// NaN values in `raw_bar` are replaced with 0 before normalization.
    std::vector<double> normalize(const std::vector<double>& raw_bar);

    /// Reset all internal state (for new session).
    void reset();

    /// Number of bars seen so far.
    int bars_seen() const;

    /// Number of feature columns.
    int n_features() const;
};
```

**Internal state (per feature column):**
- Circular buffer of the last `window` values (or all values if window == 0)
- Running count of values in the buffer
- For efficient mean/variance: store the raw values and compute on each call
  - With window <= 2000, a simple O(window) pass is fast enough per bar
  - Welford's algorithm is not needed — the reference implementation uses a two-pass approach over the window

**Algorithm (matches reference exactly):**
```
For each column j:
    1. Replace NaN with 0 in raw_bar[j]
    2. Append raw_bar[j] to circular buffer[j]
    3. Determine window: last min(window, bars_seen) values
    4. Compute mean = sum(window_values) / n_window
    5. Compute pop_variance = sum((x - mean)^2) / n_window
    6. std_dev = sqrt(pop_variance)
    7. z = (raw_bar[j] - mean) / std_dev, or 0 if std_dev == 0
    8. Clip z to [-5, 5]
```

This must produce **identical output** to calling `normalize_features()` on all bars up to and including the current one and reading the last row.

### Python Bindings

**File:** `src/bindings/bindings.cpp`

```python
norm = core.StreamingNormalizer(n_features=22, window=2000)
z_scores = norm.normalize([0.5, 1.2, ...])  # list[float] -> list[float]
norm.reset()
norm.bars_seen()  # int
norm.n_features()  # int
```

## Edge Cases

- **First bar:** Mean = raw value, variance = 0, z-score = 0 for all features.
- **Constant feature column:** All values identical → variance = 0 → z-score = 0.
- **NaN in input:** Replaced with 0 before any computation.
- **Window = 0:** Expanding window (all bars seen).
- **Window = 1:** Mean = current value, variance = 0, z-score = 0 always.
- **n_features mismatch:** `normalize()` throws if `raw_bar.size() != n_features`.
- **Very large values:** Z-score clipped to [-5, 5].
- **After reset():** State returns to initial (bars_seen = 0).

## Acceptance Criteria

1. `StreamingNormalizer` exists with the API above.
2. **Bit-exact match:** For any sequence of N bars, the streaming normalizer's output at bar i must equal `normalize_features(raw[0:i+1], i+1, n_features, window)[i * n_features : (i+1) * n_features]` from the reference C++ implementation. Tolerance: `1e-12`.
3. Works correctly with window = 2000 (default), window = 0 (expanding), and small windows.
4. Python bindings work.
5. All existing tests pass.
6. ~15 new tests.

## Files to Create

| File | Role |
|------|------|
| `include/lob/barrier/streaming_normalizer.h` | Header |
| `src/barrier/streaming_normalizer.cpp` | Implementation |

## Files to Modify

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add streaming_normalizer.cpp to lob_core target |
| `src/bindings/bindings.cpp` | Add StreamingNormalizer binding |

## Test Categories

1. **Single-bar normalization (~3):** First bar → all zeros. Second bar → correct z-scores.
2. **Window behavior (~3):** Window = 5 with 10 bars (verify old bars drop off). Window = 0 (expanding).
3. **Bit-exact regression (~4):** Generate random sequences of 50-200 bars, compare streaming output bar-by-bar against `normalize_features()`.
4. **Edge cases (~3):** NaN handling, constant column, n_features mismatch.
5. **Python bindings (~2):** Create, normalize, reset, check bars_seen.

## Dependencies

- References `normalize_features()` from `src/barrier/feature_compute.cpp` for regression tests
- Independent of Phases 1-8 (can be built in parallel)
