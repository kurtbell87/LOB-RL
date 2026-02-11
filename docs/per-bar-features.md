# Per-Bar Feature Cache

## Overview

Extend the barrier precompute pipeline to output **per-bar normalized features** alongside the existing lookback-assembled features. Sequence models need `(n_valid_bars, 22)` per-bar features rather than `(n_usable, 220)` lookback-assembled features.

## What to Build

### 1. C++ Changes: Add `bar_features` to `BarrierPrecomputedDay`

**File:** `include/lob/barrier/barrier_precompute.h`

Add a new field to the `BarrierPrecomputedDay` struct:

```cpp
// Per-bar normalized features (n_trimmed rows, each N_FEATURES floats, row-major)
// After warmup trimming and z-score normalization, BEFORE lookback assembly.
std::vector<float> bar_features;
int n_trimmed = 0;  // number of rows in bar_features (n_bars - REALIZED_VOL_WARMUP)
```

**File:** `src/barrier/barrier_precompute.cpp`

In `barrier_precompute_impl()`, after normalizing features (step 3) and before lookback assembly (step 4):
- Convert the `normed` vector (doubles) to float32
- Store as `day.bar_features`
- Set `day.n_trimmed = n_trimmed`

The normalized features already exist as `normed` in the current code (line 144). We just need to save them before they're consumed by `assemble_lookback()`.

### 2. Pybind11 Binding: Expose `bar_features`

**File:** `src/bindings/bindings.cpp`

Add to the result dict in the `barrier_precompute` lambda:

```python
result["bar_features"]  # ndarray(n_trimmed, N_FEATURES) float32
result["n_trimmed"]     # int
```

Use the existing `to_numpy_2d()` helper to create a 2D numpy array from the flat float vector.

### 3. Python Cache Script: Save `bar_features`

**File:** `scripts/precompute_barrier_cache.py`

The `process_session()` function already does `np.savez_compressed(output_path, **result)`. Since `bar_features` will be in the result dict returned by `barrier_precompute()`, it will be saved automatically. No code changes needed in the script itself — the new key flows through.

Verify: load an existing .npz and confirm `bar_features` key is present with shape `(n_trimmed, 22)`.

### 4. Python Loader: Add `load_session_features()`

**File:** `python/lob_rl/barrier/first_passage_analysis.py`

Add a new function to load per-bar features and labels for sequence model consumption:

```python
def load_session_features(cache_dir: str) -> list[dict]:
    """Load per-bar features and labels for each session.

    Returns a list of session dicts, each containing:
        - features: ndarray(n_bars, 22) float32 — per-bar normalized features
        - Y_long: ndarray(n_bars,) bool — long barrier hit
        - Y_short: ndarray(n_bars,) bool — short barrier hit
        - date: str — YYYYMMDD from filename

    Sessions are sorted chronologically by date.
    Only includes sessions that have the 'bar_features' key.

    The label arrays are aligned with bar_features: both are trimmed to
    exclude the REALIZED_VOL_WARMUP=19 warmup bars from the start of session.
    So n_bars = n_trimmed = original_n_bars - 19.
    """
```

**Label alignment:** The `bar_features` array has `n_trimmed = n_bars - 19` rows (warmup trimmed). The label arrays (`label_values`, `short_label_values`) have `n_bars` rows. The loader must trim the first 19 label rows to align: `label_values[19:]`.

**Y_long/Y_short semantics:** Same as `load_binary_labels()`:
- `Y_long = (label_values == 1)` — long hits profit barrier
- `Y_short = (short_label_values == -1)` — short hits profit barrier

### 5. Temporal Split Helper

Add or expose a helper for splitting sessions chronologically:

```python
def temporal_split(sessions: list[dict], train_frac=0.6, val_frac=0.2) -> tuple[list, list, list]:
    """Split sessions into train/val/test by chronological order.

    Returns (train_sessions, val_sessions, test_sessions).
    Sessions are already sorted by date from load_session_features().

    Default: 60/20/20 split matching exp-006.
    """
```

If `temporal_split()` already exists in `first_passage_analysis.py`, reuse it. If it only works with the flat `session_boundaries` format, add a wrapper that works with session dicts.

## Edge Cases

1. **Sessions with insufficient data:** If `n_trimmed == 0` (fewer than 20 bars), the session should be skipped by the loader.
2. **Missing `bar_features` key:** Old cache files won't have this key. The loader should skip them gracefully.
3. **Label alignment off-by-one:** Labels are 0-indexed from bar 0. After trimming warmup, bar_features[0] corresponds to bar 19's features and bar 19's label. Test this explicitly.
4. **Empty sessions:** If a session has bars but no valid labels after trimming (e.g., all timeouts at edges), include it — the model can learn from timeouts too.

## Acceptance Criteria

1. `BarrierPrecomputedDay` has `bar_features` (float32 vector) and `n_trimmed` (int) fields.
2. `barrier_precompute()` populates `bar_features` with z-score normalized features (same values as the per-row features that go into lookback assembly).
3. Python binding returns `bar_features` as `ndarray(n_trimmed, 22)` float32 and `n_trimmed` as int.
4. `load_session_features()` returns a list of session dicts with aligned features and labels.
5. `temporal_split()` works with session dict lists.
6. All existing tests pass (no regressions).
7. The values in `bar_features[i]` match the values in `features[i - lookback + 1, 0:22]` (the most recent bar's features in the first usable lookback window). This is the critical alignment test.

## Tests (~15-20)

### C++ Tests
- `bar_features` populated with correct shape (n_trimmed, N_FEATURES)
- `n_trimmed == n_bars - REALIZED_VOL_WARMUP`
- `bar_features` values match first N_FEATURES of each row in normalized feature matrix
- `bar_features` is empty when n_bars <= REALIZED_VOL_WARMUP
- `bar_features` values are float32, in [-5, 5] range (clipped)

### Python Binding Tests
- `barrier_precompute()` result dict has `bar_features` key with shape (n_trimmed, 22)
- `bar_features` dtype is float32
- `n_trimmed` matches bar_features.shape[0]

### Python Loader Tests
- `load_session_features()` returns list of dicts with correct keys
- Features shape is (n_bars, 22) per session
- Y_long and Y_short are boolean arrays aligned with features
- Sessions sorted chronologically
- Sessions with missing `bar_features` key are skipped
- Sessions with n_trimmed == 0 are skipped
- Label alignment: bar_features[0] corresponds to label_values[WARMUP]
- `temporal_split()` splits correctly (60/20/20)
- `temporal_split()` preserves chronological order

## Key Files

| File | Action |
|------|--------|
| `include/lob/barrier/barrier_precompute.h` | **MODIFY** — add `bar_features`, `n_trimmed` fields |
| `src/barrier/barrier_precompute.cpp` | **MODIFY** — populate `bar_features` after normalization |
| `src/bindings/bindings.cpp` | **MODIFY** — expose `bar_features` and `n_trimmed` in dict |
| `python/lob_rl/barrier/first_passage_analysis.py` | **MODIFY** — add `load_session_features()`, `temporal_split()` for session dicts |
| `scripts/precompute_barrier_cache.py` | **NO CHANGE** — `bar_features` flows through `**result` automatically |

## Dependencies

- `feature_compute.h`: `N_FEATURES`, `REALIZED_VOL_WARMUP` constants
- `barrier_precompute.h`: `BarrierPrecomputedDay` struct
- Existing normalization: `normalize_features()` produces z-score normalized, [-5,5] clipped values
- Existing label alignment: labels are indexed from bar 0, features start at bar REALIZED_VOL_WARMUP
