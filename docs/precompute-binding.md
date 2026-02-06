# Spec: Python Binding for `precompute()`

## Overview

Expose the C++ `precompute()` function to Python via pybind11, returning numpy arrays directly. This is the bridge between C++ pre-computation and pure-Python training.

## Interface

Add to `src/bindings/bindings.cpp`:

```python
# Python usage:
import lob_rl_core

cfg = lob_rl_core.SessionConfig.default_rth()
obs, mid, spread, num_steps = lob_rl_core.precompute("/path/to/day.bin", cfg)
# obs: numpy ndarray shape (N, 43), dtype float32
# mid: numpy ndarray shape (N,), dtype float64
# spread: numpy ndarray shape (N,), dtype float64
# num_steps: int
```

## Implementation details

- Include `<pybind11/numpy.h>` for `py::array_t`
- Bind only the string-path overload (the IMessageSource overload is for C++ testing)
- Call `precompute(path, cfg)` to get `PrecomputedDay`
- Convert `obs` vector to `py::array_t<float>` with shape `(num_steps, 43)` — zero-copy if possible, or copy from the vector
- Convert `mid` to `py::array_t<double>` with shape `(num_steps,)`
- Convert `spread` to `py::array_t<double>` with shape `(num_steps,)`
- Return a tuple of `(obs, mid, spread, num_steps)`
- If `num_steps == 0`, return arrays with shape `(0, 43)`, `(0,)`, `(0,)` respectively

## Files to modify

- `src/bindings/bindings.cpp` — add the `precompute` binding

## Files NOT to modify

- `include/lob/precompute.h` — already complete
- `src/env/precompute.cpp` — already complete
- Any existing test files

## Test plan (Python only, file: `python/tests/test_precompute_binding.py`)

### Test 1: precompute returns tuple of 4 elements
- Call `lob_rl_core.precompute(path, cfg)` with a fixture file
- Assert result is a tuple of length 4

### Test 2: obs is numpy array with correct shape and dtype
- Assert `obs.shape == (N, 43)` where N > 0
- Assert `obs.dtype == np.float32`

### Test 3: mid is numpy array with correct shape and dtype
- Assert `mid.shape == (N,)` where N matches obs.shape[0]
- Assert `mid.dtype == np.float64`

### Test 4: spread is numpy array with correct shape and dtype
- Assert `spread.shape == (N,)` where N matches obs.shape[0]
- Assert `spread.dtype == np.float64`

### Test 5: num_steps matches array shapes
- Assert `num_steps == obs.shape[0] == mid.shape[0] == spread.shape[0]`

### Test 6: all obs values are finite
- Assert `np.all(np.isfinite(obs))`

### Test 7: all mid and spread values are finite and positive
- Assert `np.all(np.isfinite(mid))` and `np.all(mid > 0)`
- Assert `np.all(np.isfinite(spread))` and `np.all(spread > 0)`

### Test 8: SessionConfig.default_rth() works for precompute
- This is the standard usage pattern — verify it doesn't crash

### Test 9: custom SessionConfig works
- Create a SessionConfig with custom rth_open_ns/rth_close_ns
- Verify precompute returns results (may differ from default)

### Test 10: invalid file path raises exception
- Call with a nonexistent path — assert it raises RuntimeError or similar

## Test fixtures

Use the same `precompute_rth.bin` fixture created in Phase 1 (located at `tests/fixtures/precompute_rth.bin`). The Python tests need to find this file relative to the project root.

Use this path pattern in conftest.py or directly in tests:
```python
import pathlib
FIXTURE_DIR = pathlib.Path(__file__).parent.parent.parent / "tests" / "fixtures"
```

## Acceptance criteria

- `lob_rl_core.precompute(path, cfg)` callable from Python
- Returns numpy arrays (not Python lists)
- obs shape is 2D `(N, 43)`, mid and spread are 1D `(N,)`
- All existing tests continue to pass
- Release build required for Python tests (ASAN + pybind11 on macOS crashes)
