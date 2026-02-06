# B1: Expose BinaryFileSource and SessionConfig in Python Bindings

## Problem

Python bindings (`src/bindings/bindings.cpp`) only expose `LOBEnv` with a hardcoded `SyntheticSource`. There is no way from Python to:
- Create a `LOBEnv` backed by `BinaryFileSource` (i.e., pass a file path)
- Set `SessionConfig` (RTH open/close times, warmup messages)
- Configure `steps_per_episode`

The entire Step 2 C++ infrastructure is unreachable from the training layer.

## What to Build

Extend the pybind11 bindings to expose additional constructors and the `SessionConfig` struct.

## Interface

### `SessionConfig` (new Python class)

```python
cfg = lob_rl_core.SessionConfig()
cfg.rth_open_ns = 48_600_000_000_000   # 13:30 UTC in ns
cfg.rth_close_ns = 72_000_000_000_000  # 20:00 UTC in ns
cfg.warmup_messages = -1               # -1 = use all pre-RTH

cfg = lob_rl_core.SessionConfig.default_rth()  # factory method
```

### `LOBEnv` (additional constructors)

Keep existing default constructor (SyntheticSource). Add:

1. **File-based constructor**: `LOBEnv(file_path: str, steps_per_episode: int = 50)`
   - Creates `BinaryFileSource` from the file path
   - Uses the basic (non-session) constructor

2. **File + session constructor**: `LOBEnv(file_path: str, session_config: SessionConfig, steps_per_episode: int = 50)`
   - Creates `BinaryFileSource` from the file path
   - Uses the session-aware constructor

3. **SyntheticSource with steps_per_episode**: `LOBEnv(steps_per_episode: int)` — optional, allows configuring step count with synthetic data

### `LOBEnv` additional attributes (read-only)

- `steps_per_episode` — int, read-only property

## Requirements

1. `SessionConfig` exposed with all three fields readable/writable
2. `SessionConfig.default_rth()` static method works from Python
3. `LOBEnv(file_path)` creates env backed by `BinaryFileSource`
4. `LOBEnv(file_path, session_config, steps_per_episode)` creates session-aware env
5. `LOBEnv(file_path, steps_per_episode)` creates basic env with configurable steps
6. Default `LOBEnv()` constructor still works (backward compatible)
7. All constructors produce a functional env (step/reset work)

## Acceptance Criteria

- Test: Create `LOBEnv()` (default) — step/reset works (existing behavior)
- Test: Create `LOBEnv("path/to/file.bin")` — step/reset works with file data
- Test: Create `LOBEnv("path/to/file.bin", config, 100)` — step/reset works with session
- Test: `SessionConfig.default_rth()` returns config with correct values
- Test: `SessionConfig` fields are readable and writable
- Test: Invalid file path raises an exception
- Test: All existing Python tests still pass
