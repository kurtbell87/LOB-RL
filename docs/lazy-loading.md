# Lazy Loading for MultiDayEnv

## Problem

`MultiDayEnv.__init__()` eagerly loads ALL `.npz` files into memory via `_load_from_cache_dir()`. With 8 `SubprocVecEnv` workers × 249 days × ~37 MB/day = ~74 GB baseline RAM. Additionally, `train.py` passes the full `cache_dir` to all workers instead of just the train-split files.

## Requirements

### 1. MultiDayEnv lazy loading (cache_dir mode)

When constructed with `cache_dir`, `MultiDayEnv` must:

- **Store file paths only** — `_load_from_cache_dir()` must NOT load arrays into memory. It should only collect and validate the list of `.npz` file paths and extract `instrument_id` from each file (this is a single uint32, negligible memory).
- **Load on `reset()`** — When `reset()` selects the next day, load that day's `.npz` arrays and create the inner env (`PrecomputedEnv` or `BarLevelEnv`). The previous day's arrays should be released (no reference kept).
- **`self._precomputed_days` is removed** — Replace with `self._npz_paths: list[str]` (file paths) and `self._contract_ids: list[int | None]` (instrument_id per day, extracted at init).
- **Bar-size filtering at init** — `_filter_days_for_bar_size()` currently loads all data to check bar counts. Instead, load each `.npz` temporarily, check bar count, record the path if valid, then release the arrays. This is a one-time init cost that does NOT persist arrays in memory.
- **`file_paths` mode unchanged** — When constructed with `file_paths=` (raw `.bin` files requiring C++ precompute), the behavior is unchanged — arrays are still held in memory since there's no file to reload from. This mode is not used in training.

### 2. train.py train-split filtering

When using `--cache-dir`, `train.py` currently passes the full `cache_dir` path to `MultiDayEnv`. This means all 249 days are loaded by every worker, not just the 170 train days. Fix:

- **Replace `cache_dir` with explicit file list** — After splitting `.npz` files into train/val/test, pass only the train-split file paths to each `SubprocVecEnv` worker.
- **New MultiDayEnv parameter: `cache_files`** — Accept an explicit list of `.npz` file paths (as an alternative to `cache_dir`). When `cache_files` is provided, use those exact files instead of globbing a directory.
- **Mutual exclusivity** — Exactly one of `file_paths`, `cache_dir`, or `cache_files` must be provided.

### 3. Backward compatibility

- `MultiDayEnv(cache_dir=...)` must still work exactly as before (glob `*.npz`, sort, use all files) — just without holding arrays in memory.
- `MultiDayEnv(file_paths=...)` must still work exactly as before (eagerly precompute and hold arrays).
- All existing tests must pass without modification.
- `contract_ids` property must still return the full list of instrument_ids.

## Interface changes

### MultiDayEnv constructor

```python
MultiDayEnv(
    file_paths=None,        # list of .bin paths (eager load, unchanged)
    cache_dir=None,         # path to dir with .npz files (lazy load)
    cache_files=None,       # explicit list of .npz paths (lazy load)
    session_config=None,
    steps_per_episode=50,
    reward_mode="pnl_delta",
    lambda_=0.0,
    shuffle=False,
    seed=None,
    execution_cost=False,
    participation_bonus=0.0,
    step_interval=1,
    bar_size=0,
)
# Exactly one of file_paths, cache_dir, or cache_files must be provided.
```

### train.py make_train_env

```python
# Before:
make_train_env(cache_dir=train_cache_dir, ...)

# After:
make_train_env(cache_files=train_npz_paths, ...)
```

## Acceptance criteria

1. **Memory**: A `MultiDayEnv(cache_dir=...)` with 100 `.npz` files holds no numpy arrays after `__init__()`. Only file paths and instrument_ids are stored.
2. **Correctness**: `reset()` returns the same observations as the eager-load implementation for the same day/seed.
3. **Shuffle**: Shuffle ordering works identically — file ordering, RNG seeding, epoch wraparound all preserved.
4. **Contract tracking**: `contract_ids` property returns instrument_ids extracted at init time (not at reset time).
5. **Bar filtering**: Days with < 2 bars are still filtered out at init (load temporarily, check, discard arrays).
6. **Train split**: `train.py --cache-dir` only passes train-split `.npz` paths to workers, not the full directory.
7. **All existing tests pass** without modification.

## Edge cases

- `.npz` file that fails to load at init (during instrument_id extraction or bar filtering): warn and skip, same as today.
- `.npz` file that fails to load at `reset()` time: raise an error (this is a runtime failure, not a filtering issue).
- `cache_files=[]` (empty list): raise `ValueError`, same as `file_paths=[]`.
- `cache_files` and `cache_dir` both provided: raise `ValueError`.
