# Precompute Cache — Save to Disk, Load Everywhere

## Problem

Every training run re-precomputes all days from raw .bin files (C++ book engine). With 8 SubprocVecEnv workers, that's 160 redundant precompute() calls per run. The output is deterministic — same .bin + same SessionConfig = same arrays every time.

This makes hyperparameter iteration slow: ~30-60 sec startup before training even begins.

## Solution

Add a `precompute_cache.py` script that precomputes all days once and saves to disk as `.npz` files. Modify `MultiDayEnv` and `train.py` to load from cache instead of calling C++ precompute.

## Requirements

### 1. `scripts/precompute_cache.py` — CLI tool

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py \
  --data-dir ../data/mes --out ../cache/mes/
```

- Reads manifest.json from `--data-dir`
- Calls `lob_rl_core.precompute(path, session_config)` once per day
- Saves each day as `{date}.npz` containing `obs`, `mid`, `spread` arrays
- Skips days that already exist in cache (unless `--force`)
- Prints summary: how many days cached, array shapes, total size on disk
- Default SessionConfig is `default_rth()` (same as train.py)

### 2. `PrecomputedEnv.from_cache()` — new classmethod

```python
PrecomputedEnv.from_cache(npz_path, reward_mode="pnl_delta", lambda_=0.0,
                          execution_cost=False, participation_bonus=0.0,
                          step_interval=1)
```

- Loads `obs`, `mid`, `spread` from `.npz` file
- Passes to existing `__init__()` (which handles step_interval, temporal features, etc.)
- This is the hot path — must be fast (numpy load + array slicing only)

### 3. `MultiDayEnv` — accept cache dir

```python
MultiDayEnv(file_paths=None,           # existing: list of .bin paths
            cache_dir=None,            # NEW: path to cache dir with .npz files
            session_config=None,       # ignored when cache_dir is set
            steps_per_episode=50,
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False,
            participation_bonus=0.0,
            step_interval=1)
```

- Exactly one of `file_paths` or `cache_dir` must be provided (raise ValueError if both or neither)
- When `cache_dir` is set:
  - Glob for `*.npz` files, sorted by name (date order)
  - Load via `PrecomputedEnv.from_cache()` instead of `PrecomputedEnv.from_file()`
  - Skip `.npz` files that fail to load (same pattern as current .bin skip)
- When `file_paths` is set: existing behavior unchanged

### 4. `scripts/train.py` — `--cache-dir` flag

```bash
python scripts/train.py --cache-dir cache/mes/ --step-interval 10 --execution-cost
```

- New flag: `--cache-dir` (mutually exclusive with `--data-dir`)
- When `--cache-dir` is set:
  - Pass `cache_dir=` to `MultiDayEnv` instead of `file_paths=`
  - Train/val/test split by sorted .npz file order (same logic as current .bin split)
  - Skip manifest.json loading entirely
- When `--data-dir` is set: existing behavior unchanged

### 5. `evaluate_sortino()` — support cache

- Accept `cache_path=None` parameter (path to single .npz file)
- When set, use `PrecomputedEnv.from_cache()` instead of `PrecomputedEnv.from_file()`

## Cache file format

Each `.npz` file contains:
- `obs`: float32 array, shape (N, 43) — C++ feature rows (no position, no temporal)
- `mid`: float64 array, shape (N,) — mid prices
- `spread`: float64 array, shape (N,) — bid-ask spreads

Temporal features and step_interval subsampling are NOT cached — they're computed at env construction since they depend on runtime parameters.

## Edge cases

- `--cache-dir` with empty directory → error "No .npz files found"
- `.npz` file with wrong keys → skip with warning (same as current .bin skip)
- `--cache-dir` and `--data-dir` both set → error
- Neither `--cache-dir` nor `--data-dir` → error
- `--force` flag on precompute_cache.py → re-cache even if .npz exists
- Holiday .bin files that produce 0 steps → not cached (skipped with message)

## Acceptance criteria

1. `precompute_cache.py` creates one `.npz` per valid trading day
2. `PrecomputedEnv.from_cache()` produces identical behavior to `from_file()` for the same day
3. `MultiDayEnv(cache_dir=...)` produces identical obs/rewards as `MultiDayEnv(file_paths=...)`
4. `train.py --cache-dir` runs training with zero C++ precompute calls
5. Cache files are reusable across step_interval values (temporal features computed at runtime)
6. Existing `--data-dir` workflow is unchanged (backward compatible)

## Scope

- Python only — no C++ changes
- Files to create: `scripts/precompute_cache.py`
- Files to modify: `precomputed_env.py`, `multi_day_env.py`, `train.py`
