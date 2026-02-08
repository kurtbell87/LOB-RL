# Shuffle Split for train.py

## What to build

Add `--shuffle-split` and `--seed` CLI flags to `scripts/train.py` so that the train/val/test split can be randomized instead of strictly chronological.

## Why

The current chronological split (Jan-Jul train, Aug-Dec test) conflates overfitting with regime shift (2022 H1 vs H2 were very different markets). Since each episode is one self-contained trading day (no cross-day state in `MultiDayEnv.reset()`), random splits are valid and give a fairer assessment of generalization.

## Requirements

### New CLI arguments

1. `--shuffle-split` — `action='store_true'`, default `False`. When set, `all_files` is shuffled before splitting into train/val/test.
2. `--seed` — `type=int`, default `42`. Seeds the RNG used for shuffling. Only meaningful when `--shuffle-split` is set.

### Behavior

- **Default (no `--shuffle-split`):** Behavior is unchanged — chronological split.
- **With `--shuffle-split`:** After loading `all_files` (line 238 for `cache_dir` path, line 268 for `data_dir` path), shuffle `all_files` in-place using `random.Random(args.seed).shuffle(all_files)` before the train/val/test split.
- **Date printing:** After the split, print the date ranges for each set so users can verify reproducibility. Format:
  ```
  Train dates: 2022-01-03, 2022-01-05, ... (N days)
  Val dates: 2022-03-15, 2022-06-22, ... (N days)
  Test dates: 2022-02-10, 2022-08-01, ... (N days)
  ```
  Print dates for both `cache_dir` and `data_dir` paths. The date is `file_tuple[0]` (first element of each tuple in `all_files`).

### Implementation scope

- Only `scripts/train.py` is modified.
- No changes to `MultiDayEnv`, `PrecomputedEnv`, `BarLevelEnv`, or any other module.
- Import `random` from stdlib (already available, just not imported yet).

## Edge cases

- `--seed` without `--shuffle-split` — accepted silently (no error), but has no effect.
- `--shuffle-split` with fewer than 3 files — works fine, some splits may be empty (existing behavior handles empty val/test).
- Reproducibility — same `--seed` must produce identical splits.

## Acceptance criteria

1. `--shuffle-split` flag exists with `action='store_true'`.
2. `--seed` flag exists with `type=int`, `default=42`.
3. When `--shuffle-split` is set, `all_files` is shuffled before split (both `cache_dir` and `data_dir` paths).
4. Shuffling uses `random.Random(args.seed)` for reproducibility (not `numpy.random`).
5. Default behavior (no `--shuffle-split`) is unchanged — chronological order preserved.
6. Train/val/test date lists are printed after the split.
7. `import random` is present in `train.py`.
8. No other files are modified.

## Test approach

Tests should follow the existing pattern in `test_training_pipeline_v2.py`: read `train.py` source and use regex/string matching to verify structure. Additionally, test the shuffle logic functionally by importing and calling the shuffle with a known seed to verify determinism.
