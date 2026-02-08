# Frame Stacking for train.py

## What to build

Add `--frame-stack N` CLI flag to `scripts/train.py`. When N > 1, insert `VecFrameStack` into the env wrapping chain to give the agent multi-bar temporal context.

## Why

The MLP sees one bar at a time (21-dim obs with bar_size=1000). Frame stacking concatenates N consecutive observations, giving the agent N bars of history without architectural changes. With `--frame-stack 4`, the obs becomes 84-dim (4 × 21).

## Requirements

### New CLI argument

1. `--frame-stack` — `type=int`, `default=1`. When > 1, wraps envs with `VecFrameStack(env, n_stack=N)`. When 1, no wrapping (default behavior unchanged).

### New import

Add `VecFrameStack` to the existing `stable_baselines3.common.vec_env` import line:
```python
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecNormalize
```

### Training env chain

When `frame_stack > 1`, the wrapping order is:
```
SubprocVecEnv → VecFrameStack → VecNormalize
```

The `VecFrameStack` insertion happens after `SubprocVecEnv` creation (after line 275 / line 304) but BEFORE `VecNormalize` (line 306-307). This means:
- After both the `cache_dir` and `data_dir` branches create `env = SubprocVecEnv(...)`, and before the `if not args.no_norm:` block, add:
```python
if args.frame_stack > 1:
    env = VecFrameStack(env, n_stack=args.frame_stack)
```

### Evaluation changes

`evaluate_sortino()` needs a new `frame_stack` parameter (default 1). When `frame_stack > 1`:
- After creating `venv = DummyVecEnv([lambda: env])` (line 143)
- Before loading VecNormalize (line 145), insert:
```python
if frame_stack > 1:
    venv = VecFrameStack(venv, n_stack=frame_stack)
```

Both val and test eval calls (lines 348, 359) must forward `frame_stack=args.frame_stack`.

### Implementation scope

- Only `scripts/train.py` is modified.
- No changes to `MultiDayEnv`, `PrecomputedEnv`, `BarLevelEnv`, or any other module.

## Edge cases

- `--frame-stack 1` — no VecFrameStack wrapper, identical to current behavior.
- `--frame-stack 0` or negative — not validated (user's responsibility); `VecFrameStack` with n_stack=0 would error at SB3 level.
- Obs space change — VecFrameStack automatically updates the observation space. The PPO policy sees the stacked obs.

## Acceptance criteria

1. `--frame-stack` flag exists with `type=int`, `default=1`.
2. `VecFrameStack` is imported from `stable_baselines3.common.vec_env`.
3. When `frame_stack > 1`, training env chain is `SubprocVecEnv → VecFrameStack → VecNormalize`.
4. When `frame_stack == 1`, no `VecFrameStack` wrapper (default behavior unchanged).
5. `evaluate_sortino()` accepts `frame_stack` parameter.
6. Eval wraps `DummyVecEnv` with `VecFrameStack` before `VecNormalize.load()` when `frame_stack > 1`.
7. Both val and test eval calls forward `frame_stack`.
8. No other files are modified.

## Test approach

Tests should follow the existing pattern in `test_training_pipeline_v2.py` and `test_shuffle_split.py`: read `train.py` source and use regex/string matching to verify structure.
