# Step Interval — Coarser Time Sampling

## Problem

The precomputed data captures every BBO change event (~4.6 events/sec for /MES). At this granularity, consecutive mid-price returns are dominated by bid-ask bounce (autocorr = -0.75). The agent learns to exploit this artifact rather than real price dynamics.

## Solution

Add a `step_interval` parameter that subsamples the precomputed arrays. With `step_interval=N`, the agent sees every Nth BBO snapshot. This reduces bid-ask bounce autocorrelation and gives the agent a more realistic decision cadence.

## Requirements

### 1. `PrecomputedEnv.__init__()` — new `step_interval` parameter

- **New parameter:** `step_interval: int = 1` (default preserves current behavior)
- **Validation:** `step_interval >= 1`, raise `ValueError` if not
- **Subsampling:** Before calling `_precompute_temporal_features()`, subsample `self._obs`, `self._mid`, `self._spread` using `array[::step_interval]`
- **Minimum size check:** After subsampling, the arrays must still have at least 2 rows. Raise `ValueError` if `obs.shape[0] < 2` (existing check already does this — just needs to happen after subsampling, not before)
- **Temporal features:** Computed on the subsampled data (so returns/volatility/etc. reflect the agent's actual observation cadence)
- Everything else (reward, stepping, termination) works unchanged on the subsampled arrays

### 2. `PrecomputedEnv.from_file()` — forward `step_interval`

- Accept `step_interval=1` keyword argument
- Pass through to `__init__()`

### 3. `MultiDayEnv.__init__()` — forward `step_interval`

- Accept `step_interval=1` keyword argument
- Store and pass to each `PrecomputedEnv` constructor during `_make_env()`

### 4. `scripts/train.py` — CLI flag

- **New flag:** `--step-interval N` (default 1)
- Forward to `make_train_env()`, `make_env()`, and `evaluate_sortino()`

## Interface changes

```python
# PrecomputedEnv
PrecomputedEnv(obs, mid, spread,
               reward_mode="pnl_delta", lambda_=0.0,
               execution_cost=False, participation_bonus=0.0,
               step_interval=1)  # NEW

PrecomputedEnv.from_file(path, session_config=None,
                         reward_mode="pnl_delta", lambda_=0.0,
                         execution_cost=False, participation_bonus=0.0,
                         step_interval=1)  # NEW

# MultiDayEnv
MultiDayEnv(file_paths, session_config=None, steps_per_episode=50,
            reward_mode="pnl_delta", lambda_=0.0, shuffle=False, seed=None,
            execution_cost=False, participation_bonus=0.0,
            step_interval=1)  # NEW

# train.py CLI
--step-interval N   # default 1
```

## Edge cases

- `step_interval=1` — no-op, identical to current behavior
- `step_interval` larger than data length — raises `ValueError` ("obs must have at least 2 rows" after subsampling)
- `step_interval=0` or negative — raises `ValueError`
- Holiday files with 0 steps — already skipped by MultiDayEnv, no interaction

## Acceptance criteria

1. `step_interval=1` produces identical behavior to current code (no regression)
2. `step_interval=10` on a day with 137,718 steps produces ~13,771 steps
3. Temporal features are computed on subsampled data (not full-resolution data)
4. Reward uses subsampled mid prices (larger price deltas per step)
5. `--step-interval` CLI flag works end-to-end
6. Invalid values (0, negative, non-integer) raise `ValueError`

## Scope

- **Python only** — no C++ changes needed
- Files to modify: `precomputed_env.py`, `multi_day_env.py`, `train.py`
- No changes to observation size (still 54 dims)
- No changes to action space or reward formula
