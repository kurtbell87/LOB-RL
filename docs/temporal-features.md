# Temporal Features — Give the Model Eyes

## Problem

The observation is a static 44-float LOB snapshot (10 bid/ask price+size levels, spread, imbalance, time_left, position). No temporal information. The agent can't see momentum, volatility, or how the book is changing. It's trying to predict direction from a photograph — it needs a video.

## What to Build

Add 10 temporal features computed in Python from the precomputed arrays. No C++ changes. OBS_SIZE goes from 44 to 54.

### New features (10)

| # | Name | Formula | Intuition |
|---|------|---------|-----------|
| 1 | `mid_return_1` | `(mid[t] - mid[t-1]) / mid[t-1]` | Immediate price direction |
| 2 | `mid_return_5` | `(mid[t] - mid[t-5]) / mid[t-5]` | Very short-term momentum |
| 3 | `mid_return_20` | `(mid[t] - mid[t-20]) / mid[t-20]` | Short-term momentum |
| 4 | `mid_return_50` | `(mid[t] - mid[t-50]) / mid[t-50]` | Medium-term trend |
| 5 | `volatility_20` | rolling std of 1-step returns over last 20 steps | Recent volatility |
| 6 | `imbalance_delta_5` | `imbalance[t] - imbalance[t-5]` | Book pressure change (short) |
| 7 | `imbalance_delta_20` | `imbalance[t] - imbalance[t-20]` | Book pressure change (medium) |
| 8 | `microprice_offset` | `(ask0*bidvol0 + bid0*askvol0)/(bidvol0+askvol0) / mid - 1` | Volume-weighted fair price vs mid |
| 9 | `total_volume_imbalance` | `(sum(bid_sizes) - sum(ask_sizes)) / (sum(bid_sizes) + sum(ask_sizes))` | Depth-weighted supply/demand |
| 10 | `spread_change_5` | `rel_spread[t] - rel_spread[t-5]` | Liquidity change |

### Observation layout (after change)

```
[0-9]    bid prices (10 levels)          # from C++
[10-19]  bid sizes (10 levels)           # from C++
[20-29]  ask prices (10 levels)          # from C++
[30-39]  ask sizes (10 levels)           # from C++
[40]     spread/mid                      # from C++
[41]     imbalance                       # from C++
[42]     time_left                       # from C++
[43]     mid_return_1                    # NEW - computed in Python
[44]     mid_return_5                    # NEW
[45]     mid_return_20                   # NEW
[46]     mid_return_50                   # NEW
[47]     volatility_20                   # NEW
[48]     imbalance_delta_5               # NEW
[49]     imbalance_delta_20              # NEW
[50]     microprice_offset               # NEW
[51]     total_volume_imbalance          # NEW
[52]     spread_change_5                 # NEW
[53]     position                        # moved from [43] to [53]
```

OBS_SIZE: 44 → 54.

### Implementation approach

All temporal features are computed in Python. No C++ changes. The precomputed arrays (`obs`, `mid`, `spread`) contain all the raw data needed.

#### 1. `PrecomputedEnv` (`python/lob_rl/precomputed_env.py`)

In `__init__()`, precompute temporal feature arrays from `mid`, `spread`, and `obs`:

```python
def _precompute_temporal_features(self):
    N = self._obs.shape[0]
    mid = self._mid

    # Mid-price returns at various lookbacks
    self._mid_ret_1 = np.zeros(N, dtype=np.float32)
    self._mid_ret_5 = np.zeros(N, dtype=np.float32)
    self._mid_ret_20 = np.zeros(N, dtype=np.float32)
    self._mid_ret_50 = np.zeros(N, dtype=np.float32)

    for lag, arr in [(1, self._mid_ret_1), (5, self._mid_ret_5),
                     (20, self._mid_ret_20), (50, self._mid_ret_50)]:
        arr[lag:] = (mid[lag:] - mid[:-lag]) / np.where(mid[:-lag] != 0, mid[:-lag], 1.0)

    # Volatility: rolling std of 1-step returns over 20 steps
    ret1 = np.zeros(N, dtype=np.float64)
    ret1[1:] = (mid[1:] - mid[:-1]) / np.where(mid[:-1] != 0, mid[:-1], 1.0)
    self._vol_20 = np.zeros(N, dtype=np.float32)
    for t in range(20, N):
        self._vol_20[t] = np.std(ret1[t-20:t]).astype(np.float32)

    # Imbalance deltas
    imbalance = self._obs[:, 41]  # IMBALANCE index = 4*DEPTH + 1
    self._imb_delta_5 = np.zeros(N, dtype=np.float32)
    self._imb_delta_20 = np.zeros(N, dtype=np.float32)
    self._imb_delta_5[5:] = imbalance[5:] - imbalance[:-5]
    self._imb_delta_20[20:] = imbalance[20:] - imbalance[:-20]

    # Microprice offset: computed per-step from obs array
    # bid0 = obs[:, 0], bidsize0 = obs[:, 10], ask0 = obs[:, 20], asksize0 = obs[:, 30]
    bid0 = self._obs[:, 0]
    bidsize0 = self._obs[:, 10]
    ask0 = self._obs[:, 20]
    asksize0 = self._obs[:, 30]
    denom = bidsize0 + asksize0
    microprice = np.where(denom > 0,
        (ask0 * bidsize0 + bid0 * asksize0) / denom,
        (bid0 + ask0) / 2.0)
    mid_f = mid.astype(np.float32)
    self._microprice_offset = np.where(mid_f != 0,
        microprice / mid_f - 1.0, 0.0).astype(np.float32)

    # Total volume imbalance across all 10 levels
    bid_sizes_sum = self._obs[:, 10:20].sum(axis=1)  # DEPTH=10, bid sizes at [10,20)
    ask_sizes_sum = self._obs[:, 30:40].sum(axis=1)  # ask sizes at [30,40)
    total = bid_sizes_sum + ask_sizes_sum
    self._total_vol_imb = np.where(total > 0,
        (bid_sizes_sum - ask_sizes_sum) / total, 0.0).astype(np.float32)

    # Spread change over 5 steps
    rel_spread = self._obs[:, 40]  # SPREAD index = 4*DEPTH
    self._spread_change_5 = np.zeros(N, dtype=np.float32)
    self._spread_change_5[5:] = rel_spread[5:] - rel_spread[:-5]
```

In `_build_obs()`, build the full 54-feature observation:

```python
def _build_obs(self):
    t = self._t
    obs = np.empty(54, dtype=np.float32)
    obs[:43] = self._obs[t]  # original C++ features
    obs[43] = self._mid_ret_1[t]
    obs[44] = self._mid_ret_5[t]
    obs[45] = self._mid_ret_20[t]
    obs[46] = self._mid_ret_50[t]
    obs[47] = self._vol_20[t]
    obs[48] = self._imb_delta_5[t]
    obs[49] = self._imb_delta_20[t]
    obs[50] = self._microprice_offset[t]
    obs[51] = self._total_vol_imb[t]
    obs[52] = self._spread_change_5[t]
    obs[53] = np.float32(self._position)
    return obs
```

Update `observation_space` to `Box(-inf, inf, shape=(54,), dtype=np.float32)`.

**Early timestep handling:** For `t < max_lookback` (e.g., t < 50), features with insufficient history are 0.0. This is correct — the agent sees "no historical data available" as zero, which VecNormalize handles naturally.

#### 2. `MultiDayEnv` (`python/lob_rl/multi_day_env.py`)

Update `observation_space` to shape `(54,)`. The inner `PrecomputedEnv` handles the temporal features automatically.

#### 3. `LOBGymEnv` (`python/lob_rl/gym_env.py`)

**Do NOT change.** LOBGymEnv wraps the C++ env which produces 44-dim obs. Temporal features would require C++ FeatureBuilder changes (future work).

#### 4. `scripts/train.py`

Update `make_env()` to use `PrecomputedEnv.from_file()` instead of `LOBGymEnv` for evaluation. This ensures eval uses the same 54-dim obs space as training.

```python
def make_env(file_path, reward_mode='pnl_delta', lambda_=0.0,
             execution_cost=False, participation_bonus=0.0):
    return PrecomputedEnv.from_file(
        file_path,
        session_config=DEFAULT_SESSION_CONFIG,
        reward_mode=reward_mode,
        lambda_=lambda_,
        execution_cost=execution_cost,
        participation_bonus=participation_bonus,
    )
```

Update the import to include `PrecomputedEnv`.

## Edge Cases

- **`t < lookback`**: All temporal features with insufficient history are 0.0. For t=0, all 10 temporal features are zero. This is safe — VecNormalize handles the warmup naturally.
- **`mid[t-lag] == 0`**: Division by zero guarded with `np.where`. Returns 0.0.
- **`bidsize0 + asksize0 == 0`**: Microprice falls back to `(bid0 + ask0) / 2`. Returns 0.0 offset.
- **NaN/Inf in precomputed data**: `np.where` guards and `np.isfinite` checks where needed. Non-finite values replaced with 0.0.
- **Very short episodes (< 50 steps)**: Works correctly — most temporal features will be 0, but the agent can still use the 43 C++ features + position.
- **Backward compatibility**: `LOBGymEnv` still produces 44-dim obs. Only `PrecomputedEnv` produces 54-dim.

## Acceptance Criteria

1. `PrecomputedEnv.observation_space.shape == (54,)`.
2. `MultiDayEnv.observation_space.shape == (54,)`.
3. For `t >= 50`, all 10 temporal features are non-zero (given non-constant price data).
4. For `t == 0`, all 10 temporal features are 0.0.
5. `mid_return_1` at step `t` equals `(mid[t] - mid[t-1]) / mid[t-1]` for `t >= 1`.
6. `volatility_20` at step `t` equals `np.std(1-step-returns[t-20:t])` for `t >= 20`.
7. `microprice_offset` correctly computes volume-weighted fair price relative to mid.
8. `total_volume_imbalance` sums all 10 bid/ask size levels.
9. Position is at index 53 (last element).
10. `make_env` in `train.py` returns a `PrecomputedEnv` (not `LOBGymEnv`).
11. All existing PrecomputedEnv tests still pass (reward logic unchanged).
12. All existing MultiDayEnv tests still pass.
13. Gymnasium `check_env` passes on the 54-dim env.

## Files to Modify

- `python/lob_rl/precomputed_env.py` — add temporal feature precomputation, update obs space and `_build_obs()`
- `python/lob_rl/multi_day_env.py` — update observation_space shape
- `scripts/train.py` — switch `make_env` to use `PrecomputedEnv.from_file()`

## Files NOT to Modify

- `include/lob/feature_builder.h` — C++ obs unchanged (still 43+1)
- `include/lob/env.h` — C++ env unchanged
- `src/bindings/bindings.cpp` — bindings unchanged
- `python/lob_rl/gym_env.py` — LOBGymEnv keeps 44-dim obs
