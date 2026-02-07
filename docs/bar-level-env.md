# Bar-Level Environment — Aggregate Ticks into Decision Bars

## Problem

PPO is stepping at individual BBO changes (~137k steps/day) or every 10th tick (~13k steps/day). Credit assignment over hundreds of near-zero-reward micro-steps is intractable. Every training run produces degenerate policies (stay flat or exploit bonuses). The actual decision frequency should match a human trader's timescale: one decision per 200–2000 ticks.

## Solution

Aggregate tick-level precomputed data into N-tick bars. One `env.step()` = one completed bar. Episode length drops from 13k–137k steps to 100–700 bars. Bar-level features (OHLCV-like, spread stats, order flow) carry more signal than individual BBO snapshots.

## Requirements

### 1. `aggregate_bars()` — pure function

```python
def aggregate_bars(obs, mid, spread, bar_size):
    """Aggregate tick-level arrays into bar-level features.

    Args:
        obs: (N, 43) float32 — C++ precomputed tick features
        mid: (N,) float64 — mid prices per tick
        spread: (N,) float64 — bid-ask spreads per tick
        bar_size: int — number of ticks per bar

    Returns:
        bar_features: (B, NUM_BAR_FEATURES) float32 — bar-level features
        bar_mid_close: (B,) float64 — closing mid price per bar
        bar_spread_close: (B,) float64 — closing spread per bar
    """
```

Bars are formed by grouping consecutive ticks into chunks of `bar_size`. The last chunk is dropped if it has fewer than `bar_size // 4` ticks (avoid partial-bar noise).

**Intra-bar features (13 dims):**

| Index | Name | Computation |
|-------|------|-------------|
| 0 | `bar_return` | `(mid_close - mid_open) / mid_open` |
| 1 | `bar_range` | `(mid_high - mid_low) / mid_open` |
| 2 | `bar_volatility` | `std(mid_ticks) / mid_open` (0 if <2 ticks) |
| 3 | `spread_mean` | `mean(spread_ticks)` |
| 4 | `spread_close` | `spread` at last tick of bar |
| 5 | `imbalance_mean` | `mean(obs[:, 41])` over ticks in bar |
| 6 | `imbalance_close` | `obs[-1, 41]` at last tick |
| 7 | `bid_volume_mean` | `mean(sum(obs[:, 10:20], axis=1))` — avg total bid depth |
| 8 | `ask_volume_mean` | `mean(sum(obs[:, 30:40], axis=1))` — avg total ask depth |
| 9 | `volume_imbalance` | `mean((sum_bid - sum_ask) / (sum_bid + sum_ask))` per tick |
| 10 | `microprice_offset` | At bar close: `(ask0*bidsize0 + bid0*asksize0) / (bidsize0 + asksize0) / mid_close - 1` |
| 11 | `time_remaining` | `obs[-1, 42]` at last tick |
| 12 | `n_ticks_norm` | `actual_ticks / bar_size` (1.0 for full bars) |

Where: `obs[:, 10:20]` = bid sizes, `obs[:, 30:40]` = ask sizes, `obs[:, 41]` = imbalance, `obs[:, 42]` = time_remaining. Tick indices `0`, `20` are top-of-book bid/ask prices; `10`, `30` are top-of-book bid/ask sizes.

### 2. `BarLevelEnv` — new gymnasium.Env

```python
BarLevelEnv(obs, mid, spread,
            bar_size=500,
            reward_mode="pnl_delta",
            lambda_=0.0,
            execution_cost=False,
            participation_bonus=0.0)
```

**Constructor behavior:**
1. Call `aggregate_bars(obs, mid, spread, bar_size)` → `bar_features`, `bar_mid_close`, `bar_spread_close`
2. Compute cross-bar temporal features from `bar_features` (see below)
3. Store aggregated arrays for stepping

**Cross-bar temporal features (7 dims):**

| Index | Name | Computation |
|-------|------|-------------|
| 13 | `return_lag1` | `bar_return[t-1]` (0 if t<1) |
| 14 | `return_lag3` | `bar_return[t-3]` (0 if t<3) |
| 15 | `return_lag5` | `bar_return[t-5]` (0 if t<5) |
| 16 | `cumulative_return_5` | `sum(bar_return[t-5:t])` (partial sum if t<5) |
| 17 | `rolling_vol_5` | `std(bar_return[t-5:t])` (0 if t<2) |
| 18 | `imb_delta_3` | `imbalance_close[t] - imbalance_close[t-3]` (0 if t<3) |
| 19 | `spread_delta_3` | `spread_close[t] - spread_close[t-3]` (0 if t<3) |

**Full observation: 21 dims** = 13 intra-bar + 7 cross-bar temporal + 1 position

```
observation_space: Box(-inf, inf, shape=(21,), float32)
action_space: Discrete(3)  # 0=short, 1=flat, 2=long
```

**Stepping:**

```
reset():
    bar_index = 0, position = 0
    return obs_for_bar[0]  # (21,) with position=0

step(action):
    position = action - 1  # {-1, 0, +1}
    bar_index += 1
    reward = position * (bar_mid_close[bar_index] - bar_mid_close[bar_index - 1])
    if execution_cost:
        reward -= bar_spread_close[bar_index - 1] / 2 * |position - prev_position|
    if participation_bonus > 0:
        reward += participation_bonus * |position|
    terminated = (bar_index >= num_bars - 1)
    if terminated and position != 0:
        reward -= flattening_penalty  # spread/2 * |position| at session end
    return obs_for_bar[bar_index], reward, terminated, False, info
```

Note: reward uses close-to-close mid prices between consecutive bars. For intraday consecutive tick bars, `bar[t+1].open ≈ bar[t].close`, so this is equivalent to the user's `position * (mid_close - mid_open)` formulation.

**Classmethods:**

```python
BarLevelEnv.from_cache(npz_path, bar_size=500, reward_mode="pnl_delta",
                       lambda_=0.0, execution_cost=False,
                       participation_bonus=0.0)

BarLevelEnv.from_file(path, session_config=None, bar_size=500,
                      reward_mode="pnl_delta", lambda_=0.0,
                      execution_cost=False, participation_bonus=0.0)
```

Same pattern as `PrecomputedEnv`: `from_cache` loads `.npz`, `from_file` calls C++ precompute.

### 3. `MultiDayEnv` — `bar_size` parameter

```python
MultiDayEnv(file_paths=None,
            cache_dir=None,
            session_config=None,
            steps_per_episode=0,
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False,
            participation_bonus=0.0,
            step_interval=1,
            bar_size=0)           # NEW: 0 = tick-level (existing), >0 = bar-level
```

- When `bar_size > 0`: creates `BarLevelEnv` instances instead of `PrecomputedEnv`
- `step_interval` is ignored when `bar_size > 0` (bars replace subsampling)
- `steps_per_episode` is respected (0 = full day, >0 = truncate)
- `observation_space` matches inner env (21 dims for bar, 54 for tick)

### 4. `scripts/train.py` — new flags

```bash
python scripts/train.py --cache-dir cache/mes/ --bar-size 500 \
    --execution-cost --policy-arch 256,256 --activation relu
```

New flags:
- `--bar-size N` (int, default 0): 0 = tick-level, >0 = bar-level. Forwarded to `MultiDayEnv`.
- `--policy-arch DIMS` (str, default "64,64"): comma-separated hidden layer sizes. Parsed to `policy_kwargs=dict(net_arch=dict(pi=[...], vf=[...]), activation_fn=...)`.
- `--activation` (str, default "tanh", choices=["tanh", "relu"]): activation function for policy/value networks.

When `bar_size > 0`, `step_interval` is ignored with a warning.

### 5. `evaluate_sortino()` — support bar_size

Accept `bar_size=0` parameter. When >0, construct `BarLevelEnv.from_cache()` or `BarLevelEnv.from_file()` for evaluation.

### 6. `supervised_diagnostic.py` — support bar_size

Accept `--bar-size N` flag. When >0, load bar-level features and compute oracle labels on bar returns. This validates the MLP architecture can learn from bar features before running PPO.

## Edge cases

- `bar_size` larger than total ticks in a day → day has 0 or 1 bars → skip day with warning
- Last partial bar with fewer than `bar_size // 4` ticks → dropped
- Day with only 1 valid bar → episode terminates immediately on first step
- `bar_size=0` → existing tick-level behavior, completely unchanged
- `--bar-size` and `--step-interval` both set → warning, `step_interval` ignored
- Division by zero in `volume_imbalance` when `sum_bid + sum_ask == 0` → 0.0
- Division by zero in `microprice_offset` when `bidsize0 + asksize0 == 0` → 0.0
- `bar_volatility` with <2 ticks in a bar → 0.0

## Acceptance criteria

1. `aggregate_bars()` produces correct bar features from tick data (verified against hand-computed values)
2. `BarLevelEnv` is a valid `gymnasium.Env` (passes `check_env`)
3. `BarLevelEnv.from_cache()` and `from_file()` produce identical behavior for same day
4. `MultiDayEnv(bar_size=500)` produces bar-level episodes (~100-700 steps/day for N=500)
5. `MultiDayEnv(bar_size=0)` behavior is identical to current (backward compatible)
6. `train.py --bar-size 500 --policy-arch 256,256 --activation relu` runs training successfully
7. `supervised_diagnostic.py --bar-size 500` produces classification results on bar features
8. All existing tick-level tests pass unchanged

## Scope

- Python only — no C++ changes
- Files to create: `python/lob_rl/bar_aggregation.py`
- Files to modify: `python/lob_rl/precomputed_env.py` (or new `bar_level_env.py`), `python/lob_rl/multi_day_env.py`, `scripts/train.py`, `scripts/supervised_diagnostic.py`
- New file: `python/lob_rl/bar_level_env.py` — preferred over extending PrecomputedEnv (different obs dims, different feature logic)
