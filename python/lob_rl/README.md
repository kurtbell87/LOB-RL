# python/lob_rl — Python Package

Gymnasium wrappers and utilities on top of the C++ `lob_rl_core` module.

## Files

| File | Role |
|---|---|
| `__init__.py` | Package init. Exports `PrecomputedEnv`, `LOBGymEnv`, `MultiDayEnv`. |
| `_config.py` | Shared `make_session_config(dict) -> SessionConfig` helper. |
| `gym_env.py` | `LOBGymEnv` — single-day `gymnasium.Env` wrapping `lob_rl_core.LOBEnv`. |
| `precomputed_env.py` | `PrecomputedEnv` — single-day env backed by precomputed numpy arrays. Zero C++ at runtime. |
| `multi_day_env.py` | `MultiDayEnv` — cycles through multiple day files. **Lazy-loads** `.npz` on `reset()` (cache modes). Supports `cache_files=` for explicit file lists. **This is what `train.py` uses.** |
| `bar_aggregation.py` | `aggregate_bars(obs, mid, spread, bar_size)` — tick → bar feature aggregation. Returns (bar_features, bar_mid_close, bar_spread_close). |
| `bar_level_env.py` | `BarLevelEnv` — bar-level `gymnasium.Env`. 21-dim obs (13 intra-bar + 7 temporal + 1 position). `from_cache()`, `from_file()`. |
| `_obs_layout.py` | C++ observation layout constants (index slices, sizes). Shared by `precomputed_env.py` and `bar_aggregation.py`. |
| `_reward.py` | Shared reward/flatten logic: `compute_forced_flatten()`, `compute_step_reward()`. Used by `precomputed_env.py` and `bar_level_env.py`. |
| `_statistics.py` | Shared statistical utilities: `rolling_std(arr, window, warmup=True)`. Used by `precomputed_env.py` and `bar_level_env.py`. |
| ~~`convert_dbn.py`~~ | Deleted (PR #11). |

## API Signatures

### Reward helpers (`_reward.py`)

```python
compute_forced_flatten(spread, prev_position, action)
# Returns (reward, info_dict). reward = -spread/2 * |prev_position|.
# info_dict = {"forced_flatten": True, "forced_flatten_cost": float, "intended_action": int}

compute_step_reward(position, prev_position, mid_now, mid_prev,
                    spread_prev, reward_mode, lambda_,
                    execution_cost, participation_bonus)
# Returns reward (float). PnL delta with optional penalty, exec cost, bonus.
```

### Statistics (`_statistics.py`)

```python
rolling_std(arr, window, warmup=True)
# Returns ndarray(N,) float32. Rolling std of arr over *window* steps.
# warmup=True: indices 2..window-1 use growing window. warmup=False: only indices >= window filled.
# Uses O(N) cumulative-sum algorithm.
```

### PrecomputedEnv (`precomputed_env.py`)

```python
PrecomputedEnv(obs, mid, spread,
               reward_mode="pnl_delta",       # or "pnl_delta_penalized"
               lambda_=0.0,
               execution_cost=False,           # if True, charges spread/2 * |delta_pos|
               participation_bonus=0.0,        # per-step bonus * abs(position)
               step_interval=1)               # subsample every Nth BBO snapshot

# obs: ndarray (N, 43) float32 — precomputed C++ feature rows
# mid: ndarray (N,) float64 — mid prices
# spread: ndarray (N,) float64 — bid-ask spreads
# step_interval: subsamples arrays[::step_interval] BEFORE computing temporal features
# At construction, _precompute_temporal_features() builds self._temporal (M, 10) where M=N//step_interval

PrecomputedEnv.from_file(path, session_config=None,
                         reward_mode="pnl_delta", lambda_=0.0,
                         execution_cost=False, participation_bonus=0.0,
                         step_interval=1)

# Gym interface
obs, info = env.reset()                      # obs: ndarray(54,) float32
obs, reward, terminated, truncated, info = env.step(action)  # action: 0=short, 1=flat, 2=long

# Terminal step: forced flatten
# Position forced to 0.0 regardless of action. Reward = -spread/2 * |prev_position|.
# info = {"forced_flatten": True, "forced_flatten_cost": float, "intended_action": int}
```

**Observation layout (54 dims):**
- `[0:43]` — C++ precomputed features (book state, levels, etc.)
- `[43:53]` — Temporal features: mid_return_{1,5,20,50}, volatility_20, imb_delta_{5,20}, microprice_offset, total_vol_imb, spread_change_5
- `[53]` — Agent position (-1, 0, 1)

```python
PrecomputedEnv.from_cache(npz_path,
                          reward_mode="pnl_delta", lambda_=0.0,
                          execution_cost=False, participation_bonus=0.0,
                          step_interval=1)
# Loads obs, mid, spread from .npz file. Validates required keys.
```

### MultiDayEnv (`multi_day_env.py`)

```python
MultiDayEnv(file_paths=None,           # list of .bin paths (eager load, C++ precompute)
            cache_dir=None,            # path to dir with .npz files (lazy load)
            cache_files=None,          # explicit list of .npz paths (lazy load)
            session_config=None,       # dict or None (uses default RTH); ignored in cache modes
            steps_per_episode=50,      # 0 = run to session close
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False,
            participation_bonus=0.0,
            step_interval=1,           # forwarded to PrecomputedEnv
            bar_size=0)                # 0=tick-level, >0=bar-level

# Exactly one of file_paths, cache_dir, or cache_files must be provided.
# cache_dir/cache_files: LAZY LOADING — only file paths stored at init,
#   one .npz loaded per reset(), previous day released. ~37 MB/worker.
# file_paths: EAGER — arrays held in memory (legacy, not used in training).
# Each reset() advances to next day.
# observation_space: (54,) tick-level or (21,) bar-level
# reset() info: {"day_index": int, "instrument_id": int|None, "contract_roll": bool}

# Contract boundary tracking:
env.contract_ids  # list[int|None] — instrument_id per day (extracted at init, not at reset)
```

### LOBGymEnv (`gym_env.py`)

```python
LOBGymEnv(file_path=None,
          session_config=None,
          steps_per_episode=50,
          reward_mode="pnl_delta",
          lambda_=0.0,
          execution_cost=False,
          participation_bonus=0.0)

# Wraps C++ LOBEnv. observation_space: (44,) — NO temporal features (C++ only).
```

## Constants

- `PrecomputedEnv.observation_space`: Box(-inf, inf, shape=(54,), float32) — 43 C++ + 10 temporal + 1 position
- `LOBGymEnv.observation_space`: Box(-inf, inf, shape=(44,), float32) — 43 C++ + 1 position (no temporal)
- `action_space`: Discrete(3) — 0=short, 1=flat, 2=long
- `reward_mode` values: `"pnl_delta"`, `"pnl_delta_penalized"`

## Dependencies

- **Depends on:** `lob_rl_core` (C++ pybind11 module from `build-release/`), gymnasium, numpy
- **Internal deps:** `bar_level_env.py` and `precomputed_env.py` both import from `_statistics.py` and `_reward.py`. `multi_day_env.py` imports from `precomputed_env.py` and lazily from `bar_level_env.py`.
- **Depended on by:** `scripts/train.py`, `python/tests/`

## Modification hints

- **New env parameter:** Add to `PrecomputedEnv.__init__()`, forward in `MultiDayEnv.reset()` to inner env, add to `LOBGymEnv.__init__()` (forwards to C++), add to `from_file()` classmethod
- **New reward mode:** Add string handling in `PrecomputedEnv.step()`, also update C++ `parse_reward_mode()` in bindings
- **New temporal feature:** Add computation in `_precompute_temporal_features()`, append to `np.column_stack()`, update obs size 54→N+1, update `_build_obs()` slice indices, update `observation_space`
- **New statistical utility:** Add to `_statistics.py`. Already used by both `precomputed_env.py` (rolling_std with warmup=False) and `bar_level_env.py` (rolling_std with warmup=True).
