# python/lob_rl — Python Package

Gymnasium wrappers and utilities on top of the C++ `lob_rl_core` module.

## Files

| File | Role |
|---|---|
| `__init__.py` | Package init. Exports `PrecomputedEnv`, `LOBGymEnv`, `MultiDayEnv`. |
| `_config.py` | Shared `make_session_config(dict) -> SessionConfig` helper. |
| `gym_env.py` | `LOBGymEnv` — single-day `gymnasium.Env` wrapping `lob_rl_core.LOBEnv`. |
| `precomputed_env.py` | `PrecomputedEnv` — single-day env backed by precomputed numpy arrays. Zero C++ at runtime. |
| `multi_day_env.py` | `MultiDayEnv` — cycles through multiple day files. Precomputes all at construction. **This is what `train.py` uses.** |
| `convert_dbn.py` | Converts Databento `.dbn.zst` to flat binary `.bin`. CLI entry point. |

## API Signatures

### PrecomputedEnv (`precomputed_env.py`)

```python
PrecomputedEnv(obs, mid, spread,
               reward_mode="pnl_delta",       # or "pnl_delta_penalized"
               lambda_=0.0,
               execution_cost=False,           # if True, charges spread/2 * |delta_pos|
               participation_bonus=0.0)        # per-step bonus * abs(position)

# obs: ndarray (N, 43) float32 — precomputed C++ feature rows
# mid: ndarray (N,) float64 — mid prices
# spread: ndarray (N,) float64 — bid-ask spreads
# At construction, _precompute_temporal_features() builds self._temporal (N, 10)

PrecomputedEnv.from_file(path, session_config=None,
                         reward_mode="pnl_delta", lambda_=0.0,
                         execution_cost=False, participation_bonus=0.0)

# Gym interface
obs, info = env.reset()                      # obs: ndarray(54,) float32
obs, reward, terminated, truncated, info = env.step(action)  # action: 0=short, 1=flat, 2=long
```

**Observation layout (54 dims):**
- `[0:43]` — C++ precomputed features (book state, levels, etc.)
- `[43:53]` — Temporal features: mid_return_{1,5,20,50}, volatility_20, imb_delta_{5,20}, microprice_offset, total_vol_imb, spread_change_5
- `[53]` — Agent position (-1, 0, 1)

### MultiDayEnv (`multi_day_env.py`)

```python
MultiDayEnv(file_paths,                # list of .bin paths
            session_config=None,       # dict or None (uses default RTH)
            steps_per_episode=50,      # 0 = run to session close
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False,
            participation_bonus=0.0)

# Each reset() advances to next day. Precomputes all days at construction.
# observation_space: (54,) — same as PrecomputedEnv
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
- **Depended on by:** `scripts/train.py`, `python/tests/`

## Modification hints

- **New env parameter:** Add to `PrecomputedEnv.__init__()`, forward in `MultiDayEnv.reset()` to inner env, add to `LOBGymEnv.__init__()` (forwards to C++), add to `from_file()` classmethod
- **New reward mode:** Add string handling in `PrecomputedEnv.step()`, also update C++ `parse_reward_mode()` in bindings
- **New temporal feature:** Add computation in `_precompute_temporal_features()`, append to `np.column_stack()`, update obs size 54→N+1, update `_build_obs()` slice indices, update `observation_space`
