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
               reward_mode="pnl_delta",  # or "pnl_delta_penalized"
               lambda_=0.0,
               execution_cost=False)     # if True, charges spread/2 * |delta_pos|

# obs: ndarray (N, 43) float32 — precomputed feature rows
# mid: ndarray (N,) float64 — mid prices
# spread: ndarray (N,) float64 — bid-ask spreads

PrecomputedEnv.from_file(path, session_config=None,
                         reward_mode="pnl_delta", lambda_=0.0,
                         execution_cost=False)

# Gym interface
obs, info = env.reset()                      # obs: ndarray(44,) float32
obs, reward, terminated, truncated, info = env.step(action)  # action: 0=short, 1=flat, 2=long
```

### MultiDayEnv (`multi_day_env.py`)

```python
MultiDayEnv(file_paths,                # list of .bin paths
            session_config=None,       # dict or None (uses default RTH)
            steps_per_episode=50,      # 0 = run to session close
            reward_mode="pnl_delta",
            lambda_=0.0,
            shuffle=False,
            seed=None,
            execution_cost=False)

# Each reset() advances to next day. Precomputes all days at construction.
```

### LOBGymEnv (`gym_env.py`)

```python
LOBGymEnv(file_path=None,
          session_config=None,
          steps_per_episode=50,
          reward_mode="pnl_delta",
          lambda_=0.0,
          execution_cost=False)

# Wraps C++ LOBEnv. Same Gym interface as PrecomputedEnv.
```

## Constants

- `observation_space`: Box(-inf, inf, shape=(44,), float32)
- `action_space`: Discrete(3) — 0=short, 1=flat, 2=long
- `reward_mode` values: `"pnl_delta"`, `"pnl_delta_penalized"`

## Dependencies

- **Depends on:** `lob_rl_core` (C++ pybind11 module from `build-release/`), gymnasium, numpy
- **Depended on by:** `scripts/train.py`, `python/tests/`

## Modification hints

- **New env parameter:** Add to `PrecomputedEnv.__init__()`, forward in `MultiDayEnv.reset()` to inner env, add to `LOBGymEnv.__init__()` (forwards to C++), add to `from_file()` classmethod
- **New reward mode:** Add string handling in `PrecomputedEnv.step()`, also update C++ `parse_reward_mode()` in bindings
