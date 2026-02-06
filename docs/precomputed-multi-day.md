# Precomputed Multi-Day Environment

## Motivation

Since the agent has no market impact (actions don't affect the order book), all market observations can be precomputed once before training. The current `MultiDayEnv` wraps `LOBGymEnv` which replays every MBO message through the C++ Book engine on every episode — millions of messages per day, extremely slow. A precomputed version precomputes all 43-float observations, mid prices, and spreads once per day file at construction time, then training runs on pure numpy arrays with zero C++ calls.

## What to Build

### `PrecomputedMultiDayEnv` class (`python/lob_rl/multi_day_env.py`)

Replace the current `MultiDayEnv` implementation. The new version:

1. **Constructor** `__init__(self, file_paths, session_config=None, steps_per_episode=0, reward_mode="pnl_delta", lambda_=0.0, shuffle=False, seed=None)`:
   - Accepts the same arguments as the current `MultiDayEnv` for backward compatibility
   - `steps_per_episode` is accepted but ignored (precomputed envs always run full session)
   - At construction time, calls `lob_rl_core.precompute(path, cfg)` for **each** file path
   - Stores the resulting numpy arrays `(obs, mid, spread)` per day
   - Skips any day files that produce fewer than 2 BBO snapshots (warns via print)
   - Raises `ValueError` if no usable days remain after filtering
   - Sets `observation_space = Box(-inf, inf, shape=(44,), dtype=float32)` and `action_space = Discrete(3)`

2. **`reset(self, *, seed=None, options=None)`**:
   - Cycles through days in order (sequential) or shuffled order
   - Creates a `PrecomputedEnv` from the current day's precomputed arrays
   - Returns `(obs_44, info_dict)`
   - `info_dict` includes `{"day_index": int}` identifying which day file is active

3. **`step(self, action)`**:
   - Delegates to the current inner `PrecomputedEnv`
   - Returns the standard 5-tuple

4. **Shuffle behavior** (same as current `MultiDayEnv`):
   - `shuffle=False`: visit days in order, wrap at epoch boundary
   - `shuffle=True`: randomize order, re-shuffle at each epoch boundary
   - `seed` parameter controls RNG for reproducibility
   - `reset(seed=N)` reseeds the RNG

### Fix existing `from_file` test fixture

The `conftest.py` `EPISODE_FILE` points to `episode_200records.bin` which has epoch-era timestamps (not RTH). The `from_file()` tests in `test_precomputed_env.py` use this fixture, causing `precompute()` to return 0 steps.

**Fix**: Update `conftest.py` to also export a `PRECOMPUTE_EPISODE_FILE` pointing to `session_180records.bin` (which has real Jan 15, 2025 RTH timestamps). The `test_precomputed_env.py` `from_file` tests should use this fixture.

### Update `python/lob_rl/__init__.py`

Export `PrecomputedMultiDayEnv` (or the renamed `MultiDayEnv`) from the package.

## Interface

```python
from lob_rl.multi_day_env import MultiDayEnv  # now precomputed

env = MultiDayEnv(
    file_paths=["data/mes/20250106.bin", "data/mes/20250107.bin", ...],
    session_config={"rth_open_ns": 48_600_000_000_000, "rth_close_ns": 72_000_000_000_000, "warmup_messages": -1},
    reward_mode="pnl_delta",
    shuffle=True,
)

obs, info = env.reset()  # precomputes all days, returns first obs
# info == {"day_index": 3}  (which day file is active)

obs, reward, terminated, truncated, info = env.step(2)  # pure numpy, fast
```

## Edge Cases

- Day file with < 2 BBO changes during RTH: skip with warning, don't crash
- All day files produce < 2 BBO changes: raise `ValueError`
- Empty `file_paths` list: raise `ValueError`
- Single day file: works (cycles the same day every reset)
- `session_config=None`: use `SessionConfig.default_rth()`

## Acceptance Criteria

1. `MultiDayEnv` precomputes all days at construction (C++ calls happen once)
2. `step()` and `reset()` use pure numpy (no C++ calls during training)
3. Sequential and shuffle modes work correctly
4. Seed determinism: same seed produces same day ordering
5. Passes `gymnasium.utils.env_checker.check_env()`
6. Passes SB3 `DummyVecEnv` compatibility
7. `test_precomputed_env.py` `from_file` tests pass (fixture fix)
8. All existing `test_multi_day_env.py` tests either pass or are updated for the new precomputed behavior
9. All existing `test_precomputed_env.py` tests pass
