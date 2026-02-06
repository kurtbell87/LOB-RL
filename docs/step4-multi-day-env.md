# Step 4: Multi-Day Training Environment

## Problem

`scripts/train.py` line 117 creates a `DummyVecEnv` backed by a single day's data file (`train_files[0]`). The agent trains on the same ~500K-2M market events repeatedly, guaranteeing overfitting to one day's microstructure. The PRD specifies 20 training days, 5 validation days, and 2 test days.

## What to Build

A `MultiDayEnv` Gymnasium wrapper (`python/lob_rl/multi_day_env.py`) that cycles through multiple data files so the agent sees diverse market conditions across training.

### Interface

```python
class MultiDayEnv(gym.Env):
    def __init__(
        self,
        file_paths: list[str],
        session_config: dict | None = None,
        steps_per_episode: int = 0,
        reward_mode: str = "pnl_delta",
        lambda_: float = 0.0,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """
        file_paths: list of .bin file paths (one per day)
        shuffle: if True, randomize day order; if False, cycle sequentially
        seed: RNG seed for reproducible shuffling
        Other args forwarded to LOBGymEnv.
        """
```

### Behavior

1. **Initialization**: Stores the list of file paths and env config. Creates an initial `LOBGymEnv` for the first file.
2. **`reset()`**: Advances to the next day's file. If `shuffle=True`, randomizes the order when all days have been visited (epoch boundary). Creates a fresh `LOBGymEnv` for the new file. Returns the new env's `reset()` result.
3. **`step(action)`**: Delegates to the current day's `LOBGymEnv.step()`.
4. **`observation_space` / `action_space`**: Same as `LOBGymEnv` (Box(44,), Discrete(3)).
5. **Cycling**: After visiting all days, wraps around. With `shuffle=True`, re-shuffles the order each epoch.
6. **Seed handling**: `reset(seed=X)` seeds the internal RNG for shuffle reproducibility.

### Edge Cases

- Empty `file_paths` list: raise `ValueError`
- Single file: works fine, behaves like regular `LOBGymEnv`
- File that fails to load: raise immediately (don't silently skip)

## Update `scripts/train.py`

- Replace single-day `DummyVecEnv` with `MultiDayEnv` using all `train_files`
- Use `DummyVecEnv([lambda: MultiDayEnv(train_paths, ...)])` for SB3 compatibility
- Increase default `--total-timesteps` from 100K to 500K (more data to train on)

## Acceptance Criteria

1. `MultiDayEnv` passes `gymnasium.utils.env_checker.check_env()`
2. Sequential mode (`shuffle=False`) cycles through files in order, wrapping around
3. Shuffle mode randomizes day order, re-shuffles at epoch boundaries
4. Seeded shuffle produces deterministic day ordering
5. `step()` delegates correctly to current inner env
6. Empty file list raises `ValueError`
7. `train.py` uses all training days via `MultiDayEnv`

## Files to Create/Modify

- **Create**: `python/lob_rl/multi_day_env.py`
- **Create**: `python/tests/test_multi_day_env.py`
- **Modify**: `scripts/train.py` — use `MultiDayEnv`
- **Modify**: `python/lob_rl/__init__.py` — export `MultiDayEnv`
