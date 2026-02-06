# Step 3c: Gym Wrapper (gymnasium.Env Subclass)

## Problem

There is no `gymnasium.Env` subclass wrapping `lob_rl_core.LOBEnv`. SB3 and standard RL tooling require a Gym-compatible interface.

## What to Build

A `LOBGymEnv` class in `python/lob_rl/gym_env.py` that subclasses `gymnasium.Env` and delegates to `lob_rl_core.LOBEnv`.

## Interface

```python
import gymnasium as gym
from lob_rl.gym_env import LOBGymEnv

# With SyntheticSource (for quick testing)
env = LOBGymEnv()

# With file-backed data
env = LOBGymEnv(file_path="path/to/data.bin")

# With session config
env = LOBGymEnv(
    file_path="path/to/data.bin",
    session_config={"rth_open_ns": 48600000000000, "rth_close_ns": 72000000000000},
    steps_per_episode=1000,
    reward_mode="pnl_delta_penalized",
    lambda_=0.001,
)
```

### gymnasium.Env interface

- `observation_space`: `gymnasium.spaces.Box(low=-inf, high=inf, shape=(44,), dtype=np.float32)`
- `action_space`: `gymnasium.spaces.Discrete(3)`
- `reset(seed=None, options=None)` -> `(obs, info)` where obs is `np.ndarray` shape `(44,)` and info is `{}`
- `step(action)` -> `(obs, reward, terminated, truncated, info)` where:
  - `obs`: `np.ndarray` shape `(44,)`
  - `reward`: float
  - `terminated`: bool (episode done due to session close or source exhausted)
  - `truncated`: bool (always False — we don't truncate)
  - `info`: `{}` (empty dict for now)

### Notes

- The `seed` parameter in `reset()` is accepted but ignored (data replay is deterministic)
- `terminated` maps to `done` from the C++ `StepResult`
- `truncated` is always `False`

## Requirements

1. `LOBGymEnv` is a valid `gymnasium.Env` subclass
2. `observation_space` is `Box(shape=(44,), dtype=float32)`
3. `action_space` is `Discrete(3)`
4. `reset()` returns `(ndarray, dict)` tuple
5. `step()` returns 5-tuple `(ndarray, float, bool, bool, dict)`
6. Observation is numpy array of shape `(44,)` and dtype `float32`
7. Works with `gymnasium.utils.env_checker.check_env()` (no errors)
8. Works with SB3 `check_env()` if available
9. Supports all constructor configurations (synthetic, file, session, reward mode)

## Acceptance Criteria

- Test: `LOBGymEnv()` creates valid env with correct spaces
- Test: `reset()` returns correct tuple format with ndarray obs
- Test: `step()` returns correct 5-tuple format
- Test: `check_env()` passes
- Test: Full episode runs without error
- Test: File-backed constructor works
- Test: Session + reward mode constructor works
- Test: Multiple episodes (reset after done) work
- Test: All existing tests still pass
