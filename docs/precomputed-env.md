# Spec: PrecomputedEnv

## Overview

A pure-Python `gymnasium.Env` that operates on pre-computed numpy arrays instead of calling C++ per step. This enables ~1M+ steps/sec training throughput.

## File: `python/lob_rl/precomputed_env.py`

## Interface

```python
class PrecomputedEnv(gymnasium.Env):
    """Pure-numpy gymnasium env for pre-computed LOB data."""

    metadata = {"render_modes": []}

    def __init__(self, obs, mid, spread, reward_mode="pnl_delta", lambda_=0.0):
        """
        Args:
            obs: ndarray shape (N, 43), dtype float32 — market observations (no position)
            mid: ndarray shape (N,), dtype float64 — mid prices
            spread: ndarray shape (N,), dtype float64 — spreads
            reward_mode: "pnl_delta" or "pnl_delta_penalized"
            lambda_: penalty coefficient for pnl_delta_penalized mode
        """

    @classmethod
    def from_file(cls, file_path, session_config=None, reward_mode="pnl_delta", lambda_=0.0):
        """Convenience: calls lob_rl_core.precompute() then constructs the env."""

    def reset(self, *, seed=None, options=None):
        """Returns (obs_44, info) where obs_44 has position appended at index 43."""

    def step(self, action):
        """Returns (obs_44, reward, terminated, truncated, info)."""
```

## Behavior

### Constructor
- Store `obs` (N, 43), `mid` (N,), `spread` (N,) as numpy arrays
- `observation_space = Box(low=-inf, high=inf, shape=(44,), dtype=float32)`
- `action_space = Discrete(3)`
- Validate: `obs.shape[0] >= 2` (need at least 2 snapshots for 1 transition)
- Store `reward_mode` and `lambda_`

### `from_file()` classmethod
- If `session_config` is None, use `lob_rl_core.SessionConfig.default_rth()`
- If `session_config` is a dict, create `lob_rl_core.SessionConfig()` and set fields
- Call `lob_rl_core.precompute(file_path, cfg)` → `(obs, mid, spread, num_steps)`
- Return `cls(obs, mid, spread, reward_mode, lambda_)`

### `reset()`
- Set `t = 0`, `position = 0.0`
- Build 44-float obs: `np.append(obs[0], [0.0])` (position=0 at index 43)
- Return `(obs_44, {})`

### `step(action)`
- Map action: `position = action - 1` (0→-1, 1→0, 2→+1)
- Compute reward: `position * (mid[t+1] - mid[t])`
- If `reward_mode == "pnl_delta_penalized"`: subtract `lambda_ * abs(position)`
- Increment `t`
- Check if `t >= N - 1` (terminal — at last snapshot, no more transitions)
- If terminal: add flattening penalty `-abs(position) * spread[t] / 2`
- Build 44-float obs: `np.append(obs[t], [position])`
- Return `(obs_44, reward, terminated, False, {})`

### Episode length
- N snapshots → N-1 steps (N-1 transitions between consecutive snapshots)
- Terminal at `t == N - 1`

## Modify: `python/lob_rl/__init__.py`
- Add import: `from lob_rl.precomputed_env import PrecomputedEnv`

## Test plan (Python only, file: `python/tests/test_precomputed_env.py`)

### Test 1: Constructor stores arrays correctly
- Create small synthetic arrays: `obs(5,43)`, `mid(5,)`, `spread(5,)`
- Verify env has correct observation_space and action_space

### Test 2: reset returns 44-float observation with position=0
- Call `reset()`, get `(obs_44, info)`
- Assert `obs_44.shape == (44,)`
- Assert `obs_44.dtype == np.float32`
- Assert `obs_44[43] == 0.0` (position=0 at reset)
- Assert `obs_44[:43]` matches `obs[0]`
- Assert `info == {}`

### Test 3: step returns correct 5-tuple format
- `reset()`, then `step(1)` (flat)
- Assert returns `(obs, reward, terminated, truncated, info)` with correct types
- Assert `terminated == False` (first step)
- Assert `truncated == False`

### Test 4: action mapping {0→-1, 1→0, 2→+1}
- Test each action and verify position in obs
- `step(0)` → `obs_44[43] == -1.0`
- `step(1)` → `obs_44[43] == 0.0`
- `step(2)` → `obs_44[43] == 1.0`

### Test 5: reward = position * (mid[t+1] - mid[t]) for PnLDelta
- Create env with known mid prices: `[100, 101, 99]`
- `reset()`, `step(2)` (go long, position=+1)
- reward = +1 * (101 - 100) = 1.0

### Test 6: PnLDeltaPenalized subtracts lambda * |position|
- Create env with `reward_mode="pnl_delta_penalized"`, `lambda_=0.5`
- `reset()`, `step(2)` (long, position=+1)
- reward = 1 * (mid[1] - mid[0]) - 0.5 * 1.0

### Test 7: flat action (1) has zero reward when mid doesn't change
- Create env with constant mid
- `step(1)` → reward == 0.0

### Test 8: episode terminates at t == N-1
- Create env with 5 snapshots → 4 steps
- Take 4 steps, on the 4th step `terminated == True`

### Test 9: flattening penalty at terminal step
- Create env with 3 snapshots, spread=[0.5, 0.5, 0.5]
- `reset()`, `step(2)` (go long), `step(2)` (stay long, terminal)
- Terminal reward includes `-|1| * 0.5 / 2 = -0.25`

### Test 10: no flattening penalty when flat at terminal
- Same as above but `step(1)` at terminal → no penalty added

### Test 11: from_file() classmethod works
- Call `PrecomputedEnv.from_file(fixture_path)`
- Verify it returns a valid env that can be reset and stepped

### Test 12: passes gymnasium check_env()
- Create env from file, run `gymnasium.utils.env_checker.check_env()`

### Test 13: reset resets position and time to 0
- Step through some of the episode
- Call `reset()` again — verify obs is from t=0 with position=0

### Test 14: obs at each step comes from correct time index
- Verify `obs_44[:43]` at step t matches `obs_array[t]`

### Test 15: constructor raises ValueError if obs has < 2 rows
- `obs(1, 43)` → should raise `ValueError`

### Test 16: from_file with custom session_config dict
- Pass `session_config={"rth_open_ns": ..., "rth_close_ns": ...}`
- Verify it works

## Acceptance criteria

- `PrecomputedEnv` is a valid `gymnasium.Env` (passes `check_env()`)
- Pure Python/numpy — no C++ calls during `step()` or `reset()`
- Episode length = N-1 steps for N snapshots
- Flattening penalty applied only at terminal step
- Importable from `lob_rl` package
