# Execution Cost on Position Changes

## Summary

Add per-step execution cost (half-spread) whenever the agent changes position.
Currently, position changes are free mid-episode — only the end-of-episode
flattening penalty charges spread cost. This makes the simulation unrealistic
and produces degenerate learning signals.

## Motivation

The baseline PPO agent has negative Sortino (-1.05 val, -14.4 test) and showed
entropy collapse. A key contributor: the agent can flip long↔short for free,
so there's no penalty for random trading. In real markets, every contract traded
costs at least half the bid-ask spread. Without this friction, the reward signal
is noisy and the optimal policy is indeterminate.

## Design

### Execution cost formula

```
execution_cost = spread[t] / 2.0 * |new_position - old_position|
```

Where `|new_position - old_position|` is the number of contracts traded:
- 0 if position unchanged
- 1 if flat→long, flat→short, long→flat, short→flat
- 2 if long→short or short→long

This is consistent with the existing flattening penalty (which is the special
case where new_position=0).

### Step reward formula

```
reward = position * (mid[t+1] - mid[t])        # PnL from exposure
       - spread[t] / 2.0 * |pos - prev_pos|    # execution cost
       - lambda * |position|                    # inventory penalty (if enabled)
```

### Parameter: `execution_cost` (bool, default=False)

Add an `execution_cost` boolean parameter (default `False` for backwards
compatibility) to:

1. **`PrecomputedEnv.__init__()`** — store flag, track `_prev_position`
2. **`PrecomputedEnv.step()`** — if enabled, subtract execution cost from reward
3. **`PrecomputedEnv.reset()`** — reset `_prev_position` to 0.0
4. **`MultiDayEnv.__init__()`** — accept and forward to `PrecomputedEnv`
5. **`LOBGymEnv.__init__()`** — accept and forward to C++ `LOBEnv`
6. **C++ `RewardCalculator`** — add `execution_cost(old_pos, new_pos, spread)` method
7. **C++ `LOBEnv`** — track `prev_position_`, call execution cost on step, accept bool flag
8. **pybind11 bindings** — expose the new parameter in all LOBEnv constructors
9. **`train.py`** — add `--execution-cost` CLI flag, pass to env constructors

### Backwards compatibility

- Default `execution_cost=False` preserves existing behavior exactly
- Existing reward modes (`pnl_delta`, `pnl_delta_penalized`) work unchanged
- The execution cost is orthogonal to reward mode — it applies to any mode

## Changes by file

### Python: `python/lob_rl/precomputed_env.py`

- Add `execution_cost=False` parameter to `__init__`
- Add `self._prev_position = 0.0` in `__init__` and `reset()`
- In `step()`: before updating position, save old position. After computing
  PnL reward, subtract `spread[t] / 2 * |new_pos - old_pos|` if enabled.
  Then update `_prev_position = self._position`.

### Python: `python/lob_rl/multi_day_env.py`

- Add `execution_cost=False` parameter to `__init__`
- Forward to `PrecomputedEnv` constructor in `reset()`

### Python: `python/lob_rl/gym_env.py`

- Add `execution_cost=False` parameter to `__init__`
- Forward to C++ `LOBEnv` constructor

### C++: `include/lob/reward.h`

- Add method to `RewardCalculator`:
  ```cpp
  float execution_cost(float old_position, float new_position, double spread) const {
      return -std::abs(new_position - old_position) * static_cast<float>(spread / 2.0);
  }
  ```

### C++: `include/lob/env.h`

- Add `bool execution_cost_ = false;` member
- Add `float prev_position_ = 0.0f;` member
- Update constructors to accept `bool execution_cost` parameter

### C++: `src/env/lob_env.cpp`

- In `reset()`: set `prev_position_ = 0.0f`
- In `step()`: save old position before updating, compute execution cost if
  enabled, add to reward. Update `prev_position_` after.

### C++: `src/bindings/bindings.cpp`

- Update LOBEnv constructor bindings to accept `execution_cost` parameter

### Python: `scripts/train.py`

- Add `--execution-cost` flag (store_true, default False)
- Pass to `MultiDayEnv` and `make_env` in evaluation

## Edge cases

- **Position unchanged:** execution cost = 0 (no trade)
- **First step after reset:** prev_position = 0, so going long costs spread/2
- **Episode end flattening:** The existing flattening penalty still applies.
  The execution cost is NOT double-counted because the flattening penalty
  represents the cost of closing the position at session end, while the
  execution cost covers mid-episode trades. On the terminal step, both apply
  if the agent changed position AND has a nonzero position.
- **Spread = 0 or NaN:** If spread is not finite, execution cost = 0

## Acceptance criteria

1. `PrecomputedEnv(execution_cost=False)` produces identical rewards to current behavior
2. `PrecomputedEnv(execution_cost=True)` subtracts `spread/2 * |delta_pos|` each step
3. Position change long→short (delta=2) costs more than flat→long (delta=1)
4. No position change costs nothing
5. C++ `LOBEnv` with `execution_cost=true` matches Python behavior
6. `MultiDayEnv` and `LOBGymEnv` forward the parameter correctly
7. `train.py --execution-cost` flag activates the feature
8. All existing tests still pass (backwards compatibility)
