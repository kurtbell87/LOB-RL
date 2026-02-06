# Participation Bonus — Reward for Market Exposure

## Problem

The PPO agent learns to stay flat (position=0) because holding no position produces zero PnL delta and avoids all risk. With execution cost, this is even more attractive — any trade incurs cost. The agent needs an incentive to participate in the market so it can learn directional signals from PnL.

## What to Build

Add a per-step participation bonus: `reward += participation_bonus * abs(position)`. This rewards being in the market (long or short) without biasing direction. The agent must still learn *which* direction from PnL.

### Reward formula (after this change)

```
reward = position * (mid[t+1] - mid[t])                       # PnL delta
       - lambda * abs(position)                                # inventory penalty (if penalized mode)
       - (spread/2) * abs(position - prev_position)            # execution cost (if enabled)
       + participation_bonus * abs(position)                   # NEW: participation bonus
```

The bonus is applied per step. When `position == 0`, the bonus is 0. When `position == +1` or `-1`, the bonus is `participation_bonus`.

### Changes required

#### 1. C++ `RewardCalculator` (`include/lob/reward.h`)

Add a `participation_bonus()` static method:

```cpp
static double participation_bonus(double position, double bonus) {
    return bonus * std::abs(position);
}
```

#### 2. C++ `LOBEnv` (`include/lob/env.h`)

Add `participation_bonus` parameter (float, default 0.0) to both constructors. Store as member. Call `RewardCalculator::participation_bonus()` in `step()` and add to reward.

#### 3. pybind11 bindings

Add `participation_bonus` keyword argument to all LOBEnv constructor overloads (default 0.0).

#### 4. Python `PrecomputedEnv` (`python/lob_rl/precomputed_env.py`)

Add `participation_bonus=0.0` parameter to `__init__()` and `from_file()`. In `step()`, after existing reward computation:

```python
if self._participation_bonus > 0:
    reward += self._participation_bonus * abs(self._position)
```

Apply the bonus BEFORE the terminal flattening penalty (same step, just add to reward).

#### 5. Python `MultiDayEnv` (`python/lob_rl/multi_day_env.py`)

Add `participation_bonus=0.0` parameter. Forward to inner `PrecomputedEnv`.

#### 6. Python `LOBGymEnv` (`python/lob_rl/gym_env.py`)

Add `participation_bonus=0.0` parameter. Forward to C++ `LOBEnv`.

#### 7. `scripts/train.py`

Add CLI flag `--participation-bonus` (float, default 0.0). Forward to `MultiDayEnv` and to `make_env()` for evaluation.

### New CLI flag

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--participation-bonus` | float | 0.0 | Per-step bonus for being in the market: `bonus * abs(position)` |

## Edge Cases

- `participation_bonus=0.0` (default): No bonus applied. Behavior identical to current code.
- `participation_bonus` with `execution_cost`: Both apply. The bonus offsets some execution cost, encouraging trades even when costly.
- `participation_bonus` with `pnl_delta_penalized` and `lambda > 0`: The inventory penalty penalizes position size while the participation bonus rewards it. These are intentionally opposing forces — the agent must balance them. Users should ensure `participation_bonus < lambda` if they want net penalization, or `participation_bonus > lambda` if they want net encouragement.
- Terminal step: The participation bonus still applies on the final step (before flattening penalty).
- Negative `participation_bonus`: Technically valid (would penalize being in the market). Not prohibited.

## Acceptance Criteria

1. `PrecomputedEnv(obs, mid, spread, participation_bonus=0.01)` applies a 0.01 bonus per step when position != 0.
2. `PrecomputedEnv` with `participation_bonus=0.0` (default) produces identical rewards to current code.
3. `MultiDayEnv` forwards `participation_bonus` to inner env.
4. `LOBGymEnv` forwards `participation_bonus` to C++ `LOBEnv`.
5. C++ `RewardCalculator::participation_bonus(1.0, 0.01)` returns 0.01.
6. C++ `RewardCalculator::participation_bonus(0.0, 0.01)` returns 0.0.
7. `train.py --participation-bonus 0.01` passes the value through the full stack.
8. `train.py --help` shows the new flag with default 0.0.
9. Evaluation (`evaluate_sortino`) forwards `participation_bonus` correctly.
10. All existing tests continue to pass.

## Files to Modify

- `include/lob/reward.h` — add `participation_bonus()` method
- `include/lob/env.h` — add parameter, wire into `step()`
- `src/bindings/bindings.cpp` — add kwarg to pybind11 overloads
- `python/lob_rl/precomputed_env.py` — add parameter, apply in `step()`
- `python/lob_rl/multi_day_env.py` — forward parameter
- `python/lob_rl/gym_env.py` — forward parameter
- `scripts/train.py` — add CLI flag, forward to envs and eval

## Files NOT to Modify

- `src/env/precompute.cpp` — precomputation doesn't change
- `python/lob_rl/_config.py` — session config doesn't change
- C++ test files — new C++ tests will be added separately
