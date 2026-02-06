# Step 3b: RewardCalculator (PnLDelta, PnLDeltaPenalized)

## Problem

The reward calculation is hardcoded inline in `LOBEnv::step()` as `position * (mid_now - mid_prev)`. The PRD requires a configurable reward function with two options:
- **PnLDelta**: Change in mark-to-market PnL (current behavior)
- **PnLDeltaPenalized**: PnL - λ|position| (inventory penalty to discourage holding)

The flattening penalty (half-spread at session close) should also be handled by the reward calculator.

## What to Build

A `RewardCalculator` class with an enum-based reward mode, integrated into `LOBEnv`.

## Interface

### `RewardCalculator` class (new: `include/lob/reward.h`, `src/env/reward_calculator.cpp`)

```cpp
enum class RewardMode { PnLDelta, PnLDeltaPenalized };

class RewardCalculator {
public:
    explicit RewardCalculator(RewardMode mode = RewardMode::PnLDelta, float lambda = 0.0f);

    // Calculate step reward
    float compute(float position, double current_mid, double prev_mid) const;

    // Calculate flattening penalty at session close
    float flattening_penalty(float position, double spread) const;

    RewardMode mode() const;
    float lambda() const;
};
```

### Reward formulas

- **PnLDelta**: `position * (mid_now - mid_prev)`
- **PnLDeltaPenalized**: `position * (mid_now - mid_prev) - lambda * |position|`

### Flattening penalty (unchanged from BUG2 fix)

`-|position| * spread / 2`

### LOBEnv integration

- `LOBEnv` constructors accept an optional `RewardMode` and `lambda` parameter
- Default: `RewardMode::PnLDelta`, `lambda = 0.0`
- The inline reward calculation in `step()` is replaced by `reward_calculator_.compute()`
- The inline flattening penalty is replaced by `reward_calculator_.flattening_penalty()`
- Python bindings: `LOBEnv` constructors accept optional `reward_mode` (string: "pnl_delta" or "pnl_delta_penalized") and `lambda_` (float)

## Requirements

1. `RewardCalculator` with `PnLDelta` mode computes `position * (mid_now - mid_prev)`
2. `RewardCalculator` with `PnLDeltaPenalized` mode computes `position * (mid_now - mid_prev) - lambda * |position|`
3. `flattening_penalty()` returns `-|position| * spread / 2`
4. Non-finite mid values result in 0 reward (same as current behavior)
5. `LOBEnv` uses `RewardCalculator` instead of inline reward code
6. Default reward mode is `PnLDelta` (backward compatible)
7. Python bindings expose reward mode configuration

## Acceptance Criteria

- Test: PnLDelta reward correct for long/short/flat positions
- Test: PnLDeltaPenalized reward includes lambda penalty
- Test: Lambda=0 with PnLDeltaPenalized is equivalent to PnLDelta
- Test: Flattening penalty correct for various positions and spreads
- Test: Non-finite mid returns 0 reward
- Test: LOBEnv with PnLDelta mode matches current behavior
- Test: LOBEnv with PnLDeltaPenalized produces different (lower) rewards than PnLDelta when position != 0
- Test: Python can construct LOBEnv with reward mode
- Test: All existing tests still pass
