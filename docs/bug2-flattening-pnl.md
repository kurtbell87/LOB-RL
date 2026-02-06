# BUG2: Flattening PnL Not Reflected in Reward at Session Close

## Problem

In `src/env/lob_env.cpp`, when a session ends (`done && session_filter_`):
1. Reward is computed on line 112 using the current position
2. Position is forced to 0 on line 130
3. The cost of closing the position (crossing the spread to flatten) is never accounted for

The agent receives free exits, which biases learning — it learns that holding a position at session end costs nothing.

## What to Build

When the episode ends in session-aware mode and the agent has a non-zero position, the reward on the final step should include the cost of flattening that position.

## Flattening Cost

- If position is +1 (long), flattening means selling at the bid. The cost is `position * (bid - mid)` which equals `1 * (bid - mid)` = `-spread/2` approximately.
- If position is -1 (short), flattening means buying at the ask. The cost is `position * (ask - mid)` which equals `-1 * (ask - mid)` = `-spread/2` approximately.
- More precisely: flattening cost = `-|position| * spread / 2`
- If position is 0, no flattening cost.

The simplest correct approach: when `done && session_filter_ && position_ != 0`, subtract `|position_| * spread / 2` from the reward before zeroing the position.

## Interface

No interface changes. `step()` signature unchanged. Only the reward value on the final step changes.

## Requirements

1. On session close with position +1: reward includes penalty of approximately `spread / 2`
2. On session close with position -1: reward includes penalty of approximately `spread / 2`
3. On session close with position 0: no penalty (existing behavior)
4. Non-session-aware mode: no change (position not forced to 0)
5. Steps before session close: no change

## Acceptance Criteria

- Test: Session ends with long position — reward is reduced by half-spread
- Test: Session ends with short position — reward is reduced by half-spread
- Test: Session ends flat — reward unchanged
- Test: Non-session mode — no flattening penalty applied
- Test: All existing tests still pass
