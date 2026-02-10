# T7: Reward Accounting Unit Test

## What to Build

A new module `python/lob_rl/barrier/reward_accounting.py` implementing reward computation logic for the barrier-hit trading environment (spec Section 4.4–4.5). This is a pure-computation module — no Gymnasium dependency — that calculates rewards, manages position state transitions, and applies action masking.

Hand-computed reward sequences serve as regression tests that catch bugs in: barrier reversal for shorts, mark-to-market at timeout, transaction cost accounting, and position state transitions.

**Spec reference:** Section 4.4 (Transition and Reward), Section 4.5 (Reward Definitions), Section 8 step 7.

## Dependencies

- `label_pipeline.py` — `BarrierLabel` with `.label ∈ {+1, -1, 0}`, `.entry_price`, `.resolution_bar`, `.tau`
- `bar_pipeline.py` — `TradeBar` with `.close`, `.high`, `.low`
- `__init__.py` — `TICK_SIZE = 0.25`

## Reward Logic (from spec Section 4.4–4.5)

### Constants (defaults)

| Symbol | Definition | Default |
|--------|-----------|---------|
| `a` | Profit barrier distance (ticks) | 20 |
| `b` | Stop barrier distance (ticks) | 10 |
| `G` | Profit reward (normalized) = `a/b` | 2.0 |
| `L` | Loss penalty (normalized) = `b/b` | 1.0 |
| `C` | Transaction cost (normalized) = `2 ticks / b` | 0.2 |
| `T_max` | Maximum holding period (bars) | 40 |

### Position States

- `FLAT = 0` — no position
- `LONG = +1` — long position
- `SHORT = -1` — short position

### Actions

- `LONG = 0` — enter long
- `SHORT = 1` — enter short
- `FLAT = 2` — stay flat (choose not to enter)
- `HOLD = 3` — maintain position

### Action Masking

| Current position | Valid actions | Invalid actions |
|------------------|-------------|----------------|
| Flat (0) | {long, short, flat} | {hold} |
| Long (+1) | {hold} | {long, short, flat} |
| Short (-1) | {hold} | {long, short, flat} |

### Entry (from flat)

When `position == 0` and agent selects `long` or `short`:

```
position = +1 (long) or -1 (short)
entry_price = bar.close  (C_k in spec)
hold_counter = 0
reward = 0  (no immediate reward on entry)

For long:
    profit_barrier = entry_price + a * tick_size
    stop_barrier   = entry_price - b * tick_size

For short:
    profit_barrier = entry_price - a * tick_size
    stop_barrier   = entry_price + b * tick_size
```

### Holding

At each bar `j` while holding:

```
hold_counter += 1

Check barrier breach using bar.high, bar.low:

For long:
    profit hit: bar.high >= profit_barrier
    stop hit:   bar.low <= stop_barrier

For short:
    profit hit: bar.low <= profit_barrier
    stop hit:   bar.high >= stop_barrier

If profit barrier hit:
    reward = +G - C = +2.0 - 0.2 = +1.8
    position = 0

If stop barrier hit:
    reward = -L - C = -1.0 - 0.2 = -1.2
    position = 0

If both hit (dual breach):
    Use label from compute_labels() to resolve

If hold_counter == T_max:
    reward = position * (bar.close - entry_price) / (b * tick_size) - C
    position = 0

Otherwise:
    reward = 0
    position unchanged
```

### Staying Flat

```
reward = 0
position = 0
```

### Unrealized PnL

```
unrealized_pnl = position * (current_close - entry_price)  (in ticks: / tick_size)
```

For long: `+1 * (current - entry)` — positive when price goes up.
For short: `-1 * (current - entry)` — positive when price goes down.

## Module API

### `RewardConfig`

Dataclass holding reward parameters.

```python
@dataclass
class RewardConfig:
    a: int = 20          # profit barrier distance (ticks)
    b: int = 10          # stop barrier distance (ticks)
    G: float = 2.0       # profit reward (a/b)
    L: float = 1.0       # loss penalty (b/b)
    C: float = 0.2       # transaction cost (2 ticks / b)
    T_max: int = 40      # maximum holding period
    tick_size: float = 0.25
```

### `PositionState`

Dataclass tracking position state.

```python
@dataclass
class PositionState:
    position: int = 0          # -1, 0, +1
    entry_price: float = 0.0
    hold_counter: int = 0
    profit_barrier: float = 0.0
    stop_barrier: float = 0.0
```

### Action Constants

```python
ACTION_LONG  = 0
ACTION_SHORT = 1
ACTION_FLAT  = 2
ACTION_HOLD  = 3
```

### `get_action_mask(position) -> list[bool]`

Returns a 4-element boolean mask indicating valid actions for the current position.

**Parameters:**
- `position: int` — Current position: -1, 0, or +1.

**Returns:**
- `list[bool]` of length 4: `[long_valid, short_valid, flat_valid, hold_valid]`.

### `compute_entry(bar, action, config) -> PositionState`

Computes the new position state after an entry action.

**Parameters:**
- `bar: TradeBar` — Current bar (entry price = `bar.close`).
- `action: int` — `ACTION_LONG` (0) or `ACTION_SHORT` (1).
- `config: RewardConfig` — Reward parameters.

**Returns:**
- `PositionState` with position, entry_price, barriers, hold_counter=0.

### `compute_hold_reward(bar, state, config) -> tuple[float, PositionState]`

Computes the reward and new state after a hold step.

**Parameters:**
- `bar: TradeBar` — Current bar to check for barrier breach.
- `state: PositionState` — Current position state.
- `config: RewardConfig` — Reward parameters.

**Returns:**
- `reward: float` — The reward for this step.
- `new_state: PositionState` — Updated position state (may be flat if barrier hit or timeout).

### `compute_unrealized_pnl(state, current_close) -> float`

Computes unrealized PnL in ticks.

**Parameters:**
- `state: PositionState` — Current position state.
- `current_close: float` — Current bar close price.

**Returns:**
- `float` — Unrealized PnL in ticks. 0.0 if flat.

### `compute_reward_sequence(bars, action, start_bar_idx, config) -> list[dict]`

Trace through a full trade from entry to exit, returning the reward at each step.

**Parameters:**
- `bars: list[TradeBar]` — Sequence of bars.
- `action: int` — Entry action (`ACTION_LONG` or `ACTION_SHORT`).
- `start_bar_idx: int` — Index into bars where entry occurs.
- `config: RewardConfig` — Reward parameters.

**Returns:**
- `list[dict]` — One dict per step from entry to exit. Each dict contains:
  - `bar_idx: int`
  - `reward: float`
  - `position: int`
  - `hold_counter: int`
  - `unrealized_pnl: float`
  - `exit_type: str | None` — `"profit"`, `"stop"`, `"timeout"`, or `None` if still holding.

## Test Cases

### Hand-Computed Reward Sequences — Long Entry (5 tests)

For these tests, construct 5 synthetic bar sequences with known prices where the barrier outcome is deterministic.

1. **`test_long_profit_hit`**: Entry at close=4000.00, next bar high reaches 4005.00 (>= profit_barrier 4005.00). Reward = +G - C = +1.8. Position returns to flat.

2. **`test_long_stop_hit`**: Entry at close=4000.00, next bar low reaches 3997.50 (<= stop_barrier 3997.50). Reward = -L - C = -1.2. Position returns to flat.

3. **`test_long_hold_then_profit`**: Entry at close=4000.00, bars 1-3 don't hit barriers, bar 4 high reaches 4005.00. Rewards: [0, 0, 0, 0, +1.8]. Hold counter increments correctly.

4. **`test_long_timeout`**: Entry at close=4000.00, no barrier hit for T_max=5 bars (use small T_max for testing), final bar close=4001.25. Reward at timeout = +1 * (4001.25 - 4000.00) / (10 * 0.25) - 0.2 = 1.25/2.5 - 0.2 = 0.5 - 0.2 = 0.3.

5. **`test_long_timeout_negative_mtm`**: Entry at close=4000.00, no barrier hit for T_max=5 bars, final bar close=3999.00. Reward = +1 * (3999.00 - 4000.00) / (10 * 0.25) - 0.2 = -1.0/2.5 - 0.2 = -0.4 - 0.2 = -0.6.

### Hand-Computed Reward Sequences — Short Entry (5 tests)

Same bar sequences, but entering short. Barriers are reversed.

6. **`test_short_profit_hit`**: Entry at close=4000.00, next bar low reaches 3995.00 (<= profit_barrier 3995.00). Reward = +G - C = +1.8.

7. **`test_short_stop_hit`**: Entry at close=4000.00, next bar high reaches 4002.50 (>= stop_barrier 4002.50). Reward = -L - C = -1.2.

8. **`test_short_hold_then_profit`**: Entry at close=4000.00, bars 1-3 don't hit barriers, bar 4 low reaches 3995.00. Rewards: [0, 0, 0, 0, +1.8].

9. **`test_short_timeout`**: Entry at close=4000.00, T_max=5, final bar close=3998.75. Reward = -1 * (3998.75 - 4000.00) / (10 * 0.25) - 0.2 = -1 * (-1.25)/2.5 - 0.2 = 0.5 - 0.2 = 0.3.

10. **`test_short_timeout_negative_mtm`**: Entry at close=4000.00, T_max=5, final bar close=4001.00. Reward = -1 * (4001.00 - 4000.00) / (10 * 0.25) - 0.2 = -1 * 1.0/2.5 - 0.2 = -0.4 - 0.2 = -0.6.

### MTM Normalization (2 tests)

11. **`test_mtm_normalization_by_b`**: Verify timeout reward uses `/ (b * tick_size)` for normalization. With b=10, tick_size=0.25: denominator = 2.5. Price move of 5 ticks (1.25 points) → MTM = 1.25 / 2.5 = 0.5.

12. **`test_mtm_zero_at_entry_price`**: Timeout at exact entry price → MTM = 0 - C = -0.2. Pure cost.

### Transaction Cost Accounting (3 tests)

13. **`test_cost_deducted_once_on_profit`**: Profit hit reward is exactly +G - C, not +G - 2C.

14. **`test_cost_deducted_once_on_stop`**: Stop hit reward is exactly -L - C.

15. **`test_cost_deducted_once_on_timeout`**: Timeout reward is MTM - C, cost deducted once.

### Position State Transitions (4 tests)

16. **`test_flat_to_long_to_flat`**: Enter long → hold → profit hit → back to flat. Verify position transitions: 0 → +1 → +1 → 0.

17. **`test_flat_to_short_to_flat`**: Enter short → hold → stop hit → back to flat. Position: 0 → -1 → -1 → 0.

18. **`test_entry_sets_barriers_long`**: After long entry at 4000.00 with a=20, b=10: profit_barrier=4005.00, stop_barrier=3997.50.

19. **`test_entry_sets_barriers_short`**: After short entry at 4000.00 with a=20, b=10: profit_barrier=3995.00, stop_barrier=4002.50.

### Unrealized PnL (4 tests)

20. **`test_unrealized_pnl_long_positive`**: Long entry at 4000.00, current close 4001.25. PnL = +1 * (4001.25 - 4000.00) / 0.25 = 5.0 ticks.

21. **`test_unrealized_pnl_long_negative`**: Long entry at 4000.00, current close 3999.00. PnL = +1 * (3999.00 - 4000.00) / 0.25 = -4.0 ticks.

22. **`test_unrealized_pnl_short_positive`**: Short entry at 4000.00, current close 3998.75. PnL = -1 * (3998.75 - 4000.00) / 0.25 = 5.0 ticks.

23. **`test_unrealized_pnl_short_negative`**: Short entry at 4000.00, current close 4001.00. PnL = -1 * (4001.00 - 4000.00) / 0.25 = -4.0 ticks.

### Action Masking (4 tests)

24. **`test_mask_flat_position`**: When flat: long=True, short=True, flat=True, hold=False.

25. **`test_mask_long_position`**: When long: long=False, short=False, flat=False, hold=True.

26. **`test_mask_short_position`**: When short: long=False, short=False, flat=False, hold=True.

27. **`test_mask_is_list_of_bool`**: Return type is `list[bool]` of length 4.

### Full Reward Sequence (3 tests)

28. **`test_reward_sequence_long_profit`**: Full sequence from entry to profit hit. Verify each step's reward, position, hold_counter, unrealized_pnl.

29. **`test_reward_sequence_short_timeout`**: Full sequence from short entry to timeout. Verify intermediate unrealized_pnl values and final MTM reward.

30. **`test_reward_sequence_immediate_hit`**: Entry at close, immediate barrier hit on next bar. Sequence has exactly 2 elements: entry (reward=0) + hit.

### Edge Cases (3 tests)

31. **`test_hold_counter_increments`**: Hold counter goes from 0 at entry to T_max at timeout.

32. **`test_config_defaults`**: RewardConfig defaults match spec: G=2.0, L=1.0, C=0.2, T_max=40, a=20, b=10, tick_size=0.25.

33. **`test_entry_price_is_bar_close`**: Entry price is always bar.close, not bar.open or bar.vwap.

## Implementation Notes

- This module is pure computation — no Gymnasium, no numpy beyond basic arithmetic. It should be trivially testable.
- All reward values use the normalized units from spec Section 4.5 (barrier units, not ticks or dollars).
- The `compute_unrealized_pnl` function returns PnL in **ticks** (divided by tick_size), not in barrier units. This is what the environment will expose in the observation.
- For the test hand-computations, use small T_max (e.g., 5) to keep bar sequences short while still testing timeout logic.
- The module does NOT resolve dual-breach ties — that's handled by `compute_labels()` from T2. The `compute_hold_reward` function checks barriers sequentially (profit first, then stop) for the simple case, but in the environment, pre-computed labels will resolve ambiguity.
- Action constants match spec Section 4.3: long=0, short=1, flat=2, hold=3.
