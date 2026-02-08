# Contract Boundary Guard

## Summary

Ensure the RL agent cannot hold positions across contract roll boundaries. Add forced end-of-day flattening, contract metadata in the cache, and runtime assertions at roll boundaries.

## Motivation

The agent trades MES futures which roll quarterly. If a position is held from one contract (e.g., MESH2) into the next (MESM2), the PnL is nonsensical — these are different instruments with different prices. The current system implicitly prevents this because each day is a separate episode with position=0 on reset, but there is no explicit enforcement, no contract metadata, and no assertion. This feature adds hard structural guarantees.

## Requirements

### 1. Store `instrument_id` in `.npz` cache files

**File:** `scripts/precompute_cache.py`

Currently saves: `np.savez(path, obs=obs, mid=mid, spread=spread)`

Change to: `np.savez(path, obs=obs, mid=mid, spread=spread, instrument_id=np.array([instrument_id], dtype=np.uint32))`

The `instrument_id` is a scalar uint32 stored as a 1-element array (numpy savez doesn't support scalar metadata cleanly).

### 2. Forced flatten on terminal step — `PrecomputedEnv`

**File:** `python/lob_rl/precomputed_env.py`

On the terminal step of each episode, the agent's position MUST be forced to flat (0.0), regardless of the action chosen. This replaces the current "terminal flatten penalty" with a cleaner mechanism.

**Current terminal behavior (to be replaced):**
```python
self._position = self._ACTION_MAP[action]  # agent's choice
reward = self._position * (mid[t+1] - mid[t])  # PnL with held position
# ... execution cost for entering ...
if terminated:
    reward -= abs(self._position) * spread[t] / 2.0  # penalty for holding
```

**New terminal behavior:**
```python
# On terminal step:
intended_action = action
self._position = 0.0  # FORCED flat
reward = 0.0  # No PnL from last tick (position is 0)

# Close cost: always charged, regardless of execution_cost flag
# This is the mandatory liquidation cost of closing prev_position
close_cost = self._spread[self._t] / 2.0 * abs(0.0 - self._prev_position)
reward -= close_cost

# No separate terminal penalty (forced flat replaces it)
```

**On non-terminal steps:** behavior is unchanged.

**Info dict on terminal step:**
```python
info["forced_flatten"] = True  # always True on terminal step
info["forced_flatten_cost"] = close_cost  # the spread cost charged
info["intended_action"] = intended_action  # what the agent wanted to do
```

**Important:** the close cost uses `self._prev_position` (position from the previous step), NOT the agent's intended action. This is because the agent is being liquidated from whatever position it held at the end of the prior step. The agent's action on the terminal step is completely ignored.

### 3. Forced flatten on terminal step — `BarLevelEnv`

**File:** `python/lob_rl/bar_level_env.py`

Same forced flatten logic as PrecomputedEnv. On the terminal step:
- Position forced to 0.0
- Reward = 0.0 (no bar PnL)
- Close cost = `bar_spread_close[bar_index] / 2.0 * abs(prev_position)` — always charged
- Info dict with `forced_flatten`, `forced_flatten_cost`, `intended_action`

Replace the current terminal flatten penalty:
```python
# OLD (remove this)
if terminated and self._position != 0.0:
    reward -= self._bar_spread_close[self._bar_index] / 2.0 * abs(self._position)
```

### 4. MultiDayEnv contract boundary tracking

**File:** `python/lob_rl/multi_day_env.py`

#### 4a. Load instrument_id from cache

When loading from `cache_dir`, read `instrument_id` from each `.npz` file. Store as `self._contract_ids: list[int | None]` parallel to `self._precomputed_days`.

- If the `.npz` has `instrument_id` key: use `int(data["instrument_id"][0])`
- If the `.npz` lacks `instrument_id` key (backward compat): use `None`

When loading from `file_paths` (non-cache path), set all contract_ids to `None`.

#### 4b. Contract boundary assertion

In `reset()`, after creating the new inner env:
- Track `self._prev_contract_id` (initially `None`)
- If the new day's contract_id is not `None` AND `self._prev_contract_id` is not `None` AND they differ:
  - This is a contract boundary crossing
  - Log/warn: `"Contract roll: {prev_contract} → {new_contract} on day {day_index}"`
- Update `self._prev_contract_id = current_contract_id`

The forced flatten in the inner envs guarantees position == 0 at every episode boundary, so no assertion is needed here — the structural guarantee is sufficient. But do include the contract roll info in the returned `info` dict:

```python
info["instrument_id"] = self._contract_ids[file_idx]  # may be None
info["contract_roll"] = (prev != current and prev is not None and current is not None)
```

#### 4c. Expose contract_ids property

Add a read-only property `contract_ids` that returns the list of contract IDs (for testing/debugging).

### 5. Do NOT change

- `bar_aggregation.py` — no changes needed
- The `execution_cost` flag behavior on non-terminal steps — unchanged
- The `participation_bonus` on non-terminal steps — unchanged
- The `observation_space` and `action_space` — unchanged
- The `_build_obs()` method — unchanged

## Edge Cases

1. **Agent chooses flat on terminal step:** `forced_flatten_cost` should be `spread/2 * |prev_position|`. If prev_position was already 0, cost is 0.
2. **All days same contract (no rolls):** `contract_roll` is always False. Forced flatten still applies at end of each day.
3. **Cache files without instrument_id (backward compat):** `contract_ids` entry is `None`, `contract_roll` is always False for these days, `instrument_id` in info is `None`.
4. **Shuffle mode:** Days are visited in random order. Contract boundary detection still works because it compares the current day's contract to the previous episode's contract (in visit order, not calendar order). Every episode ends with forced flatten regardless.
5. **Single-day MultiDayEnv:** No contract boundary possible. Forced flatten still applies.
6. **prev_position is 0 at terminal step:** Close cost is 0 (no position to close). `forced_flatten` is still True.

## Acceptance Criteria

### PrecomputedEnv
- [ ] Terminal step forces position to 0.0 regardless of action
- [ ] Terminal step reward is `-spread/2 * |prev_position|` (no PnL component)
- [ ] Terminal step info contains `forced_flatten=True`, `forced_flatten_cost`, `intended_action`
- [ ] Non-terminal steps are completely unchanged
- [ ] If prev_position is 0 at terminal, close cost is 0

### BarLevelEnv
- [ ] Same forced flatten behavior as PrecomputedEnv (adapted for bar-level)
- [ ] Terminal step info contains same keys
- [ ] Non-terminal steps unchanged

### MultiDayEnv
- [ ] Loads `instrument_id` from `.npz` when present
- [ ] Handles missing `instrument_id` gracefully (backward compat)
- [ ] `reset()` info includes `instrument_id` and `contract_roll`
- [ ] `contract_ids` property returns list of contract IDs
- [ ] Contract roll detection works across shuffled day order

### precompute_cache.py
- [ ] Saves `instrument_id` as uint32 array in `.npz`
- [ ] Existing cache loading code handles files with or without `instrument_id`
