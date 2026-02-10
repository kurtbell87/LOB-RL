# T8: Barrier-Hit Trading Environment

## What to Build

A new Gymnasium-compatible environment `python/lob_rl/barrier/barrier_env.py` implementing the barrier-hit trading environment described in spec Section 4. One episode = one RTH trading session. The agent enters long/short positions, and exits are mechanical via barrier hits or timeout.

**Spec reference:** Section 4 (Environment), Section 8 step 8.

## Dependencies

- `bar_pipeline.py` — `TradeBar` with OHLCV, timestamps
- `label_pipeline.py` — `BarrierLabel` with label, tau, resolution_type, entry_price, resolution_bar
- `feature_pipeline.py` — `build_feature_matrix()` returns (N-h+1, 13*h) feature array
- `reward_accounting.py` — `RewardConfig`, `PositionState`, `compute_entry()`, `compute_hold_reward()`, `compute_unrealized_pnl()`, `get_action_mask()`, action constants
- `gymnasium` — `gymnasium.Env`, `gymnasium.spaces.Discrete`, `gymnasium.spaces.Box`
- `numpy`

## Environment Design

### Constructor

```python
class BarrierEnv(gymnasium.Env):
    def __init__(self, bars, labels, features, config=None):
```

**Parameters:**
- `bars: list[TradeBar]` — Bars for one session.
- `labels: list[BarrierLabel]` — Pre-computed barrier labels aligned with bars.
- `features: np.ndarray` — Pre-computed feature matrix from `build_feature_matrix()`, shape `(N_usable, 13*h)`.
- `config: RewardConfig | None` — Reward parameters. Defaults to `RewardConfig()`.

**Spaces:**
- `observation_space = Box(low=-np.inf, high=np.inf, shape=(13*h + 2,), dtype=np.float32)` — 130 features + position + unrealized_pnl = 132-dim.
- `action_space = Discrete(4)` — long=0, short=1, flat=2, hold=3.

### Observation Layout

```
obs[0:130]  = feature vector Z_k (13 features × 10 lookback)
obs[130]    = position (-1, 0, or +1)
obs[131]    = unrealized PnL in ticks (0.0 if flat)
```

### Episode Structure

One episode = one session's bars. The environment steps through bars sequentially (bar index advances by 1 each step).

- `reset()` → sets bar index to 0, position to flat, returns first observation.
- `step(action)` → processes action, checks barriers, returns (obs, reward, terminated, truncated, info).
- Episode terminates (`terminated=True`) when all bars are consumed (bar index reaches end).
- If holding a position at episode end, force-close at MTM (treat as timeout).
- `truncated` is always `False` (no time limit beyond session end).

### Step Logic

1. Get current bar at `self._bar_idx`.
2. If flat (`position == 0`):
   - If action is `long` or `short`: call `compute_entry()`, reward = 0.
   - If action is `flat`: reward = 0, stay flat.
   - If action is `hold`: invalid (masked), but handle gracefully — treat as `flat`.
3. If holding (`position != 0`):
   - Increment bar index (agent is forced to hold, the next bar is checked for barrier).
   - Call `compute_hold_reward()` on the new bar.
   - If barrier hit or timeout: position returns to flat, reward is the exit reward.
   - If neither: reward = 0, continue holding.
4. Advance bar index.
5. If bar index >= number of usable bars: episode terminates. If still holding, force-close.
6. Build and return observation.

### Action Masking

The environment exposes `action_masks()` returning a numpy array of shape `(4,)` with dtype `np.int8` (1 = valid, 0 = invalid). This is the SB3 MaskablePPO convention.

### Info Dict

The `info` dict returned by `step()` contains:
- `position: int` — Current position after this step.
- `bar_idx: int` — Current bar index.
- `exit_type: str | None` — `"profit"`, `"stop"`, `"timeout"`, `"force_close"`, or `None`.
- `entry_price: float | None` — Entry price if holding, `None` if flat.
- `n_trades: int` — Cumulative number of completed trades in this episode.

### `from_bars` Factory

```python
@classmethod
def from_bars(cls, bars, a=20, b=10, t_max=40, h=10, config=None):
```

Convenience factory that computes labels and features from raw bars. For testing only — production code pre-computes these.

## Test Cases

### Gymnasium API Compliance (5 tests)

1. **`test_reset_returns_obs_and_info`**: `reset()` returns `(obs, info)` where obs is np.ndarray with correct dtype and shape.

2. **`test_step_returns_five_tuple`**: `step(action)` returns `(obs, reward, terminated, truncated, info)`.

3. **`test_observation_shape`**: Observation shape is `(132,)` = 130 features + 2 position state (with h=10).

4. **`test_action_space_discrete_4`**: `env.action_space` is `Discrete(4)`.

5. **`test_observation_space_box`**: `env.observation_space` is `Box` with correct shape and dtype `float32`.

### Observation Content (4 tests)

6. **`test_obs_features_match_feature_matrix`**: The first 130 elements of obs match the corresponding row of the feature matrix.

7. **`test_obs_position_field`**: `obs[130]` reflects the current position (-1, 0, or +1).

8. **`test_obs_unrealized_pnl_field`**: `obs[131]` reflects the current unrealized PnL in ticks.

9. **`test_obs_initial_flat`**: After reset, `obs[130] == 0` (flat) and `obs[131] == 0.0` (no PnL).

### Action Masking (5 tests)

10. **`test_action_masks_flat`**: When flat, mask is `[1, 1, 1, 0]` (long, short, flat valid; hold invalid).

11. **`test_action_masks_long`**: When long, mask is `[0, 0, 0, 1]` (only hold valid).

12. **`test_action_masks_short`**: When short, mask is `[0, 0, 0, 1]` (only hold valid).

13. **`test_action_masks_shape_and_dtype`**: `action_masks()` returns np.ndarray of shape `(4,)` with dtype `np.int8`.

14. **`test_action_masks_updates_after_entry`**: Mask changes from `[1,1,1,0]` to `[0,0,0,1]` after entering a position.

### Episode Lifecycle (5 tests)

15. **`test_episode_terminates_at_end`**: Episode terminates after all bars consumed.

16. **`test_episode_length_matches_usable_bars`**: Number of steps in episode matches number of usable bars from feature matrix.

17. **`test_force_close_at_session_end`**: If holding when episode ends, position is force-closed with MTM reward.

18. **`test_truncated_always_false`**: `truncated` is always `False`.

19. **`test_reset_clears_state`**: After reset, position is flat, bar_idx is 0, PnL is 0.

### Reward Accounting Integration (6 tests)

20. **`test_entry_reward_zero`**: Entering a position yields reward = 0.

21. **`test_profit_hit_reward`**: When a bar triggers profit barrier, reward = +G - C.

22. **`test_stop_hit_reward`**: When a bar triggers stop barrier, reward = -L - C.

23. **`test_timeout_reward_mtm`**: When hold_counter reaches T_max, reward = MTM - C.

24. **`test_flat_action_reward_zero`**: Staying flat yields reward = 0.

25. **`test_force_close_reward_mtm`**: Force close at session end yields MTM - C reward.

### Position State Transitions (4 tests)

26. **`test_flat_to_long`**: After entry with action=0 (long), position is +1.

27. **`test_flat_to_short`**: After entry with action=1 (short), position is -1.

28. **`test_long_to_flat_on_barrier`**: After profit/stop hit on long position, position returns to 0.

29. **`test_position_persists_during_hold`**: Position stays constant while holding without barrier hit.

### Random Agent (3 tests)

30. **`test_random_agent_completes_episode`**: A random agent (sampling from valid actions) completes an episode without crashing.

31. **`test_random_agent_100_episodes`**: Run 100 episodes with random actions. All complete without error.

32. **`test_random_agent_mean_reward`**: Mean per-trade reward across 100 episodes is approximately -0.20 (within ±0.50 — wide tolerance for synthetic data).

### Info Dict (3 tests)

33. **`test_info_has_required_keys`**: Info dict contains `position`, `bar_idx`, `exit_type`, `entry_price`, `n_trades`.

34. **`test_info_exit_type_on_barrier_hit`**: `exit_type` is `"profit"` or `"stop"` when barrier fires.

35. **`test_info_n_trades_increments`**: `n_trades` increments by 1 each time a trade completes.

### Edge Cases (4 tests)

36. **`test_invalid_action_when_holding`**: If an invalid action (e.g., `long` when already long) is passed, the environment handles it gracefully (defaults to `hold`).

37. **`test_invalid_action_when_flat`**: If `hold` action is passed when flat, the environment handles it gracefully (defaults to `flat`).

38. **`test_single_bar_episode`**: Environment with only 1 usable bar terminates immediately after one step.

39. **`test_config_forwarded_to_reward`**: Custom `RewardConfig` values are used in reward computation (e.g., different C value changes reward).

### Factory Method (2 tests)

40. **`test_from_bars_creates_env`**: `BarrierEnv.from_bars(bars)` creates a valid environment.

41. **`test_from_bars_obs_shape_correct`**: Environment from `from_bars()` has correct observation shape.

## Implementation Notes

- The environment uses pre-computed labels and features. It does NOT recompute labels or features during `step()`. This ensures determinism and avoids lookahead.
- The feature matrix from `build_feature_matrix()` drops the first `h-1` bars (insufficient lookback). The environment should align bar indices accordingly: usable bar index `i` corresponds to `bars[i + h - 1]`.
- Labels are also aligned: `labels[i]` corresponds to `bars[i]`. The environment uses `labels[i + h - 1]` for the bar at feature row `i`.
- The `compute_hold_reward()` from reward_accounting.py handles barrier checking. The environment just feeds it the current bar and state.
- Action masking uses np.int8 `[1,1,1,0]` or `[0,0,0,1]` format (SB3 MaskablePPO convention).
- For the random agent tests, use synthetic bars from `generate_random_walk()` with enough bars to complete meaningful episodes.
- The environment should work with any bar count >= 1.
- `from_bars()` factory is for convenience/testing. It calls `compute_labels()` and `build_feature_matrix()` internally.
- Force-close at session end uses the same MTM calculation as timeout (from `compute_hold_reward` with hold_counter set to T_max).
- The unrealized PnL in the observation is in **ticks** (not barrier units), matching `compute_unrealized_pnl()` from reward_accounting.
