# RecurrentPPO for train.py

## What to build

Add `--recurrent` flag to `scripts/train.py` that switches from PPO/MlpPolicy to RecurrentPPO/MlpLstmPolicy from `sb3-contrib`.

## Why

The MLP sees one bar at a time (21-dim). While frame stacking concatenates N bars explicitly, an LSTM can learn to attend to relevant history across the full episode. This tests whether temporal recurrence improves OOS generalization.

## Requirements

### New CLI argument

1. `--recurrent` — `action='store_true'`, default `False`. When set, uses `RecurrentPPO` from `sb3_contrib` instead of `PPO`.

### Mutual exclusivity

`--recurrent` + `--frame-stack > 1` should raise a parser error. These are alternative approaches to temporal context and should not be combined. The validation should be:
```python
if args.recurrent and args.frame_stack > 1:
    parser.error("--recurrent and --frame-stack > 1 are mutually exclusive")
```

### Conditional import

When `--recurrent` is set, import `RecurrentPPO` from `sb3_contrib`:
```python
if args.recurrent:
    from sb3_contrib import RecurrentPPO
```
This import should be inside `main()`, after arg parsing, only when needed. `sb3_contrib` is NOT imported at the top level.

### Model creation

When `--recurrent` is set (line ~318 area), replace:
```python
model = PPO('MlpPolicy', env, ...)
```
with:
```python
model = RecurrentPPO('MlpLstmPolicy', env, ...)
```
All other PPO hyperparameters (learning_rate, n_steps, batch_size, n_epochs, gamma, ent_coef, vf_coef, max_grad_norm, clip_range, policy_kwargs, tensorboard_log) stay the same.

### Evaluation changes

`evaluate_sortino()` needs a new `is_recurrent` parameter (default `False`). When `is_recurrent=True`, the predict loop changes to track LSTM state:

```python
lstm_states = None
episode_start = np.ones((1,), dtype=bool)
# in the loop:
action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
episode_start = dones
```

Both val and test eval calls must forward `is_recurrent=args.recurrent`.

### New dependency

`sb3-contrib` must be installed. The spec requires `uv pip install sb3-contrib` to be run beforehand. The test should verify that the import path (`from sb3_contrib import RecurrentPPO`) is correct.

### Implementation scope

- Only `scripts/train.py` is modified.
- No changes to `MultiDayEnv`, `PrecomputedEnv`, `BarLevelEnv`, or any other module.

## Edge cases

- `--recurrent` without `--frame-stack` — works fine (frame_stack defaults to 1).
- `--recurrent` with `--frame-stack 1` — works fine (1 means no stacking, no conflict).
- `--recurrent` with `--frame-stack 4` — parser error.
- `policy_kwargs` — same for both PPO and RecurrentPPO (MlpLstmPolicy respects net_arch and activation_fn).

## Acceptance criteria

1. `--recurrent` flag exists with `action='store_true'`.
2. `--recurrent` + `--frame-stack > 1` raises parser error.
3. When `--recurrent`, `RecurrentPPO` is imported from `sb3_contrib`.
4. When `--recurrent`, model is `RecurrentPPO('MlpLstmPolicy', ...)`.
5. When not `--recurrent`, model is `PPO('MlpPolicy', ...)` (unchanged).
6. `evaluate_sortino()` accepts `is_recurrent` parameter (default `False`).
7. When `is_recurrent=True`, eval uses LSTM state tracking (`state=lstm_states`, `episode_start=`).
8. Both val and test eval calls forward `is_recurrent=args.recurrent`.
9. `sb3_contrib` is NOT imported at the top level (conditional import only).
10. No other files are modified.

## Test approach

Tests should follow the existing pattern: read `train.py` source and use regex/string matching to verify structure. Additionally, verify that `sb3_contrib` is importable (as a dependency check).
