# Checkpointing & Resume for train.py

## Goal

Add `--checkpoint-freq N` and `--resume PATH` CLI arguments to `scripts/train.py` so that training can be periodically saved and resumed after interruption (pod crash, spot preemption, or intentional stop).

## Requirements

### `--checkpoint-freq N`

- New CLI argument on `train.py`. Type: `int`, default: `0` (disabled).
- When `N > 0`, save a checkpoint every `N` timesteps during training.
- Use SB3's built-in `CheckpointCallback` (from `stable_baselines3.common.callbacks`).
- Checkpoints saved to `{output_dir}/checkpoints/` directory (created automatically).
- Checkpoint filenames follow SB3 default: `rl_model_{N}_steps.zip`.
- **Also save VecNormalize stats** alongside each checkpoint when VecNormalize is active (`--no-norm` is NOT set). Save to `{output_dir}/checkpoints/rl_model_{N}_steps_vecnormalize.pkl`.
- VecNormalize saving requires a custom callback (or extending CheckpointCallback) since SB3's CheckpointCallback only saves the model, not the VecNormalize wrapper.

### `--resume PATH`

- New CLI argument on `train.py`. Type: `str`, default: `None`.
- `PATH` points to a previously saved model checkpoint (`.zip` file).
- When provided:
  - Load the model from the checkpoint using `PPO.load()` or `RecurrentPPO.load()` (matching `--recurrent` flag).
  - Pass the current training `env` to `.load()` so the model attaches to the new environment.
  - If a VecNormalize stats file exists alongside the checkpoint (same path but with `_vecnormalize.pkl` suffix instead of `.zip`), load it. Specifically: derive the VecNormalize path by replacing `.zip` with `_vecnormalize.pkl` in the resume path. If this file exists, load VecNormalize stats via `VecNormalize.load()` and wrap the env.
  - Training continues with `model.learn()` using the remaining timesteps.
  - `reset_num_timesteps=False` must be passed to `model.learn()` so that the timestep counter continues from the checkpoint.
- When `--resume` is NOT provided, behavior is unchanged (train from scratch, `reset_num_timesteps=True` is the default).

### Interaction with existing flags

- `--resume` works with both `--recurrent` and non-recurrent modes.
- `--resume` works with `--checkpoint-freq` (resume training and continue checkpointing).
- `--resume` requires the same `--cache-dir` / `--data-dir` as the original run (but this is the user's responsibility, not validated).
- `--resume` with `--no-norm`: skip VecNormalize loading entirely.

## Acceptance Criteria

1. `--checkpoint-freq 500 --total-timesteps 1000` produces at least one checkpoint file in `{output_dir}/checkpoints/`.
2. When VecNormalize is active, checkpoint also saves `_vecnormalize.pkl` alongside each model checkpoint.
3. `--resume PATH` loads a model and continues training with `reset_num_timesteps=False`.
4. `--resume PATH` auto-loads VecNormalize stats if the `_vecnormalize.pkl` file exists next to the checkpoint.
5. `--resume PATH --no-norm` skips VecNormalize loading.
6. `--resume PATH --recurrent` uses `RecurrentPPO.load()` instead of `PPO.load()`.
7. The `--checkpoint-freq` argument appears in `--help` output.
8. The `--resume` argument appears in `--help` output.

## Edge Cases

- `--checkpoint-freq 0` (default): no checkpointing, no callbacks. Existing behavior unchanged.
- `--resume` with non-existent path: should raise a clear error (SB3 will raise `FileNotFoundError`).
- `--resume` with a path that has no adjacent `_vecnormalize.pkl`: proceed without loading VecNormalize stats (print a warning).
- `--checkpoint-freq` without `--resume`: save checkpoints from a fresh training run.

## Implementation Notes

- `CheckpointCallback` is imported from `stable_baselines3.common.callbacks`.
- For VecNormalize checkpoint saving, create a simple custom callback that saves `env.save()` at each checkpoint interval. This can be a `BaseCallback` subclass or an `EveryNTimesteps` wrapper.
- Use `CallbackList` to combine `CheckpointCallback` with the VecNormalize-saving callback.
- The `model.learn()` call currently has no `callback` parameter — add `callback=callback_list` when `--checkpoint-freq > 0`.
- For `--resume`, the model loading happens INSTEAD of creating a new `PPO(...)` or `RecurrentPPO(...)`. The `env` is still created the same way.

## Files to Modify

- `scripts/train.py` — add args, callbacks, resume logic.
