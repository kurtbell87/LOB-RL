# T9b: Barrier Training Script

## What to Build

The standalone CLI training script `scripts/train_barrier.py` that wires together all T9 modules (MultiSessionBarrierEnv, barrier_vec_env, training_diagnostics) into an end-to-end training pipeline for the barrier-hit PPO agent. This is the missing piece from T9 — all supporting modules exist and are tested; this script provides the CLI entry point.

**Spec reference:** Section 5 (PPO Configuration), orchestrator T9.

## Dependencies

- `lob_rl.barrier.multi_session_env` — `MultiSessionBarrierEnv`
- `lob_rl.barrier.barrier_vec_env` — `make_barrier_vec_env`
- `lob_rl.barrier.training_diagnostics` — `BarrierDiagnosticCallback`, `linear_schedule`
- `lob_rl.barrier._sb3_compat` — SB3 compatibility shim
- `lob_rl.barrier.bar_pipeline` — `build_session_bars`
- `lob_rl.barrier.label_pipeline` — `compute_labels`
- `lob_rl.barrier.feature_pipeline` — `build_feature_matrix`
- `lob_rl.barrier.reward_accounting` — `RewardConfig`
- `sb3_contrib` — `MaskablePPO`
- `stable_baselines3` — `VecNormalize`, callbacks
- `torch.nn` — `ReLU`
- `argparse`, `pathlib`, `json`, `numpy`

## Script Design

### File: `scripts/train_barrier.py`

```python
#!/usr/bin/env python3
"""Train barrier-hit PPO agent using MaskablePPO."""

def parse_args():
    """Parse CLI arguments."""
    # --data-dir PATH        Directory with .dbn.zst or precomputed session data
    # --output-dir PATH      Where to save model, logs, metrics
    # --bar-size INT         Trade bar size (default: 500)
    # --lookback INT         Feature lookback h (default: 10)
    # --n-envs INT           Parallel environments (default: 4)
    # --total-timesteps INT  Total training steps (default: 2_000_000)
    # --eval-freq INT        Steps between eval runs (default: 10_000)
    # --checkpoint-freq INT  Steps between checkpoints (default: 50_000)
    # --train-frac FLOAT     Fraction of sessions for training (default: 0.8)
    # --seed INT             Random seed (default: 42)
    # --resume PATH          Resume from checkpoint
    # --no-normalize         Skip VecNormalize wrapper

def load_session_data(data_dir, bar_size, lookback, config):
    """Load and preprocess session data from .dbn.zst files.

    Returns list[dict] where each dict has 'bars', 'labels', 'features' keys.
    """

def split_sessions(session_data, train_frac, seed):
    """Split session data into train/val/test.

    Returns (train_data, val_data, test_data).
    Default split: train_frac / (1-train_frac)/2 / (1-train_frac)/2.
    """

def build_model(env, seed, resume_path=None):
    """Create or load MaskablePPO model with Section 5.2 hyperparameters.

    policy_kwargs = dict(
        net_arch=[256, 256, dict(pi=[64], vf=[64])],
        activation_fn=nn.ReLU,
    )
    MaskablePPO(
        'MlpPolicy', env,
        learning_rate=linear_schedule(1e-4),
        n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    )
    """

def main():
    """Main training loop."""
    # 1. Parse args
    # 2. Load session data
    # 3. Split train/val/test
    # 4. Create train vec env + optional VecNormalize
    # 5. Create eval vec env
    # 6. Create model with Section 5.2 hyperparameters
    # 7. Set up callbacks (diagnostic, eval, checkpoint)
    # 8. Train
    # 9. Save final model + diagnostics summary
```

### Section 5.2 Hyperparameters (hardcoded)

| Parameter | Value |
|-----------|-------|
| Learning rate | `1e-4` with linear decay to 0 |
| n_steps (batch) | 2048 |
| batch_size (mini-batch) | 256 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| net_arch | `[256, 256, dict(pi=[64], vf=[64])]` |
| activation_fn | `nn.ReLU` |

### Data Loading Strategy

The script supports two data source modes:
1. **Raw .dbn.zst**: Discover files in `--data-dir`, call `build_session_bars()` → `compute_labels()` → `build_feature_matrix()` for each.
2. **Precomputed .npz**: If `--data-dir` contains `.npz` files with `bars`, `labels`, `features` keys, load directly. (Future optimization.)

For now, only mode 1 is implemented. Mode 2 is a performance optimization for repeated training.

## Test Cases

### Script Argument Parsing (4 tests)

1. **`test_parse_args_defaults`**: Default values match spec (bar_size=500, lookback=10, n_envs=4, total_timesteps=2_000_000, etc.).

2. **`test_parse_args_custom`**: Custom values override defaults.

3. **`test_parse_args_resume`**: `--resume PATH` correctly stored.

4. **`test_parse_args_no_normalize`**: `--no-normalize` flag works.

### Session Data Loading (3 tests)

5. **`test_split_sessions_proportions`**: 80/10/10 split with correct proportions.

6. **`test_split_sessions_deterministic`**: Same seed gives same split.

7. **`test_split_sessions_no_overlap`**: Train/val/test have no overlapping sessions.

### Model Building (3 tests)

8. **`test_build_model_hyperparameters`**: Model has Section 5.2 hyperparameters (lr schedule, n_steps, batch_size, etc.).

9. **`test_build_model_architecture`**: Model network has [256,256] shared + [64] pi + [64] vf.

10. **`test_build_model_resume`**: `build_model(resume_path=...)` loads from checkpoint.

### End-to-End Smoke Test (2 tests)

11. **`test_main_smoke_synthetic`**: Run `main()` with synthetic data for 2048 steps. Produces output dir with model + diagnostics.

12. **`test_main_smoke_with_checkpoint`**: Run, checkpoint, verify checkpoint files exist.

## Implementation Notes

- The script imports from existing tested modules. No new logic — just CLI wiring.
- `load_session_data()` requires .dbn.zst files which depend on the databento C++ library. Tests use synthetic data via `generate_random_walk()` and `build_bars_from_trades()`.
- The script writes to `--output-dir`: model.zip, diagnostics.csv, vec_normalize.pkl (if used), tb/ (tensorboard logs).
- For testing, override `--total-timesteps 2048 --n-envs 1 --eval-freq 2048` to keep tests fast.
- `split_sessions()` shuffles with seed then splits sequentially. The split is deterministic.
- VecNormalize is optional (`--no-normalize`). Features are already z-scored but position/PnL aren't.
