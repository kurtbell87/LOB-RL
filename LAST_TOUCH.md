# Last Touch — Cold-Start Briefing

## What to do next

**Training pipeline v2 is DONE** (PR #4 merged). The training script now has VecNormalize, 8 parallel envs, entropy regularization, and all key hyperparameters exposed as CLI flags. Next: run a training experiment.

### Immediate next step

Run the improved training pipeline and evaluate:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --execution-cost --total-timesteps 2000000
```

Default flags now include `--ent-coef 0.01`, `--n-envs 8`, `--batch-size 256`, `--n-epochs 5`, VecNormalize on. Monitor in TensorBoard.

### After the training run

1. Check TensorBoard for entropy, policy loss, value loss curves.
2. If entropy still collapses, increase `--ent-coef` to 0.03 or 0.1.
3. If training is stable, extend to `--total-timesteps 5000000`.
4. Try `--reward-mode pnl_delta_penalized --lambda 0.01` for inventory penalty.
5. Try larger network: add `policy_kwargs=dict(net_arch=[256, 256])` to PPO.

### What was just built

**Training pipeline v2** (PR #4) — fixed critical training issues:

| Change | Before | After |
|---|---|---|
| `ent_coef` | 0.0 (!) | 0.01 (CLI tunable) |
| Observation normalization | None | VecNormalize (obs + reward) |
| Parallel envs | 1 (DummyVecEnv) | 8 (SubprocVecEnv) |
| Batch size | 64 | 256 |
| Epochs per rollout | 10 | 5 |
| Eval execution_cost | Not forwarded (bug) | Forwarded correctly |
| Eval VecNormalize | None | Loaded in eval mode |
| Hyperparameter CLI flags | 7 | 13 |

Also: minor C++ refactor (eliminated intermediate buffer in `precompute.cpp`).

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point. PPO with SB3, VecNormalize, SubprocVecEnv. 13 CLI flags. |
| `python/lob_rl/precomputed_env.py` | Pure-numpy env. `execution_cost` param charges spread on position change. |
| `python/lob_rl/multi_day_env.py` | Multi-day env. Forwards `execution_cost` to inner `PrecomputedEnv`. |
| `include/lob/reward.h` | `RewardCalculator` with `compute()`, `flattening_penalty()`, `execution_cost()`. |
| `include/lob/env.h` | `LOBEnv` — two constructors, both accept `execution_cost` bool. |

## Don't waste time on

- **Build verification** — `build-release/` is current, 434 C++ tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead (every key dir has one, now with full API signatures).

## Architecture overview

```
data/mes/*.bin  →  BinaryFileSource (C++)  →  Book (C++)  →  LOBEnv (C++)
                                                                   ↓
                                                           precompute() (C++)
                                                                   ↓
                                                         numpy arrays (Python)
                                                                   ↓
                                                    MultiDayEnv (Python/Gym)
                                                                   ↓
                                                      SB3 PPO (scripts/train.py)
```

## Test coverage

- **434 C++ tests** — `cd build-release && ./lob_tests`
- **580 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1014 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| Run training with v2 pipeline | High | VecNormalize + ent_coef + 8 envs; evaluate Sortino |
| Hyperparameter sweep if needed | Medium | ent_coef, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
