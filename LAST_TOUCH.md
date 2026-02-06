# Last Touch — Cold-Start Briefing

## What to do next

**Execution cost feature is DONE** (PR #3 open). Next: re-train with execution cost enabled and tune hyperparameters.

### Immediate next steps (pick one)

1. **Re-train with execution cost:** `--execution-cost --total-timesteps 5000000` (~25 min). This is the most important next step — the baseline was trained without execution cost so the reward signal was degenerate.
2. **Observation normalization:** Wrap env with `VecNormalize` in `train.py`. Standardizes observations and rewards, helps PPO converge.
3. **Inventory penalty + execution cost combo:** `--execution-cost --reward-mode pnl_delta_penalized --lambda 0.01`
4. **Network architecture:** Try larger MLP (256x256) via SB3 `policy_kwargs`.

### To re-run training with execution cost

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --execution-cost --total-timesteps 5000000
```

### What was just built

**Execution cost** — charges `spread/2 * |new_pos - old_pos|` on every step where the agent changes position. Implemented across the full stack:

| Layer | Change |
|---|---|
| C++ `RewardCalculator` | `execution_cost(old_pos, new_pos, spread)` method |
| C++ `LOBEnv` | `execution_cost` bool param, tracks `prev_position_` |
| pybind11 bindings | `execution_cost` kwarg on all 5 constructor overloads |
| `PrecomputedEnv` | `execution_cost` param, `_prev_position` tracking |
| `MultiDayEnv` | Forwards `execution_cost` to inner env |
| `LOBGymEnv` | Forwards `execution_cost` to C++ |
| `train.py` | `--execution-cost` CLI flag |

**Also:** `.gitignore` now excludes `build-release/`, `data/mes/`, `runs/`. Directory READMEs updated with full API signatures per new CLAUDE.md breadcrumb format.

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point. PPO with SB3, Sortino evaluation. `--execution-cost` flag. |
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
- **508 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **942 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| Re-train with execution cost | High | Baseline was without cost; need new baseline |
| Observation normalization (VecNormalize) | High | Helps PPO with unbounded obs space |
| Hyperparameter tuning | Medium | LR schedule, entropy coeff, clip range |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
