# Last Touch — Cold-Start Briefing

## What to do next

**Participation bonus DONE** (PR #5 merged). Agent was converging to "don't trade" because holding flat = zero risk. The participation bonus rewards market exposure: `bonus * abs(position)`.

### Immediate next step

Train with participation bonus to incentivize market participation:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --participation-bonus 0.01 --total-timesteps 2000000
```

If the agent trades but loses money, try with execution cost too:
```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --participation-bonus 0.01 --execution-cost --total-timesteps 2000000
```

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |

### What was just built

**Participation bonus** (PR #5) — per-step reward for market exposure:

```
reward += participation_bonus * abs(position)
```

Full-stack implementation: C++ `RewardCalculator::participation_bonus()`, `LOBEnv`, pybind11 bindings, `PrecomputedEnv`, `MultiDayEnv`, `LOBGymEnv`, `train.py --participation-bonus` flag. Also refactored: extracted `DEFAULT_SESSION_CONFIG` in train.py, cleaned up bindings.cpp.

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point. PPO with SB3, VecNormalize, SubprocVecEnv. 14 CLI flags incl `--participation-bonus`. |
| `python/lob_rl/precomputed_env.py` | Pure-numpy env. `execution_cost`, `participation_bonus` params. |
| `python/lob_rl/multi_day_env.py` | Multi-day env. Forwards all reward params to inner `PrecomputedEnv`. |
| `include/lob/reward.h` | `RewardCalculator` with `compute()`, `flattening_penalty()`, `execution_cost()`, `participation_bonus()`. |
| `include/lob/env.h` | `LOBEnv` — two constructors, both accept `execution_cost` and `participation_bonus`. |

## Don't waste time on

- **Build verification** — `build-release/` is current, 460 C++ tests pass.
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

- **460 C++ tests** — `cd build-release && ./lob_tests`
- **626 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1086 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| Train with participation bonus | High | `--participation-bonus 0.01`; check if agent trades |
| Train with bonus + exec cost | High | `--participation-bonus 0.01 --execution-cost`; realistic setup |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
