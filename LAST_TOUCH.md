# Last Touch — Cold-Start Briefing

## What to do next

**Temporal features DONE** (PR #6 merged). Observation expanded from 44 to 54 dimensions. 10 new Python-computed features: mid-price returns (4 lookbacks), volatility_20, imbalance deltas (5/20), microprice offset, total volume imbalance, spread change. The model now has momentum/volatility/book-pressure signals.

### Immediate next step

Train with temporal features to see if the agent can learn directionality:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --participation-bonus 0.01 --total-timesteps 2000000
```

If results improve, try with execution cost for realism:
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
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |

### What was just built

**Temporal features** (PR #6) — 10 engineered features expanding obs from 44 to 54:

| Index | Feature | Description |
|-------|---------|-------------|
| 43 | mid_return_1 | 1-step mid-price return |
| 44 | mid_return_5 | 5-step mid-price return |
| 45 | mid_return_20 | 20-step mid-price return |
| 46 | mid_return_50 | 50-step mid-price return |
| 47 | volatility_20 | 20-step rolling std of returns (vectorized O(N)) |
| 48 | imb_delta_5 | 5-step book imbalance change |
| 49 | imb_delta_20 | 20-step book imbalance change |
| 50 | microprice_offset | (microprice - mid) / spread |
| 51 | total_vol_imb | (bid_vol - ask_vol) / (bid_vol + ask_vol) across 10 levels |
| 52 | spread_change_5 | 5-step spread change |
| 53 | position | Agent's current position (moved from index 43) |

Implementation: `PrecomputedEnv._precompute_temporal_features()` computes all at construction from numpy arrays. `_build_obs()` copies 10-element slice per step. `LOBGymEnv` still produces 44-dim (C++ only).

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point. `make_env()` returns `PrecomputedEnv`. 14 CLI flags. |
| `python/lob_rl/precomputed_env.py` | Pure-numpy env. 54-dim obs. Temporal features, execution_cost, participation_bonus. |
| `python/lob_rl/multi_day_env.py` | Multi-day env. 54-dim obs. Forwards all params to inner `PrecomputedEnv`. |
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
                                              PrecomputedEnv + temporal features (Python)
                                                                   ↓
                                                    MultiDayEnv (Python/Gym)
                                                                   ↓
                                                      SB3 PPO (scripts/train.py)
```

## Test coverage

- **460 C++ tests** — `cd build-release && ./lob_tests`
- **696 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1156 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| Train with temporal features | High | 54-dim obs, `--participation-bonus 0.01`, 2M steps |
| Train with temporal + exec cost | High | Add `--execution-cost` for realistic setup |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
