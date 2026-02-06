# Last Touch — Cold-Start Briefing

## What to do next

**Temporal features DONE** (PR #6 merged). Observation expanded from 44 to 54 dimensions. 10 new Python-computed features: mid-price returns (4 lookbacks), volatility_20, imbalance deltas (5/20), microprice offset, total volume imbalance, spread change. The model now has momentum/volatility/book-pressure signals.

### Immediate next step

**Fix the spread calculation in C++ precompute.** The `spread` array from `lob_rl_core.precompute()` is wrong:
- Always negative (100% of values) — should be positive (ask - bid)
- Mean magnitude ~37 points (~150 ticks) — should be ~0.25-0.50 for /MES
- This breaks execution cost and terminal flattening penalty

After fixing spread, need **coarser time sampling** to reduce bid-ask bounce dominance. Current sampling (~4.6 steps/sec) gives autocorrelation of -0.75, making trivial mean reversion the dominant strategy. A coarser time scale (e.g., 1 step/sec or 1 step per N messages) would shift the signal toward genuine directional moves.

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |
| **Temporal features** | 2M steps, 54-dim obs, participation_bonus=0.01 | Sortino inf val. **But PnL is bid-ask bounce mean reversion, not real alpha.** Autocorr(1)=-0.75. Simple mean-reversion rule matches agent PnL. Entropy 0.07. |

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
| **Fix spread in C++ precompute** | **Critical** | `spread` is negative and ~150x too large. Must fix before execution cost or any realistic training. Check `env.h` or `precompute` logic. |
| **Coarser time sampling** | **High** | Current ~4.6 steps/sec has autocorr -0.75 (bid-ask bounce). Need step_interval or time-based sampling to get genuine directional signal. |
| Re-train with correct spread + exec cost | High | After spread fix, verify mean reversion no longer dominates |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
