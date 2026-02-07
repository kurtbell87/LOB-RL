# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Verify the fix with real data, then re-train.** The precompute event handling is fixed (PR #7). Next:

1. **Re-run precompute** on real data and verify spread is positive (~0.25 for /MES) and mid prices are realistic. Quick smoke test:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python -c "
   import lob_rl_core as core
   day = core.precompute('../data/mes/20241226.bin')
   import numpy as np
   spreads = np.array(day.spread)
   print(f'Steps: {day.num_steps}, Spread: min={spreads.min():.4f} mean={spreads.mean():.4f} max={spreads.max():.4f}')
   print(f'Negative spreads: {(spreads < 0).sum()} / {len(spreads)}')
   "
   ```

2. **Re-train with `--execution-cost`** — with correct spread, execution costs will properly penalize the bid-ask bounce strategy:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
     --data-dir ../data/mes --participation-bonus 0.01 --execution-cost
   ```

3. **Add coarser time sampling** — current ~4.6 steps/sec gives autocorr=-0.75 (bid-ask bounce). Need step_interval or time-based sampling.

### What was just completed

**Fix precompute event handling (PR #7).** Four C++ changes:

1. `include/lob/message.h` — Added `uint8_t flags = 0` to `Message` struct
2. `src/data/binary_file_source.cpp` — Copies `rec.flags` to `msg.flags` in `convert()`
3. `src/engine/book.cpp` — `Book::apply()` treats `Action::Trade` as no-op (Databento spec)
4. `src/env/precompute.cpp` — Flag-aware snapshotting mode:
   - Auto-detects flag-aware vs legacy mode (checks if any RTH message has non-zero flags)
   - Buffers mid-event messages, only applies complete events on F_LAST
   - Skips F_SNAPSHOT records entirely
   - Rejects crossed/locked books (spread <= 0)
   - Legacy mode preserved for backward compatibility with synthetic test data

Also updated 5 old Book tests that expected Trade to modify the book — now correctly expect no-op per Databento spec.

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |
| **Temporal features** | 2M steps, 54-dim obs, participation_bonus=0.01 | Sortino inf val. **But PnL is bid-ask bounce mean reversion, not real alpha.** Autocorr(1)=-0.75. Caused by broken spread/precompute. |
| **Post-fix** | Not yet run | Spread should now be correct. Re-train needed. |

## Key files for current task

| File | Role |
|---|---|
| `src/env/precompute.cpp` | Flag-aware precompute — dual-mode (legacy + flag-aware) |
| `scripts/train.py` | Training entry point — 14 CLI flags |
| `python/lob_rl/precomputed_env.py` | PrecomputedEnv — uses precompute() output |
| `python/lob_rl/multi_day_env.py` | MultiDayEnv — wraps multiple days |

## Reference material

- **Databento DBN spec cloned to `references/dbn/`** — authoritative source for MBO record layout, flag definitions, and event semantics
- Key files: `references/dbn/rust/dbn/src/flags.rs` (flag constants), `references/dbn/rust/dbn/src/record.rs` (MboMsg struct), `references/dbn/rust/dbn/src/enums.rs` (Action/Side enums)

## Don't waste time on

- **Build verification** — `build-release/` is current, 489 C++ tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating the Python converter** — it's correct. No changes needed.
- **Investigating lookahead in reward or temporal features** — fully audited, all clean.
- **Investigating walk-forward / VecNormalize leakage** — fully audited, all clean.
- **The precompute fix** — it's done and merged. PR #7.

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

- **489 C++ tests** — `cd build-release && ./lob_tests`
- **696 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1185 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| **Verify spread fix on real data** | **Critical** | Run precompute on real data, check spreads are positive |
| **Re-train with correct spread + exec cost** | **High** | First real training run on clean data |
| **Coarser time sampling** | **High** | Step_interval or time-based sampling to reduce autocorr from -0.75. |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
