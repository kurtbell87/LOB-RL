# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Re-train with `--step-interval 10 --execution-cost` (drop participation bonus).** The previous run (500k steps, exec cost + participation bonus 0.01) showed the agent exploiting participation bonus: it takes a position early and holds forever, collecting ~$1370/episode in bonus alone. Entropy collapsed to 0.01.

The step-interval feature is now available (PR #8). Next experiment:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --data-dir ../data/mes --execution-cost --step-interval 10 --total-timesteps 2000000
```

With `--step-interval 10`, each day has ~13k steps instead of ~137k, which:
- Reduces bid-ask bounce autocorrelation (the original motivation)
- Makes participation bonus less dominant (if used at all)
- Gives the agent a ~2.2 sec decision cadence instead of ~0.22 sec

If the agent still collapses to flat, try `--step-interval 10 --participation-bonus 0.001` (10x smaller bonus).

### What was just completed

**Step interval feature (PR #8).** Python-only change — subsamples precomputed arrays:
1. `python/lob_rl/precomputed_env.py` — `step_interval=1` parameter, subsamples obs/mid/spread before temporal feature computation
2. `python/lob_rl/multi_day_env.py` — forwards `step_interval` to inner env
3. `scripts/train.py` — `--step-interval N` CLI flag, forwarded through make_env/make_train_env/evaluate_sortino

Also refactored: extracted `_lagged_diff()` helper, added obs index constants, deduplicated `make_realistic_obs()` into conftest.py.

**Spread verification DONE.** All days have min spread=0.25 (one /MES tick), mean~0.37, no negative or zero spreads.

**Training run (500k, exec cost + participation bonus 0.01):**
- Val: mean_return=1273, sortino=inf, 4/4 positive — but driven by participation bonus accumulation, not real PnL
- Entropy collapsed to 0.01 by 300k steps
- Agent learned "take position, hold forever, collect bonus"

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |
| Temporal features | 2M steps, 54-dim obs, participation_bonus=0.01 | Sortino inf val. But PnL is bid-ask bounce mean reversion, not real alpha. Autocorr(1)=-0.75. Caused by broken spread/precompute. |
| **Post-fix + exec cost + bonus** | 500k steps, exec_cost, participation_bonus=0.01 | Val: mean=1273, sortino=inf. Entropy collapsed (0.01). Agent exploits participation bonus (hold forever). |
| **Next: step-interval** | 2M steps, exec_cost, step_interval=10, no bonus | Not yet run. |

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point — 15 CLI flags incl. `--step-interval` |
| `python/lob_rl/precomputed_env.py` | PrecomputedEnv — step_interval subsampling + temporal features |
| `python/lob_rl/multi_day_env.py` | MultiDayEnv — wraps multiple days, forwards step_interval |
| `python/tests/test_step_interval.py` | 62 tests covering step_interval feature |
| `python/tests/conftest.py` | Shared `make_realistic_obs()` helper |

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
- **Spread verification** — done, all positive, min=0.25 for /MES.

## Architecture overview

```
data/mes/*.bin  →  BinaryFileSource (C++)  →  Book (C++)  →  LOBEnv (C++)
                                                                   ↓
                                                           precompute() (C++)
                                                                   ↓
                                                         numpy arrays (Python)
                                                                   ↓
                                              PrecomputedEnv + temporal features (Python)
                                              [step_interval subsampling happens here]
                                                                   ↓
                                                    MultiDayEnv (Python/Gym)
                                                                   ↓
                                                      SB3 PPO (scripts/train.py)
```

## Test coverage

- **489 C++ tests** — `cd build-release && ./lob_tests`
- **758 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1247 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| **Re-train with step-interval + exec cost** | **Critical** | 2M steps, step_interval=10, no participation bonus |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch, step_interval values |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
