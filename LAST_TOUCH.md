# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Build the cache, then iterate on reward design.**

Step 1 — Generate the precompute cache:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py \
  --data-dir ../data/mes --out ../cache/mes/
```

Step 2 — Train using cached data (instant startup, no C++ precompute):

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --execution-cost --step-interval 10 --total-timesteps 2000000
```

The core unsolved problem is **entropy collapse**. Every training configuration so far results in the agent converging to a degenerate policy (stay flat, or exploit participation bonus). The reward signal needs work. With precompute cache eliminating startup overhead, iteration on reward design becomes fast.

### What was just completed

**Precompute cache (PR #9).** Eliminates redundant C++ precompute calls:
1. `scripts/precompute_cache.py` — CLI tool: reads manifest, calls C++ precompute once per day, saves `{date}.npz`
2. `python/lob_rl/precomputed_env.py` — `from_cache()` classmethod loads obs/mid/spread from `.npz`
3. `python/lob_rl/multi_day_env.py` — `cache_dir=` parameter (mutually exclusive with `file_paths=`)
4. `scripts/train.py` — `--cache-dir` flag (mutually exclusive with `--data-dir`)
5. `src/env/warmup.h` — extracted shared warmup logic (C++ refactor)

Also: `evaluate_sortino()` accepts `cache_path=` for cached evaluation.

**Training run (2M, step-interval 10 + exec cost, no bonus):**
- Val: mean_return=-15.94, sortino=-1.07
- Entropy collapsed to 0.03 by 500k steps
- Agent learned to stay flat again

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |
| Temporal features | 2M steps, 54-dim obs, participation_bonus=0.01 | Sortino inf val. But PnL is bid-ask bounce mean reversion, not real alpha. Autocorr(1)=-0.75. Caused by broken spread/precompute. |
| Post-fix + exec cost + bonus | 500k steps, exec_cost, participation_bonus=0.01 | Val: mean=1273, sortino=inf. Entropy collapsed (0.01). Agent exploits participation bonus (hold forever). |
| **Step-interval + exec cost** | **2M steps, exec_cost, step_interval=10, no bonus** | **Val: mean=-15.94, sortino=-1.07. Entropy collapsed (0.03). Agent stays flat.** |

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point — 16 CLI flags incl. `--cache-dir`, `--step-interval` |
| `scripts/precompute_cache.py` | CLI tool to build `.npz` cache from `.bin` files |
| `python/lob_rl/precomputed_env.py` | PrecomputedEnv — `from_file()`, `from_cache()`, step_interval, temporal features |
| `python/lob_rl/multi_day_env.py` | MultiDayEnv — wraps multiple days, supports `file_paths=` or `cache_dir=` |
| `python/tests/test_from_cache.py` | 22 tests for PrecomputedEnv.from_cache() |
| `python/tests/test_cache_multi_day.py` | 25 tests for MultiDayEnv cache_dir |
| `python/tests/test_train_cache_dir.py` | 15 tests for train.py --cache-dir |
| `python/tests/test_precompute_cache.py` | 18 tests for precompute_cache.py script |
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
- **Precompute cache** — it's done and merged. PR #9.

## Architecture overview

```
data/mes/*.bin  →  precompute_cache.py  →  cache/mes/*.npz (one-time)
                                                   ↓
                                     PrecomputedEnv.from_cache() (Python)
                                     [step_interval subsampling + temporal features]
                                                   ↓
                                        MultiDayEnv (Python/Gym)
                                                   ↓
                                          SB3 PPO (scripts/train.py --cache-dir)
```

Alternative (legacy, slower): `data/mes/*.bin → C++ precompute() → PrecomputedEnv.from_file()`

## Test coverage

- **489 C++ tests** — `cd build-release && ./lob_tests`
- **847 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1336 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| **Reward signal design** | **Critical** | Entropy collapse in every run. Need better reward shaping. |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch, step_interval values |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
