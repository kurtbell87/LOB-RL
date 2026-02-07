# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Build cache, run supervised diagnostic on bar features, then train with bar-level env.**

Step 1 — Generate the precompute cache (one-time):

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py \
  --data-dir ../data/mes --out ../cache/mes/
```

Step 2 — Run supervised diagnostic to validate architecture on bar features:

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/supervised_diagnostic.py \
  --data-dir ../data/mes --bar-size 500 --step-interval 10
```

This tests whether a 2x256 ReLU MLP can predict optimal bar-level actions from state. If it can't beat majority-class baseline even with oracle labels, the problem is features/frequency, not RL.

Step 3 — Train with bar-level env (if supervised diagnostic passes):

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --bar-size 500 --execution-cost \
  --policy-arch 256,256 --activation relu --total-timesteps 2000000
```

With `--bar-size 500`, each day has ~270 bars instead of ~137k ticks. Episodes are 270 steps — tractable for PPO credit assignment.

### What was just completed

**Bar-level environment (PR #10).** Aggregates tick-level MBO data into N-tick bars:
1. `python/lob_rl/bar_aggregation.py` — `aggregate_bars()` pure function (tick → bar features)
2. `python/lob_rl/bar_level_env.py` — `BarLevelEnv` gymnasium env (21-dim obs)
3. `python/lob_rl/multi_day_env.py` — `bar_size=` parameter
4. `scripts/train.py` — `--bar-size`, `--policy-arch`, `--activation` flags
5. `scripts/supervised_diagnostic.py` — MLP capacity diagnostic tool

Bar features (21 dims): 13 intra-bar (return, range, volatility, spread, imbalance, volume, microprice, time, activity) + 7 cross-bar temporal + 1 position.

**Precompute cache (PR #9).** Eliminates redundant C++ precompute calls:
- `scripts/precompute_cache.py`, `PrecomputedEnv.from_cache()`, `MultiDayEnv(cache_dir=...)`, `train.py --cache-dir`

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k, ent_coef=0.0, 1 env | Sortino -1.05. Entropy collapsed. |
| v2 + exec cost | 2M, ent_coef=0.01, 8 envs, exec_cost | Entropy stable (0.70). Agent stays flat. |
| v2 no exec cost | 2M, ent_coef=0.01, 8 envs | Entropy collapsed (0.09). Sortino -1.05. |
| v2 + participation bonus | 2M, participation_bonus=0.01 | Sortino -0.91. Wrong direction. |
| Post-fix + exec cost + bonus | 500k, exec_cost, bonus=0.01 | Entropy collapsed (0.01). Exploits bonus. |
| Step-interval + exec cost | 2M, exec_cost, step_interval=10 | Entropy collapsed (0.03). Stays flat. |
| **Next: bar-level** | **2M, bar_size=500, 256x256 ReLU** | **Not yet run.** |

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point — `--bar-size`, `--policy-arch`, `--activation`, `--cache-dir` |
| `scripts/precompute_cache.py` | CLI tool to build `.npz` cache from `.bin` files |
| `scripts/supervised_diagnostic.py` | Supervised MLP capacity diagnostic |
| `python/lob_rl/bar_aggregation.py` | `aggregate_bars()` — tick → bar feature aggregation |
| `python/lob_rl/bar_level_env.py` | `BarLevelEnv` — bar-level gymnasium env (21-dim obs) |
| `python/lob_rl/precomputed_env.py` | `PrecomputedEnv` — tick-level env (54-dim obs) |
| `python/lob_rl/multi_day_env.py` | `MultiDayEnv` — wraps multiple days, supports `bar_size=` |

## Don't waste time on

- **Build verification** — `build-release/` is current, 489 C++ tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard, torch all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating lookahead/leakage** — fully audited, all clean.
- **Precompute fix** — done, PR #7.
- **Spread verification** — done, all positive.
- **Precompute cache** — done, PR #9.
- **Bar-level env** — done, PR #10.

## Architecture overview

```
data/mes/*.bin  →  precompute_cache.py  →  cache/mes/*.npz (one-time)
                                                   ↓
                              ┌─── bar_size=0: PrecomputedEnv (54-dim, tick-level)
                              │
         MultiDayEnv ─────────┤
                              │
                              └─── bar_size>0: BarLevelEnv (21-dim, bar-level)
                                       ↑
                                  aggregate_bars() + cross-bar temporal
                                                   ↓
                                          SB3 PPO (scripts/train.py)
```

## Test coverage

- **489 C++ tests** — `cd build-release && ./lob_tests`
- **1001 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1490 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| **Run supervised diagnostic** | **Critical** | Validate MLP can learn from bar features before training |
| **Train with bar-level env** | **Critical** | First bar-level training run |
| Hyperparameter sweep | Medium | ent_coef, LR, bar_size (200-1000), arch size |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| More MBO data from Databento | Low | Current: 27 days. More data helps generalization. |

---

For project history, see `git log`.
