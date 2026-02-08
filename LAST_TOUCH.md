# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Ingest new data from `data/GLBX-20260207-L953CAPU5B.zip`, cache it, retrain, and validate.**

1. **Unzip and place `.dbn.zst` files** in `data/mes/`:
```bash
cd data && unzip GLBX-20260207-L953CAPU5B.zip -d mes/
```

2. **Find the instrument ID** for /MES front-month. Check `symbology.json` in the zip or use Databento docs. Example: `/MES` continuous front = instrument_id `42005347` (verify).

3. **Precompute cache** for the new data (`--instrument-id` is now required):
```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_cache.py \
  --data-dir ../data/mes/ --out ../cache/mes/ --instrument-id <ID> --force
```

4. **Retrain with the winning config** on the full dataset:
```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --total-timesteps 2000000 \
  --ent-coef 0.05 --learning-rate 0.001 \
  --train-days 170
```

5. **Validate on proper OOS split** — with ~250 days, use 170 train / 40 val / 40 test (adjust `--train-days`). The current 21-day dataset only has 1 OOS day.

6. **Consider longer training** — with 170 train days (vs 20), may need more than 2M steps. Try 5M or 10M.

### What was just completed

**Native DBN source (this session, PR #11).** Replaced custom `.bin` pipeline with direct `.dbn.zst` reading via databento-cpp:
- `DbnFileSource` reads `.dbn.zst` files natively (also auto-detects legacy `.bin`)
- `map_mbo_to_message()` shared mapper (reusable for live data in Phase 4)
- `instrument_id` parameter added to `precompute()`, `LOBEnv`, `precompute_cache.py`
- Deleted: `BinaryFileSource`, `convert_dbn.py`, and all their tests
- Vectorized `BarLevelEnv._precompute_temporal()` with numpy
- 403 C++ + 949 Python = 1352 tests pass

**Hyperparameter sweep (prior session).** 7 configs tested, all 2M steps:

| # | Config | Return | Entropy | Wall time |
|---|--------|--------|---------|-----------|
| 0 | Baseline (ent=0.01, bar=500, lr=3e-4) | 67.5 | -0.20 | ~6 min |
| 1 | ent_coef=0.05 | 111.0 | -0.55 | 382s |
| 2 | ent_coef=0.1 | 101.0 | -0.77 | 379s |
| 3 | bar_size=200 | 3.75 | -0.21 | 372s |
| 4 | bar_size=1000 | 128.0 | -0.16 | 415s |
| 5 | lr=1e-3 | 116.25 | -0.30 | 387s |
| 6 | lr=1e-4 | 18.0 | -0.20 | 386s |
| **7** | **bar=1000 + ent=0.05 + lr=1e-3** | **139.5** | **-0.48** | **432s** |

**Key findings:**
- **Best config: `bar_size=1000, ent_coef=0.05, lr=1e-3`** — 139.5 return, stable entropy (-0.48), explained_var 0.98.
- **21/21 days positive** on full-dataset eval (but only 1 true OOS day).
- `bar_size=200` is a dud (too noisy). `bar_size=1000` (coarser bars) works best.
- Higher `ent_coef` (0.05) keeps entropy healthy and improves returns.
- Higher `lr` (1e-3) learns faster in 2M steps.
- Inventory penalty deemed unnecessary — the entropy fix solved the "sits flat" problem.
- **Training runs on Apple silicon CPU at ~5,650 FPS, ~7 min per 2M-step run.**

**Saved models** in `build-release/runs/`:
- `sweep_ent005/`, `sweep_ent01/`, `sweep_bar200/`, `sweep_bar1000/`, `sweep_lr1e3/`, `sweep_lr1e4/`, `sweep_combined/`
- Each contains `ppo_lob.zip`, `vec_normalize.pkl`, `tb_logs/`

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k, ent_coef=0.0, 1 env | Sortino -1.05. Entropy collapsed. |
| v2 + exec cost | 2M, ent_coef=0.01, 8 envs, exec_cost | Entropy stable (0.70). Agent stays flat. |
| v2 no exec cost | 2M, ent_coef=0.01, 8 envs | Entropy collapsed (0.09). Sortino -1.05. |
| v2 + participation bonus | 2M, participation_bonus=0.01 | Sortino -0.91. Wrong direction. |
| Post-fix + exec cost + bonus | 500k, exec_cost, bonus=0.01 | Entropy collapsed (0.01). Exploits bonus. |
| Step-interval + exec cost | 2M, exec_cost, step_interval=10 | Entropy collapsed (0.03). Stays flat. |
| Bar-level v1 | 2M, bar_size=500, 256x256 ReLU, exec_cost | Return 67.5. Entropy 0.20. First positive result. |
| Sweep: ent_coef=0.05 | 2M, bar_size=500, ent=0.05 | Return 111.0. Entropy -0.55. |
| Sweep: ent_coef=0.1 | 2M, bar_size=500, ent=0.1 | Return 101.0. Entropy -0.77. |
| Sweep: bar_size=200 | 2M, bar_size=200 | Return 3.75. Too noisy. |
| Sweep: bar_size=1000 | 2M, bar_size=1000 | Return 128.0. Entropy -0.16. |
| Sweep: lr=1e-3 | 2M, bar_size=500, lr=1e-3 | Return 116.25. Entropy -0.30. |
| Sweep: lr=1e-4 | 2M, bar_size=500, lr=1e-4 | Return 18.0. Too slow. |
| **Combined best** | **2M, bar=1000, ent=0.05, lr=1e-3** | **Return 139.5. Entropy -0.48. Best result.** |

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point — `--bar-size`, `--policy-arch`, `--activation`, `--cache-dir`, `--train-days` |
| `scripts/precompute_cache.py` | CLI tool to build `.npz` cache from `.dbn.zst` files. Requires `--instrument-id`. |
| `src/data/dbn_file_source.h/.cpp` | `DbnFileSource` — reads `.dbn.zst` (or `.bin`) via databento-cpp |
| `src/data/dbn_message_map.h/.cpp` | `map_mbo_to_message()` — MBO record → Message mapping (shared with future live source) |
| `python/lob_rl/bar_aggregation.py` | `aggregate_bars()` — tick → bar feature aggregation |
| `python/lob_rl/bar_level_env.py` | `BarLevelEnv` — bar-level gymnasium env (21-dim obs) |
| `python/lob_rl/precomputed_env.py` | `PrecomputedEnv` — tick-level env (54-dim obs) |
| `python/lob_rl/multi_day_env.py` | `MultiDayEnv` — wraps multiple days, supports `bar_size=` |
| `data/GLBX-20260207-L953CAPU5B.zip` | New 1-year dataset (not yet extracted) |

## Don't waste time on

- **Build verification** — `build-release/` is current, 403 C++ + 949 Python = 1352 tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard, torch, databento-cpp all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating lookahead/leakage** — fully audited, all clean.
- **Hyperparameter sweep** — done, best config identified.
- **Inventory penalty** — decided against; entropy fix solved the flat-agent problem.
- **bar_size=200** — confirmed too noisy, return near zero.
- **Native DBN source** — done (PR #11). `.bin` pipeline replaced with `.dbn.zst`.
- **Precompute fix, spread verification, precompute cache, bar-level env** — all done.

## Architecture overview

```
data/mes/*.mbo.dbn.zst  →  precompute_cache.py --instrument-id  →  cache/mes/*.npz
                                  (DbnFileSource + map_mbo_to_message)
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

- **403 C++ tests** — `cd build-release && ./lob_tests` (15 skipped: need `.dbn.zst` fixture)
- **949 Python tests** — `PYTHONPATH=build-release:python uv run pytest python/tests/` (4 skipped: fixture-dependent)
- **1352 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| ~~Hyperparameter sweep~~ | ~~Done~~ | Best: bar=1000, ent=0.05, lr=1e-3, return 139.5. |
| ~~Native DBN source~~ | ~~Done~~ | PR #11. `.dbn.zst` reading, `instrument_id`, deleted `.bin` pipeline. |
| **Ingest new data** | **Critical** | Unzip `data/GLBX-20260207-L953CAPU5B.zip`, find instrument_id, precompute cache. |
| **Retrain on full dataset** | **Critical** | 170/40/40 split, winning config, possibly 5M+ steps. |
| **Proper OOS validation** | **Critical** | Current results are almost entirely in-sample. |
| Bar-level supervised diagnostic | Low | Script lacks `--bar-size` support. Nice-to-have. |
| More data (second year) | Low | 2023-2024 as complement if 2022 generalizes well. |

---

For project history, see `git log`.
