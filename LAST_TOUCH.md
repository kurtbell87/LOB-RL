# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Set up RunPod for GPU training.** Local CPU training is too slow for LSTM (~422 fps vs ~4200 fps for MLP) and results so far are negative OOS. Need GPU compute to run longer experiments (5M–10M steps) and LSTM.

1. **RunPod setup** — containerize the training pipeline, push cache to cloud storage, configure GPU instances.
2. **Run LSTM experiment on RunPod** — killed locally at 15% (too slow). Needs GPU.
3. **Run longer experiments** — 2M steps isn't enough. Try 5M–10M with all 3 architectures on GPU.
4. **Investigate negative OOS** — both MLP and frame-stack are negative on shuffle-split too, so it's not just chronological regime shift. The agent is not generalizing. Consider: reward shaping, observation normalization debugging, different architectures, or more data.

### Roll calendar

`data/mes/roll_calendar.json` maps each date to the front-month instrument_id. `precompute_cache.py --roll-calendar` uses this to filter each day's `.dbn.zst` to exactly ONE contract (no spreads, no back months).

| Period | Contract | Instrument ID |
|--------|----------|--------------|
| Jan 1 – Mar 11 | MESH2 | 11355 |
| Mar 12 – Jun 10 | MESM2 | 13615 |
| Jun 11 – Sep 9 | MESU2 | 10039 |
| Sep 10 – Dec 31 | MESZ2 | 10299 |

Source: `data/symbology.json` from Databento download. Roll dates are ~1 week before CME 3rd-Friday expiry.

### What was just completed

**Local comparative experiments (2M steps, shuffle-split, seed 42).** Both completed runs are negative OOS:

| Model | Val Return | Test Return | Val Sortino | Test Sortino | Notes |
|-------|-----------|-------------|-------------|--------------|-------|
| Baseline MLP | -51.5 | -62.5 | -2.09 | -1.51 | 0/10 positive test episodes |
| Frame Stack (4) | -48.4 | -50.2 | -1.82 | -1.08 | 2/10 positive test episodes |
| LSTM | — | — | — | — | Killed at 15% — 422 fps too slow for local CPU |

**Key finding:** Shuffle-split doesn't fix OOS. The negative results aren't just chronological regime shift — the agent genuinely isn't generalizing at 2M steps.

**Shuffle split, frame stacking, and RecurrentPPO (prior session, PRs #14–16).** Three features to improve OOS evaluation and give the agent temporal context:

- **Shuffle split (PR #14):** `--shuffle-split` and `--seed 42` on `train.py`. Reproducible random train/val/test splits instead of chronological. Valid because episodes (days) are independent — no cross-day state. 42 new tests.
- **Frame stacking (PR #15):** `--frame-stack N` on `train.py`. `VecFrameStack` wrapper between SubprocVecEnv and VecNormalize. Eval also wraps with VecFrameStack. bar_size=1000 + frame_stack=4 → 84-dim obs. 40 new tests.
- **RecurrentPPO (PR #16):** `--recurrent` flag on `train.py`. `RecurrentPPO('MlpLstmPolicy')` from sb3-contrib. LSTM state tracking in eval (lstm_states, episode_start). Mutually exclusive with `--frame-stack > 1`. 59 new tests.
- **Shared reward module:** Extracted `compute_forced_flatten()` and `compute_step_reward()` into `python/lob_rl/_reward.py`.
- **Refactoring:** `conftest.py` shared helpers (`make_tick_data`, `save_cache_with_instrument_id`), `test_helpers.h` flag constants, `SyntheticSource` named constants, `precompute.cpp` extracted `process_rth_legacy()` / `process_rth_flag_aware()`.
- 418 C++ + 1304 Python = 1722 tests pass.

**Prior session context:** Model trained on 170 days (5M steps, winning config) showed return 139.5 in-sample but -53.8 val / -36.6 test OOS on chronological split. Hypothesis: chronological split conflates overfitting with regime shift (2022 H1 vs H2 were different markets).

**Lazy loading (PR #13).** OOM fix for 200+ day training.

**Contract boundary guard (PR #12).** Forced flatten on terminal step.

**Hyperparameter sweep.** Best: bar=1000, ent=0.05, lr=1e-3 → return 139.5.

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
| **Full dataset (chrono)** | **5M, 170 train / 40 val / 40 test** | **139.5 in-sample, -53.8 val, -36.6 test** |
| **MLP shuffle-split** | **2M, 170 train, shuffle, seed 42** | **Val -51.5, Test -62.5. Sortino -1.51.** |
| **Frame-stack shuffle** | **2M, frame_stack=4, shuffle, seed 42** | **Val -48.4, Test -50.2. Sortino -1.08.** |
| **LSTM shuffle** | **2M, recurrent, shuffle, seed 42** | **Killed at 15% — too slow locally (422 fps).** |

## Key files for current task

| File | Role |
|---|---|
| `scripts/train.py` | Training entry point — `--bar-size`, `--cache-dir`, `--train-days`, `--shuffle-split`, `--seed`, `--frame-stack`, `--recurrent` |
| `scripts/precompute_cache.py` | CLI tool to build `.npz` cache. Use `--roll-calendar` or `--instrument-id`. |
| `data/mes/roll_calendar.json` | Maps each date → front-month instrument_id. Used by `--roll-calendar`. |
| `data/mes/*.mbo.dbn.zst` | 312 daily files, Jan–Dec 2022, 57GB. All /MES instruments per file. |
| `python/lob_rl/bar_level_env.py` | `BarLevelEnv` — bar-level gymnasium env (21-dim obs) |
| `python/lob_rl/precomputed_env.py` | `PrecomputedEnv` — tick-level env (54-dim obs) |
| `python/lob_rl/multi_day_env.py` | `MultiDayEnv` — wraps multiple days, lazy-loads `.npz` |
| `python/lob_rl/_reward.py` | Shared `compute_forced_flatten()`, `compute_step_reward()` |

## Don't waste time on

- **Build verification** — `build-release/` is current, 418 C++ + 1304 Python = 1722 tests pass.
- **Dependency checks** — SB3, sb3-contrib, gymnasium, numpy, tensorboard, torch, databento-cpp all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating lookahead/leakage** — fully audited, all clean.
- **Hyperparameter sweep** — done, best config identified.
- **Shuffle split, frame stacking, RecurrentPPO** — all done (PRs #14–16).
- **Lazy loading, contract boundary guard, native DBN source** — all done (PRs #11–13).
- **Precompute fix, spread verification, precompute cache, bar-level env** — all done.

## Architecture overview

```
data/mes/*.mbo.dbn.zst  →  precompute_cache.py --roll-calendar  →  cache/mes/*.npz
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
                           SubprocVecEnv → [VecFrameStack] → VecNormalize
                                                   ↓
                                  SB3 PPO / RecurrentPPO (scripts/train.py)
```

## Test coverage

- **418 C++ tests** — `cd build-release && ./lob_tests` (15 skipped: need `.dbn.zst` fixture)
- **1304 Python tests** — `PYTHONPATH=build-release:python uv run pytest python/tests/` (4 skipped: fixture-dependent)
- **1722 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| ~~Hyperparameter sweep~~ | ~~Done~~ | Best: bar=1000, ent=0.05, lr=1e-3, return 139.5. |
| ~~Native DBN source~~ | ~~Done~~ | PR #11. |
| ~~Extract new data~~ | ~~Done~~ | 312 files in `data/mes/`, 57GB. |
| ~~Contract boundary guard~~ | ~~Done~~ | PR #12. |
| ~~Lazy loading~~ | ~~Done~~ | PR #13. |
| ~~Shuffle split~~ | ~~Done~~ | PR #14. `--shuffle-split --seed 42`. |
| ~~Frame stacking~~ | ~~Done~~ | PR #15. `--frame-stack N`. |
| ~~RecurrentPPO~~ | ~~Done~~ | PR #16. `--recurrent`. |
| ~~Precompute cache~~ | ~~Done~~ | 249 `.npz` in `cache/mes/`, built with `--roll-calendar`. Use `--workers 8` if rebuilding. |
| ~~Comparative experiments~~ | ~~Done (partial)~~ | MLP and frame-stack done locally (both negative OOS). LSTM killed — needs GPU. |
| ~~Proper OOS validation~~ | ~~Done~~ | Shuffle-split also negative. Not just regime shift — agent doesn't generalize at 2M steps. |
| **RunPod GPU training** | **Critical** | LSTM too slow locally (422 fps). Need GPU for LSTM + longer runs (5M–10M steps). |
| **Investigate negative OOS** | **Critical** | Consider: reward shaping, obs normalization debugging, different architectures, more data. |
| Bar-level supervised diagnostic | Low | Script lacks `--bar-size` support. Nice-to-have. |
| More data (second year) | Low | 2023-2024 as complement if 2022 generalizes well. |

---

For project history, see `git log`.
