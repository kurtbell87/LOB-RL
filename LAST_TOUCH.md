# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Constellation integration is on `feat/constellation-integration` branch.** All 6 phases complete. The branch has 6 commits on top of `main` and is ready for review/merge.

What was integrated:
- **Phase 1:** Constellation C++ source tree (131 files) imported to `src/constellation/`
- **Phase 2:** Adapter layer — Book class now wraps Constellation's LimitOrderBook (same API, new internals)
- **Phase 3:** OrdersEngine exposed via pybind11 + new `OrderSimulationEnv` Gymnasium environment
- **Phase 4:** BatchBacktestEngine, MarketBook, replay config exposed via pybind11
- **Phase 5:** Constellation feature system (11 feature types, FeatureManager) exposed via pybind11
- **Phase 6:** 35 integration tests for all Constellation bindings

**All 660 C++ + 2246 Python tests pass (2906 total).** Plus 65/67 Constellation Catch2 tests (2 pre-existing failures from Constellation's original partial fill tests).

### What the integration enables

1. **Order simulation:** `OrderSimulationEnv` — Gymnasium env with Discrete(7) actions (hold, buy/sell market, buy/sell limit, cancel buy/sell). Uses Constellation's OrdersEngine for realistic order lifecycle.
2. **High-performance replay:** `BatchBacktestEngine` can process .dbn.zst files at >1M msg/sec from Python.
3. **Real-time features:** 11 Constellation feature types (micro price, rolling volatility, order imbalance, etc.) available for research scripts.
4. **MarketBook:** Multi-instrument order book with global event counters, accessible from Python.

### Outstanding research work (unchanged)

The research pipeline on `main` continues independently. See `RESEARCH_LOG.md` for experiment status. 13 experiments completed (2 confirmed, 11 refuted). Architecture exploration closed — signal is the bottleneck, not model capacity.

### Roll calendar

`data/mes/roll_calendar.json` maps each date to the front-month instrument_id. `precompute_cache.py --roll-calendar` uses this to filter each day's `.dbn.zst` to exactly ONE contract (no spreads, no back months).

| Period | Contract | Instrument ID |
|--------|----------|--------------|
| Jan 1 – Mar 11 | MESH2 | 11355 |
| Mar 12 – Jun 10 | MESM2 | 13615 |
| Jun 11 – Sep 9 | MESU2 | 10039 |
| Sep 10 – Dec 31 | MESZ2 | 10299 |

### What was just completed

**Constellation Integration — 6 phases (2026-02-12).** Long-lived feature branch `feat/constellation-integration` bringing Constellation's production-grade C++ engine into LOB-RL:

- **Phase 0:** Branch `feat/constellation-integration` from `main`. Verified all 2871 tests pass.
- **Phase 1:** Imported 131 Constellation C++ files into `src/constellation/`. Added `constellation_core` static library + `constellation_tests` (Catch2) to CMake. Fixed databento v0.47.0 optional schema API. 65/67 Constellation tests pass.
- **Phase 2:** Replaced Book internals with Constellation's LimitOrderBook. Created message adapter (`to_mbo_msg`/`to_message`). Preserved all LOB-RL behavioral quirks (trade no-ops, modify-as-add, uint32 saturation). Removed databento type shims. All 2871 existing tests unchanged.
- **Phase 3:** Exposed OrdersEngine, OrderSpec, OrderInfo, OrderStatus, OrderType, OrderSide, TimeInForce enums via pybind11. Created `OrderSimulationEnv` Gymnasium environment.
- **Phase 4:** Exposed BatchBacktestEngine, BatchAggregatorConfig, FillRecord, MarketBook via pybind11.
- **Phase 5:** Exposed 11 Constellation feature factory functions, IFeature, IFeatureManager, BookSide enum via pybind11.
- **Phase 6:** Added 35 Python integration tests covering all Constellation bindings.

6 commits: `7755b43` → `8334b57` → `34dea58` → `df90aa7` → `4f2c6a5` → `65dc407`.

## Key files for current task

| File | Role |
|---|---|
| `src/constellation/` | Constellation C++ source tree (131 files) |
| `src/constellation/adapters/message_adapter.h` | LOB-RL Message ↔ Constellation MboMsg conversion |
| `include/lob/book.h` | Book class wrapping Constellation's LimitOrderBook |
| `src/engine/book.cpp` | Book implementation delegating to LimitOrderBook |
| `src/bindings/bindings.cpp` | All pybind11 bindings (LOB-RL + Constellation) |
| `python/lob_rl/order_sim_env.py` | OrderSimulationEnv — Gymnasium env with order simulation |
| `python/tests/test_constellation_bindings.py` | 35 integration tests for Constellation bindings |
| `CMakeLists.txt` | Build config — `constellation_core` library + `constellation_tests` |

## Don't waste time on

- **Build verification** — `build-release/` is current, 660 C++ + 2246 Python = 2906 tests pass.
- **Dependency checks** — SB3, sb3-contrib, gymnasium, numpy, tensorboard, torch, databento-cpp, Catch2 all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.

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

NEW (Constellation integration):

OrderSimulationEnv ← OrdersEngine ← Constellation's CQRS order engine
                   ← MarketBook   ← LimitOrderBook (replaces old Book internals)

BatchBacktestEngine → process .dbn.zst → fills, market state, features
```

## Test coverage

- **660 C++ tests** — `cd build-release && ./lob_tests` (15 skipped: need `.dbn.zst` fixture)
- **65 Constellation Catch2 tests** — `cd build-release && ./constellation_tests` (2 pre-existing failures)
- **2246 Python tests** (2211 core+barrier + 35 constellation) — `PYTHONPATH=build-release:python uv run pytest python/tests/`
- **2906 total LOB-RL tests**, all passing. **2971 including Constellation Catch2.**

---

For project history, see `git log`.
