# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**LOB bug fixes are complete and validated.** The `feat/constellation-integration` branch now has Constellation's LimitOrderBook matching the Databento reference implementation exactly. All fixes have been merged to main.

**Next:** Choose a research direction from `RESEARCH_LOG.md`:
- Phase 2b bar-size sweep (B ∈ {200,500,1000,2000}) — different observation scales
- Conditional signal detection (regime filtering)
- Accept the null and pivot features/targets
- Architecture exploration using the now-correct LOB

### What was just completed

**LimitOrderBook Bug Fixes (2026-02-12).** Four bugs fixed in Constellation's `LimitOrderBook` to match Databento's reference implementation (`references/known_working/example_lob.cpp`):

1. **IsTob flag handling (Bug 1 — root cause of 161K+ mismatches):** When `mbo.flags.IsTob()` on Add, clear entire side via `ClearSide()`, then add one synthetic level with `count=0`. TOB orders are not tracked in `orders_`.
2. **Partial cancel (Bug 2):** Cancel now subtracts `mbo.size` (not `old.size`). Order only removed when remaining size == 0.
3. **Modify-as-Add (Bug 3):** `HandleModify` calls `HandleAdd(mbo)` when order_id not found, instead of silently returning.
4. **Trade/Fill no-ops (Bug 4):** Trade and Fill actions no longer call `HandleFillOrCancel`. Counter `trade_count_` still increments for telemetry.

**Validation results:**
- DBEQ GOOG/GOOGL (286K events, 161K BBO checks): **0 mismatches**
- MES futures (8.6M events, 7.5M BBO checks): **0 mismatches**
- Reference BBO output files saved to `references/known_working/`
- 6 new Catch2 unit tests + 6 gtest validation tests

### Constellation integration (prior session)

- **Phase 0-6 complete.** Book wraps Constellation's LimitOrderBook. OrdersEngine, BatchBacktestEngine, MarketBook, 11 feature types exposed via pybind11. `OrderSimulationEnv` Gymnasium env. 35 integration tests.

### Outstanding research work

The research pipeline on `main` continues independently. See `RESEARCH_LOG.md` for experiment status. 13 experiments completed (2 confirmed, 11 refuted). Architecture exploration closed — signal is the bottleneck, not model capacity.

### Roll calendar

`data/mes/roll_calendar.json` maps each date to the front-month instrument_id. `precompute_cache.py --roll-calendar` uses this to filter each day's `.dbn.zst` to exactly ONE contract (no spreads, no back months).

| Period | Contract | Instrument ID |
|--------|----------|--------------|
| Jan 1 – Mar 11 | MESH2 | 11355 |
| Mar 12 – Jun 10 | MESM2 | 13615 |
| Jun 11 – Sep 9 | MESU2 | 10039 |
| Sep 10 – Dec 31 | MESZ2 | 10299 |

## Key files for current task

| File | Role |
|---|---|
| `src/constellation/modules/orderbook/src/LimitOrderBook.cpp` | Fixed LOB implementation — all 4 bugs |
| `src/constellation/modules/orderbook/include/orderbook/LimitOrderBook.hpp` | Added `ClearSide()` declaration |
| `tests/test_lob_vs_databento_reference.cpp` | 6 validation tests (DBEQ + MES) |
| `references/known_working/example_lob.cpp` | Databento reference implementation (spec) |
| `references/known_working/mes_reference_bbo_20220103.txt` | Reference BBO output for MES 2022-01-03 |
| `references/known_working/mes_constellation_bbo_20220103.txt` | Constellation BBO output (identical to reference) |
| `src/constellation/modules/orderbook/tests/TestOrderBookExtended.cpp` | 6 new unit tests (IsTob, partial cancel, modify-as-add, trade noop) |
| `src/constellation/adapters/message_adapter.h` | LOB-RL Message ↔ Constellation MboMsg conversion |
| `include/lob/book.h` | Book class wrapping Constellation's LimitOrderBook |

## Don't waste time on

- **Build verification** — `build-release/` is current, 660 C++ + 2246 Python = 2906 tests pass.
- **LOB correctness** — Validated against Databento reference on both equities and futures data. 0 mismatches.
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

Constellation integration:

OrderSimulationEnv ← OrdersEngine ← Constellation's CQRS order engine
                   ← MarketBook   ← LimitOrderBook (now matches Databento reference)

BatchBacktestEngine → process .dbn.zst → fills, market state, features
```

## Test coverage

- **660 C++ tests** — `cd build-release && ./lob_tests` (15 skipped: need `.dbn.zst` fixture)
- **71 Constellation Catch2 tests** — `cd build-release && ./constellation_tests` (2 pre-existing failures in orders module)
- **6 LOB validation tests** — `./build-release/lob_validation_tests` (run from repo root; needs `data/` files)
- **2246 Python tests** (2211 core+barrier + 35 constellation) — `PYTHONPATH=build-release:python uv run pytest python/tests/`
- **2906 total LOB-RL tests**, all passing. **2977 including Constellation Catch2 + validation.**

---

For project history, see `git log`.
