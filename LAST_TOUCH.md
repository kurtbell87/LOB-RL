# Last Touch

Read this file first to understand current project state. Update it when you finish your work.

---

## Current State: Step 1 (Vertical Slice) Complete

Full pipeline working end-to-end: `SyntheticSource` → `Book` → `LOBEnv` → pybind11 → Python.

### What Exists and Works

| Layer | Files | Status |
|-------|-------|--------|
| Headers | `include/lob/{message,source,book,env,reward}.h` | All implemented per TDS |
| Data | `src/data/synthetic_source.{h,cpp}` | Deterministic MBO generator (seeded RNG, 40% adds / 30% cancels / 15% modifies / 15% trades) |
| Engine | `src/engine/book.cpp` | Full order book reconstruction (add/cancel/modify/trade/clear), `std::map` for levels, `unordered_map` for orders |
| Env | `src/env/{lob_env,feature_builder,reward_calculator}.cpp` | Step/reset cycle, 44-float obs, mark-to-market PnL, `PnLDelta`/`PnLDeltaPenalized` rewards |
| Bindings | `src/bindings/bindings.cpp` | pybind11 module `lob_rl_core` with numpy arrays |
| Python | `python/lob_rl/{__init__,wrappers}.py`, `python/setup.py` | Package + Gymnasium wrapper |
| Tests | `tests/test_{source,book,env}.cpp`, `python/tests/test_gym.py` | 30 C++ (ASAN/UBSAN clean) + 6 Python, all passing |

### How to Build and Test

```bash
# From project root
cd build
cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=ON
cmake --build .
ctest                                                                    # C++ tests
PYTHONPATH=. uv run --with pytest python3 -m pytest ../python/tests/     # Python tests
```

### Key Design Notes

- Prices are fixed-point `int64_t` (price * 1e9). /MES tick = 0.25 → 250,000,000 in fixed-point.
- Positions are {-1, 0, +1}. Actions: 0=short, 1=flat, 2=long.
- Fills are instant at BBO (bid for sells, ask for buys).
- `EnvConfig.trades_per_step` controls how many MBO messages are processed per `step()` call.
- Python package management uses `uv`, never `pip`.

---

## Your Task: Step 2 (Real Data)

Per [PRD.md](PRD.md) Section 9 Step 2 and [TDS.md](TDS.md):

1. **`DatabentoSource`** (`src/data/databento_source.{h,cpp}`) — Parse Databento MBO binary format for /MES. Implement `IMessageSource` interface. Build with `-DWITH_DATABENTO=ON`.
2. **Session boundary handling** — Use `Session` struct in `EnvConfig` to filter messages to US RTH window. Skip pre-market, stop at session end.
3. **Full 10-level book validation** — Replay real data and verify BBO matches expected values from Databento BBO feed.
4. **Gymnasium integration test** — Install `gymnasium` and verify `LOBEnvGym` wrapper passes `check_env` from `stable-baselines3`.

### Constraints

- Do not modify `include/lob/*.h` without explicit sign-off.
- All tests must remain passing (C++ and Python).
- ASAN/UBSAN clean.
- Update this file when done.
