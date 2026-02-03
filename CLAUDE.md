# LOB-RL Project Context

## What This Is

RL environment for training directional trading agents on /MES futures using Databento MBO data.

**Read first:** [PRD.md](PRD.md) (requirements), [TDS.md](TDS.md) (technical spec)

---

## Core Decisions (Do Not Deviate)

| Aspect | Decision |
|--------|----------|
| Strategy | Directional, not market-making |
| Position | {-1, 0, +1} only |
| Orders | Market orders, instant fill at BBO |
| Book | Read-only reconstruction (no matching engine) |
| Episode | One session, flat at open/close |
| Observation | 44 floats (10-level book + features) |
| Reward | C++ enum-configured, no Python callbacks |

---

## Architecture

```
Python (Gym, training)
        │
        │ pybind11
        ▼
    LOBEnv ─── RewardCalculator
        │
        ▼
      Book ─── FeatureBuilder
        │
        ▼
  IMessageSource
   ├── SyntheticSource (testing)
   └── DatabentoSource (real data)
```

---

## Module Ownership

| Module | Path | Interface |
|--------|------|-----------|
| Data | `src/data/` | `include/lob/source.h` |
| Engine | `src/engine/` | `include/lob/book.h` |
| Env | `src/env/` | `include/lob/env.h` |
| Bindings | `src/bindings/` | Gym-compatible |

**Rule:** Each module depends only on `include/lob/*.h`. Never reach into another module's `src/`.

---

## Interface Contracts

### MBOMessage (the currency between modules)

```cpp
struct MBOMessage {
    uint64_t timestamp_ns;
    uint64_t order_id;
    int64_t  price;          // fixed-point (price * 1e9)
    uint32_t quantity;
    Side     side;           // Bid, Ask
    Action   action;         // Add, Cancel, Modify, Trade, Clear
};
```

### Book Queries

```cpp
Level bid(int depth) const;  // depth 0 = best
Level ask(int depth) const;
int64_t mid_price() const;
int64_t spread() const;
double imbalance(int levels) const;
```

### Environment

```cpp
StepResult reset();
StepResult step(int action);  // 0=short, 1=flat, 2=long
```

---

## Observation Layout (K=10)

| Index | Feature |
|-------|---------|
| 0-9 | Bid prices (relative to mid) |
| 10-19 | Bid sizes |
| 20-29 | Ask prices (relative to mid) |
| 30-39 | Ask sizes |
| 40 | Spread |
| 41 | Imbalance |
| 42 | Time remaining |
| 43 | Position |

---

## Build Order

1. **Vertical slice:** SyntheticSource → minimal Book → minimal LOBEnv → Python binding
2. **Real data:** DatabentoSource, full Book
3. **Training:** Full features, reward calc, Gym wrapper, train agent

---

## Testing

- Use `SyntheticSource` for all unit tests (deterministic, no external files)
- Test fixtures live in `tests/test_utils/`
- Every module must pass ASAN/UBSAN
- Run: `ctest` (C++), `pytest python/tests/` (Python)

---

## What NOT To Do

- No limit order simulation
- No matching engine / queue priority
- No Python in the hot path
- No dependencies beyond those in TDS.md
- No "improvements" outside assigned module
- No changes to `include/lob/*.h` without sign-off

---

## When You're Done

1. All interface methods implemented
2. Unit tests pass
3. ASAN/UBSAN clean
4. Module README.md updated
5. Compiles in isolation
