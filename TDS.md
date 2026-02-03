# LOB-RL: Technical Design Specification

---

## 1. Directory Structure

```
LOB-RL/
├── CMakeLists.txt
├── include/                    # Public interfaces (shared across modules)
│   ├── lob/
│   │   ├── message.h           # MBOMessage, Side, Action enums
│   │   ├── source.h            # IMessageSource interface
│   │   ├── book.h              # Book interface
│   │   ├── env.h               # LOBEnv, EnvConfig, StepResult
│   │   └── reward.h            # RewardType enum, RewardCalculator
├── src/
│   ├── data/                   # Data layer
│   │   ├── README.md
│   │   ├── synthetic_source.h
│   │   ├── synthetic_source.cpp
│   │   ├── databento_source.h
│   │   └── databento_source.cpp
│   ├── engine/                 # Book reconstruction
│   │   ├── README.md
│   │   ├── book.cpp
│   │   └── price_level.h
│   ├── env/                    # RL environment
│   │   ├── README.md
│   │   ├── lob_env.cpp
│   │   ├── feature_builder.h
│   │   ├── feature_builder.cpp
│   │   ├── reward_calculator.h
│   │   └── reward_calculator.cpp
│   └── bindings/               # Python bindings
│       ├── README.md
│       └── bindings.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_utils/             # Shared test utilities (compiled lib)
│   │   ├── fixtures.h
│   │   └── fixtures.cpp
│   ├── test_book.cpp
│   ├── test_env.cpp
│   └── test_source.cpp
├── python/
│   ├── setup.py
│   ├── lob_rl/
│   │   ├── __init__.py
│   │   └── wrappers.py         # Gym wrapper
│   └── tests/
│       └── test_gym.py
└── scripts/
    └── train.py                # Training entrypoint
```

---

## 2. Interface Contracts

### 2.1 Message Types (`include/lob/message.h`)

```cpp
#pragma once
#include <cstdint>

namespace lob {

enum class Side : uint8_t { Bid, Ask };

enum class Action : uint8_t {
    Add,
    Cancel,
    Modify,
    Trade,
    Clear
};

struct MBOMessage {
    uint64_t timestamp_ns;
    uint64_t order_id;
    int64_t  price;          // fixed-point (price * 1e9)
    uint32_t quantity;
    Side     side;
    Action   action;
};

static_assert(sizeof(MBOMessage) <= 40, "MBOMessage should be compact");

}  // namespace lob
```

### 2.2 Message Source (`include/lob/source.h`)

```cpp
#pragma once
#include "lob/message.h"
#include <memory>

namespace lob {

class IMessageSource {
public:
    virtual ~IMessageSource() = default;
    virtual bool has_next() const = 0;
    virtual MBOMessage next() = 0;
    virtual void reset() = 0;
    virtual uint64_t message_count() const = 0;
};

}  // namespace lob
```

### 2.3 Book (`include/lob/book.h`)

```cpp
#pragma once
#include "lob/message.h"
#include <cstdint>

namespace lob {

struct Level {
    int64_t  price;
    uint32_t quantity;
};

class Book {
public:
    void apply(const MBOMessage& msg);
    void clear();

    // Queries (depth 0 = best)
    Level bid(int depth) const;
    Level ask(int depth) const;
    int bid_depth() const;
    int ask_depth() const;

    // Derived
    int64_t mid_price() const;
    int64_t spread() const;
    double imbalance(int levels) const;
};

}  // namespace lob
```

### 2.4 Environment (`include/lob/env.h`)

```cpp
#pragma once
#include "lob/source.h"
#include "lob/reward.h"
#include <vector>
#include <memory>
#include <string>

namespace lob {

struct Session {
    int start_hour_utc;
    int start_minute_utc;
    int end_hour_utc;
    int end_minute_utc;
};

// Predefined sessions
namespace sessions {
    constexpr Session US_RTH_EST = {14, 30, 21, 0};  // EST (no DST)
    constexpr Session US_RTH_EDT = {13, 30, 20, 0};  // EDT (DST)
}

struct EnvConfig {
    std::string data_path;
    int book_depth = 10;
    int trades_per_step = 100;
    RewardType reward_type = RewardType::PnLDelta;
    double inventory_penalty = 0.0;
    Session session = sessions::US_RTH_EST;
};

struct Observation {
    std::vector<float> data;
    static int size(int book_depth);  // 4*book_depth + 4
};

struct StepResult {
    Observation obs;
    double reward;
    bool done;

    // Info (for debugging/logging)
    int position;
    double pnl;
    uint64_t timestamp_ns;
};

class LOBEnv {
public:
    LOBEnv(EnvConfig config, std::unique_ptr<IMessageSource> source);

    StepResult reset();
    StepResult step(int action);  // 0=short, 1=flat, 2=long

    int observation_size() const;
    static constexpr int action_size() { return 3; }
    const EnvConfig& config() const;
};

}  // namespace lob
```

### 2.5 Reward (`include/lob/reward.h`)

```cpp
#pragma once

namespace lob {

enum class RewardType {
    PnLDelta,
    PnLDeltaPenalized
};

class Book;  // forward decl

class RewardCalculator {
public:
    RewardCalculator(RewardType type, double inventory_penalty = 0.0);

    double calculate(
        int prev_position, double prev_pnl,
        int curr_position, double curr_pnl,
        const Book& book
    );

private:
    RewardType type_;
    double inventory_penalty_;
};

}  // namespace lob
```

---

## 3. Module README Template

Each module's README.md follows this format (20 lines max):

```markdown
# Module Name

## Purpose
One sentence.

## Interface
- `ClassName::method()` — what it does

## Dependencies
- Depends on: X, Y
- Depended on by: Z

## Example
\`\`\`cpp
// 5 lines showing basic usage
\`\`\`

## Tests
Run: `ctest -R test_modulename`
```

---

## 4. Build Configuration

### CMake Targets

| Target | Type | Description |
|--------|------|-------------|
| `lob_core` | STATIC | Core C++ library (book, env, sources) |
| `lob_rl` | MODULE | pybind11 Python module |
| `test_utils` | STATIC | Test fixtures and helpers |
| `tests` | EXECUTABLE | Google Test runner |

### Compiler Settings

- C++20
- `-Wall -Wextra -Werror`
- Debug: `-fsanitize=address,undefined`
- Release: `-O3 -march=native -DNDEBUG`

### Dependencies (FetchContent)

| Dependency | Purpose |
|------------|---------|
| pybind11 | Python bindings |
| Google Test | Unit tests |
| spdlog | Logging (header-only) |
| robin_hood | Fast hash map (header-only) |

Databento SDK is optional — build with `-DWITH_DATABENTO=ON`.

---

## 5. Testing Strategy

### Unit Tests

| Test File | Covers | Key Cases |
|-----------|--------|-----------|
| `test_source.cpp` | SyntheticSource | Determinism, message validity |
| `test_book.cpp` | Book | Add/cancel/modify/trade, edge cases |
| `test_env.cpp` | LOBEnv | Step/reset, episode boundaries, PnL |

### Test Fixtures (`test_utils/`)

```cpp
namespace fixtures {
    // Deterministic message sequences
    std::unique_ptr<IMessageSource> simple_session();   // 1000 msgs, normal activity
    std::unique_ptr<IMessageSource> empty_book();       // clears then nothing
    std::unique_ptr<IMessageSource> one_sided();        // bids only
    std::unique_ptr<IMessageSource> rapid_trades();     // high trade frequency
}
```

### Python Tests

| Test | Covers |
|------|--------|
| `test_gym.py` | Gym interface compliance, numpy dtypes, episode rollout |

---

## 6. Observation Vector Layout

For `book_depth = K`:

| Index | Count | Feature |
|-------|-------|---------|
| 0 | K | Bid prices (relative to mid, normalized) |
| K | K | Bid sizes (log-normalized) |
| 2K | K | Ask prices (relative to mid, normalized) |
| 3K | K | Ask sizes (log-normalized) |
| 4K | 1 | Spread (normalized) |
| 4K+1 | 1 | Imbalance [-1, 1] |
| 4K+2 | 1 | Time remaining [0, 1] |
| 4K+3 | 1 | Position {-1, 0, 1} |

**Total size: 4K + 4** (44 floats for K=10)

---

## 7. Parallel Work Assignment

| Agent/Team | Module | Depends On | Delivers |
|------------|--------|------------|----------|
| A | `src/data/` | `message.h` only | SyntheticSource, DatabentoSource |
| B | `src/engine/` | `message.h` only | Book |
| C | `src/env/` | Book interface | LOBEnv, FeatureBuilder, RewardCalculator |
| D | `src/bindings/` | LOBEnv interface | Python module |

**Integration order:** A+B can work in parallel → C integrates → D wraps.

---

## 8. Definition of Done

A module is done when:

1. All interface methods implemented
2. Unit tests pass
3. ASAN/UBSAN clean
4. README.md updated
5. Compiles in isolation (only depends on `include/`)
