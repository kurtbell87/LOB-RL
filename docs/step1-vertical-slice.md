# Step 1: Vertical Slice

## Goal

Prove the full pipeline works end-to-end: C++ engine processes messages, maintains a book, exposes a Gym-like `step()/reset()` interface, and Python can call it via pybind11.

## Components

### 1. Message (header-only struct)

**File:** `include/lob/message.h`

```cpp
struct Message {
    enum class Side : uint8_t { Bid, Ask };
    enum class Action : uint8_t { Add, Cancel, Modify, Trade };

    uint64_t order_id;
    Side side;
    Action action;
    double price;
    uint32_t qty;
    uint64_t ts_ns;  // nanosecond timestamp
};
```

### 2. IMessageSource (interface)

**File:** `include/lob/source.h`

```cpp
class IMessageSource {
public:
    virtual ~IMessageSource() = default;
    virtual bool next(Message& msg) = 0;  // returns false when exhausted
    virtual void reset() = 0;             // rewind to start
};
```

### 3. SyntheticSource

**Files:** `src/data/synthetic_source.h`, `src/data/synthetic_source.cpp`

Generates a deterministic sequence of ~100 messages that builds a simple book:
- Phase 1 (msgs 0-9): Add 5 bid levels + 5 ask levels around mid=1000.00, tick=0.25
- Phase 2 (msgs 10-49): Random modifications and cancels
- Phase 3 (msgs 50-99): Trades that cross the spread

Must be deterministic (seeded RNG) so tests are reproducible. `reset()` rewinds to message 0 with the same seed.

### 4. Book

**Files:** `include/lob/book.h`, `src/engine/book.cpp`

Maintains bid and ask sides. For this vertical slice, only needs:

```cpp
class Book {
public:
    void apply(const Message& msg);  // process one message
    void reset();                     // clear all state

    double best_bid() const;         // NaN if empty
    double best_ask() const;         // NaN if empty
    double mid_price() const;        // (bid+ask)/2, NaN if either side empty
    double spread() const;           // ask-bid, NaN if either side empty

    int bid_depth() const;           // number of bid levels
    int ask_depth() const;           // number of ask levels
};
```

Internally, use `std::map<double, uint32_t>` for price-level aggregation (bids descending, asks ascending). Handle all four message actions:
- **Add:** Insert or increment qty at price level
- **Cancel:** Decrement qty (remove level if qty reaches 0)
- **Modify:** Update qty at price level (need order tracking via `std::unordered_map<uint64_t, OrderInfo>`)
- **Trade:** Decrement qty at price level

### 5. LOBEnv

**Files:** `include/lob/env.h`, `src/env/lob_env.cpp`

Minimal Gym-like environment:

```cpp
struct StepResult {
    std::vector<float> obs;   // observation vector
    float reward;
    bool done;
};

class LOBEnv {
public:
    explicit LOBEnv(std::unique_ptr<IMessageSource> source,
                    int steps_per_episode = 50);

    StepResult reset();
    StepResult step(int action);  // action in {0, 1, 2}
};
```

**Behavior:**
- `reset()`: Resets source and book. Replays messages until book has valid BBO. Returns initial observation.
- `step(action)`: Advance N messages (default: 1), apply to book, compute obs and reward.
- Observation: minimal 4-float vector for this slice: `[best_bid, best_ask, spread, position]` (raw values, normalization comes in Step 3)
- Reward: simple PnL delta = `position * (mid_now - mid_prev)`
- Done: when source is exhausted or `steps_per_episode` reached
- Action mapping: 0 = short (-1), 1 = flat (0), 2 = long (+1)

### 6. Python Bindings

**File:** `src/bindings/bindings.cpp`

pybind11 module named `lob_rl_core`:

```python
import lob_rl_core

env = lob_rl_core.LOBEnv()  # uses SyntheticSource by default
obs = env.reset()            # returns list of floats
obs, reward, done = env.step(2)  # action=2 (go long)
```

Bind:
- `LOBEnv` class with `reset()` and `step(action)` methods
- `step()` returns a tuple `(obs, reward, done)` where obs is a Python list

## Build System

CMake with:
- C++17
- pybind11 (fetched via FetchContent)
- GoogleTest (fetched via FetchContent)
- Targets: `lob_core` (static lib), `lob_rl_core` (pybind11 module), test executables

## Edge Cases & Error Conditions

- Book with no bids or asks: `best_bid()`/`best_ask()` return NaN
- Cancel for unknown order: no-op (log warning, don't crash)
- Action out of range: clamp to {0, 1, 2}
- Source exhausted: `done=true`
- Empty book mid_price: return NaN

## Acceptance Criteria

1. **Book tests:** Apply known sequence of messages, verify BBO at each step
2. **SyntheticSource tests:** Verify deterministic output (same messages on repeated calls), verify reset rewinds
3. **LOBEnv tests:** `reset()` returns valid observation, `step()` advances state, episode terminates correctly
4. **Python binding tests:** Import `lob_rl_core`, create env, call `reset()` and `step()`, verify return types and shapes
5. **Integration:** Step through entire synthetic episode from Python, verify no crashes and `done` eventually becomes `True`
