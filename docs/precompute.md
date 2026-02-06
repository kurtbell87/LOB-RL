# Spec: C++ `precompute()` Function

## Overview

Pre-compute all LOB snapshots for a trading day in a single C++ pass. Since the agent has zero market impact, book evolution is deterministic — we can replay once, record observations at each BBO change, and train on pure numpy arrays without any Python↔C++ overhead per step.

## Interface

### Header: `include/lob/precompute.h`

```cpp
#pragma once
#include "lob/source.h"
#include "lob/session.h"
#include <vector>
#include <string>

struct PrecomputedDay {
    std::vector<float> obs;     // N × 43, row-major (no position — that's agent state)
    std::vector<double> mid;    // N mid-prices
    std::vector<double> spread; // N spreads
    int num_steps = 0;          // N (number of snapshots)
};

// Primary overload: works with any IMessageSource (testable with ScriptedSource)
PrecomputedDay precompute(IMessageSource& source, const SessionConfig& cfg);

// Convenience overload: creates BinaryFileSource from path internally
PrecomputedDay precompute(const std::string& path, const SessionConfig& cfg);
```

### Implementation: `src/env/precompute.cpp`

## Logic

1. **Warmup phase**: Apply all pre-market messages to the Book (same logic as `LOBEnv::reset_with_session` — if `cfg.warmup_messages < 0`, apply all; if `> 0`, apply the last N; if `== 0`, skip)
2. **RTH replay**: For each RTH message:
   - Record `prev_bid = book.best_bid()`, `prev_ask = book.best_ask()` before apply
   - Call `book.apply(msg)`
   - Check if `best_bid` or `best_ask` changed from the previous values
   - If BBO changed AND both bid and ask are finite:
     - Compute `time_remaining = 1.0 - session_filter.session_progress(msg.ts_ns)`
     - Call `feature_builder.build(book, 0.0f, time_remaining)` — position=0 because position is agent state, not market state
     - Append first 43 floats (indices 0-42, excluding position at index 43) to `obs`
     - Append `book.mid_price()` to `mid`
     - Append `book.spread()` to `spread`
     - Increment `num_steps`
3. **Stop** when source returns `false` or message is PostMarket
4. The source is NOT reset — caller manages lifecycle

## Existing components reused (DO NOT modify)

- `Book` (`include/lob/book.h`, `src/engine/book.cpp`)
- `FeatureBuilder` (`include/lob/feature_builder.h`, `src/env/feature_builder.cpp`)
- `SessionFilter` / `SessionConfig` (`include/lob/session.h`)
- `BinaryFileSource` (`src/data/binary_file_source.h`, `src/data/binary_file_source.cpp`)
- `IMessageSource` (`include/lob/source.h`)

## New files

- `include/lob/precompute.h` — header with struct + function declarations
- `src/env/precompute.cpp` — implementation

## CMakeLists.txt changes

- Add `src/env/precompute.cpp` to `lob_core` STATIC library sources
- Add `tests/test_precompute.cpp` to `lob_tests` sources

## Test plan (C++ only, file: `tests/test_precompute.cpp`)

### Test 1: Empty source returns zero steps
- Create a ScriptedSource with no messages
- Call `precompute(source, SessionConfig::default_rth())`
- Assert `num_steps == 0`, `obs.empty()`, `mid.empty()`, `spread.empty()`

### Test 2: Pre-market only (no RTH messages) returns zero steps
- Create messages with timestamps all in pre-market (before 13:30 UTC)
- Assert `num_steps == 0`

### Test 3: RTH messages without BBO change produce no snapshots
- Create pre-market messages establishing BBO (bid=999.75, ask=1000.25)
- Create RTH messages that add orders at deeper levels (bid=999.50, ask=1000.50)
- Assert `num_steps == 0` (BBO didn't change)

### Test 4: Single BBO change produces exactly 1 snapshot
- Pre-market: establish book with bid=999.75/100, ask=1000.25/100
- RTH: one message that improves best bid to 1000.00
- Assert `num_steps == 1`
- Assert `obs.size() == 43`
- Assert `mid[0]` is approximately `(1000.00 + 1000.25) / 2 = 1000.125`
- Assert `spread[0]` is approximately `0.25`
- Assert all 43 obs values are finite

### Test 5: Multiple BBO changes produce correct count
- Pre-market: establish book
- RTH: 5 messages that each change the best bid or best ask
- Assert `num_steps == 5`
- Assert `obs.size() == 5 * 43`
- Assert `mid.size() == 5` and `spread.size() == 5`

### Test 6: Post-market messages are excluded
- Pre-market: establish book
- RTH: 2 BBO-changing messages
- Post-market: 3 BBO-changing messages (timestamp after 20:00 UTC)
- Assert `num_steps == 2` (post-market messages ignored)

### Test 7: obs contains first 43 floats from FeatureBuilder (no position)
- Create a scenario with one BBO change
- Verify `obs[42]` is time_remaining (between 0 and 1, since it's RTH)
- Verify obs has exactly 43 floats (not 44 — no position)

### Test 8: time_remaining decreases across snapshots
- Create multiple BBO changes spread across RTH
- Assert `obs[i*43 + 42]` (time_remaining) is monotonically decreasing across snapshots

### Test 9: mid-prices track actual book mid
- Create known price changes and verify `mid` vector matches expected values

### Test 10: Warmup respects SessionConfig.warmup_messages
- Create 100 pre-market messages, set `warmup_messages = 5`
- Only last 5 pre-market messages should be applied to book
- Verify the book state reflects only those 5 messages (check BBO)

### Test 11: String overload creates BinaryFileSource
- Use a test fixture binary file (from `tests/fixtures/`)
- Call `precompute(path, cfg)` and verify it produces results
- (This tests the convenience overload)

### Test 12: BBO change requires both bid and ask to be finite
- Start with only bids (no asks) in pre-market
- RTH: change bid — should NOT produce snapshot (ask is infinite)
- RTH: add first ask — NOW should produce snapshot (both finite)

## Acceptance criteria

- All tests pass with Debug build (`cd build && cmake --build . && ctest`)
- `PrecomputedDay` struct is row-major for efficient numpy conversion
- No modifications to existing source files (Book, FeatureBuilder, etc.)
- obs vector excludes position (index 43) — only 43 floats per snapshot
