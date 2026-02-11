# BarBuilder: Single-Pass Bar Construction with Book Accumulation

## Context

The barrier pipeline needs to build trade bars AND accumulate book feature data from MBO events. The Python pipeline does this in two passes: one for trades (`extract_trades_from_mbo`), one for all MBO events (`extract_all_mbo`). A single C++ pass through the `.dbn.zst` stream via `DbnFileSource` + `Book` replaces both.

The existing C++ infrastructure provides:
- `DbnFileSource` — reads `.dbn.zst` files, produces `Message` objects (with `map_action` mapping: A→Add, C→Cancel, M→Modify, T/F→Trade; R/others→skipped)
- `Book` — maintains LOB state with `apply()`, `best_bid()`, `best_ask()`, `best_bid_qty()`, `best_ask_qty()`, `total_bid_depth(n)`, `total_ask_depth(n)`, `weighted_mid()`, `vamp(n)`, `spread()`
- `SessionFilter` — classifies timestamps into PreMarket/RTH/PostMarket
- `SessionConfig` — configurable RTH open/close times

## Requirements

### 1. DbnFileSource Change: Map 'R' (Clear) to Cancel

Currently `map_action()` in `src/data/dbn_file_source.cpp` returns `false` for action='R', skipping Clear messages entirely. These messages indicate order removal and should update the book.

**Change:** Map 'R' to `Message::Action::Cancel` so the book processes order removals correctly.

### 2. TradeBar Struct

**New file:** `include/lob/barrier/trade_bar.h`

```cpp
struct TradeBar {
    int bar_index;
    double open, high, low, close;
    int volume;            // total trade qty
    double vwap;           // volume-weighted average price
    int64_t t_start;       // first trade timestamp (ns)
    int64_t t_end;         // last trade timestamp (ns)

    // Full trade sequences for tiebreaking
    std::vector<double> trade_prices;
    std::vector<int32_t> trade_sizes;
};
```

### 3. BarBookAccum Struct

**New file:** `include/lob/barrier/bar_builder.h`

Per-bar book feature accumulator. Captures everything needed by the feature computation in Cycle 3.

```cpp
struct BarBookAccum {
    // BBO at bar close
    uint32_t bid_qty = 0;
    uint32_t ask_qty = 0;

    // Depth at bar close
    uint32_t total_bid_3 = 0, total_ask_3 = 0;     // depth(3)
    uint32_t total_bid_5 = 0, total_ask_5 = 0;     // depth(5)
    uint32_t total_bid_10 = 0, total_ask_10 = 0;   // depth(10)

    // Cancel counts within bar
    int bid_cancels = 0;
    int ask_cancels = 0;

    // OFI: signed volume of Add messages at BBO level
    double ofi_signed_volume = 0.0;
    double total_add_volume = 0.0;

    // Weighted mid: first valid and end-of-bar
    double wmid_first = NaN;    // first valid wmid after an in-bar event
    double wmid_end = NaN;      // wmid at bar close

    // Spread samples collected during bar
    std::vector<double> spread_samples;

    // VAMP at bar temporal midpoint and bar end
    double vamp_at_mid = NaN;
    double vamp_at_end = NaN;

    // Aggressor volumes (passive side determines direction)
    double buy_aggressor_vol = 0.0;    // passive ask → buy
    double sell_aggressor_vol = 0.0;   // passive bid → sell

    // Trade and cancel counts within bar
    int n_trades = 0;
    int n_cancels = 0;
};
```

### 4. BarBuilder Class

```cpp
class BarBuilder {
public:
    BarBuilder(int bar_size, const SessionConfig& cfg);

    // Process one MBO message. May emit completed bars.
    void process(const Message& msg);

    // Flush any incomplete bar (call at end of stream).
    // Returns true if a partial bar was emitted.
    bool flush();

    // Access completed bars and accumulators.
    const std::vector<TradeBar>& bars() const;
    const std::vector<BarBookAccum>& accums() const;

    // RTH boundaries for feature computation.
    int64_t rth_open_ns() const;
    int64_t rth_close_ns() const;

private:
    Book book_;
    SessionFilter filter_;
    int bar_size_;
    // ... internal state for pending trades, current accum
};
```

**Processing logic for each `Message`:**
1. Classify timestamp: `filter_.classify(msg.ts_ns)` → Phase
2. **PreMarket:** `book_.apply(msg)` for warmup. No bar building.
3. **PostMarket:** Ignore (don't apply to book or build bars).
4. **RTH:**
   a. Apply to book: `book_.apply(msg)`
   b. Update current `BarBookAccum`:
      - If Cancel: increment `bid_cancels` or `ask_cancels`, increment `n_cancels`
      - If Add: accumulate OFI (signed_volume at BBO level), total_add_volume
      - If Trade/Fill: add to pending trade arrays, increment `n_trades`, accumulate aggressor volumes
      - After every event: sample spread, track wmid, track VAMP at bar midpoint
   c. When pending trades reach `bar_size_`:
      - Build `TradeBar` from pending trades (OHLCV, VWAP, timestamps)
      - Snapshot book state into `BarBookAccum` (BBO qty, depth, wmid_end, vamp_end)
      - Push both to completed vectors
      - Reset pending trades and accum state

**RTH boundary computation:** `BarBuilder` needs actual RTH open/close in nanoseconds-since-epoch (not time-of-day) for feature computation. The first RTH message's timestamp is used to compute the calendar day, then RTH boundaries are computed from that day. Expose via `rth_open_ns()` / `rth_close_ns()`.

### 5. CMakeLists.txt Changes

Add `src/barrier/bar_builder.cpp` to `lob_core` sources. Create `include/lob/barrier/` directory.

## Files to Change

| File | Change |
|------|--------|
| `src/data/dbn_file_source.cpp` | Map 'R' → Cancel in `map_action()` |
| `include/lob/barrier/trade_bar.h` | **NEW** — TradeBar struct |
| `include/lob/barrier/bar_builder.h` | **NEW** — BarBookAccum, BarBuilder class |
| `src/barrier/bar_builder.cpp` | **NEW** — Implementation |
| `CMakeLists.txt` | Add src/barrier/bar_builder.cpp and test file |
| `tests/test_barrier_bar_builder.cpp` | **NEW** — Tests |

## Test Plan

### TradeBar tests (~5):
1. Default construction has zero values
2. Trade prices/sizes vectors populated correctly
3. OHLCV from known trade sequence
4. VWAP hand-computed from known trades
5. Timestamps match first/last trade

### BarBuilder basic tests (~10):
6. Empty stream → no bars
7. Single bar of exactly bar_size trades → 1 bar
8. Two complete bars → 2 bars with correct indices
9. Incomplete final bar → not emitted without flush
10. flush() emits partial bar
11. Bar OHLCV values correct
12. Bar VWAP hand-computed
13. Trade prices/sizes arrays stored in bar
14. Bar timestamps match first/last trade in bar
15. Volume equals sum of trade sizes

### RTH filtering tests (~5):
16. Pre-market messages → book warmup only, no bars
17. Post-market messages → ignored
18. Mixed pre-market + RTH → only RTH trades form bars
19. Pre-market book state carries into RTH (book not reset)
20. rth_open_ns() and rth_close_ns() return correct values

### BarBookAccum tests (~15):
21. BBO qty snapshot at bar close
22. Depth(3/5/10) snapshot at bar close
23. Cancel counts per side
24. OFI: Add at BBO increases signed volume
25. OFI: Add away from BBO does not affect OFI
26. wmid_first captured after first in-bar event
27. wmid_end captured at bar close
28. Spread samples collected
29. VAMP at bar midpoint
30. VAMP at bar end
31. Aggressor volumes: passive ask → buy aggressor
32. Aggressor volumes: passive bid → sell aggressor
33. n_trades count
34. n_cancels count
35. Accumulator resets between bars

### DbnFileSource 'R' action test (~2):
36. 'R' action maps to Cancel
37. Book handles 'R'-mapped Cancel correctly

### Integration test (~3):
38. Full stream with warmup + RTH → correct bar count
39. Bar close prices form valid price series
40. Accum values are non-default for populated bars

## Acceptance Criteria

- All existing C++ tests still pass (449 + 15 skipped)
- New tests pass (~40 cases)
- `DbnFileSource` 'R' mapping doesn't break existing tests
- TradeBar matches Python `TradeBar` fields
- BarBookAccum captures all data needed by feature_compute (Cycle 3)
