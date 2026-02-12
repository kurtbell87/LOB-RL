# Phase 1: Extend IMarketBookDataSource with Depth Queries

## Problem

Constellation's `IMarketBookDataSource` interface only exposes `BestBidPrice()`, `BestAskPrice()`, and `VolumeAtPrice()`. LOB-RL's 11 book-dependent features need depth levels (3/5/10), weighted mid, and VAMP. The underlying `LimitOrderBook` already has `GetLevel(BookSide, size_t)`, `GetBids()`, `GetAsks()` — but these aren't accessible through the aggregator interface that features receive.

The `MicroDepthFeature` currently returns hardcoded zeros because it can't query depth through `IMarketBookDataSource`.

## What to Build

### 1. Add 4 pure virtual methods to `IMarketBookDataSource`

**File:** `src/constellation/interfaces/include/interfaces/orderbook/IMarketBookDataSource.hpp`

Add these methods to the `IMarketBookDataSource` class (requires `#include "interfaces/orderbook/IInstrumentBook.hpp"` for `BookSide` and `PriceLevel`; also `#include "interfaces/orderbook/IMarketStateView.hpp"` for `PriceLevel` if not already available):

```cpp
/// Return the price level at `depth_index` (0 = best) for the given instrument and side.
/// Returns nullopt if the instrument is unknown or the book has fewer than depth_index+1 levels.
virtual std::optional<PriceLevel> GetLevel(std::uint32_t instrument_id,
                                            BookSide side,
                                            std::size_t depth_index) const = 0;

/// Sum total quantity across the top `n_levels` on the given side.
/// Returns 0 if the instrument is unknown or the book is empty.
virtual std::uint64_t TotalDepth(std::uint32_t instrument_id,
                                  BookSide side,
                                  std::size_t n_levels) const = 0;

/// Volume-weighted mid-price: (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty).
/// Returns nullopt if either side has no quote. Price returned as double in real currency (not nanos).
virtual std::optional<double> WeightedMidPrice(std::uint32_t instrument_id) const = 0;

/// Volume-Adjusted Mid-Price (VAMP) using top `n_levels` on each side.
/// VAMP = sum(price_i * qty_i for all levels on both sides) / sum(qty_i for all levels on both sides).
/// Returns nullopt if either side is empty. Price returned as double in real currency.
virtual std::optional<double> VolumeAdjustedMidPrice(std::uint32_t instrument_id,
                                                      std::size_t n_levels) const = 0;
```

Note: `PriceLevel` is already defined in `IMarketStateView.hpp` and used in the codebase. `BookSide` is in `IInstrumentBook.hpp`. Both need to be included or forward-declared.

### 2. Implement in `MarketBook`

**File:** `src/constellation/modules/orderbook/include/orderbook/MarketBook.hpp` — add declarations.
**File:** `src/constellation/modules/orderbook/src/MarketBook.cpp` — add implementations.

Each method:
1. Locks `mapMutex_` to find the instrument's `LimitOrderBook*`
2. Returns nullopt/0 if instrument not found
3. Delegates to `LimitOrderBook` methods (which have their own internal locking)

Implementation details:

**GetLevel:** Delegates directly to `LimitOrderBook::GetLevel(side, depth_index)`.

**TotalDepth:** Iterates levels 0..n_levels-1, calling `LimitOrderBook::GetLevel(side, i)` and summing `total_quantity`. Stops early if a level returns nullopt.

**WeightedMidPrice:** Gets `BestBid()` and `BestAsk()` from the LOB. If either is nullopt, returns nullopt. Otherwise computes:
```
bid_price = best_bid.price / 1e9  (convert from nanos)
ask_price = best_ask.price / 1e9
bid_qty = best_bid.total_quantity
ask_qty = best_ask.total_quantity
wmid = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
```

**VolumeAdjustedMidPrice:** Gets top `n_levels` from both sides. For each level, accumulates `price * qty` and `qty`. Returns `sum(price*qty) / sum(qty)`. Prices converted from nanos to real currency. Returns nullopt if either side has zero levels.

### 3. Fix `MicroDepthFeature` to use real depth

**File:** `src/constellation/modules/features/src/primitives/MicroDepthFeature.cpp`

Replace the stub `ComputeUpdate` with a real implementation that calls `source.GetLevel(config_.instrument_id, config_.side, config_.depth_index)`. If the result is present, store `price / 1e9` and `total_quantity`. Otherwise store 0.

### 4. Expose new MarketBook methods in Python bindings

**File:** `src/bindings/bindings.cpp`

Add to the `MarketBook` binding:
```python
mb.get_level(instrument_id, side, depth_index)  # returns dict {price, quantity, order_count} or None
mb.total_depth(instrument_id, side, n_levels)    # returns int
mb.weighted_mid_price(instrument_id)             # returns float or None
mb.vamp(instrument_id, n_levels)                 # returns float or None
```

The `side` parameter should accept `BookSide.Bid` / `BookSide.Ask` enum values. The `BookSide` enum must be exposed to Python if not already.

## Edge Cases

- **Unknown instrument:** GetLevel returns nullopt, TotalDepth returns 0, WeightedMidPrice returns nullopt, VAMP returns nullopt.
- **Empty book side:** Same behavior as unknown instrument for that side.
- **depth_index beyond available levels:** GetLevel returns nullopt, TotalDepth sums only available levels.
- **n_levels = 0:** TotalDepth returns 0, VAMP returns nullopt.
- **Single-sided book (only bids or only asks):** WeightedMidPrice returns nullopt, VAMP returns nullopt.
- **All quantity at one level:** WeightedMidPrice and VAMP still compute correctly.
- **Nano price precision:** All internal prices are int64_t nanos. WeightedMidPrice and VAMP convert to double for return.

## Acceptance Criteria

1. `IMarketBookDataSource` has the 4 new pure virtual methods.
2. `MarketBook` implements all 4 methods correctly.
3. `MicroDepthFeature::ComputeUpdate` uses `source.GetLevel()` instead of returning zeros.
4. Python bindings expose `get_level`, `total_depth`, `weighted_mid_price`, `vamp` on `MarketBook`.
5. `BookSide` enum is available in Python as `core.BookSide.Bid` / `core.BookSide.Ask`.
6. All existing tests continue to pass (the new pure virtual methods must be implemented in MarketBook, which is the only concrete implementation of IMarketBookDataSource).
7. ~30 new tests covering depth levels, total depth, weighted mid, VAMP, edge cases, and Python bindings.

## Files to Modify

| File | Change |
|------|--------|
| `src/constellation/interfaces/include/interfaces/orderbook/IMarketBookDataSource.hpp` | Add 4 pure virtual methods + includes |
| `src/constellation/modules/orderbook/include/orderbook/MarketBook.hpp` | Add 4 override declarations |
| `src/constellation/modules/orderbook/src/MarketBook.cpp` | Implement 4 methods |
| `src/constellation/modules/features/src/primitives/MicroDepthFeature.cpp` | Fix ComputeUpdate |
| `src/bindings/bindings.cpp` | Expose new methods + BookSide enum |

## Files NOT to Modify

- `LimitOrderBook.hpp` / `.cpp` — already has the needed methods
- Any existing feature implementations — they don't use depth queries yet
- Any barrier pipeline code — that comes in later phases

## Test Categories

1. **GetLevel tests (~8):** bid/ask at depth 0,1,2; unknown instrument; empty book; beyond max depth
2. **TotalDepth tests (~6):** sum across 1,3,5,10 levels; partial depth (fewer levels than requested); empty book
3. **WeightedMidPrice tests (~6):** symmetric book; asymmetric quantities; single order each side; one-sided book; empty book
4. **VolumeAdjustedMidPrice tests (~6):** multi-level VAMP; n_levels > available; n_levels = 0; single level
5. **MicroDepthFeature fix (~4):** returns real prices after data update; handles missing levels
6. **Python binding tests (~5):** get_level returns dict/None; total_depth returns int; weighted_mid_price; vamp; BookSide enum
