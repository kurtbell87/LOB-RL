# Phase 3: Migrate Trade-Only Bar Features (11 features)

## Problem

LOB-RL's `compute_bar_features()` in `src/barrier/feature_compute.cpp` computes 22 features from `TradeBar` and `BarBookAccum` structs. 11 of these depend only on trade data (no book depth queries). This phase migrates those 11 features to Constellation's `IBarFeature` system, validating the pattern before tackling book-dependent features in Phase 4.

## Architecture Extension: MBO Event Access

Bar features need raw MBO event data (trade price, size, action type, side) that isn't available through `IMarketBookDataSource` or `IMarketView`. Since `databento::MboMsg` is already used in Constellation's interface layer (`IMarketBook.hpp`), we extend `IBarFeature` with MBO message access.

### Changes to Phase 2 interfaces

**File:** `src/constellation/interfaces/include/interfaces/features/IBarFeature.hpp`

Add a default virtual method (backward-compatible):
```cpp
#include "databento/record.hpp"

class IBarFeature : public virtual IFeature {
    // ... existing methods ...

    /// Called with the raw MBO message before OnDataUpdate.
    /// Default no-op — override in features that need trade/event data.
    virtual void OnMboMsg(const databento::MboMsg& mbo) { (void)mbo; }
};
```

**File:** `src/constellation/modules/features/include/features/BarFeatureManager.hpp`

Add an overloaded `OnMboEvent`:
```cpp
/// Forward an MBO event to all registered bar features.
/// First calls OnMboMsg(mbo) on each feature, then OnDataUpdate(source, market).
void OnMboEvent(
    const databento::MboMsg& mbo,
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market);
```

The existing `OnMboEvent(source, market)` overload is kept for backward compatibility.

### IConfigurableFeature

All 11 features implement `IConfigurableFeature` for `tick_size` and `rth_open_ns`/`rth_close_ns` configuration. The `IConfigurableFeature` interface from `src/constellation/interfaces/include/interfaces/features/IConfigurableFeature.hpp` provides `SetParam(key, value)`.

## Features to Implement

All features inherit from `AbstractBarFeature` and `IConfigurableFeature`. Each:
- Overrides `OnMboMsg(mbo)` to extract trade data (if needed)
- Overrides `AccumulateEvent(source, market)` (may be no-op for pure trade features)
- Overrides `ResetAccumulators()` and `FinalizeBar()`
- Overrides `GetBarValue(name)` and `HasFeature(name)`
- Overrides `SetParam(key, value)` for configuration
- Registers via `REGISTER_FEATURE()`

### Col 0: TradeFlowImbalanceBarFeature

**Value name:** `"trade_flow_imbalance"`

**Accumulate (OnMboMsg):** For each Trade/Fill action, record trade price and size. Apply tick rule: compare current trade price to previous. If higher → uptick (+1), if lower → downtick (-1), if equal → forward-fill previous direction. Multiply size by direction to get signed volume.

**Finalize:** `(buy_vol - sell_vol) / (buy_vol + sell_vol)`. Returns 0 if total is 0.

**Config:** None required (works on raw trade data).

### Col 3: BarRangeBarFeature

**Value name:** `"bar_range"`

**Accumulate (OnMboMsg):** For each Trade/Fill, track high and low trade prices.

**Finalize:** `(high - low) / tick_size`. Returns 0 if no trades.

**Config:** `tick_size` (default 0.25).

### Col 4: BarBodyBarFeature

**Value name:** `"bar_body"`

**Accumulate (OnMboMsg):** Track first and last trade prices (open/close).

**Finalize:** `(close - open) / tick_size`. Returns 0 if no trades.

**Config:** `tick_size` (default 0.25).

### Col 5: BodyRangeRatioBarFeature

**Value name:** `"body_range_ratio"`

**Accumulate (OnMboMsg):** Track open, close, high, low.

**Finalize:** `(close - open) / (high - low)`. Returns 0 if range is 0.

**Config:** None.

### Col 6: VwapDisplacementBarFeature

**Value name:** `"vwap_displacement"`

**Accumulate (OnMboMsg):** Accumulate `price * size` and `size` for VWAP. Track close, high, low.

**Finalize:** `vwap = sum(price*size) / sum(size)`. Returns `(close - vwap) / (high - low)`. Returns 0 if range is 0.

**Config:** None.

### Col 7: LogVolumeBarFeature

**Value name:** `"log_volume"`

**Accumulate (OnMboMsg):** Sum trade sizes.

**Finalize:** `log(max(volume, 1))`.

**Config:** None.

### Col 8: RealizedVolBarFeature

**Value name:** `"realized_vol"`

**Cross-bar state:** This feature maintains state across bars (running sum and sum-of-squares of log returns). `ResetAccumulators()` does NOT reset cross-bar state — only resets intra-bar accumulators (current bar's close price tracker).

**Accumulate (OnMboMsg):** Track last trade price (bar close).

**OnBarComplete:** If bar_index > 0, compute log return `log(close / prev_close)` and update running sums. If bar_index < REALIZED_VOL_WARMUP (19), store NaN. Otherwise compute `sqrt(max(E[x^2] - E[x]^2, 0))` using the running sums over all bars.

**Finalize:** Store the computed realized vol value.

**Config:** `warmup_period` (default 19).

### Col 9: SessionTimeBarFeature

**Value name:** `"session_time"`

**Accumulate (OnMboMsg):** Track latest timestamp.

**Finalize:** `clamp((t_end - rth_open_ns) / (rth_close_ns - rth_open_ns), 0, 1)`.

**Config:** `rth_open_ns`, `rth_close_ns` (both required, set via `SetParam`).

### Col 12: SessionAgeBarFeature

**Value name:** `"session_age"`

**No accumulation needed.** Uses `current_bar_index()` from `AbstractBarFeature`.

**Finalize:** `min(bar_index / 20.0, 1.0)`.

**Config:** `period` (default 20.0).

### Col 19: TradeArrivalRateBarFeature

**Value name:** `"trade_arrival_rate"`

**Accumulate (OnMboMsg):** Count Trade/Fill events.

**Finalize:** `log(1 + n_trades)`.

**Config:** None.

### Col 21: PriceImpactBarFeature

**Value name:** `"price_impact_per_trade"`

**Accumulate (OnMboMsg):** Track open, close, and trade count.

**Finalize:** `(close - open) / (max(n_trades, 1) * tick_size)`.

**Config:** `tick_size` (default 0.25).

## File Layout

All feature files go in `src/constellation/modules/features/src/bar/` (new directory) with headers in `src/constellation/modules/features/include/features/bar/`.

| File | Class |
|------|-------|
| `bar/TradeFlowImbalanceBarFeature.hpp/cpp` | `TradeFlowImbalanceBarFeature` |
| `bar/BarRangeBarFeature.hpp/cpp` | `BarRangeBarFeature` |
| `bar/BarBodyBarFeature.hpp/cpp` | `BarBodyBarFeature` |
| `bar/BodyRangeRatioBarFeature.hpp/cpp` | `BodyRangeRatioBarFeature` |
| `bar/VwapDisplacementBarFeature.hpp/cpp` | `VwapDisplacementBarFeature` |
| `bar/LogVolumeBarFeature.hpp/cpp` | `LogVolumeBarFeature` |
| `bar/RealizedVolBarFeature.hpp/cpp` | `RealizedVolBarFeature` |
| `bar/SessionTimeBarFeature.hpp/cpp` | `SessionTimeBarFeature` |
| `bar/SessionAgeBarFeature.hpp/cpp` | `SessionAgeBarFeature` |
| `bar/TradeArrivalRateBarFeature.hpp/cpp` | `TradeArrivalRateBarFeature` |
| `bar/PriceImpactBarFeature.hpp/cpp` | `PriceImpactBarFeature` |

## Files to Modify

| File | Change |
|------|--------|
| `src/constellation/interfaces/include/interfaces/features/IBarFeature.hpp` | Add `OnMboMsg` virtual method |
| `src/constellation/modules/features/include/features/BarFeatureManager.hpp` | Add overloaded `OnMboEvent` |
| `src/constellation/modules/features/src/BarFeatureManager.cpp` | Implement overloaded `OnMboEvent` |
| `CMakeLists.txt` | Add 11 new .cpp files to constellation_features target |
| `src/bindings/bindings.cpp` | Add factory functions for each feature |

## Constants

Match the existing `feature_compute.h` constants:
- `TICK_SIZE = 0.25` (configurable via `SetParam`)
- `REALIZED_VOL_WARMUP = 19`
- `SESSION_AGE_PERIOD = 20.0`
- `EPSILON = 1e-10`

## Trade Detection

In `OnMboMsg(const databento::MboMsg& mbo)`, a trade event is detected when `mbo.action == databento::Action::Trade` or `mbo.action == databento::Action::Fill`. The trade price is `static_cast<double>(mbo.price) / databento::kFixedPriceScale` and the trade size is `static_cast<int>(mbo.size)`. The trade side is determined from `mbo.side` (`databento::Side::Ask` = buyer-initiated, `databento::Side::Bid` = seller-initiated).

## Edge Cases

- **Zero trades in a bar:** All trade-dependent features return 0 (or NaN for realized vol during warmup).
- **Single trade in bar:** Trade flow imbalance returns 0 (needs >=2 for tick rule). Range = 0. Body = 0.
- **Realized vol warmup:** First 19 bars return NaN. Feature system handles NaN → 0 in normalization.
- **Session time with bad RTH params:** Clamp to [0, 1].
- **Tick size = 0:** Should not happen; features guard with max(tick_size, epsilon).

## Acceptance Criteria

1. `IBarFeature` has `OnMboMsg` with default no-op.
2. `BarFeatureManager` has overloaded `OnMboEvent(mbo, source, market)`.
3. All 11 features register via `REGISTER_FEATURE()` and are constructible by name.
4. Each feature configured via `SetParam()`.
5. **Bit-exact regression:** For each feature column, given the same trade sequence and params, the bar feature output matches `compute_bar_features()` to within `1e-12`.
6. All existing tests pass (IBarFeature extension is backward-compatible).
7. Python bindings expose factory functions for each feature.
8. ~40 new tests covering all features + edge cases + regression.

## Test Strategy

For each feature:
1. **Unit test:** Construct feature, simulate a bar with known trade data, verify output matches hand-computed expected value.
2. **Regression test:** Compare feature output against `compute_bar_features()` for the same column, using a synthetic multi-bar sequence with varied trade patterns.

For the regression test, create a helper that:
1. Builds `TradeBar`/`BarBookAccum` from the same trade sequence
2. Calls `compute_bar_features()` to get reference values
3. Runs the same trades through `BarFeatureManager` with the corresponding bar feature
4. Asserts bit-exact match for the specific column

Additionally:
- Test `REGISTER_FEATURE` registration works (create by name from `FeatureRegistry`)
- Test `SetParam` for tick_size, RTH params
- Test realized vol cross-bar state persistence
- Test Python bindings
