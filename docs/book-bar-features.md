# Phase 4: Migrate Book-Dependent Bar Features (11 features)

## Problem

11 of LOB-RL's 22 features depend on book state (BBO quantities, depth levels, weighted mid, VAMP, spread, cancels). These use the depth queries added in Phase 1 and the IBarFeature infrastructure from Phase 2. This completes the full 22-feature migration to Constellation's composable feature system.

## Architecture

Same pattern as Phase 3: each feature inherits `AbstractBarFeature` + `IConfigurableFeature`, implements `AccumulateEvent()` and optionally `OnMboMsg()`, registers via `REGISTER_FEATURE()`.

Key difference from Phase 3: these features query `IMarketBookDataSource` in `AccumulateEvent()` (book state per event) and/or extract cancel/add actions from `OnMboMsg()`.

## Features to Implement

### Col 1: BboImbalanceBarFeature

**Value name:** `"bbo_imbalance"`

**AccumulateEvent:** No-op during bar. Only reads book state at finalize.

**OnMboMsg:** No-op.

**FinalizeBar:** Query `source.GetLevel(instrument_id, Bid, 0)` and `GetLevel(instrument_id, Ask, 0)` from the **last** `AccumulateEvent` call's source. Store bid_qty and ask_qty. Compute `bid_qty / (bid_qty + ask_qty)`. Default `0.5` if total is 0.

**Implementation note:** Store a pointer to the last-seen `IMarketBookDataSource*` from `AccumulateEvent`. In `FinalizeBar()`, use it to read final BBO. This is safe because `FinalizeBar` is called immediately after the last `AccumulateEvent` of the bar.

**Config:** `instrument_id` (required).

### Col 2: DepthImbalanceBarFeature

**Value name:** `"depth_imbalance"`

**FinalizeBar:** `source.TotalDepth(instrument_id, Bid, 5)` and `TotalDepth(instrument_id, Ask, 5)`. Compute `bid / (bid + ask)`. Default `0.5` if total is 0.

**Config:** `instrument_id`, `n_levels` (default 5).

### Col 10: CancelAsymmetryBarFeature

**Value name:** `"cancel_asymmetry"`

**OnMboMsg:** If action is Cancel, check `mbo.side`. `databento::Side::Bid` → increment bid_cancels. `databento::Side::Ask` → increment ask_cancels.

**FinalizeBar:** `(bid_cancels - ask_cancels) / (bid_cancels + ask_cancels + 1e-10)`.

**Config:** None.

### Col 11: SpreadMeanBarFeature

**Value name:** `"spread_mean"`

**AccumulateEvent:** On every event, sample `best_ask_price - best_bid_price` (from `source.BestBidPrice/BestAskPrice`, converted to real currency). Append to running sum and count. Skip if either side is nullopt.

**FinalizeBar:** `sum / count`. Default `1.0` if no samples.

**Config:** `instrument_id`.

### Col 13: OrderFlowImbalanceBarFeature (OFI)

**Value name:** `"ofi"`

**OnMboMsg:** If action is Add:
- Extract side and size from MBO message
- `databento::Side::Bid` → signed_volume += size
- `databento::Side::Ask` → signed_volume -= size
- total_add_volume += size (unsigned)

**FinalizeBar:** If both signed_volume and total_add_volume are 0, return 0. Otherwise `clamp(signed_volume / (total_add_volume + 1e-10), -1, 1)`.

**Config:** None.

### Col 14: DepthRatioBarFeature

**Value name:** `"depth_ratio"`

**FinalizeBar:**
```
total_3 = TotalDepth(inst, Bid, 3) + TotalDepth(inst, Ask, 3)
total_10 = TotalDepth(inst, Bid, 10) + TotalDepth(inst, Ask, 10)
if total_10 == 0: return 0.5
return total_3 / (total_10 + 1e-10)
```

**Config:** `instrument_id`.

### Col 15: WeightedMidDisplacementBarFeature

**Value name:** `"wmid_displacement"`

**AccumulateEvent:** On first event of bar, sample `source.WeightedMidPrice(instrument_id)` → store as `wmid_first`. On every event, update `wmid_end`.

**FinalizeBar:** If either wmid_first or wmid_end is NaN/nullopt, return 0. Otherwise `(wmid_end - wmid_first) / tick_size`.

**Config:** `instrument_id`, `tick_size` (default 0.25).

### Col 16: SpreadStdBarFeature

**Value name:** `"spread_std"`

**AccumulateEvent:** Same as SpreadMeanBarFeature — sample spread on every event. Accumulate sum and sum-of-squares.

**FinalizeBar:** If < 2 samples, return 0. Otherwise compute population std: `sqrt(E[x^2] - E[x]^2)`.

**Config:** `instrument_id`.

### Col 17: VampDisplacementBarFeature

**Value name:** `"vamp_displacement"`

**AccumulateEvent:** At the midpoint of the bar (approximated: at event count / 2), sample `source.VolumeAdjustedMidPrice(instrument_id, n_levels)` → `vamp_at_mid`. At every event, update `vamp_at_end`.

**Implementation note:** Since we don't know the total event count in advance, track event_count and capture vamp_at_mid on the first event past the halfway mark. We track the running event count and at bar completion, the "mid" sample is the one closest to event_count/2. Alternative simpler approach: capture vamp_at_mid at the first event and vamp_at_end at the last event (matches how `BarBuilder::snapshot_accum()` works — it captures `vamp_at_mid` at the trade count midpoint of bar construction). For exact match with the reference implementation: the reference `BarBuilder` samples `vamp_at_mid` when `trade_count_in_bar >= bar_size/2` for the first time. Since we don't know bar_size, we'll take the simpler approach of tracking all events and computing the "mid event" sample. However, since the reference accum's `vamp_at_mid` is set at `bar_size/2` trades, we need to match this.

**Config:** `instrument_id`, `n_levels` (default 5), `tick_size` (default 0.25), `bar_size` (for midpoint calculation).

### Col 18: AggressorImbalanceBarFeature

**Value name:** `"aggressor_imbalance"`

**OnMboMsg:** If action is Trade or Fill:
- Check `mbo.side`: `databento::Side::Ask` → buyer-initiated (buy_aggressor_vol += size), `databento::Side::Bid` → seller-initiated (sell_aggressor_vol += size).

**FinalizeBar:** If total is 0, return 0. Otherwise `(buy - sell) / (buy + sell + 1e-10)`.

**Config:** None.

### Col 20: CancelTradeRatioBarFeature

**Value name:** `"cancel_trade_ratio"`

**OnMboMsg:** Count Cancel events (n_cancels) and Trade/Fill events (n_trades).

**FinalizeBar:** `log(1 + n_cancels / max(n_trades, 1))`.

**Config:** None.

## File Layout

Same pattern as Phase 3: headers in `include/features/bar/`, sources in `src/bar/`.

| File | Class |
|------|-------|
| `bar/BboImbalanceBarFeature.hpp/cpp` | `BboImbalanceBarFeature` |
| `bar/DepthImbalanceBarFeature.hpp/cpp` | `DepthImbalanceBarFeature` |
| `bar/CancelAsymmetryBarFeature.hpp/cpp` | `CancelAsymmetryBarFeature` |
| `bar/SpreadMeanBarFeature.hpp/cpp` | `SpreadMeanBarFeature` |
| `bar/OrderFlowImbalanceBarFeature.hpp/cpp` | `OrderFlowImbalanceBarFeature` |
| `bar/DepthRatioBarFeature.hpp/cpp` | `DepthRatioBarFeature` |
| `bar/WeightedMidDisplacementBarFeature.hpp/cpp` | `WeightedMidDisplacementBarFeature` |
| `bar/SpreadStdBarFeature.hpp/cpp` | `SpreadStdBarFeature` |
| `bar/VampDisplacementBarFeature.hpp/cpp` | `VampDisplacementBarFeature` |
| `bar/AggressorImbalanceBarFeature.hpp/cpp` | `AggressorImbalanceBarFeature` |
| `bar/CancelTradeRatioBarFeature.hpp/cpp` | `CancelTradeRatioBarFeature` |

## Files to Modify

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add 11 new .cpp files |
| `src/bindings/bindings.cpp` | Factory functions for each feature |

## Design: Last-Event Book State Access

Features that read book state at bar completion (BboImbalance, DepthImbalance, DepthRatio, WeightedMidDisplacement, VampDisplacement) need access to `IMarketBookDataSource` during `FinalizeBar()`. But `FinalizeBar()` has no parameters.

**Solution:** In `AccumulateEvent(source, market)`, store a const pointer to `source` as a protected member of `AbstractBarFeature` (or a member the derived class manages). Since `FinalizeBar()` is called synchronously by `BarFeatureManager::NotifyBarComplete()` after the last `OnMboEvent()`, the pointer is still valid.

This is already the pattern: store `const IMarketBookDataSource* last_source_ = nullptr` in the feature, set it in `AccumulateEvent()`, read it in `FinalizeBar()`.

## Edge Cases

- **Empty book:** BboImbalance returns 0.5, DepthImbalance returns 0.5, DepthRatio returns 0.5.
- **One-sided book:** Imbalances return 0 or 1 (all on one side).
- **No cancels in bar:** CancelAsymmetry returns 0, CancelTradeRatio returns log(1+0)=0.
- **No trades in bar:** AggressorImbalance returns 0, CancelTradeRatio uses max(n_trades,1)=1.
- **No spread samples:** SpreadMean returns 1.0, SpreadStd returns 0.
- **WeightedMid unavailable:** WeightedMidDisplacement returns 0.
- **VAMP unavailable:** VampDisplacement returns 0.

## Acceptance Criteria

1. All 11 book-dependent features implement `IBarFeature` with correct accumulation.
2. Each registers via `REGISTER_FEATURE()`.
3. **Bit-exact regression** for each column against `compute_bar_features()` (tolerance 1e-12).
4. Python bindings expose all 11 features.
5. All existing tests pass.
6. ~40 new tests.

## Test Strategy

Same as Phase 3:
1. Unit tests per feature with synthetic book state
2. Regression tests comparing against `compute_bar_features()` for the matching column
3. Full 22-feature regression: register all 22 bar features (11 from Phase 3 + 11 from Phase 4), process a synthetic multi-bar sequence, compare entire feature vector against `compute_bar_features()`
