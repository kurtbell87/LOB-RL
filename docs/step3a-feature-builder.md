# Step 3a: FeatureBuilder (44-Float Observation Space)

## Problem

`LOBEnv::make_obs()` returns 4 raw floats `[best_bid, best_ask, spread, position]`. The PRD requires 44 normalized floats using 10 levels of book depth.

## What to Build

A `FeatureBuilder` class in C++ that takes a `Book` reference and agent state, and produces a 44-float observation vector. Then integrate it into `LOBEnv` so `make_obs()` uses it.

## Observation Layout (PRD Section 5)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-9 | Bid prices (relative to mid) | `(bid_price - mid) / mid` — negative values, closer to 0 = closer to mid |
| 10-19 | Bid sizes | `size / max_size` where max_size is the max across all 20 levels in this snapshot |
| 20-29 | Ask prices (relative to mid) | `(ask_price - mid) / mid` — positive values, closer to 0 = closer to mid |
| 30-39 | Ask sizes | `size / max_size` (same denominator as bid sizes) |
| 40 | Spread | `spread / mid` |
| 41 | Imbalance | `(bid_qty_top - ask_qty_top) / (bid_qty_top + ask_qty_top)` — range [-1, 1] |
| 42 | Time remaining | `1.0 - session_progress` — range [0, 1], 1 at open, 0 at close. If no session, always 0.5. |
| 43 | Position | Raw value: -1, 0, or 1 |

## Interface

### `FeatureBuilder` class (new file: `include/lob/feature_builder.h`, `src/env/feature_builder.cpp`)

```cpp
class FeatureBuilder {
public:
    static constexpr int OBS_SIZE = 44;
    static constexpr int DEPTH = 10;

    // Build observation vector from current book state and agent state
    std::vector<float> build(const Book& book, float position, float time_remaining) const;
};
```

### LOBEnv integration

- `LOBEnv` gets a `FeatureBuilder` member
- `make_obs()` calls `feature_builder_.build(book_, position_, time_remaining)` where `time_remaining` comes from `session_filter_->session_progress()` if available, else 0.5
- Old 4-float observation is replaced by 44-float observation

## Edge Cases

- Empty book (no bids/asks): mid is NaN → use 0.0 for all price-relative features, 0.0 for sizes, 0.0 for spread, 0.0 for imbalance
- Fewer than 10 levels: Book already pads with NaN/0 via `top_bids(10)`/`top_asks(10)` — NaN prices become 0.0 in relative form, 0 sizes stay 0
- All sizes zero (shouldn't happen, but defensive): max_size = 0 → sizes all 0.0
- Mid price is 0 (shouldn't happen): avoid division by zero, return 0.0

## Requirements

1. `FeatureBuilder::build()` returns exactly 44 floats
2. Bid prices are relative to mid, normalized by mid
3. Ask prices are relative to mid, normalized by mid
4. Sizes normalized by max size across all 20 levels
5. Spread normalized by mid
6. Imbalance in [-1, 1]
7. Time remaining in [0, 1]
8. Position is raw {-1, 0, 1}
9. `LOBEnv::make_obs()` returns 44 floats (replaces old 4-float obs)
10. All values are finite (no NaN/Inf in output)

## Acceptance Criteria

- Test: `FeatureBuilder::build()` returns vector of size 44
- Test: With known book state, verify each index range has correct values
- Test: Empty book produces all-zero features (except position and time_remaining)
- Test: Imbalance calculation correct for various bid/ask qty ratios
- Test: Size normalization uses max across all 20 levels
- Test: `LOBEnv.step()` / `LOBEnv.reset()` return 44-element obs from Python
- Test: All existing tests updated for 44-float obs (some will need adjustment)
