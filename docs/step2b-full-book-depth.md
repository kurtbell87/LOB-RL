# Step 2b: Full Book Depth (10 Levels)

## Goal

Extend the `Book` class to expose top-K price levels (default K=10) for both bid and ask sides, enabling the full 44-float observation vector defined in the PRD.

## Changes to Book

**File:** `include/lob/book.h`, `src/engine/book.cpp`

Add the following methods:

```cpp
class Book {
public:
    // Existing methods unchanged...

    // New: top-K level queries
    struct PriceLevel {
        double price;
        uint32_t qty;
    };

    // Returns top K bid levels (best first, descending price)
    // If fewer than K levels exist, remaining entries have price=NaN, qty=0
    std::vector<PriceLevel> top_bids(int k = 10) const;

    // Returns top K ask levels (best first, ascending price)
    // If fewer than K levels exist, remaining entries have price=NaN, qty=0
    std::vector<PriceLevel> top_asks(int k = 10) const;

    // Total qty at best bid/ask
    uint32_t best_bid_qty() const;  // 0 if empty
    uint32_t best_ask_qty() const;  // 0 if empty
};
```

## Behavior

### `top_bids(k)`
- Returns exactly `k` entries
- Entries are sorted best-first (highest price first)
- If the book has fewer than `k` bid levels, pad with `{NaN, 0}`
- If `k <= 0`, returns empty vector

### `top_asks(k)`
- Returns exactly `k` entries
- Entries are sorted best-first (lowest price first)
- If the book has fewer than `k` ask levels, pad with `{NaN, 0}`
- If `k <= 0`, returns empty vector

### `best_bid_qty()` / `best_ask_qty()`
- Returns the aggregate quantity at the best bid/ask price level
- Returns 0 if that side is empty

## Edge Cases

- Empty book: all levels are `{NaN, 0}`
- Book with 3 bid levels, requesting 10: first 3 are real, remaining 7 are `{NaN, 0}`
- Book with 15 levels, requesting 10: only top 10 returned
- After cancel removes a level: depth decreases, levels shift up
- k=0: returns empty vector
- k=1: equivalent to best_bid/best_ask info in a struct

## Acceptance Criteria

1. **top_bids tests:**
   - Empty book returns K entries all NaN/0
   - Add 3 bid levels → first 3 populated, rest NaN/0
   - Add 12 bid levels → only top 10 returned
   - Levels are in descending price order
   - Cancel best bid → next level becomes top
   - Quantities are correct per level (including aggregated orders at same price)

2. **top_asks tests:**
   - Symmetric to bid tests but ascending price order

3. **best_bid_qty / best_ask_qty tests:**
   - Empty book → 0
   - Single order → that order's qty
   - Multiple orders at best price → sum of quantities

4. **Integration:**
   - Process SyntheticSource messages, verify top-10 snapshot at various points
   - Process BinaryFileSource test fixture, verify depth consistency
