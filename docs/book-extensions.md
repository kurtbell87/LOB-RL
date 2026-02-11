# Book Extensions: Depth Aggregation and Weighted Prices

## Context

The C++ `Book` class (`include/lob/book.h`, `src/engine/book.cpp`) already has:
- `top_bids(k)` / `top_asks(k)` returning `vector<PriceLevel>` with `{price, qty}`
- `best_bid_qty()` / `best_ask_qty()` returning `uint32_t`
- `best_bid()` / `best_ask()` / `mid_price()` / `spread()` returning `double`
- `bid_depth()` / `ask_depth()` returning `size_t` (number of levels)

The Python `OrderBook` class (`python/lob_rl/barrier/lob_reconstructor.py`) has four additional methods needed by the barrier feature pipeline:
- `total_bid_depth(n)` / `total_ask_depth(n)` — cumulative quantity across top n levels
- `weighted_mid_price()` — imbalance-weighted mid price
- `vamp(n)` — volume-adjusted mid price using top n levels each side

The C++ `Book` class needs these same 4 methods to support the C++ barrier precompute pipeline.

## Requirements

### 1. `total_bid_depth(int n) -> uint32_t`

Sum the `qty` field across the top `n` bid levels (best first, descending price).

- **n <= 0:** return 0
- **Empty book (no bids):** return 0
- **Fewer than n bid levels:** sum all available levels
- **n >= number of levels:** sum all levels (no error)
- Uses saturating addition if overflow would occur (consistent with existing Book pattern)

### 2. `total_ask_depth(int n) -> uint32_t`

Sum the `qty` field across the top `n` ask levels (best first, ascending price).

- Same edge cases as `total_bid_depth`

### 3. `weighted_mid() -> double`

Imbalance-weighted mid price: `(bid_qty * ask_price + ask_qty * bid_price) / (bid_qty + ask_qty)`.

Where `bid_qty = best_bid_qty()`, `ask_qty = best_ask_qty()`, `bid_price = best_bid()`, `ask_price = best_ask()`.

- **Empty book (no bids or no asks):** return `NaN`
- **Both BBO quantities are 0:** return `NaN`
- **One side qty = 0:** return `NaN` (cannot weight)
- Note: C++ `Book` returns NaN for empty (unlike Python which returns 0.0). This is the C++ convention — feature code handles NaN with `std::isfinite()` checks.

### 4. `vamp(int n) -> double`

Volume-adjusted mid price using the top `n` levels on each side:
```
sum(price * qty for all top-n bid levels) + sum(price * qty for all top-n ask levels)
─────────────────────────────────────────────────────────────────────────────────────
      sum(qty for all top-n bid levels) + sum(qty for all top-n ask levels)
```

- **Empty book (no bids or no asks):** return `NaN`
- **n <= 0:** return `NaN`
- **Total qty across both sides = 0:** return `NaN`
- **Fewer than n levels on one side:** use all available levels on that side
- Uses `double` arithmetic to avoid overflow (prices * qty can exceed uint32_t)

## Files to Change

| File | Change |
|------|--------|
| `include/lob/book.h` | Add 4 method declarations |
| `src/engine/book.cpp` | Implement 4 methods |
| `tests/test_book_depth.cpp` | Add ~10 new test cases |

## Test Plan

All tests go in `tests/test_book_depth.cpp` under `BookDepth` test suite (existing file).

### total_bid_depth / total_ask_depth tests:
1. Empty book → returns 0
2. Single level → returns that level's qty
3. Multiple levels → returns sum across top n
4. Fewer levels than n → returns sum of all
5. n = 0 → returns 0
6. Symmetric book (make_symmetric_book) → known values

### weighted_mid tests:
7. Empty book → returns NaN
8. Only bids, no asks → returns NaN
9. Symmetric BBO → equals simple mid_price
10. Asymmetric BBO → weighted toward larger side

### vamp tests:
11. Empty book → returns NaN
12. n = 0 → returns NaN
13. Single level each side → equals weighted_mid
14. Multiple levels → known hand-computed value
15. Symmetric book (make_symmetric_book) → known value

## Acceptance Criteria

- All existing 418 C++ tests still pass
- New tests pass
- Methods match Python `OrderBook` semantics (with NaN instead of 0.0 for empty)
