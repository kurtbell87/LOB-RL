# BUG1: Book::Modify Must Handle Price Changes

## Problem

`Book::apply()` for `Action::Modify` ignores price changes. When a Modify message arrives with a different price than the order's current price, the code updates quantity at the *old* price level and never moves the order to the new price. In Databento MBO data, Modify can change both price and quantity.

**Current behavior (lines 36-52 of book.cpp):**
- Looks up order by `order_id`
- Replaces old qty with new qty at `entry.price` (the old price)
- Updates `entry.qty` but never `entry.price`
- `msg.price` is completely ignored when the order already exists

**Correct behavior:**
- If `msg.price != entry.price`, remove the old quantity from the old price level, add new quantity to the new price level, and update `entry.price`
- If `msg.price == entry.price`, update quantity in place (current behavior is fine for this case)
- Clean up empty price levels after removal

## Interface

No interface changes. `Book::apply(const Message&)` signature stays the same.

## Requirements

1. **Modify with same price**: Quantity at the price level changes from old to new. Order entry updated.
2. **Modify with different price**: Old quantity removed from old price level. New quantity added at new price level. Order entry updated with new price and quantity. Old price level removed if it reaches zero.
3. **Modify unknown order**: Treated as Add (existing behavior, keep it).
4. **Modify that changes price across levels with other orders**: Only the modified order's quantity moves. Other orders at the old price level are unaffected.

## Acceptance Criteria

- Test: Add order at price 100, modify to price 101 with new qty. Verify old level is gone (or reduced), new level has correct qty.
- Test: Add two orders at price 100, modify one to price 101. Verify price 100 still has the other order's qty.
- Test: Add order, modify with same price but different qty. Verify level qty updated correctly.
- Test: Modify with price change updates `best_bid()` / `best_ask()` correctly.
- Test: All existing book tests still pass.
