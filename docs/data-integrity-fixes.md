# Data Integrity Fixes Spec

## Overview

Fix critical undefined behavior and edge cases identified during codebase audit. These issues could cause crashes, memory corruption, or silent data errors during training on real data.

## Issues to Fix

### CRITICAL 1: `synthetic_source.cpp:75` — UB in `pick_order()` with empty `live_orders`

**Location:** `src/data/synthetic_source.cpp` lines 74-78

**Problem:** The lambda `pick_order` creates `std::uniform_int_distribution<size_t>(0, live_orders.size() - 1)`. When `live_orders.size() == 0`, this computes `0 - 1` on an unsigned type, wrapping to `SIZE_MAX`. The distribution constructor with `(0, SIZE_MAX)` is undefined behavior per the C++ standard.

**Current mitigation:** The caller checks `live_orders.empty()` on line 85 before calling `pick_order()`. However:
- The lambda is a latent bug waiting to happen if someone refactors
- Defensive programming requires the function itself to be safe

**Fix:** Add an assertion or guard inside `pick_order()`. If `live_orders` is empty, either:
- Throw an exception (assert failure in debug, runtime_error in release), OR
- Return a sentinel indicating "no order available"

**Acceptance criteria:**
- `pick_order()` must not exhibit UB when `live_orders` is empty
- Tests must verify behavior when called with empty `live_orders`

---

### CRITICAL 2: `binary_file_source.cpp:41` — signed-to-unsigned cast of `gcount()`

**Location:** `src/data/binary_file_source.cpp` lines 40-42

**Problem:**
```cpp
auto bytes_read = file.gcount();  // std::streamsize (signed, typically int64_t)
auto full_records = static_cast<uint32_t>(bytes_read / sizeof(FlatRecord));
```

If `gcount()` returns a negative value (which shouldn't happen in normal use but could in error conditions), casting to `uint32_t` produces a very large positive number, causing `records_.resize()` to allocate huge memory or corrupt data.

**Fix:** Add a guard to ensure `bytes_read >= 0` before the cast. If negative, treat as 0 records read.

**Acceptance criteria:**
- Negative `gcount()` values must be handled safely (treated as 0 bytes read)
- Tests must verify truncated file handling still works correctly

---

### HIGH: `message.h` — No validation for invalid field values

**Location:** `include/lob/message.h`

**Problem:** The `Message` struct accepts any values without validation:
- Negative prices (invalid for financial data)
- NaN prices (will propagate through calculations)
- Zero quantities (orders with qty=0 are meaningless)
- Zero order_id (typically invalid)

**Impact:** Invalid data could propagate through the Book and produce garbage observations.

**Fix:** Add a static validation method `Message::is_valid()` that returns `false` for:
- `price < 0` or `!std::isfinite(price)`
- `qty == 0`
- `order_id == 0`

The validation method should be available for callers to use. Whether to enforce validation (and where) is a design decision — start with providing the method.

**Acceptance criteria:**
- `Message::is_valid()` method exists and correctly identifies invalid messages
- Tests cover: negative price, NaN price, zero qty, zero order_id

---

## Out of Scope for This Spec

The following MEDIUM issues are deferred:
- `session.h:29` — No validation `rth_open_ns < rth_close_ns`
- `binary_file_source.cpp:62` — `int64_t → double` precision loss
- `book.cpp:48` — `level_qty + msg.qty` could overflow

The feature_builder.cpp:45 division by zero issue appears to already be guarded by the `if (has_mid)` check (where `has_mid` requires `mid > 0.0`). Verify this is true and remove from the audit list if confirmed.

---

## Test Strategy

1. **SyntheticSource tests:**
   - Direct call to `pick_order` equivalent with empty vector should not crash
   - Verify existing generation still produces valid messages

2. **BinaryFileSource tests:**
   - Create a fixture file that triggers truncated read path
   - Verify `gcount()` edge cases are handled

3. **Message validation tests:**
   - Test `is_valid()` returns false for each invalid condition
   - Test `is_valid()` returns true for normal messages
