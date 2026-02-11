# src/engine — Limit Order Book Engine

Core matching engine. `Book` maintains a price-time priority order book.

## Files

| File | Role |
|---|---|
| `book.cpp` | `Book::apply(Message)` — processes Add/Cancel/Modify/Trade. Maintains `top_bids(k)`/`top_asks(k)` with NaN/0 padding. |

## Key interfaces (in `include/lob/`)

- `book.h` — `Book` class: `apply()`, `best_bid()`, `best_ask()`, `mid_price()`, `spread()`, `top_bids(k)`, `top_asks(k)`, `best_bid_qty()`, `best_ask_qty()`, `total_bid_depth(n)`, `total_ask_depth(n)`, `weighted_mid()`, `vamp(n)`
- `message.h` — `Message` struct: `Side`, `Action` enums, order fields, `is_valid()`
