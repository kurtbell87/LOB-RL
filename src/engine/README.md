# src/engine — Limit Order Book Engine

Core matching engine. `Book` maintains a price-time priority order book.

## Files

| File | Role |
|---|---|
| `book.cpp` | `Book::apply(Message)` — processes Add/Cancel/Modify/Trade. Maintains `top_bids(k)`/`top_asks(k)` with NaN/0 padding. |

## Key interfaces (in `include/lob/`)

- `book.h` — `Book` class: `apply()`, `best_bid()`, `best_ask()`, `mid()`, `spread()`, `top_bids(k)`, `top_asks(k)`
- `message.h` — `Message` struct: `Side`, `Action` enums, order fields, `is_valid()`
