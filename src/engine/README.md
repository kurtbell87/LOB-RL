# Engine Module

## Purpose
Order book reconstruction from MBO messages.

## Interface
- `Book` — defined in `include/lob/book.h`
- `apply(msg)` — process one MBO message
- `bid(depth)`, `ask(depth)` — query book levels
- `mid_price()`, `spread()`, `imbalance()` — derived metrics

## Dependencies
- Depends on: `message.h` only
- Depended on by: `LOBEnv`, `FeatureBuilder`

## Status
Not implemented.
