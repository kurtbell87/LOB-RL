# src/constellation/ — Constellation C++ Engine

Constellation's production-grade C++ trading engine, imported from the standalone Constellation repository and integrated into LOB-RL as the `constellation_core` static library.

## Directory Structure

| Directory | Role |
|---|---|
| `interfaces/` | Abstract interfaces (IOrdersEngine, IMarketView, IFeature, IStrategy, etc.) |
| `modules/orderbook/` | LimitOrderBook, MarketBook, AugmentedPriceMap |
| `modules/orders/` | OrdersEngine — CQRS order lifecycle (Market/Limit/Stop/Bracket) |
| `modules/features/` | 11 feature classes (micro price, rolling vol, order imbalance, etc.) |
| `modules/market_data/` | DbnFileFeed, DataBentoMboFeed |
| `modules/strategy/` | IStrategy impl, SampleBatchStrategy |
| `applications/replay/` | BatchBacktestEngine, BatchAggregator — high-perf MBO replay |
| `applications/orchestrator/` | Orchestrator, SpscRingBuffer — 3-thread pipeline (not yet exposed to Python) |
| `applications/engine/` | TradingEngine application |
| `adapters/` | LOB-RL ↔ Constellation type conversion (message_adapter.h) |
| `test_data/` | Small fixture for Constellation Catch2 tests |

## Key APIs (via pybind11 in lob_rl_core)

### OrdersEngine
```python
engine = core.OrdersEngine()
oid = engine.place_order(spec)       # OrderSpec → uint64 order_id
engine.modify_order(oid, update)     # OrderUpdate
engine.cancel_order(oid)             # bool
info = engine.get_order_info(oid)    # OrderInfo
orders = engine.list_open_orders(instrument_id)  # list[OrderInfo]
```

### BatchBacktestEngine
```python
engine = core.BatchBacktestEngine()
engine.set_aggregator_config(cfg)    # BatchAggregatorConfig
engine.process_files(["path.dbn.zst"])
fills = engine.get_fills()           # list[FillRecord]
engine.reset_stats()
```

### MarketBook
```python
mb = core.MarketBook()
mb.instrument_count()                # int
mb.get_instrument_ids()              # list[int]
mb.best_bid_price(instrument_id)     # float or None
mb.best_ask_price(instrument_id)     # float or None
mb.get_global_add_count()            # int
```

### Feature System
```python
fm = core.create_feature_manager()
feat = core.create_micro_price_feature(instrument_id=1)
fm.register_feature(feat)
val = fm.get_value("micro_price")    # float
```

### Enums
- `OrderType`: Market, Limit, Stop, StopLimit
- `OrderSide`: Buy, Sell
- `OrderStatus`: New, PartiallyFilled, Filled, Canceled, Rejected, Expired, Unknown
- `TimeInForce`: Day, GTC, IOC, FOK
- `BookSide`: Bid, Ask

### Constants
- `FIXED_PRICE_SCALE = 1_000_000_000` (nano-scale price encoding)

## Build

`constellation_core` is built as a static library in CMakeLists.txt. It links against `databento::databento` (PUBLIC). `lob_core` links against `constellation_core`, making all Constellation headers available transitively.

`constellation_tests` is a separate Catch2 test executable (65/67 tests pass; 2 pre-existing partial fill failures).

## Cross-File Dependencies

- **Depends on:** `databento::databento` (FetchContent)
- **Depended on by:** `lob_core` (Book class wraps LimitOrderBook), `lob_rl_core` (pybind11 bindings)
- **Adapter:** `adapters/message_adapter.h` bridges LOB-RL `Message` ↔ `databento::MboMsg`
