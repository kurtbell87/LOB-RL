# T1: Bar Construction Pipeline

## What to Build

A Python module `python/lob_rl/barrier/bar_pipeline.py` that constructs fixed-count trade bars from Databento MBO data. This is the data foundation for the entire barrier-hit agent pipeline.

## Context

The existing `bar_aggregation.py` aggregates ticks at runtime for RL stepping. The new pipeline is different: it's an **offline batch processor** that reads raw `.dbn.zst` files, extracts matched trades, and produces bar datasets with full trade sequence retention for downstream label construction.

## Interface

### `TradeBar` (dataclass)

```python
@dataclass
class TradeBar:
    bar_index: int           # Sequential bar number within session
    open: float              # Price of first trade (O_k)
    high: float              # Max trade price (H_k)
    low: float               # Min trade price (L_k)
    close: float             # Price of last trade (C_k)
    volume: int              # Total contracts traded (V_k)
    vwap: float              # Volume-weighted average price
    t_start: int             # Timestamp of first trade (nanoseconds since epoch)
    t_end: int               # Timestamp of last trade (nanoseconds since epoch)
    session_date: str        # YYYY-MM-DD of the RTH session
    trade_prices: np.ndarray # Ordered trade prices within bar (for tiebreaking)
    trade_sizes: np.ndarray  # Corresponding trade sizes
```

### `build_bars_from_trades(trades, n, session_date)` -> `list[TradeBar]`

Takes a sequence of matched trades (price, size, timestamp) within a single RTH session. Groups them into bars of exactly `n` trades. Discards the last incomplete bar.

**Parameters:**
- `trades`: Sequence of (price, size, timestamp) tuples or structured array
- `n`: Number of trades per bar (default 500)
- `session_date`: Date string for the session

**Returns:** List of `TradeBar` objects.

### `extract_trades_from_mbo(filepath, instrument_id=None)` -> structured array

Reads a `.dbn.zst` file and extracts matched trades. Uses Databento MBO message type. Filters to the specified instrument ID if provided.

**Parameters:**
- `filepath`: Path to `.dbn.zst` file
- `instrument_id`: Optional filter for specific contract

**Returns:** Structured numpy array with fields: price (float64), size (int32), timestamp (int64, nanoseconds), side (int8, 1=buy/-1=sell)

### `filter_rth_trades(trades)` -> structured array

Filters trades to RTH session hours only: 8:30 AM - 3:00 PM CT (13:30 - 20:00 UTC during CDT, 14:30 - 21:00 UTC during CST). Also excludes any trades during CME Globex maintenance window (4:00 - 5:00 PM CT = 21:00 - 22:00 UTC CDT / 22:00 - 23:00 UTC CST).

### `build_session_bars(filepath, n=500, instrument_id=None)` -> `list[TradeBar]`

End-to-end: reads MBO file, extracts trades, filters to RTH, builds bars.

### `build_dataset(filepaths, n=500, roll_calendar=None, output_path=None)` -> `pd.DataFrame`

Batch process multiple days. Returns a DataFrame with columns matching `TradeBar` fields (trade_prices/trade_sizes stored as separate sidecar).

**Output artifacts:**
- `bars.parquet` — Bar OHLCV data (one row per bar)
- `trade_sequences/` — Directory of `.npy` files per session, each containing the trade price/size arrays indexed by bar number

## Session Boundary Rules

1. A bar must NOT straddle RTH session boundaries (8:30 AM open / 3:00 PM close CT).
2. A bar must NOT straddle the Globex maintenance window (4:00-5:00 PM CT).
3. If bar `k` is incomplete at 3:00 PM CT, **discard it**.
4. A new bar begins fresh at 8:30 AM CT the next session.
5. Bar indices reset to 0 at the start of each session.

## Trade Matching

Databento MBO data format (from `databento.DBNStore.from_file().to_df()`):

| Column | Type | Description |
|--------|------|-------------|
| `ts_recv` (index) | datetime64[ns, UTC] | Receive timestamp |
| `ts_event` | datetime64[ns] | Exchange timestamp (use this for bar timing) |
| `action` | str | 'T'=Trade, 'A'=Add, 'C'=Cancel, 'M'=Modify, 'R'=Clear, 'F'=Fill |
| `side` | str | 'B'=aggressive buy, 'A'=aggressive sell, 'N'=none |
| `price` | float64 | Trade price in index points (e.g., 4775.25) |
| `size` | int | Number of contracts |
| `instrument_id` | int | Contract identifier (filter via roll calendar) |

Filter to `action == 'T'` for matched trades. The `side` column indicates the aggressor side (needed for T3 trade flow imbalance, not T1).

Typical volume: ~200k RTH trades/day for front-month /MES, producing ~400 bars at N=500.

## Timezone Handling

- All Databento timestamps are UTC nanoseconds.
- RTH hours: 8:30 AM - 3:00 PM Central Time.
- Central Time = UTC-6 (CST, Nov-Mar) or UTC-5 (CDT, Mar-Nov).
- The training data covers 2022-01-01 to 2023-12-31, which spans both CST and CDT periods.
- Use `pytz` or `zoneinfo` with `America/Chicago` for correct DST handling.

## Default Parameters

- `N = 500` trades per bar (configurable for sweep: {200, 500, 1000, 2000})

## Tests to Write

1. Bar with exactly N trades produces correct OHLCV:
   - Create a known sequence of 500 trades with hand-computed O, H, L, C, V, VWAP.
   - Verify all fields match.

2. VWAP is bounded by [L_k, H_k] for all bars:
   - VWAP = sum(price_i * size_i) / sum(size_i) must be within [low, high].

3. `t_start_k < t_end_k` for all bars:
   - First trade timestamp < last trade timestamp.

4. `t_end_{k} <= t_start_{k+1}` for consecutive bars within a session:
   - No temporal overlap between bars.

5. Bar does not straddle RTH session boundaries:
   - Create trades spanning 2:59 PM to 3:01 PM. Verify bar is discarded.
   - Create trades spanning 8:29 AM to 8:31 AM. Verify only 8:30+ trades used.

6. Bar does not straddle Globex maintenance window:
   - This only applies to Globex session data (we're RTH-only, so this is a no-op check for RTH, but test that the filter excludes maintenance window trades if they somehow appear).

7. Incomplete bar at session end is discarded:
   - Create 750 trades. With N=500, should produce 1 bar (not 2).

8. Bars across full dataset have correct trade counts:
   - `sum(V_k) = total matched trades - remainder` (where remainder < N).

9. Trade sequence retention:
   - `bar.trade_prices` has exactly N elements.
   - `bar.trade_prices[0] == bar.open` and `bar.trade_prices[-1] == bar.close`.
   - `max(bar.trade_prices) == bar.high` and `min(bar.trade_prices) == bar.low`.

10. Volume accounting:
    - `bar.volume == sum(bar.trade_sizes)`.

## Acceptance Criteria

- All tests pass.
- Module can process a single `.dbn.zst` file and produce valid bars.
- Output parquet and trade sequence files can be loaded back and validated.
- Spot-check 10 random bars against known charting platform (manual, post-implementation).

## Edge Cases

- Empty session (market holiday, no trades in RTH window) → empty bar list.
- Fewer than N trades in entire session → no bars produced.
- DST transition days (March and November) → correct RTH boundary computation.
- Session with exactly k*N trades → k bars, no remainder.

## Dependencies

- `databento` Python library for reading `.dbn.zst` files
- `numpy`, `pandas`, `pyarrow` (for parquet)
- `zoneinfo` (stdlib) for timezone handling

## File Location

- Module: `python/lob_rl/barrier/bar_pipeline.py`
- Tests: `python/tests/barrier/test_bar_pipeline.py`
