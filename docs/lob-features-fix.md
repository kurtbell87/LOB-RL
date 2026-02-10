# LOB Reconstructor + Fix Dead Features + N_FEATURES Constant

## Summary

Fix 4 dead book features (cols 1, 2, 10, 11) in `compute_bar_features()` by:
1. Building an `OrderBook` class that replays MBO messages to reconstruct LOB state.
2. Adding `extract_all_mbo()` to `bar_pipeline.py` to extract all MBO actions (not just trades).
3. Wiring MBO data through `precompute_barrier_cache.py` into `build_feature_matrix()`.
4. Introducing `N_FEATURES = 13` constant in `__init__.py` and replacing all hardcoded `13`.

## New File: `python/lob_rl/barrier/lob_reconstructor.py`

### `OrderBook` Class

Pure Python limit order book reconstruction from MBO (Market By Order) messages.

```python
class OrderBook:
    """Stateful limit order book reconstructed from MBO messages.

    Tracks individual orders by order_id. Maintains sorted price levels
    on bid and ask sides.
    """

    def __init__(self):
        """Initialize empty order book."""
        # Internal state:
        # _bids: dict[price -> total_qty]  (buy side)
        # _asks: dict[price -> total_qty]  (sell side)
        # _orders: dict[order_id -> (side, price, size)]

    def apply(self, action: str, side: str, price: float, size: int, order_id: int) -> None:
        """Process a single MBO message.

        Parameters
        ----------
        action : str
            'A' (Add), 'C' (Cancel), 'M' (Modify), 'T' (Trade),
            'R' (Resting/Clear), 'F' (Fill).
        side : str
            'B' (Bid/Buy) or 'A' (Ask/Sell).
        price : float
            Order price.
        size : int
            Order size (qty).
        order_id : int
            Unique order identifier.

        Behavior by action:
        - 'A': Insert new order. Add size to the price level. Track order_id.
        - 'C': Cancel order by order_id. Remove its size from the price level.
                If order_id unknown, no-op (defensive).
        - 'M': Modify order by order_id. Update price/size. Adjust level totals.
                If order_id unknown, treat as Add.
        - 'T': Trade execution. Decrement order by size. If order_id tracked,
                reduce its remaining qty. Remove level if qty reaches 0.
                If order_id unknown, directly decrement the price level qty.
        - 'R': Clear/Reset. Remove order if tracked, else no-op.
        - 'F': Fill. Same behavior as 'T'.
        """

    def best_bid(self) -> float:
        """Return best bid price, or 0.0 if no bids."""

    def best_ask(self) -> float:
        """Return best ask price, or 0.0 if no asks."""

    def spread(self) -> float:
        """Return spread in price units (best_ask - best_bid).
        Returns 0.0 if either side is empty."""

    def spread_ticks(self) -> float:
        """Return spread in ticks (spread / TICK_SIZE)."""

    def best_bid_qty(self) -> int:
        """Return total quantity at best bid, or 0 if no bids."""

    def best_ask_qty(self) -> int:
        """Return total quantity at best ask, or 0 if no asks."""

    def bid_depth(self, n: int = 10) -> list[tuple[float, int]]:
        """Return top n bid levels as [(price, qty), ...], sorted descending by price."""

    def ask_depth(self, n: int = 10) -> list[tuple[float, int]]:
        """Return top n ask levels as [(price, qty), ...], sorted ascending by price."""

    def total_bid_depth(self, n: int = 10) -> int:
        """Return cumulative quantity across top n bid levels."""

    def total_ask_depth(self, n: int = 10) -> int:
        """Return cumulative quantity across top n ask levels."""

    def is_empty(self) -> bool:
        """Return True if both sides are empty."""

    def mid_price(self) -> float:
        """Return (best_bid + best_ask) / 2, or 0.0 if either side empty."""

    def weighted_mid_price(self) -> float:
        """Return imbalance-weighted mid: (bid_qty * ask + ask_qty * bid) / (bid_qty + ask_qty).
        Returns mid_price() if either BBO qty is 0."""
```

### Design Decisions

- Uses `dict[float, int]` for price levels (price -> total_qty). Levels with qty <= 0 are deleted.
- Uses `dict[int, tuple[str, float, int]]` for order tracking (order_id -> (side, price, remaining_size)).
- Sorting for `bid_depth()` / `ask_depth()` is done on-demand (not maintained). These are called once per bar (not per message), so O(n log n) sort per bar is fine.
- Defensive: unknown `order_id` on Cancel/Trade is silently ignored (real MBO data can have gaps).
- `TICK_SIZE` imported from `lob_rl.barrier`.

## Changes to `bar_pipeline.py`

### New Function: `extract_all_mbo()`

```python
def extract_all_mbo(filepath: str, instrument_id: int = None) -> pd.DataFrame:
    """Read a .dbn.zst file and extract ALL MBO messages (not just trades).

    Parameters
    ----------
    filepath : str
        Path to .dbn.zst file.
    instrument_id : int, optional
        Filter to specific contract.

    Returns
    -------
    pd.DataFrame
        Columns: action (str), side (str), price (float64), size (int32),
                 order_id (int64), ts_event (int64).
        Sorted by ts_event ascending.
        Actions: 'A' (Add), 'C' (Cancel), 'M' (Modify), 'T' (Trade),
                 'R' (Clear), 'F' (Fill).
    """
```

Uses `databento.DBNStore.from_file().to_df()` — same pattern as existing `extract_trades_from_mbo()`.

## Changes to `feature_pipeline.py`

### `compute_bar_features(bars, mbo_data=None)`

When `mbo_data` is a `pd.DataFrame` (output of `extract_all_mbo()`):

1. Create an `OrderBook` instance.
2. Iterate through all MBO messages, grouped by bar boundaries using `bar.t_start` / `bar.t_end`.
3. For each bar, snapshot the book state at bar close to compute:

**Col 1 — BBO imbalance:**
```
bid_qty = book.best_bid_qty()
ask_qty = book.best_ask_qty()
bbo_imbalance = bid_qty / (bid_qty + ask_qty)  # range [0, 1]
# If both are 0: default 0.5
```

**Col 2 — Depth imbalance:**
```
total_bid = book.total_bid_depth(5)
total_ask = book.total_ask_depth(5)
depth_imbalance = total_bid / (total_bid + total_ask)  # range [0, 1]
# If both are 0: default 0.5
```

**Col 10 — Cancel rate asymmetry:**
```
Count cancel actions ('C') within the bar, split by side.
cancel_asym = (bid_cancels - ask_cancels) / (bid_cancels + ask_cancels + 1e-10)
# range [-1, +1]
```

**Col 11 — Mean spread:**
```
Accumulate spread_ticks at each MBO event within the bar.
mean_spread = mean(all spread samples) or 1.0 if no samples.
# range > 0
```

When `mbo_data is None`: keep existing neutral defaults (0.5, 0.5, 0.0, 1.0). This preserves backward compatibility — existing tests pass unchanged.

### Implementation approach

Refactor `compute_bar_features()` to:
1. Compute all non-book features in the existing per-bar loop (cols 0, 3-9, 12).
2. If `mbo_data is not None`, call a private helper `_compute_book_features(bars, mbo_data)` that returns an `(n_bars, 4)` array for cols 1, 2, 10, 11.
3. If `mbo_data is None`, fill cols 1, 2, 10, 11 with defaults.

The helper `_compute_book_features()`:
- Creates `OrderBook()`.
- Pre-groups MBO rows by bar via `np.searchsorted` on `bar.t_end` timestamps.
- For each bar: replay messages, count cancels, accumulate spread samples, snapshot book at bar close.
- Returns `np.ndarray` shape `(n_bars, 4)`.

## Changes to `__init__.py`

Add:
```python
N_FEATURES = 13  # Number of bar-level feature columns
```

## Changes to `barrier_env.py`

Replace:
```python
self._h = self._feature_dim // 13
```
with:
```python
from lob_rl.barrier import N_FEATURES
self._h = self._feature_dim // N_FEATURES
```

## Changes to `supervised_diagnostic.py`

Replace all hardcoded `13` with `N_FEATURES`:
- Line 27 docstring: "compute 13 bar-level features" -> use N_FEATURES
- Line 41: `np.ndarray of shape (n_usable, 13 * h)` -> `N_FEATURES * h`
- Line 48: `np.empty((0, 13 * h), ...)` -> `np.empty((0, N_FEATURES * h), ...)`

## Changes to `conftest.py`

Replace:
```python
DEFAULT_FEATURE_DIM = 13 * DEFAULT_H  # 130
```
with:
```python
from lob_rl.barrier import N_FEATURES
DEFAULT_FEATURE_DIM = N_FEATURES * DEFAULT_H
```

## Changes to `precompute_barrier_cache.py`

### `process_session()` updates

1. Import `extract_all_mbo` from `bar_pipeline`.
2. After calling `build_session_bars()`, call `extract_all_mbo(filepath, instrument_id)` to get the full MBO DataFrame.
3. Pass `mbo_data=mbo_df` to `build_feature_matrix(bars, h=lookback, mbo_data=mbo_df)`.
4. Store `n_features` metadata in the `.npz` file:
   ```python
   result["n_features"] = np.array(N_FEATURES, dtype=np.int32)
   ```

### `load_session_from_cache()` updates

Add version check: if `n_features` is present in the `.npz` and differs from `N_FEATURES`, raise `ValueError` with a clear message telling the user to re-precompute.

## Docstring/Comment Updates in `regime_switch.py`

Update shape references in docstrings:
- `ks_test_features()`: `shape (N, 13)` -> `shape (N, N_FEATURES)`
- `measure_normalization_adaptation()`: `shape (N, 13)` -> `shape (N, N_FEATURES)`

(No logic changes needed — these functions already use `features.shape[1]` dynamically.)

## Edge Cases

1. **Empty MBO data**: If `mbo_data` DataFrame is empty, fall back to neutral defaults.
2. **MBO timestamps outside bar range**: Messages before first bar or after last bar are applied to the book but don't contribute to bar-level stats.
3. **Order book empty at bar close**: If book has no bids/asks at snapshot time, use neutral defaults.
4. **Duplicate order_ids**: Later messages for the same order_id overwrite earlier state.
5. **Negative quantities**: Clamp to 0. Remove level if qty <= 0.
6. **Price=0 orders**: Skip (defensive against bad data).

## Acceptance Criteria

1. **OrderBook unit tests**: Add, Cancel, Modify, Trade, Fill, Clear all work correctly. BBO, depth, spread computed correctly. Edge cases (empty book, unknown order_id, zero-price) handled.
2. **Book feature integration tests**: Given synthetic MBO data with known bid/ask structure, `compute_bar_features()` produces correct BBO imbalance, depth imbalance, cancel asymmetry, and mean spread.
3. **Backward compatibility**: `compute_bar_features(bars)` (no mbo_data) produces identical output to current implementation. All 550 existing barrier tests pass unchanged.
4. **N_FEATURES constant**: All files reference `N_FEATURES` instead of literal `13`. Changing `N_FEATURES` in `__init__.py` propagates correctly.
5. **Precompute wiring**: `process_session()` extracts MBO data and passes it through. The `.npz` includes `n_features` metadata.
6. **Cache version check**: `load_session_from_cache()` raises `ValueError` when loaded cache has wrong `n_features`.

## Test Strategy

### OrderBook tests (~25)
- `test_add_order_creates_level`: Add bid order -> best_bid, best_bid_qty correct.
- `test_add_multiple_orders_same_level`: Two adds at same price -> qty sums.
- `test_add_orders_different_levels`: Multiple levels -> depth sorted correctly.
- `test_cancel_known_order`: Cancel reduces qty, removes level when 0.
- `test_cancel_unknown_order`: No crash, no state change.
- `test_modify_existing_order`: Changes price and/or size correctly.
- `test_modify_unknown_order_acts_as_add`: Defensive behavior.
- `test_trade_decrements_qty`: Trade reduces order, removes when fully filled.
- `test_trade_unknown_order`: Decrements price level directly.
- `test_fill_same_as_trade`: 'F' action behaves like 'T'.
- `test_clear_removes_order`: 'R' action removes tracked order.
- `test_spread`: Correct spread from bid/ask prices.
- `test_spread_empty_book`: Returns 0.0.
- `test_spread_ticks`: Spread in tick units.
- `test_bid_depth_sorted`: Descending by price.
- `test_ask_depth_sorted`: Ascending by price.
- `test_total_bid_depth`: Cumulative qty across levels.
- `test_total_ask_depth`: Cumulative qty across levels.
- `test_depth_n_limits`: Only returns top N levels.
- `test_mid_price`: (best_bid + best_ask) / 2.
- `test_weighted_mid_price`: Imbalance-weighted mid.
- `test_is_empty`: True when both sides empty.
- `test_price_zero_ignored`: Zero-price orders silently skipped.
- `test_negative_qty_clamped`: Over-cancel doesn't produce negative qty.
- `test_replay_sequence`: Multi-message sequence produces correct final state.

### Book Feature Integration tests (~15)
- `test_bbo_imbalance_with_mbo`: Synthetic MBO with known BBO -> correct imbalance.
- `test_depth_imbalance_with_mbo`: Synthetic MBO with known depth -> correct ratio.
- `test_cancel_asymmetry_with_mbo`: Known cancel pattern -> correct asymmetry.
- `test_mean_spread_with_mbo`: Known spreads -> correct mean.
- `test_features_without_mbo_unchanged`: No mbo_data -> same neutral defaults as before.
- `test_features_shape_unchanged`: Output shape is still (N, 13).
- `test_empty_mbo_falls_back`: Empty DataFrame -> neutral defaults.
- `test_mbo_timestamps_align_with_bars`: Messages assigned to correct bars.
- `test_book_features_multiple_bars`: Book state persists across bars correctly.
- `test_spread_samples_within_bar`: Mean computed from all events in bar.
- `test_book_empty_at_close_uses_defaults`: If book empties during bar, neutral defaults.
- `test_cancel_count_per_bar`: Cancels counted only within bar boundaries.
- `test_bbo_imbalance_range`: Output in [0, 1].
- `test_cancel_asymmetry_range`: Output in [-1, +1].
- `test_mean_spread_positive`: Output > 0.

### N_FEATURES constant tests (~5)
- `test_n_features_constant_exists`: `from lob_rl.barrier import N_FEATURES` works.
- `test_n_features_value`: `N_FEATURES == 13`.
- `test_feature_pipeline_uses_n_features`: `compute_bar_features()` output shape is (N, N_FEATURES).
- `test_barrier_env_uses_n_features`: BarrierEnv computes `h` using N_FEATURES.
- `test_conftest_default_dim`: `DEFAULT_FEATURE_DIM == N_FEATURES * DEFAULT_H`.

### Precompute wiring tests (~5)
- `test_process_session_stores_n_features`: `.npz` contains `n_features` key.
- `test_load_session_version_check`: Mismatched `n_features` raises `ValueError`.
- `test_load_session_backward_compat`: Missing `n_features` key doesn't crash (old caches still loadable with warning).
- `test_extract_all_mbo_columns`: Output DataFrame has correct columns.
- `test_extract_all_mbo_includes_all_actions`: Not filtered to trades only.

Total estimated: ~50 tests.
