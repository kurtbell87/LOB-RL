# Phase 2 Microstructure Features (Cols 17-21)

## Summary

Expand the barrier feature set from 17 to 22 features by adding 5 new LOB microstructure features. These features capture trade-level dynamics and market impact information that complement the order-book features from Phase 1. Bump `N_FEATURES` from 17 to 22 in `__init__.py`.

## New Features

| Col | Feature | Range | Default (no MBO) | Computation |
|-----|---------|-------|-------------------|-------------|
| 17 | VAMP displacement | signed | 0.0 | Delta of volume-adjusted mid-price within bar, in ticks. |
| 18 | Aggressor imbalance | [-1, +1] | 0.0 | Net signed volume from Trade messages with explicit side. |
| 19 | Trade arrival rate | >= 0 | 0.0 | `log(1 + n_trades_in_bar)`. |
| 20 | Cancel-to-trade ratio | >= 0 | 0.0 | `log(1 + n_cancels / max(n_trades, 1))`. |
| 21 | Price impact per trade | signed | 0.0 | `(close - open) / (max(n_trades, 1) * TICK_SIZE)`. |

New observation dim: 22 * 10 + 2 = 222.

## Feature Computation Details

### Col 17: VAMP Displacement

Volume-Adjusted Mid-Price (VAMP) uses depth from multiple levels to compute a more robust mid-price. The displacement measures how VAMP changes within a bar.

```
VAMP = (sum(ask_price_i * ask_qty_i for top 3 ask levels) +
        sum(bid_price_i * bid_qty_i for top 3 bid levels)) /
       (sum(ask_qty_i for top 3) + sum(bid_qty_i for top 3))

vamp_start = VAMP at start of bar (before in-bar messages)
vamp_end = VAMP at end of bar (after all messages)

displacement = (vamp_end - vamp_start) / TICK_SIZE
```

If book has fewer than 1 level on either side at start or end, default to 0.0.

When `mbo_data is None`: default 0.0.

### Col 18: Aggressor Imbalance

More accurate than the tick-rule trade flow imbalance (col 0) because it uses the explicit side from MBO Trade messages rather than inferring from price changes.

```
For each Trade ('T' or 'F') message in the bar:
  - If the trade's stored order is on the bid side (passive bid, so aggressor is seller):
    sell_vol += size
  - If the trade's stored order is on the ask side (passive ask, so aggressor is buyer):
    buy_vol += size

aggressor_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + eps)
# Clamp to [-1, +1]
```

Note: In MBO data, the `side` field on a Trade message indicates the PASSIVE side (the resting order that was hit). The aggressor is the opposite side.

When `mbo_data is None`: default 0.0 (neutral).

### Col 19: Trade Arrival Rate

Captures trading urgency/activity within the bar. Uses log scale to compress the wide range.

```
n_trades = count of Trade ('T' or 'F') messages within the bar's time window
trade_arrival = log(1 + n_trades)
```

When `mbo_data is None`: default 0.0.

### Col 20: Cancel-to-Trade Ratio

Proxy for quote-stuffing or market-making activity. High cancel-to-trade ratio suggests frequent requoting.

```
n_cancels = count of Cancel ('C') messages within bar
n_trades = count of Trade ('T' or 'F') messages within bar

cancel_trade_ratio = log(1 + n_cancels / max(n_trades, 1))
```

When `mbo_data is None`: default 0.0.

### Col 21: Price Impact per Trade

Measures the market's ability to absorb order flow. Higher impact = less liquid market.

```
n_trades = count of Trade ('T' or 'F') messages within bar
impact = (bar.close - bar.open) / (max(n_trades, 1) * TICK_SIZE)
```

When `mbo_data is None`: default 0.0.

## Changes to `__init__.py`

```python
N_FEATURES = 22  # was 17
```

## Changes to `feature_pipeline.py`

### `compute_bar_features()`

1. Update the feature array allocation: use `N_FEATURES` (now 22).
2. Update docstring to show 22 features with new cols 17-21.
3. When `book_features` is available, assign cols 17-21 from `book_features[:, 8:13]`.
4. When `mbo_data is None`, set defaults: cols 17-21 all = 0.0.

### `_compute_book_features()`

Expand from shape `(n, 8)` to `(n, 13)`:
- Cols 0-7: existing (BBO imbalance, depth imbalance, cancel asymmetry, mean spread, OFI, depth ratio, wmid displacement, spread std)
- Col 8: VAMP displacement
- Col 9: Aggressor imbalance
- Col 10: Trade arrival rate
- Col 11: Cancel-to-trade ratio
- Col 12: Price impact per trade

**New defaults for cols 8-12:** all 0.0. Append to `_BOOK_DEFAULTS`.

**Implementation within bar loop:**
- Track `vamp_start` at bar start, `vamp_end` at bar close.
- Count trades and accumulate buy/sell aggressor volumes.
- Count cancels and trades for ratio.
- At bar close: compute VAMP displacement, aggressor imbalance, arrival rate, cancel ratio, price impact.

### `_BOOK_DEFAULTS`

Expand from 8 to 13 values:
```python
_BOOK_DEFAULTS = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#                 BBO  Depth Cancel Spread OFI  DepthR WMid SpreadStd VAMP Aggr TrdArr C/T Impact
```

### `build_feature_matrix()` docstring

Update: `(M, 22*h)` instead of `(M, 17*h)`.

## VAMP Helper on OrderBook

Add `vamp(n=3)` method to `OrderBook` class in `lob_reconstructor.py`:

```python
def vamp(self, n=3):
    """Volume-adjusted mid-price using top n levels on each side.

    Returns 0.0 if either side has no levels.
    """
    bid_levels = self.bid_depth(n)
    ask_levels = self.ask_depth(n)
    if not bid_levels or not ask_levels:
        return 0.0
    total_qty = sum(q for _, q in bid_levels) + sum(q for _, q in ask_levels)
    if total_qty == 0:
        return 0.0
    weighted = (sum(p * q for p, q in bid_levels) + sum(p * q for p, q in ask_levels))
    return weighted / total_qty
```

## Downstream Impact

All downstream code uses `N_FEATURES` dynamically:
- `barrier_env.py`: `self._h = self._feature_dim // N_FEATURES` — already uses `N_FEATURES`.
- `conftest.py`: `DEFAULT_FEATURE_DIM = N_FEATURES * DEFAULT_H` — already uses `N_FEATURES`.
- `supervised_diagnostic.py`: uses `compute_bar_features()` output shape dynamically.
- `precompute_barrier_cache.py`: stores `n_features` metadata. Old caches will fail version check.

No code changes needed in these files beyond what `N_FEATURES` propagates.

## Edge Cases

1. **No Trade messages in bar**: Aggressor imbalance = 0.0, trade arrival = 0.0, cancel-to-trade = log(1 + n_cancels / 1).
2. **No Cancel messages in bar**: Cancel-to-trade ratio = 0.0.
3. **Book empty at bar start/end**: VAMP displacement = 0.0.
4. **Fewer than 3 levels on a side**: VAMP uses however many levels are available (via `bid_depth(3)`/`ask_depth(3)`).
5. **Zero trades, zero cancels**: Cancel-to-trade ratio = 0.0.
6. **bar.close == bar.open**: Price impact = 0.0.

## Acceptance Criteria

1. `N_FEATURES == 22` and all files that imported it reflect the new value.
2. `compute_bar_features()` returns shape `(N, 22)`.
3. Without MBO data, new cols 17-21 have correct neutral defaults (all 0.0).
4. With synthetic MBO data, all 5 new features compute correctly.
5. All existing barrier tests pass (shape changes propagate cleanly).
6. `build_feature_matrix()` returns shape `(M, 22*h)`.
7. `BarrierEnv` observation dim = 22*10 + 2 = 222 (with h=10).
8. `OrderBook.vamp(n)` computes correctly.

## Test Strategy

### VAMP tests (~4)
- `test_vamp_symmetric_book`: Equal depth both sides -> VAMP = mid price.
- `test_vamp_asymmetric_book`: Unequal depth -> VAMP skews toward heavier side.
- `test_vamp_empty_side`: One side empty -> returns 0.0.
- `test_vamp_single_level`: Only 1 level per side -> uses that level.

### VAMP displacement tests (~4)
- `test_vamp_displacement_positive`: Book shifts up -> positive.
- `test_vamp_displacement_negative`: Book shifts down -> negative.
- `test_vamp_displacement_no_change`: Book unchanged -> 0.0.
- `test_vamp_displacement_default_no_mbo`: Without mbo_data, col 17 = 0.0.

### Aggressor imbalance tests (~5)
- `test_aggressor_imbalance_buy_heavy`: More buy aggression -> positive.
- `test_aggressor_imbalance_sell_heavy`: More sell aggression -> negative.
- `test_aggressor_imbalance_balanced`: Equal -> near 0.0.
- `test_aggressor_imbalance_no_trades`: No trades -> 0.0.
- `test_aggressor_imbalance_default_no_mbo`: Without mbo_data, col 18 = 0.0.

### Trade arrival rate tests (~4)
- `test_trade_arrival_many_trades`: Many trades -> higher value.
- `test_trade_arrival_few_trades`: Few trades -> lower value.
- `test_trade_arrival_no_trades`: No trades -> 0.0.
- `test_trade_arrival_default_no_mbo`: Without mbo_data, col 19 = 0.0.

### Cancel-to-trade ratio tests (~5)
- `test_cancel_trade_ratio_high`: Many cancels, few trades -> high ratio.
- `test_cancel_trade_ratio_low`: Few cancels, many trades -> low ratio.
- `test_cancel_trade_ratio_no_cancels`: No cancels -> 0.0.
- `test_cancel_trade_ratio_no_trades`: Cancels but no trades -> log(1 + n_cancels).
- `test_cancel_trade_ratio_default_no_mbo`: Without mbo_data, col 20 = 0.0.

### Price impact tests (~5)
- `test_price_impact_positive`: Close > open -> positive impact.
- `test_price_impact_negative`: Close < open -> negative impact.
- `test_price_impact_no_change`: Close == open -> 0.0.
- `test_price_impact_many_trades`: Same move with more trades -> lower impact.
- `test_price_impact_default_no_mbo`: Without mbo_data, col 21 = 0.0.

### Integration tests (~5)
- `test_n_features_now_22`: `N_FEATURES == 22`.
- `test_compute_bar_features_shape_22`: Output shape (N, 22).
- `test_build_feature_matrix_shape_22h`: Output shape (M, 22*h).
- `test_barrier_env_obs_dim_222`: With h=10, obs = 222.
- `test_all_existing_barrier_tests_pass`: Existing tests pass.

Total estimated: ~32 tests.
