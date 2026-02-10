# Barrier Label Construction

## What to Build

A Python module `python/lob_rl/barrier_labels.py` that constructs three-outcome barrier-hit labels from bar-level price data. This is the labeling foundation for the barrier-hit PPO agent — it determines whether price hits an upper target, lower stop, or times out within a forward-looking window.

## Context

The existing `bar_supervised_diagnostic.py` uses 1-step-ahead oracle labels (next-bar price direction). This module replaces those with multi-bar barrier labels that match the agent's actual decision problem: "if I enter here, does price hit my target or stop first?"

The existing `.npz` cache stores tick-level `mid` (N,) arrays per day. The `aggregate_bars()` function in `bar_aggregation.py` computes bar features but does not expose bar high/low prices. This module provides utilities to extract bar OHLC from tick-level data and construct barrier labels from the resulting bar series.

## Dependencies

- `numpy` (already installed)
- `python/lob_rl/bar_aggregation.py` — for `aggregate_bars()` (existing, unchanged)
- `.npz` cache files — `obs` (N,43), `mid` (N,), `spread` (N,)

## Module: `python/lob_rl/barrier_labels.py`

### Function 1: `extract_bar_ohlc`

```python
def extract_bar_ohlc(mid: np.ndarray, bar_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract bar-level OHLC from tick-level mid-price array.

    Args:
        mid: (N,) float64 — tick-level mid prices.
        bar_size: int — number of ticks per bar. Must be >= 1.

    Returns:
        bar_open:  (B,) float64
        bar_high:  (B,) float64
        bar_low:   (B,) float64
        bar_close: (B,) float64

    Bar count follows the same convention as aggregate_bars():
    - N // bar_size full bars
    - Plus one partial bar if the remainder >= bar_size // 4
    - Partial bars with fewer than bar_size // 4 ticks are discarded
    """
```

**Edge cases:**
- `mid` is empty → return four empty arrays
- `bar_size > len(mid)` → return empty if len(mid) < bar_size // 4, else one partial bar
- All prices identical → open == high == low == close for every bar

### Function 2: `construct_barrier_labels`

```python
def construct_barrier_labels(
    bar_close: np.ndarray,
    bar_high: np.ndarray,
    bar_low: np.ndarray,
    upper_ticks: float,
    lower_ticks: float,
    t_max: int,
    tick_size: float = 0.25,
) -> tuple[np.ndarray, dict]:
    """Construct three-outcome barrier-hit labels.

    For each bar k, defines barriers relative to bar_close[k]:
        Upper barrier: U_k = bar_close[k] + upper_ticks * tick_size
        Lower barrier: D_k = bar_close[k] - lower_ticks * tick_size

    Then scans forward bars j = k+1, k+2, ..., k+t_max:
        - If bar_high[j] >= U_k before bar_low[j] <= D_k → label = +1
        - If bar_low[j] <= D_k before bar_high[j] >= U_k → label = -1
        - If both barriers breached in the same bar j (dual hit) → label = 0 (ambiguous)
        - If neither barrier breached within t_max bars → label = 0 (timeout)

    Args:
        bar_close: (B,) float64 — bar close prices.
        bar_high:  (B,) float64 — bar high prices.
        bar_low:   (B,) float64 — bar low prices.
        upper_ticks: float — upper barrier distance in ticks (e.g., 20).
        lower_ticks: float — lower barrier distance in ticks (e.g., 10).
        t_max: int — maximum lookahead bars (timeout window). Must be >= 1.
        tick_size: float — price per tick (default 0.25 for /MES).

    Returns:
        labels: (B,) int64 — values in {-1, 0, +1}.
            Bars within t_max of the end get label 0 (insufficient lookahead).
        stats: dict with keys:
            'n_upper_hit': int — count of +1 labels
            'n_lower_hit': int — count of -1 labels
            'n_timeout': int — count of 0 labels (timeout, no hit)
            'n_dual_hit': int — count of 0 labels due to dual barrier breach
            'n_insufficient': int — count of 0 labels due to insufficient lookahead
            'dual_hit_rate': float — n_dual_hit / (n_upper_hit + n_lower_hit + n_dual_hit)
            'mean_time_to_upper': float — mean bars until upper hit (NaN if no upper hits)
            'mean_time_to_lower': float — mean bars until lower hit (NaN if no lower hits)
    """
```

**Key semantics:**
- Barriers are checked using bar high/low (not close). This means a barrier can be breached intrabar.
- When both `bar_high[j] >= U_k` AND `bar_low[j] <= D_k` for the SAME bar j, this is a "dual hit." Without intrabar trade sequences, we cannot determine which barrier was hit first. Label as 0 (ambiguous). Track the count in `stats['n_dual_hit']`.
- Bars near the end of the array (where `k + t_max >= B`) get label 0 with `n_insufficient` incremented.
- The function must handle arrays of any length, including length 0 and 1.

**Performance requirement:** Must process 200,000+ bars in under 5 seconds. Use vectorized numpy operations, not Python loops over bars.

### Function 3: `calibrate_t_max`

```python
def calibrate_t_max(
    bar_close: np.ndarray,
    bar_high: np.ndarray,
    bar_low: np.ndarray,
    upper_ticks: float,
    lower_ticks: float,
    tick_size: float = 0.25,
    percentile: float = 95.0,
) -> tuple[int, dict]:
    """Calibrate T_max from the winner (upper-hit) time-to-resolution distribution.

    Runs label construction with t_max=len(bar_close) (no timeout), then
    computes the percentile of the time-to-resolution for upper-hit labels.

    Args:
        bar_close, bar_high, bar_low: (B,) arrays.
        upper_ticks, lower_ticks: barrier distances in ticks.
        tick_size: price per tick.
        percentile: percentile of winner distribution to use (default 95).

    Returns:
        t_max: int — calibrated T_max (ceiling of the percentile).
        diagnostics: dict with keys:
            'p50_upper': float — median time-to-upper-hit
            'p95_upper': float — 95th percentile time-to-upper-hit
            'p50_lower': float — median time-to-lower-hit
            'p95_lower': float — 95th percentile time-to-lower-hit
            'n_upper': int — number of upper hits
            'n_lower': int — number of lower hits
    """
```

### Function 4: `generate_random_walk_bars`

```python
def generate_random_walk_bars(
    n_bars: int,
    trades_per_bar: int,
    tick_size: float = 0.25,
    p_up: float = 0.5,
    start_price: float = 4000.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic bar OHLC from a discrete random walk for testing.

    Simulates individual trade prices as a random walk on the tick grid,
    then aggregates into bars of `trades_per_bar` trades each.

    Args:
        n_bars: number of bars to generate.
        trades_per_bar: number of trades per bar.
        tick_size: price increment per tick.
        p_up: probability of +1 tick per trade.
        start_price: initial price.
        seed: random seed for reproducibility.

    Returns:
        bar_open:  (n_bars,) float64
        bar_high:  (n_bars,) float64
        bar_low:   (n_bars,) float64
        bar_close: (n_bars,) float64
    """
```

This function is primarily for testing (gambler's ruin validation) but is public API for general synthetic data generation.

## Acceptance Criteria

### AC-1: extract_bar_ohlc correctness
- Given a known tick sequence, bar OHLC values match hand-computed values.
- Bar count matches `aggregate_bars()` for the same input and bar_size.
- Empty input returns empty arrays.

### AC-2: construct_barrier_labels basic correctness
- A simple 5-bar scenario where price rises monotonically: first bar should label +1 (upper hit).
- A simple 5-bar scenario where price drops monotonically: first bar should label -1 (lower hit).
- A flat price scenario (all bars identical): all labels should be 0 (timeout).
- Last `t_max` bars should all be labeled 0 (insufficient lookahead).

### AC-3: Dual-hit detection
- When a bar's high and low both breach the barriers, label should be 0.
- `stats['n_dual_hit']` should accurately count these cases.
- `stats['dual_hit_rate']` should be `n_dual_hit / (n_upper + n_lower + n_dual)`.

### AC-4: Gambler's ruin validation (critical)
For a discrete random walk with known drift, the empirical barrier-hit frequencies must match the gambler's ruin closed form within 2 standard errors.

**Analytic formula:**
```
P(hit upper first | start at 0) = (1 - (q/p)^b) / (1 - (q/p)^(a+b))
For p = q = 0.5: P(upper) = b / (a + b)
```

Where `a` = upper_ticks, `b` = lower_ticks.

**Test configurations (a=20, b=10):**

| p_up | Analytic P(upper) | Tolerance |
|------|-------------------|-----------|
| 0.500 | 0.3333 | ± 2 SE (n=10000) |
| 0.505 | ~0.388 | ± 2 SE |
| 0.510 | ~0.445 | ± 2 SE |
| 0.490 | ~0.280 | ± 2 SE |
| 0.485 | ~0.232 | ± 2 SE |

Use `n_bars=10000`, `trades_per_bar=500`, `t_max=500` (effectively infinite for most hits).
Use `generate_random_walk_bars()` for synthetic data.

Standard error for binomial proportion: `SE = sqrt(p*(1-p)/n)` where n = number of labeled bars (excluding timeouts and insufficient lookahead).

If zero-drift results are correct but non-zero drift is wrong, there's a sign convention bug.

### AC-5: calibrate_t_max
- On synthetic data with known barrier distances, calibrated T_max should be finite and reasonable.
- P95 of upper-hit time should be greater than P95 of lower-hit time when `upper_ticks > lower_ticks`.

### AC-6: Label statistics
- `stats` dict sums: `n_upper_hit + n_lower_hit + n_timeout + n_dual_hit + n_insufficient == len(bar_close)`.
- `dual_hit_rate` is 0 when no dual hits occur.
- `mean_time_to_upper` and `mean_time_to_lower` are NaN when no hits of that type occur.

### AC-7: Performance
- `construct_barrier_labels` on 200,000 bars with t_max=40 completes in under 5 seconds.

## Implementation Notes

- All functions should be in a single file: `python/lob_rl/barrier_labels.py`.
- Use numpy vectorized operations for the label construction loop. A possible approach: for each bar k, compute barrier levels, then use `np.argmax` on the forward window to find first breach. But the key insight is that the inner loop over lookahead bars (t_max=40) is small, so a loop over lookahead offsets (not over bars) is efficient.
- The dual-hit check must happen BEFORE individual barrier checks for the same bar.
- Type hints should use `np.ndarray` for arrays and standard Python types for scalars.

## Files to Create

| File | Role |
|------|------|
| `python/lob_rl/barrier_labels.py` | Barrier label construction module |

## Files NOT to Modify

- `python/lob_rl/bar_aggregation.py` — leave unchanged
- `python/lob_rl/bar_level_env.py` — leave unchanged
- `scripts/bar_supervised_diagnostic.py` — will be extended separately, not in this TDD cycle

## Test File

`python/tests/test_barrier_labels.py`
