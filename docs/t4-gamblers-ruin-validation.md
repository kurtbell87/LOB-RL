# T4: Gambler's Ruin Validation

## What to Build

A Python module `python/lob_rl/barrier/gamblers_ruin.py` that validates the label construction pipeline (T2) against analytic Gambler's ruin results at 5 drift levels.

## Context

Before any agent training, we need to validate that the label pipeline correctly detects barrier hits. The Gambler's ruin problem has a known closed-form solution for the probability of hitting the upper barrier first in a discrete random walk. By generating synthetic trade data from a random walk with known drift, running it through the bar and label pipelines, and comparing empirical hit frequencies to the analytic formula, we can catch sign convention bugs, off-by-one errors, and barrier detection logic flaws.

## Dependencies

- `python/lob_rl/barrier/bar_pipeline.py` — T1 (PASSED). Provides `TradeBar` and `build_bars_from_trades`.
- `python/lob_rl/barrier/label_pipeline.py` — T2 (PASSED). Provides `compute_labels` and `BarrierLabel`.

## Interface

### `gamblers_ruin_analytic(a, b, p)` -> `float`

Compute the analytic probability of hitting the upper barrier first in a discrete random walk.

**Parameters:**
- `a`: Upper barrier distance in ticks (positive integer)
- `b`: Lower barrier distance in ticks (positive integer)
- `p`: Probability of an uptick (0 < p < 1)

**Returns:** P(hit upper first) using the Gambler's ruin formula:
```
If p != 0.5:  P(upper) = (1 - (q/p)^b) / (1 - (q/p)^(a+b))   where q = 1 - p
If p == 0.5:  P(upper) = b / (a + b)
```

### `generate_random_walk(n_trades, p=0.5, start_price=4000.0, tick_size=0.25, seed=None)` -> `np.ndarray`

Generate synthetic trade prices from a discrete random walk on the tick grid.

**Parameters:**
- `n_trades`: Number of trades to generate
- `p`: Probability of an uptick. Default 0.5 (zero drift).
- `start_price`: Starting price. Default 4000.0.
- `tick_size`: Tick size. Default 0.25.
- `seed`: Random seed for reproducibility.

**Returns:** Structured numpy array with fields `('price', 'size', 'side', 'ts_event')` matching the format expected by `build_bars_from_trades`. Each trade has:
- `price`: tick-grid price (start_price +/- tick_size increments)
- `size`: 1 (constant)
- `side`: 'B' or 'A' matching direction (uptick → 'B', downtick → 'A')
- `ts_event`: monotonically increasing nanosecond timestamps within a single RTH session

### `validate_drift_level(p, a=20, b=10, n_bars=10000, bar_size=500, t_max=40, seed=None)` -> `dict`

Run the full validation for a single drift level.

**Parameters:**
- `p`: Uptick probability
- `a`: Upper barrier in ticks. Default 20.
- `b`: Lower barrier in ticks. Default 10.
- `n_bars`: Minimum number of bars to generate (enough trades generated to produce this many bars). Default 10,000.
- `bar_size`: Trades per bar. Default 500.
- `t_max`: Maximum holding period. Default 40.
- `seed`: Random seed.

**Returns:** Dictionary with:
```python
{
    "p": float,                    # Uptick probability
    "analytic": float,             # Analytic P(upper hit first)
    "empirical": float,            # Empirical P(upper hit first) from pipeline
    "n_labeled": int,              # Number of bars that received labels
    "n_upper": int,                # Number of upper hits
    "n_lower": int,                # Number of lower hits
    "n_timeout": int,              # Number of timeouts
    "se": float,                   # Standard error of empirical proportion
    "z_score": float,              # (empirical - analytic) / se
    "pass": bool,                  # |z_score| <= 2.0
}
```

### `run_validation(drift_levels=None, a=20, b=10, n_bars=10000, bar_size=500, t_max=40, seed=42)` -> `list[dict]`

Run validation across all drift levels.

**Parameters:**
- `drift_levels`: List of uptick probabilities. Default `[0.500, 0.505, 0.510, 0.490, 0.485]`.
- Other params forwarded to `validate_drift_level`.

**Returns:** List of result dicts from `validate_drift_level`, one per drift level.

## Tests to Write

### Analytic formula correctness

1. **Zero drift (p=0.5):** `gamblers_ruin_analytic(20, 10, 0.5)` == `10 / 30` ≈ 0.3333.
2. **Known values at p=0.505:** Result ≈ 0.388 (within 1%).
3. **Known values at p=0.510:** Result ≈ 0.445 (within 1%).
4. **Known values at p=0.490:** Result ≈ 0.280 (within 1%).
5. **Known values at p=0.485:** Result ≈ 0.232 (within 1%).
6. **Symmetric barriers (a==b):** `gamblers_ruin_analytic(10, 10, 0.5)` == 0.5.
7. **Edge case p near 0:** Very low probability of upper hit.
8. **Edge case p near 1:** Very high probability of upper hit.
9. **Unit barriers (a=1, b=1):** `gamblers_ruin_analytic(1, 1, 0.5)` == 0.5.

### Random walk generator

10. **Output shape:** `len(result) == n_trades`.
11. **Price on tick grid:** All prices are multiples of tick_size.
12. **Monotonic timestamps:** `ts_event` strictly increasing.
13. **Correct drift direction:** With p=1.0, all prices should be non-decreasing. With p=0.0, all non-increasing.
14. **Reproducibility:** Same seed → same output.
15. **Side field matches direction:** Upticks have side='B', downticks have side='A'.

### Full pipeline validation

16. **Zero drift passes (p=0.5):** Empirical P(upper) within 2 SE of 0.3333, n >= 10,000 bars.
17. **Mild upward drift passes (p=0.505):** Within 2 SE of ~0.388.
18. **Moderate upward drift passes (p=0.510):** Within 2 SE of ~0.445.
19. **Mild downward drift passes (p=0.490):** Within 2 SE of ~0.280.
20. **Moderate downward drift passes (p=0.485):** Within 2 SE of ~0.232.
21. **All 5 drift levels pass:** Aggregate test.
22. **Label invariants hold:** All labels in {-1, 0, +1}, all tau > 0, all tau <= t_max.
23. **Timeout rate is plausible:** Not 0% and not 100% for zero drift with t_max=40.

### Edge cases

24. **Small n_bars still works:** n_bars=100 produces reasonable results (no crashes).
25. **Different barrier sizes:** a=10, b=5 with p=0.5 gives P(upper) = 5/15 = 0.3333.
26. **Deterministic seed gives identical results across runs.**

## Acceptance Criteria

- All 5 drift levels pass within 2 SE.
- Analytic formula matches known values to within 1%.
- No NaN or Inf in any outputs.
- Generates at least 10,000 bars per drift level for sufficient statistical power.

## File Location

- Module: `python/lob_rl/barrier/gamblers_ruin.py`
- Tests: `python/tests/barrier/test_gamblers_ruin.py`
