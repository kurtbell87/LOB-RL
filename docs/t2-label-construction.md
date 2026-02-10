# T2: Label Construction Pipeline

## What to Build

A Python module `python/lob_rl/barrier/label_pipeline.py` that implements barrier-hit detection with three-outcome labels and intrabar tiebreaking.

## Context

The bar pipeline (T1) produces `TradeBar` objects with OHLCV, timestamps, and retained trade sequences. The label pipeline consumes these bars and, for each bar `k`, determines whether a future upper barrier, lower barrier, or timeout occurs first.

## Dependencies

- `python/lob_rl/barrier/bar_pipeline.py` — T1 (PASSED). Provides `TradeBar` with `open`, `high`, `low`, `close`, `trade_prices`, `trade_sizes`.

## Interface

### `BarrierLabel` (dataclass)

```python
@dataclass
class BarrierLabel:
    bar_index: int          # Bar k (the candidate entry bar)
    label: int              # +1 (upper hit first), -1 (lower hit first), 0 (timeout)
    tau: int                # Stopping time: number of bars from entry to resolution
    resolution_type: str    # "upper", "lower", "timeout", "tiebreak_upper", "tiebreak_lower"
    entry_price: float      # C_k (close of entry bar)
    resolution_bar: int     # Bar index where resolution occurred (or k + T_max for timeout)
```

### `compute_labels(bars, a=20, b=10, t_max=40, direction="long")` -> `list[BarrierLabel]`

For each bar `k` in `bars`, compute the barrier-hit label by scanning forward bars.

**Parameters:**
- `bars`: List of `TradeBar` objects from a single session
- `a`: Upper barrier distance in ticks (profit target for long, stop for short). Default 20.
- `b`: Lower barrier distance in ticks (stop for long, profit target for short). Default 10.
- `t_max`: Maximum holding period in bars. Default 40.
- `direction`: "long" or "short". For "long": upper=profit, lower=stop. For "short": barriers are mirrored — profit barrier below entry, stop barrier above.

**Returns:** List of `BarrierLabel`, one per bar.

**Barrier definition (long direction):**
```
Entry price: C_k (close of bar k)
Upper barrier: U = C_k + a * tick_size   (tick_size = 0.25 for /MES)
Lower barrier: D = C_k - b * tick_size
```

**Barrier detection:**
```
For each bar j > k:
    Upper hit: H_j >= U
    Lower hit: L_j <= D
```

**Tiebreaking (dual barrier hit within single bar):**
If both `H_j >= U` and `L_j <= D` on the same bar `j`:
1. Scan `trade_prices` of bar `j` in order.
2. First trade that crosses either barrier determines which was hit first.
3. Gap-through edge case: if first trade already exceeds both barriers, resolve by gap direction from previous close.

**Short direction:**
For "short", barriers are mirrored:
```
Profit barrier: D = C_k - a * tick_size   (price goes DOWN to win)
Stop barrier:   U = C_k + b * tick_size   (price goes UP to lose)
```

### `calibrate_t_max(bars, a=20, b=10)` -> `int`

Calibrate T_max from the data per Section 2.4 of the spec.

**Procedure:**
1. Run label construction with T_max = infinity (no timeout).
2. Compute empirical distribution of `tau` for all bars where label == +1 (upper hit for long).
3. Return `ceil(P95 of this distribution)`.

**Returns:** Calibrated T_max (integer).

### `compute_tiebreak_frequency(labels)` -> `float`

Returns the fraction of labels that required tiebreaking (resolution_type contains "tiebreak").

### `compute_label_distribution(labels)` -> `dict`

Returns `{"p_plus": float, "p_minus": float, "p_zero": float}` — fraction of each label type.

## Tick Size

/MES tick size is 0.25 index points ($1.25 per tick). Barrier distances `a` and `b` are specified in ticks. To convert to price: `barrier_price = entry_price + a * 0.25`.

## Tests to Write

### Core labeling

1. **Upper barrier hit on bar j=5:** Hand-crafted bar sequence where bar 5's high crosses the upper barrier. Label should be +1, tau=5.

2. **Lower barrier hit on bar j=3:** Hand-crafted bar sequence where bar 3's low crosses the lower barrier. Label should be -1, tau=3.

3. **Timeout (neither barrier hit):** Bar sequence where all bars stay within barriers for T_max bars. Label should be 0.

### Tiebreaking

4. **Dual barrier breach — upper first:** Construct a bar where both H >= U and L <= D, but trade sequence shows upper barrier crossed first. Label should be +1 with resolution_type="tiebreak_upper".

5. **Dual barrier breach — lower first:** Same as above but trade sequence shows lower crossed first. Label should be -1 with resolution_type="tiebreak_lower".

6. **Gap-through edge case:** First trade of bar exceeds both barriers. Resolve by gap direction.

### Short direction

7. **Short barrier mirroring:** For direction="short", profit barrier is below entry (price goes DOWN to win), stop barrier is above. Verify labels are computed correctly.

### T_max calibration

8. **T_max calibration procedure:** With T_max=infinity, compute P95 of winner distribution. Verify it's a reasonable integer.

### Diagnostics

9. **Tiebreak frequency computation:** Known set of labels with some tiebreaks, verify frequency is computed correctly.

10. **Label distribution:** Known labels, verify p+, p-, p0 are computed correctly.

### Edge cases

11. **Last bars of session with insufficient lookahead:** Bar near end of session with fewer than T_max bars remaining. Should still be labeled (timeout if no barrier hit in remaining bars).

12. **Barrier hit on first bar after entry:** tau=1.

13. **Exact barrier touch:** H_j == U exactly (should count as a hit).

14. **Entry price computation:** Verify C_k is used as entry price, not O_k or VWAP_k.

### Invariants

15. **tau > 0 for all labels:** Resolution always takes at least 1 bar.

16. **tau <= T_max for all labels:** No resolution exceeds timeout.

17. **Label values are in {-1, 0, +1}:** No other values.

18. **Resolution bar is correct:** resolution_bar == bar_index + tau.

## Acceptance Criteria

- All tests pass.
- Tiebreak frequency < 5% on real data with default parameters (a=20, b=10, N=500).
- Label distribution (p+, p-, p0) is plausible: p+ < p- for 2:1 barriers under roughly driftless conditions (expected: p+ ~ 0.33, p- ~ 0.67 ignoring timeouts).

## File Location

- Module: `python/lob_rl/barrier/label_pipeline.py`
- Tests: `python/tests/barrier/test_label_pipeline.py`
