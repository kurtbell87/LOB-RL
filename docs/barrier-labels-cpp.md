# Barrier Label Computation in C++

## Context

Cycle 3 implemented feature computation. This cycle implements the C++ equivalent of Python's `label_pipeline.py` — triple-barrier label construction for trade bars.

The label algorithm scans forward from each entry bar to find which barrier (upper profit, lower stop, or timeout) is hit first. Dual barrier breaches within a single bar are resolved by scanning the trade sequence.

### Existing Infrastructure

- `TradeBar` — OHLCV + trade_prices/trade_sizes (`include/lob/barrier/trade_bar.h`)
- `feature_compute.h` — `TICK_SIZE = 0.25` constant (`include/lob/barrier/feature_compute.h`)

## Requirements

### 1. BarrierLabel Struct

**File:** `include/lob/barrier/barrier_label.h`

```cpp
struct BarrierLabel {
    int bar_index;       // entry bar index k
    int label;           // +1 (upper hit), -1 (lower hit), 0 (timeout)
    int tau;             // bars from entry to resolution (j - k)
    int resolution_bar;  // bar index where resolved (k + tau)
};
```

### 2. compute_labels()

```cpp
std::vector<BarrierLabel> compute_labels(
    const std::vector<TradeBar>& bars,
    int a = 20,
    int b = 10,
    int t_max = 40,
    double tick_size = 0.25);
```

Returns one `BarrierLabel` per bar. The function implements `direction="long"` only (matching the precompute pipeline).

**Algorithm for each entry bar k:**

1. `entry_price = bars[k].close`
2. `upper_barrier = entry_price + a * tick_size`
3. `lower_barrier = entry_price - b * tick_size`
4. Scan bars j from k+1 to min(k + t_max, n_bars - 1):
   a. `upper_hit = bars[j].high >= upper_barrier`
   b. `lower_hit = bars[j].low <= lower_barrier`
   c. If both hit (dual breach): tiebreak using trade sequence (see below)
   d. If only upper_hit: label=+1, tau=j-k
   e. If only lower_hit: label=-1, tau=j-k
5. If no barrier hit within horizon: label=0, tau=min(t_max, n_bars-1-k), minimum 1

**Tiebreak logic** (dual breach, both upper and lower barriers hit in same bar j):

1. Get trade_prices from bars[j]
2. If empty: fall back to gap direction
3. Check first trade price:
   - If first trade already at/past both barriers, or at/past either barrier: fall back to gap direction
4. Otherwise scan trade_prices sequentially:
   - First trade >= upper_barrier → label=+1
   - First trade <= lower_barrier → label=-1
5. If no trade crosses either (shouldn't happen): fall back to gap direction

**Gap direction fallback:**
- `first_trade = trade_prices[0]` if non-empty, else `bars[j].open`
- If `first_trade >= bars[j-1].close` → label=+1, else label=-1

### 3. Empty bars edge case

`compute_labels({})` returns empty vector.

### 4. CMakeLists.txt Changes

Add `src/barrier/barrier_label.cpp` to `lob_core` sources.

## Files to Change

| File | Change |
|------|--------|
| `include/lob/barrier/barrier_label.h` | **NEW** — BarrierLabel struct + compute_labels() declaration |
| `src/barrier/barrier_label.cpp` | **NEW** — Implementation |
| `CMakeLists.txt` | Add src/barrier/barrier_label.cpp and test file |
| `tests/test_barrier_labels.cpp` | **NEW** — Tests |

## Test Plan

### BarrierLabel struct tests (~2):
1. Default construction has zero values
2. Fields set correctly after construction

### Basic label tests (~6):
3. Empty bars → empty labels
4. Single bar → timeout label (tau=1 minimum)
5. Upper barrier hit: bar with high >= entry + a*tick → label=+1
6. Lower barrier hit: bar with low <= entry - b*tick → label=-1
7. Timeout: no barrier hit within t_max → label=0
8. Multiple bars: correct label count equals bar count

### Tau and resolution bar tests (~4):
9. tau = j - k for immediate next bar hit
10. tau = distance to barrier hit bar (not just 1)
11. resolution_bar = bar_index + tau
12. Timeout tau: min(t_max, remaining bars), minimum 1

### Tiebreak tests (~6):
13. Dual breach: first trade crosses upper first → label=+1
14. Dual breach: first trade crosses lower first → label=-1
15. Dual breach with empty trades: gap direction (first trade >= prev close → +1)
16. Dual breach with empty trades: gap direction (first trade < prev close → -1)
17. First trade already past a barrier → gap direction fallback
18. Dual breach resolved before scanning all trades

### Barrier distance tests (~4):
19. a=1, b=1 tight barriers → frequent hits
20. a=100, b=100 wide barriers → mostly timeouts
21. Different a and b values (asymmetric barriers)
22. tick_size parameter used correctly (not hardcoded 0.25)

### Edge case tests (~5):
23. Last bar in series: can only timeout
24. Second-to-last bar: tau capped at 1
25. t_max=1: only looks at next bar
26. All bars same price: timeout for all entries
27. Monotonically rising prices: all upper hits (except near end)

### Integration tests (~3):
28. Labels for known price sequence match hand-computed values
29. Label count equals bar count
30. All labels have valid fields (label in {-1,0,+1}, tau >= 1, resolution_bar >= bar_index)

## Acceptance Criteria

- All existing C++ tests still pass (576 + 15 skipped)
- New tests pass (~30 cases)
- Label computation matches Python `label_pipeline.py` for long direction
- Tiebreak logic handles all edge cases (empty trades, gap direction)
- tau is always >= 1
