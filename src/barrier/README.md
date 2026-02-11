# src/barrier — Barrier Feature Computation

Feature extraction pipeline for the barrier-based trading environment.

## Files

| File | Role |
|---|---|
| `bar_builder.cpp` | `BarBuilder` — aggregates MBO messages into `TradeBar` + `BarBookAccum` sequences |
| `barrier_label.cpp` | `compute_labels()` — triple-barrier labeling with intrabar tiebreaking |
| `barrier_precompute.cpp` | `barrier_precompute()` — end-to-end pipeline: DbnFileSource → BarBuilder → features → labels → `BarrierPrecomputedDay` |
| `feature_compute.cpp` | `compute_bar_features()`, `normalize_features()`, `assemble_lookback()` — raw features → z-scored → lookback windows |

## Key interfaces (in `include/lob/barrier/`)

### `trade_bar.h`

```cpp
struct TradeBar {
    int bar_index; double open, high, low, close; int volume;
    double vwap; uint64_t t_start, t_end;
    vector<double> trade_prices; vector<int> trade_sizes;
};

struct BarBookAccum {
    uint32_t bid_qty, ask_qty;
    uint32_t total_bid_3, total_ask_3, total_bid_5, total_ask_5, total_bid_10, total_ask_10;
    int bid_cancels, ask_cancels;
    double ofi_signed_volume, total_add_volume;
    double wmid_first, wmid_end;            // NaN default
    vector<double> spread_samples;
    double vamp_at_mid, vamp_at_end;        // NaN default
    double buy_aggressor_vol, sell_aggressor_vol;
    int n_trades, n_cancels;
};
```

### `barrier_label.h`

```cpp
struct BarrierLabel {
    int bar_index = 0;
    int label = 0;       // +1 (upper hit), -1 (lower hit), 0 (timeout)
    int tau = 0;          // bars from entry to resolution
    int resolution_bar = 0; // bar_index + tau
};

// Triple-barrier labels for each bar. Intrabar tiebreaking via trade scan.
vector<BarrierLabel> compute_labels(
    const vector<TradeBar>& bars,
    int a = 20, int b = 10, int t_max = 40, double tick_size = 0.25);
```

### `feature_compute.h`

```cpp
// Constants
constexpr int N_FEATURES = 22;
constexpr double TICK_SIZE = 0.25;
constexpr int REALIZED_VOL_WARMUP = 19;
constexpr double SESSION_AGE_PERIOD = 20.0;

// Raw features: flat vector of bars.size() * N_FEATURES doubles.
// Throws std::invalid_argument if bars.size() != accums.size().
vector<double> compute_bar_features(
    const vector<TradeBar>& bars, const vector<BarBookAccum>& accums,
    uint64_t rth_open_ns, uint64_t rth_close_ns);

// Z-score normalize column-wise. NaN → 0. Clipped to [-5, 5].
// window > 0: rolling window; window == 0: expanding.
vector<double> normalize_features(
    const vector<double>& raw, int n_rows, int n_cols, int window = 0);

// Lookback windows: each output row = h consecutive input rows concatenated.
// Returns float32. Empty if n_rows < h.
vector<float> assemble_lookback(
    const vector<double>& normed, int n_rows, int n_cols, int h);
```

## Feature columns (0–21)

| Col | Name | Formula |
|-----|------|---------|
| 0 | Trade flow imbalance | Tick-rule buy/sell classification, (buy-sell)/(buy+sell) |
| 1 | BBO imbalance | bid_qty / (bid_qty + ask_qty), 0.5 if both zero |
| 2 | Depth(5) imbalance | total_bid_5 / (total_bid_5 + total_ask_5) |
| 3 | Bar range (ticks) | (high - low) / TICK_SIZE |
| 4 | Bar body (ticks) | (close - open) / TICK_SIZE |
| 5 | Body/range ratio | (close - open) / (high - low), 0 if flat |
| 6 | VWAP displacement | (close - vwap) / range, 0 if flat |
| 7 | Log volume | log(max(volume, 1)) |
| 8 | Realized vol | Population std of expanding log-returns, NaN if i < 19 |
| 9 | Session time | clamp((t_end - rth_open) / rth_duration, 0, 1) |
| 10 | Cancel asymmetry | (bid_cancels - ask_cancels) / (total + 1e-10) |
| 11 | Mean spread | Mean of spread_samples, 1.0 if empty |
| 12 | Session age | min(bar_index / 20.0, 1.0) |
| 13 | OFI | clamp(ofi_signed / (total_add + 1e-10), -1, 1) |
| 14 | Depth ratio | (bid_3 + ask_3) / (bid_10 + ask_10 + 1e-10), 0.5 if zero |
| 15 | WMid displacement | (wmid_end - wmid_first) / TICK_SIZE, 0 if NaN |
| 16 | Spread std | Population std of spread_samples, 0 if < 2 |
| 17 | VAMP displacement | (vamp_end - vamp_mid) / TICK_SIZE, 0 if NaN |
| 18 | Aggressor imbalance | (buy - sell) / (buy + sell + 1e-10), 0 if both zero |
| 19 | Trade arrival rate | log(1 + n_trades) |
| 20 | Cancel-to-trade ratio | log(1 + n_cancels / max(n_trades, 1)) |
| 21 | Price impact/trade | (close - open) / (max(n_trades, 1) * TICK_SIZE) |

### `barrier_precompute.h`

```cpp
struct BarrierPrecomputedDay {
    int n_bars, n_usable, bar_size, lookback;
    vector<double> bar_open, bar_high, bar_low, bar_close, bar_vwap;
    vector<int> bar_volume;
    vector<uint64_t> bar_t_start, bar_t_end;
    vector<double> trade_prices; vector<int> trade_sizes;
    vector<int64_t> bar_trade_offsets;  // size n_bars + 1
    vector<int> label_values, label_tau, label_resolution_bar;
    vector<float> features;  // (n_usable, N_FEATURES * lookback) row-major
    int n_features;
};

// From IMessageSource (for tests)
BarrierPrecomputedDay barrier_precompute(
    IMessageSource& source, int bar_size = 500, int lookback = 10,
    int a = 20, int b = 10, int t_max = 40);

// From .dbn.zst file path (for production / pybind11)
BarrierPrecomputedDay barrier_precompute(
    const string& path, uint32_t instrument_id,
    int bar_size = 500, int lookback = 10,
    int a = 20, int b = 10, int t_max = 40);
```

## Cross-file dependencies

- **Depends on:** `include/lob/barrier/trade_bar.h`, `include/lob/barrier/barrier_label.h`, `include/lob/barrier/feature_compute.h`, `include/lob/barrier/bar_builder.h`, `include/lob/barrier/barrier_precompute.h`, `src/data/dbn_file_source.h`
- **Used by:** Python bindings (`src/bindings/bindings.cpp`), `scripts/precompute_barrier_cache.py`
