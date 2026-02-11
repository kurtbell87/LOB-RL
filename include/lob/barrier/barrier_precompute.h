#pragma once
#include "lob/barrier/feature_compute.h"
#include "lob/barrier/trade_bar.h"
#include "lob/session.h"
#include "lob/source.h"
#include <cstdint>
#include <string>
#include <vector>

struct BarrierPrecomputedDay {
    int n_bars = 0;
    int n_usable = 0;
    int bar_size = 0;
    int lookback = 0;

    // Bars: OHLCV arrays (all size n_bars)
    std::vector<double> bar_open, bar_high, bar_low, bar_close, bar_vwap;
    std::vector<int> bar_volume;
    std::vector<uint64_t> bar_t_start, bar_t_end;

    // Trade sequences (flat + offsets)
    std::vector<double> trade_prices;
    std::vector<int> trade_sizes;
    std::vector<int64_t> bar_trade_offsets;  // size n_bars + 1

    // Labels (all size n_bars) — long direction
    std::vector<int> label_values;   // +1, -1, 0
    std::vector<int> label_tau;
    std::vector<int> label_resolution_bar;

    // Short-direction labels (all size n_bars)
    std::vector<int> short_label_values;   // +1, -1, 0
    std::vector<int> short_label_tau;
    std::vector<int> short_label_resolution_bar;

    // Features (n_usable rows, each N_FEATURES * lookback elements, row-major)
    std::vector<float> features;

    // Per-bar normalized features (n_trimmed rows, each N_FEATURES floats, row-major)
    // After warmup trimming and z-score normalization, BEFORE lookback assembly.
    std::vector<float> bar_features;
    int n_trimmed = 0;

    // Metadata
    int n_features = 0;
};

// Overload for IMessageSource (used in tests with ScriptedSource)
BarrierPrecomputedDay barrier_precompute(
    IMessageSource& source,
    const SessionConfig& cfg,
    int bar_size = 500,
    int lookback = 10,
    int a = 20,
    int b = 10,
    int t_max = 40);

// Overload for file path (production use with DbnFileSource)
BarrierPrecomputedDay barrier_precompute(
    const std::string& path,
    uint32_t instrument_id,
    int bar_size = 500,
    int lookback = 10,
    int a = 20,
    int b = 10,
    int t_max = 40);
