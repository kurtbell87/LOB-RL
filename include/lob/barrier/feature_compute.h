#pragma once
#include "lob/barrier/trade_bar.h"
#include <cstdint>
#include <vector>

// Constants
constexpr int N_FEATURES = 22;
constexpr double TICK_SIZE = 0.25;
constexpr int REALIZED_VOL_WARMUP = 19;
constexpr double SESSION_AGE_PERIOD = 20.0;

// Compute raw feature vector for a sequence of bars.
// Returns a flat vector of size bars.size() * N_FEATURES.
// bars and accums must have the same size (asserts).
std::vector<double> compute_bar_features(
    const std::vector<TradeBar>& bars,
    const std::vector<BarBookAccum>& accums,
    uint64_t rth_open_ns,
    uint64_t rth_close_ns);

// Z-score normalize features column-wise using an expanding (or rolling) window.
// raw: flat vector of n_rows * n_cols doubles.
// NaN values are replaced with 0.0 before normalization.
// Output is clipped to [-5, 5].
// window: if > 0, use a rolling window of this size; otherwise expanding.
std::vector<double> normalize_features(
    const std::vector<double>& raw,
    int n_rows,
    int n_cols,
    int window = 0);

// Assemble lookback windows: each output row is the concatenation of h
// consecutive input rows. Returns float32 vector.
// Output has (n_rows - h + 1) rows, each of n_cols * h elements.
// Returns empty if n_rows < h.
std::vector<float> assemble_lookback(
    const std::vector<double>& normed,
    int n_rows,
    int n_cols,
    int h);
