#pragma once
#include "lob/barrier/trade_bar.h"
#include <vector>

struct BarrierLabel {
    int bar_index = 0;
    int label = 0;       // +1 (upper hit), -1 (lower hit), 0 (timeout)
    int tau = 0;          // bars from entry to resolution
    int resolution_bar = 0; // bar_index + tau
};

// Compute triple-barrier labels for each bar.
// a = upper barrier distance in ticks
// b = lower barrier distance in ticks
// t_max = max lookahead in bars
// tick_size = price per tick
std::vector<BarrierLabel> compute_labels(
    const std::vector<TradeBar>& bars,
    int a = 20,
    int b = 10,
    int t_max = 40,
    double tick_size = 0.25);
