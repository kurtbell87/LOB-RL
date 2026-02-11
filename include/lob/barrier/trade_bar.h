#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

struct TradeBar {
    int bar_index = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    int volume = 0;
    double vwap = 0.0;
    uint64_t t_start = 0;
    uint64_t t_end = 0;
    std::vector<double> trade_prices;
    std::vector<int> trade_sizes;
};

struct BarBookAccum {
    uint32_t bid_qty = 0;
    uint32_t ask_qty = 0;
    uint32_t total_bid_3 = 0;
    uint32_t total_ask_3 = 0;
    uint32_t total_bid_5 = 0;
    uint32_t total_ask_5 = 0;
    uint32_t total_bid_10 = 0;
    uint32_t total_ask_10 = 0;
    int bid_cancels = 0;
    int ask_cancels = 0;
    double ofi_signed_volume = 0.0;
    double total_add_volume = 0.0;
    double wmid_first = std::numeric_limits<double>::quiet_NaN();
    double wmid_end = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> spread_samples;
    double vamp_at_mid = std::numeric_limits<double>::quiet_NaN();
    double vamp_at_end = std::numeric_limits<double>::quiet_NaN();
    double buy_aggressor_vol = 0.0;
    double sell_aggressor_vol = 0.0;
    int n_trades = 0;
    int n_cancels = 0;
};
