#include "lob/barrier/feature_compute.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// compute_bar_features
// ---------------------------------------------------------------------------

// Tick-rule trade flow imbalance for a single bar.
static double compute_trade_flow_imbalance(const TradeBar& bar) {
    const auto& prices = bar.trade_prices;
    const auto& sizes = bar.trade_sizes;
    if (prices.size() < 2) return 0.0;

    // Classify each trade via tick rule: +1 uptick, -1 downtick, 0 unchanged.
    // Forward-fill unchanged prices with previous direction.
    double buy_vol = 0.0;
    double sell_vol = 0.0;
    int prev_dir = 0;

    for (size_t i = 1; i < prices.size(); ++i) {
        double diff = prices[i] - prices[i - 1];
        int dir;
        if (diff > 0.0)
            dir = 1;
        else if (diff < 0.0)
            dir = -1;
        else
            dir = prev_dir;  // forward-fill

        if (dir > 0)
            buy_vol += sizes[i];
        else if (dir < 0)
            sell_vol += sizes[i];

        prev_dir = dir;
    }

    double total = buy_vol + sell_vol;
    if (total == 0.0) return 0.0;
    return (buy_vol - sell_vol) / total;
}

std::vector<double> compute_bar_features(
    const std::vector<TradeBar>& bars,
    const std::vector<BarBookAccum>& accums,
    uint64_t rth_open_ns,
    uint64_t rth_close_ns)
{
    if (bars.size() != accums.size())
        throw std::invalid_argument("bars.size() != accums.size()");
    int n = static_cast<int>(bars.size());
    if (n == 0) return {};

    std::vector<double> out(n * N_FEATURES);
    double rth_duration = static_cast<double>(rth_close_ns - rth_open_ns);

    // Incremental realized vol: track running sum and sum-of-squares of log returns.
    // At bar i, there are i log-returns from closes[0..i].
    // population_var = sum_sq / count - (sum / count)^2
    double rv_sum = 0.0;
    double rv_sum_sq = 0.0;

    for (int i = 0; i < n; ++i) {
        const auto& bar = bars[i];
        const auto& acc = accums[i];
        double* f = &out[i * N_FEATURES];

        // Update incremental log-return stats (bar i adds return i-1 → i)
        if (i > 0) {
            double lr = std::log(bars[i].close / bars[i - 1].close);
            rv_sum += lr;
            rv_sum_sq += lr * lr;
        }

        // Col 0: Trade flow imbalance
        f[0] = compute_trade_flow_imbalance(bar);

        // Col 1: BBO imbalance = bid_qty / (bid_qty + ask_qty), default 0.5
        {
            double total = static_cast<double>(acc.bid_qty) + static_cast<double>(acc.ask_qty);
            f[1] = (total == 0.0) ? 0.5 : static_cast<double>(acc.bid_qty) / total;
        }

        // Col 2: Depth(5) imbalance = total_bid_5 / (total_bid_5 + total_ask_5), default 0.5
        {
            double total = static_cast<double>(acc.total_bid_5) + static_cast<double>(acc.total_ask_5);
            f[2] = (total == 0.0) ? 0.5 : static_cast<double>(acc.total_bid_5) / total;
        }

        // Col 3: Bar range in ticks
        double range_price = bar.high - bar.low;
        f[3] = range_price / TICK_SIZE;

        // Col 4: Bar body (signed) in ticks
        f[4] = (bar.close - bar.open) / TICK_SIZE;

        // Col 5: Body/range ratio, 0 if range==0
        f[5] = (range_price == 0.0) ? 0.0 : (bar.close - bar.open) / range_price;

        // Col 6: VWAP displacement = (close - vwap) / range, 0 if range==0
        f[6] = (range_price == 0.0) ? 0.0 : (bar.close - bar.vwap) / range_price;

        // Col 7: Log volume = log(max(volume, 1))
        f[7] = std::log(std::max(bar.volume, 1));

        // Col 8: Realized vol (population std of log returns over all closes[0..i])
        if (i < REALIZED_VOL_WARMUP) {
            f[8] = std::numeric_limits<double>::quiet_NaN();
        } else {
            // i log-returns available; population variance = E[x^2] - E[x]^2
            double count = static_cast<double>(i);
            double mean = rv_sum / count;
            double var = rv_sum_sq / count - mean * mean;
            // Guard against floating-point rounding producing tiny negatives
            f[8] = std::sqrt(std::max(var, 0.0));
        }

        // Col 9: Session time = clamp((t_end - rth_open) / rth_duration, 0, 1)
        {
            double elapsed = static_cast<double>(static_cast<int64_t>(bar.t_end) - static_cast<int64_t>(rth_open_ns));
            double t = elapsed / rth_duration;
            f[9] = std::clamp(t, 0.0, 1.0);
        }

        // Col 10: Cancel asymmetry = (bid_cancels - ask_cancels) / (total_cancels + 1e-10)
        {
            double total = static_cast<double>(acc.bid_cancels + acc.ask_cancels);
            f[10] = (acc.bid_cancels - acc.ask_cancels) / (total + 1e-10);
        }

        // Cols 11, 16: Spread mean and std (computed together to avoid iterating twice)
        {
            if (acc.spread_samples.empty()) {
                f[11] = 1.0;  // default mean spread
                f[16] = 0.0;  // < 2 samples → std = 0
            } else {
                double sum = 0.0;
                for (double s : acc.spread_samples) sum += s;
                double spread_n = static_cast<double>(acc.spread_samples.size());
                double spread_mean = sum / spread_n;
                f[11] = spread_mean;

                if (acc.spread_samples.size() < 2) {
                    f[16] = 0.0;
                } else {
                    double var = 0.0;
                    for (double s : acc.spread_samples) {
                        double d = s - spread_mean;
                        var += d * d;
                    }
                    var /= spread_n;  // population variance (ddof=0)
                    f[16] = std::sqrt(var);
                }
            }
        }

        // Col 12: Session age = min(bar_index / SESSION_AGE_PERIOD, 1.0)
        f[12] = std::min(static_cast<double>(bar.bar_index) / SESSION_AGE_PERIOD, 1.0);

        // Col 13: OFI = clamp(ofi_signed_volume / (total_add_volume + 1e-10), -1, 1)
        //         But if total_add_volume == 0, return 0.
        {
            if (acc.total_add_volume == 0.0 && acc.ofi_signed_volume == 0.0) {
                f[13] = 0.0;
            } else {
                double raw_ofi = acc.ofi_signed_volume / (acc.total_add_volume + 1e-10);
                f[13] = std::clamp(raw_ofi, -1.0, 1.0);
            }
        }

        // Col 14: Depth ratio = (total_bid_3 + total_ask_3) / (total_bid_10 + total_ask_10 + 1e-10)
        //         Default 0.5 when total_10 == 0
        {
            double total_10 = static_cast<double>(acc.total_bid_10) + static_cast<double>(acc.total_ask_10);
            if (total_10 == 0.0) {
                f[14] = 0.5;
            } else {
                double total_3 = static_cast<double>(acc.total_bid_3) + static_cast<double>(acc.total_ask_3);
                f[14] = total_3 / (total_10 + 1e-10);
            }
        }

        // Col 15: Weighted mid displacement = (wmid_end - wmid_first) / TICK_SIZE, 0 if either NaN
        {
            if (std::isnan(acc.wmid_first) || std::isnan(acc.wmid_end)) {
                f[15] = 0.0;
            } else {
                f[15] = (acc.wmid_end - acc.wmid_first) / TICK_SIZE;
            }
        }

        // Col 17: VAMP displacement = (vamp_at_end - vamp_at_mid) / TICK_SIZE, 0 if either NaN
        {
            if (std::isnan(acc.vamp_at_mid) || std::isnan(acc.vamp_at_end)) {
                f[17] = 0.0;
            } else {
                f[17] = (acc.vamp_at_end - acc.vamp_at_mid) / TICK_SIZE;
            }
        }

        // Col 18: Aggressor imbalance = (buy - sell) / (buy + sell + 1e-10), 0 if both zero
        {
            double total = acc.buy_aggressor_vol + acc.sell_aggressor_vol;
            if (total == 0.0) {
                f[18] = 0.0;
            } else {
                f[18] = (acc.buy_aggressor_vol - acc.sell_aggressor_vol) / (total + 1e-10);
            }
        }

        // Col 19: Trade arrival rate = log(1 + n_trades)
        f[19] = std::log(1.0 + acc.n_trades);

        // Col 20: Cancel-to-trade ratio = log(1 + n_cancels / max(n_trades, 1))
        {
            int denom = std::max(acc.n_trades, 1);
            f[20] = std::log(1.0 + static_cast<double>(acc.n_cancels) / denom);
        }

        // Col 21: Price impact per trade = (close - open) / (max(n_trades, 1) * TICK_SIZE)
        {
            int denom = std::max(acc.n_trades, 1);
            f[21] = (bar.close - bar.open) / (denom * TICK_SIZE);
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// normalize_features
// ---------------------------------------------------------------------------

std::vector<double> normalize_features(
    const std::vector<double>& raw,
    int n_rows,
    int n_cols,
    int window)
{
    // Replace NaN with 0 first
    std::vector<double> data(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        data[i] = std::isnan(raw[i]) ? 0.0 : raw[i];
    }

    std::vector<double> out(data.size());

    // For each row, compute z-score using an expanding or rolling window
    for (int row = 0; row < n_rows; ++row) {
        // Determine window: rows [start_row, row] inclusive
        int start_row;
        if (window > 0) {
            start_row = std::max(0, row - window + 1);
        } else {
            start_row = 0;
        }
        int n_window = row - start_row + 1;

        for (int col = 0; col < n_cols; ++col) {
            // Gather column values for the window
            double sum = 0.0;
            for (int r = start_row; r <= row; ++r) {
                sum += data[r * n_cols + col];
            }
            double mean = sum / n_window;

            double var = 0.0;
            for (int r = start_row; r <= row; ++r) {
                double d = data[r * n_cols + col] - mean;
                var += d * d;
            }
            var /= n_window;  // population variance (ddof=0)
            double std_dev = std::sqrt(var);

            double z;
            if (std_dev == 0.0) {
                z = 0.0;
            } else {
                z = (data[row * n_cols + col] - mean) / std_dev;
            }

            // Clip to [-5, 5]
            z = std::clamp(z, -5.0, 5.0);
            out[row * n_cols + col] = z;
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// assemble_lookback
// ---------------------------------------------------------------------------

std::vector<float> assemble_lookback(
    const std::vector<double>& normed,
    int n_rows,
    int n_cols,
    int h)
{
    if (n_rows < h) return {};

    int out_rows = n_rows - h + 1;
    int out_cols = n_cols * h;
    std::vector<float> out(out_rows * out_cols);

    for (int r = 0; r < out_rows; ++r) {
        for (int w = 0; w < h; ++w) {
            int src_row = r + w;
            for (int c = 0; c < n_cols; ++c) {
                out[r * out_cols + w * n_cols + c] =
                    static_cast<float>(normed[src_row * n_cols + c]);
            }
        }
    }

    return out;
}
