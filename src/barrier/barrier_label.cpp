#include "lob/barrier/barrier_label.h"
#include <algorithm>

// Resolve a dual-breach (both upper and lower barriers hit in the same bar).
// Returns +1 (upper first), -1 (lower first).
static int resolve_tiebreak(const TradeBar& bar, double upper_barrier,
                            double lower_barrier, double prev_close) {
    const auto& prices = bar.trade_prices;

    // Gap direction fallback: use first trade if available, else open.
    auto gap_direction = [&]() -> int {
        double first = prices.empty() ? bar.open : prices[0];
        return (first >= prev_close) ? 1 : -1;
    };

    // No trade data → gap direction.
    if (prices.empty()) {
        return gap_direction();
    }

    // If first trade is already at/past either barrier → gap direction.
    double first_trade = prices[0];
    if (first_trade >= upper_barrier || first_trade <= lower_barrier) {
        return gap_direction();
    }

    // Scan trades sequentially to find which barrier is crossed first.
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i] >= upper_barrier) return 1;
        if (prices[i] <= lower_barrier) return -1;
    }

    // Shouldn't happen in a real dual breach, but fall back to gap direction.
    return gap_direction();
}

std::vector<BarrierLabel> compute_labels(
    const std::vector<TradeBar>& bars,
    int a, int b, int t_max, double tick_size) {

    const int n = static_cast<int>(bars.size());
    if (n == 0) return {};

    std::vector<BarrierLabel> labels(n);

    for (int k = 0; k < n; ++k) {
        double entry_price = bars[k].close;
        double upper_barrier = entry_price + a * tick_size;
        double lower_barrier = entry_price - b * tick_size;

        BarrierLabel lbl{};
        lbl.bar_index = k;

        int j_end = std::min(k + t_max, n - 1);
        bool resolved = false;

        for (int j = k + 1; j <= j_end; ++j) {
            bool upper_hit = bars[j].high >= upper_barrier;
            bool lower_hit = bars[j].low <= lower_barrier;

            int hit_label = 0;
            if (upper_hit && lower_hit) {
                hit_label = resolve_tiebreak(bars[j], upper_barrier,
                                             lower_barrier, bars[j - 1].close);
            } else if (upper_hit) {
                hit_label = 1;
            } else if (lower_hit) {
                hit_label = -1;
            }

            if (hit_label != 0) {
                lbl.label = hit_label;
                lbl.tau = j - k;
                lbl.resolution_bar = j;
                resolved = true;
                break;
            }
        }

        if (!resolved) {
            // Timeout.
            lbl.label = 0;
            int remaining = n - 1 - k;
            lbl.tau = std::max(1, std::min(t_max, remaining));
            lbl.resolution_bar = k + lbl.tau;
        }

        labels[k] = lbl;
    }

    return labels;
}
