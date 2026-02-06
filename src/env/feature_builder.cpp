#include "lob/feature_builder.h"
#include <algorithm>
#include <cmath>

std::vector<float> FeatureBuilder::build(const Book& book, float position, float time_remaining) const {
    std::vector<float> obs(OBS_SIZE, 0.0f);

    double mid = book.mid_price();
    bool has_mid = std::isfinite(mid) && mid > 0.0;

    auto bids = book.top_bids(DEPTH);
    auto asks = book.top_asks(DEPTH);

    // Find max size across all 20 levels for normalization
    uint32_t max_size = 0;
    for (int i = 0; i < DEPTH; ++i) {
        max_size = std::max(max_size, bids[i].qty);
        max_size = std::max(max_size, asks[i].qty);
    }

    if (has_mid) {
        for (int i = 0; i < DEPTH; ++i) {
            if (!std::isnan(bids[i].price)) {
                obs[BID_PRICE + i] = static_cast<float>((bids[i].price - mid) / mid);
            }
        }
        for (int i = 0; i < DEPTH; ++i) {
            if (!std::isnan(asks[i].price)) {
                obs[ASK_PRICE + i] = static_cast<float>((asks[i].price - mid) / mid);
            }
        }
    }

    if (max_size > 0) {
        float fmax = static_cast<float>(max_size);
        for (int i = 0; i < DEPTH; ++i) {
            obs[BID_SIZE + i] = static_cast<float>(bids[i].qty) / fmax;
            obs[ASK_SIZE + i] = static_cast<float>(asks[i].qty) / fmax;
        }
    }

    if (has_mid) {
        double spread = book.spread();
        if (std::isfinite(spread)) {
            obs[SPREAD] = static_cast<float>(spread / mid);
        }

        uint32_t bid_qty_top = book.best_bid_qty();
        uint32_t ask_qty_top = book.best_ask_qty();
        uint32_t total = bid_qty_top + ask_qty_top;
        if (total > 0) {
            obs[IMBALANCE] = static_cast<float>(static_cast<int64_t>(bid_qty_top) - static_cast<int64_t>(ask_qty_top))
                     / static_cast<float>(total);
        }
    }

    obs[TIME_LEFT] = time_remaining;
    obs[POSITION] = position;

    return obs;
}
