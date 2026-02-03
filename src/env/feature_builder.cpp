#include "feature_builder.h"
#include <cmath>
#include <algorithm>

namespace lob {

FeatureBuilder::FeatureBuilder(int book_depth)
    : book_depth_(book_depth) {}

Observation FeatureBuilder::build(const Book& book, int position, double time_remaining) const {
    Observation obs;
    obs.data.resize(observation_size(), 0.0f);

    int64_t mid = book.mid_price();
    double mid_price = static_cast<double>(mid) * kPriceScale;

    // If no mid price (empty book), return zeros
    if (mid == 0) {
        obs.data[4 * book_depth_ + 3] = static_cast<float>(position);
        return obs;
    }

    // Bid prices (relative to mid, normalized by tick size)
    for (int i = 0; i < book_depth_; ++i) {
        Level level = book.bid(i);
        if (level.quantity > 0) {
            double price = static_cast<double>(level.price) * kPriceScale;
            double relative = (price - mid_price) / kTickSize;
            obs.data[i] = static_cast<float>(relative);
        }
    }

    // Bid sizes (log normalized)
    for (int i = 0; i < book_depth_; ++i) {
        Level level = book.bid(i);
        if (level.quantity > 0) {
            double size = std::log1p(static_cast<double>(level.quantity)) * kSizeScale;
            obs.data[book_depth_ + i] = static_cast<float>(size);
        }
    }

    // Ask prices (relative to mid, normalized by tick size)
    for (int i = 0; i < book_depth_; ++i) {
        Level level = book.ask(i);
        if (level.quantity > 0) {
            double price = static_cast<double>(level.price) * kPriceScale;
            double relative = (price - mid_price) / kTickSize;
            obs.data[2 * book_depth_ + i] = static_cast<float>(relative);
        }
    }

    // Ask sizes (log normalized)
    for (int i = 0; i < book_depth_; ++i) {
        Level level = book.ask(i);
        if (level.quantity > 0) {
            double size = std::log1p(static_cast<double>(level.quantity)) * kSizeScale;
            obs.data[3 * book_depth_ + i] = static_cast<float>(size);
        }
    }

    // Spread (normalized by tick size)
    double spread = static_cast<double>(book.spread()) * kPriceScale / kTickSize;
    obs.data[4 * book_depth_] = static_cast<float>(spread);

    // Imbalance [-1, 1]
    obs.data[4 * book_depth_ + 1] = static_cast<float>(book.imbalance(book_depth_));

    // Time remaining [0, 1]
    obs.data[4 * book_depth_ + 2] = static_cast<float>(std::clamp(time_remaining, 0.0, 1.0));

    // Position {-1, 0, 1}
    obs.data[4 * book_depth_ + 3] = static_cast<float>(position);

    return obs;
}

}  // namespace lob
