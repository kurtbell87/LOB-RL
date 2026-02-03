#pragma once
#include "lob/book.h"
#include "lob/env.h"
#include <vector>

namespace lob {

class FeatureBuilder {
public:
    explicit FeatureBuilder(int book_depth = 10);

    Observation build(const Book& book, int position, double time_remaining) const;

    int observation_size() const { return Observation::size(book_depth_); }

private:
    int book_depth_;

    // Normalization constants
    static constexpr double kPriceScale = 1e-9;      // Convert fixed-point to dollars
    static constexpr double kTickSize = 0.25;        // /MES tick size in dollars
    static constexpr double kSizeScale = 0.01;       // Log normalize: log(1 + qty) / 100
};

}  // namespace lob
