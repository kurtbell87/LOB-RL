#include "lob/barrier/streaming_normalizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

std::vector<double> StreamingNormalizer::normalize(const std::vector<double>& raw_bar) {
    if (static_cast<int>(raw_bar.size()) != n_features_) {
        throw std::invalid_argument(
            "raw_bar.size()=" + std::to_string(raw_bar.size()) +
            " != n_features=" + std::to_string(n_features_));
    }

    // Replace NaN with 0 and push into per-column history
    for (int j = 0; j < n_features_; ++j) {
        double val = std::isnan(raw_bar[j]) ? 0.0 : raw_bar[j];
        history_[j].push_back(val);

        // Trim to window size if rolling (at most one excess element per call)
        if (window_ > 0 && static_cast<int>(history_[j].size()) > window_) {
            history_[j].pop_front();
        }
    }

    ++bars_seen_;

    // Compute z-scores for the current (last) value in each column
    std::vector<double> z(n_features_);
    for (int j = 0; j < n_features_; ++j) {
        const auto& col = history_[j];
        int n = static_cast<int>(col.size());

        // Compute mean
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += col[i];
        }
        double mean = sum / n;

        // Compute population variance
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = col[i] - mean;
            var += d * d;
        }
        var /= n;
        double std_dev = std::sqrt(var);

        if (std_dev == 0.0) {
            z[j] = 0.0;
        } else {
            z[j] = (col.back() - mean) / std_dev;
        }

        // Clip to [-5, 5]
        z[j] = std::clamp(z[j], -5.0, 5.0);
    }

    return z;
}

void StreamingNormalizer::reset() {
    bars_seen_ = 0;
    for (auto& col : history_) {
        col.clear();
    }
}
