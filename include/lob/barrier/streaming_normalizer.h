#pragma once
#include <deque>
#include <vector>

/// Online z-score normalizer that processes one bar at a time.
/// Produces output identical to calling normalize_features() on all bars
/// up to the current one and reading the last row.
class StreamingNormalizer {
public:
    /// @param n_features Number of feature columns per bar.
    /// @param window Rolling window size. 0 = expanding (use all bars).
    ///               Default = 2000.
    explicit StreamingNormalizer(int n_features, int window = 2000)
        : n_features_(n_features), window_(window), bars_seen_(0),
          history_(n_features) {}

    /// Feed one bar of raw features and return the z-scored result.
    /// raw_bar.size() must equal n_features().
    /// NaN values are replaced with 0.0 before computation.
    /// Output is clipped to [-5, 5].
    std::vector<double> normalize(const std::vector<double>& raw_bar);

    /// Number of feature columns.
    int n_features() const { return n_features_; }

    /// Number of bars fed so far.
    int bars_seen() const { return bars_seen_; }

    /// Clear all internal state (history, counters).
    void reset();

private:
    int n_features_;
    int window_;        // 0 = expanding
    int bars_seen_;

    // Per-column history deque for rolling window support.
    std::vector<std::deque<double>> history_;
};
