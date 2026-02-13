/**
 * Tests for StreamingNormalizer (Phase 9: Streaming Feature Normalization).
 *
 * Spec: docs/streaming-normalizer.md
 *
 * The StreamingNormalizer normalizes features online as bars arrive one at a
 * time, producing identical output to calling normalize_features() on all bars
 * up to the current one and reading the last row.
 *
 * Categories:
 *   1. Construction & accessors (~3)
 *   2. Single-bar normalization (~2)
 *   3. Multi-bar z-score correctness (~3)
 *   4. Rolling window behavior (~3)
 *   5. Bit-exact regression vs normalize_features() (~4)
 *   6. Edge cases (~4)
 *   7. Reset behavior (~2)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "lob/barrier/streaming_normalizer.h"
#include "lob/barrier/feature_compute.h"  // normalize_features() reference

// ===========================================================================
// Section 1: Construction & Accessors
// ===========================================================================

TEST(StreamingNormalizer, ConstructWithFeatureCountAndDefaultWindow) {
    StreamingNormalizer norm(22);
    EXPECT_EQ(norm.n_features(), 22);
    EXPECT_EQ(norm.bars_seen(), 0);
}

TEST(StreamingNormalizer, ConstructWithCustomWindow) {
    StreamingNormalizer norm(5, 100);
    EXPECT_EQ(norm.n_features(), 5);
    EXPECT_EQ(norm.bars_seen(), 0);
}

TEST(StreamingNormalizer, ConstructWithExpandingWindow) {
    // window=0 means expanding window (all bars)
    StreamingNormalizer norm(3, 0);
    EXPECT_EQ(norm.n_features(), 3);
    EXPECT_EQ(norm.bars_seen(), 0);
}

// ===========================================================================
// Section 2: Single-bar normalization
// ===========================================================================

TEST(StreamingNormalizer, FirstBarAllZeros) {
    // First bar: mean = raw value, variance = 0, z-score = 0 for all features.
    StreamingNormalizer norm(3);
    auto z = norm.normalize({1.0, 2.0, 3.0});
    ASSERT_EQ(z.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(z[i], 0.0)
            << "First bar → std=0 → z=0 for feature " << i;
    }
    EXPECT_EQ(norm.bars_seen(), 1);
}

TEST(StreamingNormalizer, SecondBarCorrectZScores) {
    // Two bars [1.0, 3.0] (1 column):
    // Bar 0: window=[1.0], mean=1, std=0 → z=0
    // Bar 1: window=[1.0, 3.0], mean=2, std=1 → z=(3-2)/1=1.0
    StreamingNormalizer norm(1);
    auto z0 = norm.normalize({1.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);

    auto z1 = norm.normalize({3.0});
    EXPECT_NEAR(z1[0], 1.0, 1e-12);
    EXPECT_EQ(norm.bars_seen(), 2);
}

// ===========================================================================
// Section 3: Multi-bar z-score correctness
// ===========================================================================

TEST(StreamingNormalizer, ThreeBarSequenceSingleColumn) {
    // Values [1.0, 3.0, 5.0], expanding window:
    // Bar 0: window=[1] → mean=1, std=0 → z=0
    // Bar 1: window=[1,3] → mean=2, std=1 → z=(3-2)/1=1.0
    // Bar 2: window=[1,3,5] → mean=3, std=sqrt(8/3) → z=(5-3)/sqrt(8/3)
    StreamingNormalizer norm(1, 0);  // expanding window

    auto z0 = norm.normalize({1.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);

    auto z1 = norm.normalize({3.0});
    EXPECT_NEAR(z1[0], 1.0, 1e-12);

    double expected_std = std::sqrt(8.0 / 3.0);
    double expected_z2 = (5.0 - 3.0) / expected_std;
    auto z2 = norm.normalize({5.0});
    EXPECT_NEAR(z2[0], expected_z2, 1e-12);
}

TEST(StreamingNormalizer, MultiColumnIndependence) {
    // 3 bars, 2 columns — each column normalized independently.
    // Col 0: [10, 20, 30], Col 1: [100, 100, 100]
    StreamingNormalizer norm(2, 0);

    auto z0 = norm.normalize({10.0, 100.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);  // first bar → 0
    EXPECT_DOUBLE_EQ(z0[1], 0.0);

    auto z1 = norm.normalize({20.0, 100.0});
    EXPECT_NEAR(z1[0], 1.0, 1e-12);   // (20-15)/5=1.0
    EXPECT_DOUBLE_EQ(z1[1], 0.0);      // constant → std=0 → z=0

    auto z2 = norm.normalize({30.0, 100.0});
    // Col 0: window=[10,20,30], mean=20, var=((10-20)^2+(20-20)^2+(30-20)^2)/3=200/3
    // std=sqrt(200/3), z=(30-20)/sqrt(200/3)
    double std0 = std::sqrt(200.0 / 3.0);
    EXPECT_NEAR(z2[0], (30.0 - 20.0) / std0, 1e-12);
    EXPECT_DOUBLE_EQ(z2[1], 0.0);  // still constant
}

TEST(StreamingNormalizer, NegativeAndMixedValues) {
    // Values [-5, 0, 5] (1 column, expanding):
    // Bar 0: z=0
    // Bar 1: window=[-5,0], mean=-2.5, std=2.5, z=(0-(-2.5))/2.5=1.0
    // Bar 2: window=[-5,0,5], mean=0, var=(25+0+25)/3, std=sqrt(50/3), z=5/sqrt(50/3)
    StreamingNormalizer norm(1, 0);

    auto z0 = norm.normalize({-5.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);

    auto z1 = norm.normalize({0.0});
    EXPECT_NEAR(z1[0], 1.0, 1e-12);

    double std2 = std::sqrt(50.0 / 3.0);
    auto z2 = norm.normalize({5.0});
    EXPECT_NEAR(z2[0], 5.0 / std2, 1e-12);
}

// ===========================================================================
// Section 4: Rolling window behavior
// ===========================================================================

TEST(StreamingNormalizer, Window1AlwaysZero) {
    // window=1 → each bar sees only itself → mean=x, std=0 → z=0
    StreamingNormalizer norm(1, 1);

    auto z0 = norm.normalize({42.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);

    auto z1 = norm.normalize({-100.0});
    EXPECT_DOUBLE_EQ(z1[0], 0.0);

    auto z2 = norm.normalize({999.0});
    EXPECT_DOUBLE_EQ(z2[0], 0.0);
    EXPECT_EQ(norm.bars_seen(), 3);
}

TEST(StreamingNormalizer, SmallWindowDropsOldBars) {
    // window=3: after 5 bars, only uses last 3.
    // Feed [10, 20, 30, 40, 50], window=3
    // Bar 4: window=[30,40,50], mean=40, var=((30-40)^2+(40-40)^2+(50-40)^2)/3=200/3
    // std=sqrt(200/3), z=(50-40)/sqrt(200/3)
    StreamingNormalizer norm(1, 3);

    norm.normalize({10.0});
    norm.normalize({20.0});
    norm.normalize({30.0});
    norm.normalize({40.0});
    auto z4 = norm.normalize({50.0});

    double expected_std = std::sqrt(200.0 / 3.0);
    double expected_z = (50.0 - 40.0) / expected_std;
    EXPECT_NEAR(z4[0], expected_z, 1e-12);
    EXPECT_EQ(norm.bars_seen(), 5);
}

TEST(StreamingNormalizer, WindowExactlyMatchesBarCount) {
    // window=5 with exactly 5 bars → uses all bars (same as expanding).
    // Feed [2, 4, 6, 8, 10], window=5
    StreamingNormalizer norm_win(1, 5);
    StreamingNormalizer norm_exp(1, 0);

    std::vector<double> vals = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> z_win, z_exp;
    for (double v : vals) {
        z_win = norm_win.normalize({v});
        z_exp = norm_exp.normalize({v});
    }
    // At bar 4 (5th bar), both should give identical results
    EXPECT_NEAR(z_win[0], z_exp[0], 1e-12)
        << "Window=N with N bars should match expanding window";
}

// ===========================================================================
// Section 5: Bit-exact regression vs normalize_features()
// ===========================================================================

TEST(StreamingNormalizer, BitExactVsReferenceExpandingWindow) {
    // Generate 50 bars × 3 columns with deterministic data.
    // Compare streaming output bar-by-bar against normalize_features().
    const int n_bars = 50;
    const int n_cols = 3;
    std::vector<double> all_raw;
    all_raw.reserve(n_bars * n_cols);

    for (int i = 0; i < n_bars; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            // Deterministic pseudo-data: different scales per column
            all_raw.push_back(std::sin(i * 0.1 + j) * (j + 1) * 100.0);
        }
    }

    // Reference: normalize all at once
    auto ref = normalize_features(all_raw, n_bars, n_cols, /*window=*/0);

    // Streaming: feed one bar at a time
    StreamingNormalizer norm(n_cols, 0);
    for (int i = 0; i < n_bars; ++i) {
        std::vector<double> bar(all_raw.begin() + i * n_cols,
                                all_raw.begin() + (i + 1) * n_cols);
        auto z = norm.normalize(bar);
        ASSERT_EQ(z.size(), static_cast<size_t>(n_cols));
        for (int j = 0; j < n_cols; ++j) {
            EXPECT_NEAR(z[j], ref[i * n_cols + j], 1e-12)
                << "Mismatch at bar=" << i << " col=" << j;
        }
    }
}

TEST(StreamingNormalizer, BitExactVsReferenceRollingWindow5) {
    // 20 bars × 2 columns, window=5.
    const int n_bars = 20;
    const int n_cols = 2;
    std::vector<double> all_raw;
    all_raw.reserve(n_bars * n_cols);

    for (int i = 0; i < n_bars; ++i) {
        all_raw.push_back(i * 3.0 + 1.0);           // col 0: linear ramp
        all_raw.push_back(std::cos(i * 0.5) * 50.0); // col 1: cosine
    }

    auto ref = normalize_features(all_raw, n_bars, n_cols, /*window=*/5);

    StreamingNormalizer norm(n_cols, 5);
    for (int i = 0; i < n_bars; ++i) {
        std::vector<double> bar = {all_raw[i * n_cols], all_raw[i * n_cols + 1]};
        auto z = norm.normalize(bar);
        ASSERT_EQ(z.size(), static_cast<size_t>(n_cols));
        for (int j = 0; j < n_cols; ++j) {
            EXPECT_NEAR(z[j], ref[i * n_cols + j], 1e-12)
                << "Mismatch at bar=" << i << " col=" << j;
        }
    }
}

TEST(StreamingNormalizer, BitExactVsReferenceDefaultWindow2000) {
    // 100 bars × 4 columns, window=2000 (default).
    // With only 100 bars, window=2000 is effectively expanding.
    const int n_bars = 100;
    const int n_cols = 4;
    std::vector<double> all_raw;
    all_raw.reserve(n_bars * n_cols);

    for (int i = 0; i < n_bars; ++i) {
        all_raw.push_back(i * 1.5);                       // col 0: linear
        all_raw.push_back(100.0 - i * 0.8);               // col 1: decreasing
        all_raw.push_back(std::sin(i * 0.3) * 200.0);     // col 2: oscillating
        all_raw.push_back(42.0);                           // col 3: constant
    }

    auto ref = normalize_features(all_raw, n_bars, n_cols, /*window=*/2000);

    StreamingNormalizer norm(n_cols, 2000);
    for (int i = 0; i < n_bars; ++i) {
        std::vector<double> bar(all_raw.begin() + i * n_cols,
                                all_raw.begin() + (i + 1) * n_cols);
        auto z = norm.normalize(bar);
        ASSERT_EQ(z.size(), static_cast<size_t>(n_cols));
        for (int j = 0; j < n_cols; ++j) {
            EXPECT_NEAR(z[j], ref[i * n_cols + j], 1e-12)
                << "Mismatch at bar=" << i << " col=" << j;
        }
    }
}

TEST(StreamingNormalizer, BitExactVsReferenceWithNaNs) {
    // Regression test including NaN values.
    // 10 bars × 2 columns, some NaNs sprinkled in.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const int n_bars = 10;
    const int n_cols = 2;
    std::vector<double> all_raw = {
        nan,  5.0,   // bar 0
        3.0,  nan,   // bar 1
        7.0,  2.0,   // bar 2
        nan,  nan,   // bar 3
        1.0,  8.0,   // bar 4
        4.0,  4.0,   // bar 5
        9.0,  1.0,   // bar 6
        nan,  6.0,   // bar 7
        2.0,  3.0,   // bar 8
        5.0,  nan,   // bar 9
    };

    auto ref = normalize_features(all_raw, n_bars, n_cols, /*window=*/0);

    StreamingNormalizer norm(n_cols, 0);
    for (int i = 0; i < n_bars; ++i) {
        std::vector<double> bar = {all_raw[i * n_cols], all_raw[i * n_cols + 1]};
        auto z = norm.normalize(bar);
        ASSERT_EQ(z.size(), static_cast<size_t>(n_cols));
        for (int j = 0; j < n_cols; ++j) {
            EXPECT_NEAR(z[j], ref[i * n_cols + j], 1e-12)
                << "NaN regression: bar=" << i << " col=" << j;
        }
    }
}

// ===========================================================================
// Section 6: Edge cases
// ===========================================================================

TEST(StreamingNormalizer, NaNReplacedWithZero) {
    // NaN values should be replaced with 0 before any computation.
    double nan = std::numeric_limits<double>::quiet_NaN();
    StreamingNormalizer norm(2, 0);

    // Bar 0: [NaN, 5.0] → treated as [0.0, 5.0] → first bar → z=[0, 0]
    auto z0 = norm.normalize({nan, 5.0});
    EXPECT_DOUBLE_EQ(z0[0], 0.0);
    EXPECT_DOUBLE_EQ(z0[1], 0.0);

    // Bar 1: [2.0, NaN] → treated as [2.0, 0.0]
    // Col 0: window=[0.0, 2.0], mean=1, std=1, z=(2-1)/1=1.0
    // Col 1: window=[5.0, 0.0], mean=2.5, std=2.5, z=(0-2.5)/2.5=-1.0
    auto z1 = norm.normalize({2.0, nan});
    EXPECT_NEAR(z1[0], 1.0, 1e-12);
    EXPECT_NEAR(z1[1], -1.0, 1e-12);
}

TEST(StreamingNormalizer, FeatureCountMismatchThrows) {
    StreamingNormalizer norm(3);
    EXPECT_THROW(norm.normalize({1.0, 2.0}), std::exception)
        << "Should throw when raw_bar.size() != n_features (too few)";
    EXPECT_THROW(norm.normalize({1.0, 2.0, 3.0, 4.0}), std::exception)
        << "Should throw when raw_bar.size() != n_features (too many)";
}

TEST(StreamingNormalizer, ClippingToMinusFivePlusFive) {
    // Create extreme outlier scenario: many identical values then one outlier.
    // 50 bars of value 0.0, then one bar of 10000.0
    StreamingNormalizer norm(1, 0);

    for (int i = 0; i < 50; ++i) {
        norm.normalize({0.0});
    }
    // Bar 50: window=[0,...,0,10000], 51 values
    // mean=10000/51≈196.08, var=(50*(196.08)^2+(10000-196.08)^2)/51
    // z = (10000-196.08)/std should be >> 5 → clipped to 5.0
    auto z = norm.normalize({10000.0});
    EXPECT_DOUBLE_EQ(z[0], 5.0)
        << "Extreme positive z-score should be clipped to 5.0";

    // Verify the negative clip works too (with a separate normalizer)
    StreamingNormalizer norm2(1, 0);
    for (int i = 0; i < 50; ++i) {
        norm2.normalize({0.0});
    }
    auto z_neg = norm2.normalize({-10000.0});
    EXPECT_DOUBLE_EQ(z_neg[0], -5.0)
        << "Extreme negative z-score should be clipped to -5.0";
}

TEST(StreamingNormalizer, AllNaNBarProducesZeros) {
    // All-NaN bar treated as all-zero after replacement.
    double nan = std::numeric_limits<double>::quiet_NaN();
    StreamingNormalizer norm(3, 0);

    auto z0 = norm.normalize({nan, nan, nan});
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(z0[i], 0.0);
    }
    EXPECT_EQ(norm.bars_seen(), 1);
}

// ===========================================================================
// Section 7: Reset behavior
// ===========================================================================

TEST(StreamingNormalizer, ResetClearsState) {
    StreamingNormalizer norm(1, 0);

    norm.normalize({10.0});
    norm.normalize({20.0});
    EXPECT_EQ(norm.bars_seen(), 2);

    norm.reset();
    EXPECT_EQ(norm.bars_seen(), 0);

    // After reset, first bar again → z=0
    auto z = norm.normalize({10.0});
    EXPECT_DOUBLE_EQ(z[0], 0.0);
    EXPECT_EQ(norm.bars_seen(), 1);
}

TEST(StreamingNormalizer, ResetThenDifferentSequenceMatchesReference) {
    // Feed sequence A, reset, feed sequence B.
    // Result should match normalize_features() on B alone.
    StreamingNormalizer norm(1, 0);

    // Sequence A
    norm.normalize({100.0});
    norm.normalize({200.0});
    norm.normalize({300.0});

    norm.reset();

    // Sequence B
    std::vector<double> seq_b = {5.0, 10.0, 15.0, 20.0};
    auto ref = normalize_features(seq_b, 4, 1, /*window=*/0);

    for (int i = 0; i < 4; ++i) {
        auto z = norm.normalize({seq_b[i]});
        EXPECT_NEAR(z[0], ref[i], 1e-12)
            << "After reset, bar " << i << " should match reference on new sequence";
    }
}
