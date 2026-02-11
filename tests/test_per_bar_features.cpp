#include <gtest/gtest.h>
#include "lob/barrier/feature_compute.h"
#include "test_barrier_helpers.h"

// ===========================================================================
// Section 1: bar_features field existence and default state (~2)
// ===========================================================================

TEST(PerBarFeatures, DefaultConstructionHasEmptyBarFeatures) {
    BarrierPrecomputedDay day{};
    EXPECT_TRUE(day.bar_features.empty())
        << "bar_features should be empty in default-constructed BarrierPrecomputedDay";
    EXPECT_EQ(day.n_trimmed, 0)
        << "n_trimmed should be 0 in default-constructed BarrierPrecomputedDay";
}

TEST(PerBarFeatures, FieldsAccessibleOnStruct) {
    // Verify the new fields exist and can be assigned
    BarrierPrecomputedDay day{};
    day.n_trimmed = 42;
    day.bar_features.resize(42 * N_FEATURES, 1.0f);
    EXPECT_EQ(day.n_trimmed, 42);
    EXPECT_EQ(static_cast<int>(day.bar_features.size()), 42 * N_FEATURES);
}

// ===========================================================================
// Section 2: bar_features shape (n_trimmed, N_FEATURES) (~3)
// ===========================================================================

TEST(PerBarFeatures, BarFeaturesPopulatedWithCorrectSize) {
    // With enough bars, bar_features should have n_trimmed * N_FEATURES elements
    int num_bars = 30;
    int lookback = 3;
    BarrierPrecomputedDay day = run_barrier_precompute(5, num_bars, lookback);

    // n_trimmed = n_bars - REALIZED_VOL_WARMUP = 30 - 19 = 11
    int expected_trimmed = num_bars - REALIZED_VOL_WARMUP;
    ASSERT_GT(expected_trimmed, 0);

    EXPECT_EQ(day.n_trimmed, expected_trimmed)
        << "n_trimmed should equal n_bars - REALIZED_VOL_WARMUP";

    int expected_size = expected_trimmed * N_FEATURES;
    EXPECT_EQ(static_cast<int>(day.bar_features.size()), expected_size)
        << "bar_features should have n_trimmed * N_FEATURES elements"
        << " (" << expected_trimmed << " * " << N_FEATURES << " = " << expected_size << ")";
}

TEST(PerBarFeatures, NTrimmedEqualsNBarsMinusWarmup) {
    // Test the exact formula: n_trimmed = n_bars - REALIZED_VOL_WARMUP
    int bar_size = 5;

    // Test several different bar counts
    for (int num_bars : {25, 30, 40, 50}) {
        BarrierPrecomputedDay day = run_barrier_precompute(bar_size, num_bars);
        int expected = num_bars - REALIZED_VOL_WARMUP;
        EXPECT_EQ(day.n_trimmed, expected)
            << "For num_bars=" << num_bars
            << ", n_trimmed should be " << expected;
    }
}

TEST(PerBarFeatures, BarFeaturesRowCountMatchesNTrimmed) {
    // Verify row count: bar_features.size() / N_FEATURES == n_trimmed
    int num_bars = 35;
    BarrierPrecomputedDay day = run_barrier_precompute(5, num_bars);

    ASSERT_GT(day.n_trimmed, 0);
    ASSERT_FALSE(day.bar_features.empty());

    int inferred_rows = static_cast<int>(day.bar_features.size()) / N_FEATURES;
    EXPECT_EQ(inferred_rows, day.n_trimmed)
        << "bar_features.size() / N_FEATURES should equal n_trimmed";

    // Also verify no remainder
    EXPECT_EQ(static_cast<int>(day.bar_features.size()) % N_FEATURES, 0)
        << "bar_features.size() must be evenly divisible by N_FEATURES";
}

// ===========================================================================
// Section 3: bar_features values are normalized and clipped (~3)
// ===========================================================================

TEST(PerBarFeatures, AllValuesAreFinite) {
    BarrierPrecomputedDay day = run_barrier_precompute(5, 30);
    ASSERT_GT(day.n_trimmed, 0);
    EXPECT_ALL_FINITE(day.bar_features);
}

TEST(PerBarFeatures, ValuesAreFloat32InClippedRange) {
    // After z-score normalization with [-5, 5] clipping, all values should be in [-5, 5]
    BarrierPrecomputedDay day = run_barrier_precompute(5, 30);
    ASSERT_GT(day.n_trimmed, 0);

    for (size_t i = 0; i < day.bar_features.size(); ++i) {
        float v = day.bar_features[i];
        EXPECT_GE(v, -5.0f) << "bar_features[" << i << "] = " << v << " below -5";
        EXPECT_LE(v, 5.0f) << "bar_features[" << i << "] = " << v << " above 5";
    }
}

TEST(PerBarFeatures, NotAllZeros) {
    // The normalized features should not be trivially zero everywhere
    BarrierPrecomputedDay day = run_barrier_precompute(5, 30);
    ASSERT_GT(day.n_trimmed, 0);

    bool any_nonzero = false;
    for (float v : day.bar_features) {
        if (v != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_nonzero)
        << "bar_features should have at least some non-zero values";
}

// ===========================================================================
// Section 4: Edge cases (~3)
// ===========================================================================

TEST(PerBarFeatures, EmptyStreamProducesEmptyBarFeatures) {
    ScriptedSource source({});
    SessionConfig cfg = SessionConfig::default_rth();
    BarrierPrecomputedDay day = barrier_precompute(source, cfg);

    EXPECT_EQ(day.n_trimmed, 0);
    EXPECT_TRUE(day.bar_features.empty());
}

TEST(PerBarFeatures, InsufficientBarsForWarmupProducesEmpty) {
    // When n_bars <= REALIZED_VOL_WARMUP (19), no trimmed features
    int bar_size = 5;
    int num_bars = 18;  // Less than 19

    BarrierPrecomputedDay day = run_barrier_precompute(bar_size, num_bars);

    EXPECT_EQ(day.n_bars, num_bars);
    EXPECT_EQ(day.n_trimmed, 0)
        << "With " << num_bars << " bars (< " << REALIZED_VOL_WARMUP
        << " warmup), n_trimmed should be 0";
    EXPECT_TRUE(day.bar_features.empty())
        << "bar_features should be empty when n_bars <= REALIZED_VOL_WARMUP";
}

TEST(PerBarFeatures, ExactlyWarmupPlusOneBarsProducesOneRow) {
    // n_bars = REALIZED_VOL_WARMUP + 1 = 20 → n_trimmed = 1
    int bar_size = 5;
    int num_bars = REALIZED_VOL_WARMUP + 1;  // 20

    BarrierPrecomputedDay day = run_barrier_precompute(bar_size, num_bars);

    EXPECT_EQ(day.n_bars, num_bars);
    EXPECT_EQ(day.n_trimmed, 1)
        << "With exactly REALIZED_VOL_WARMUP + 1 bars, n_trimmed should be 1";
    EXPECT_EQ(static_cast<int>(day.bar_features.size()), N_FEATURES)
        << "bar_features should have exactly N_FEATURES elements for 1 row";
}

// ===========================================================================
// Section 5: Critical alignment test — bar_features matches lookback features (~2)
// ===========================================================================
// Acceptance criterion #7: bar_features[i] should match the most-recent bar's
// features in the corresponding lookback row features[i - lookback + 1, 0:N_FEATURES].
// More precisely: the last N_FEATURES columns of each lookback row should match
// the corresponding bar_features row.

TEST(PerBarFeatures, BarFeaturesAlignWithLookbackFeatures) {
    // This is THE critical alignment test from the spec.
    // bar_features[i] should match features[j]'s last N_FEATURES columns,
    // where j = i - (lookback - 1) maps bar_features row i to features row j.
    //
    // Lookback assembly creates: features[j] = [bar[j], bar[j+1], ..., bar[j+lookback-1]]
    // So features[j]'s LAST N_FEATURES columns = bar_features[j + lookback - 1].
    // Equivalently: bar_features[i] = features[i - lookback + 1]'s last N_FEATURES cols,
    //               for i >= lookback - 1 (which corresponds to features rows 0..n_usable-1).

    int lookback = 3;
    int num_bars = 30;
    BarrierPrecomputedDay day = run_barrier_precompute(5, num_bars, lookback);

    ASSERT_GT(day.n_usable, 0);
    ASSERT_GT(day.n_trimmed, 0);

    int feat_cols = N_FEATURES * lookback;

    for (int j = 0; j < day.n_usable; ++j) {
        // features row j: [bar[j], bar[j+1], ..., bar[j+lookback-1]]
        // Last N_FEATURES columns correspond to bar_features[j + lookback - 1]
        int bar_idx = j + lookback - 1;  // index into bar_features

        ASSERT_LT(bar_idx, day.n_trimmed)
            << "bar_idx " << bar_idx << " must be < n_trimmed " << day.n_trimmed;

        for (int c = 0; c < N_FEATURES; ++c) {
            float from_lookback = day.features[j * feat_cols + (lookback - 1) * N_FEATURES + c];
            float from_bar_feat = day.bar_features[bar_idx * N_FEATURES + c];
            EXPECT_FLOAT_EQ(from_lookback, from_bar_feat)
                << "Mismatch at features row " << j << " col " << c
                << ": lookback last-bar=" << from_lookback
                << " vs bar_features[" << bar_idx << "][" << c << "]=" << from_bar_feat;
        }
    }
}

TEST(PerBarFeatures, BarFeaturesFirstRowAlsoAligns) {
    // The FIRST N_FEATURES columns of features[0] should equal bar_features[0],
    // because features[0] = [bar[0], bar[1], ..., bar[lookback-1]].
    int lookback = 3;
    int num_bars = 30;
    BarrierPrecomputedDay day = run_barrier_precompute(5, num_bars, lookback);

    ASSERT_GT(day.n_usable, 0);
    ASSERT_GT(day.n_trimmed, 0);

    int feat_cols = N_FEATURES * lookback;

    // features[0]'s first N_FEATURES columns = bar_features[0]
    for (int c = 0; c < N_FEATURES; ++c) {
        float from_lookback = day.features[0 * feat_cols + 0 * N_FEATURES + c];
        float from_bar_feat = day.bar_features[0 * N_FEATURES + c];
        EXPECT_FLOAT_EQ(from_lookback, from_bar_feat)
            << "Mismatch at first-row first-window col " << c
            << ": lookback=" << from_lookback
            << " vs bar_features[0][" << c << "]=" << from_bar_feat;
    }
}

// ===========================================================================
// Section 6: bar_features independent of lookback parameter (~1)
// ===========================================================================

TEST(PerBarFeatures, BarFeaturesIdenticalAcrossLookbackValues) {
    // bar_features should be the same regardless of lookback, because they
    // are per-bar normalized features captured BEFORE lookback assembly.
    int bar_size = 5;
    int num_bars = 35;

    auto msgs = make_barrier_stream(bar_size, num_bars);

    // lookback = 3
    ScriptedSource source1(msgs);
    SessionConfig cfg = SessionConfig::default_rth();
    BarrierPrecomputedDay day1 = barrier_precompute(source1, cfg, bar_size, /*lookback=*/3);

    // lookback = 5
    ScriptedSource source2(msgs);
    BarrierPrecomputedDay day2 = barrier_precompute(source2, cfg, bar_size, /*lookback=*/5);

    // Both should have the same n_trimmed and bar_features
    EXPECT_EQ(day1.n_trimmed, day2.n_trimmed)
        << "n_trimmed should be identical for different lookback values";

    ASSERT_EQ(day1.bar_features.size(), day2.bar_features.size());

    for (size_t i = 0; i < day1.bar_features.size(); ++i) {
        EXPECT_FLOAT_EQ(day1.bar_features[i], day2.bar_features[i])
            << "bar_features[" << i << "] should be identical across lookback values";
    }
}
