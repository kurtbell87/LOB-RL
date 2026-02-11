#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include "lob/barrier/barrier_precompute.h"
#include "lob/barrier/feature_compute.h"
#include "lob/barrier/trade_bar.h"
#include "lob/barrier/bar_builder.h"
#include "lob/message.h"
#include "lob/session.h"
#include "lob/source.h"
#include "test_helpers.h"

// ===========================================================================
// Helpers: Build synthetic MBO streams for barrier_precompute() tests
// ===========================================================================

static constexpr double TICK = 0.25;

// Build a single trade message at a given price/qty/timestamp.
static Message make_trade_msg(uint64_t id, double price, uint32_t qty,
                              uint64_t ts, Message::Side side = Message::Side::Ask) {
    return make_msg(id, side, Message::Action::Trade, price, qty, ts);
}

// Build an MBO stream that produces a known number of complete bars.
// Includes pre-market book warmup + RTH trades.
// Returns the message vector.
static std::vector<Message> make_precompute_stream(int bar_size, int num_bars,
                                                    int partial_trades = 0) {
    std::vector<Message> msgs;
    uint64_t next_id = 1;
    double mid = 4000.0;

    // Pre-market: build book with 2 bid levels + 2 ask levels
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - TICK, 100, pre_ts));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + TICK, 100, pre_ts + NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - 2 * TICK, 200, pre_ts + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + 2 * TICK, 200, pre_ts + 3 * NS_PER_MIN));
    // Add deeper levels for multi-level depth features
    for (int i = 3; i <= 10; ++i) {
        msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                mid - i * TICK, 100, pre_ts + (2 * i) * NS_PER_MIN));
        msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                mid + i * TICK, 100, pre_ts + (2 * i + 1) * NS_PER_MIN));
    }

    // RTH: trades to fill bars
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    int total_trades = num_bars * bar_size + partial_trades;

    // Use a price sequence with small variation around mid
    double prices[] = {4000.25, 4000.50, 4000.00, 4000.75, 4000.25};
    int n_prices = 5;

    for (int i = 0; i < total_trades; ++i) {
        double p = prices[i % n_prices];
        Message::Side side = (i % 3 == 0) ? Message::Side::Bid : Message::Side::Ask;
        msgs.push_back(make_trade_msg(next_id++, p, 1, rth_ts + i * 1'000'000ULL, side));
    }

    return msgs;
}

// Convenience: create a stream and run barrier_precompute in one call.
static BarrierPrecomputedDay run_precompute(int bar_size, int num_bars,
                                             int lookback = 3,
                                             int a = 20, int b = 10, int t_max = 40,
                                             int partial_trades = 0) {
    auto msgs = make_precompute_stream(bar_size, num_bars, partial_trades);
    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();
    return barrier_precompute(source, cfg, bar_size, lookback, a, b, t_max);
}

// ===========================================================================
// Section 1: BarrierPrecomputedDay struct defaults (~2)
// ===========================================================================

TEST(BarrierPrecomputedDay, DefaultConstructionHasZeroValues) {
    BarrierPrecomputedDay day{};
    EXPECT_EQ(day.n_bars, 0);
    EXPECT_EQ(day.n_usable, 0);
    EXPECT_EQ(day.bar_size, 0);
    EXPECT_EQ(day.lookback, 0);
    EXPECT_TRUE(day.bar_open.empty());
    EXPECT_TRUE(day.bar_high.empty());
    EXPECT_TRUE(day.bar_low.empty());
    EXPECT_TRUE(day.bar_close.empty());
    EXPECT_TRUE(day.bar_vwap.empty());
    EXPECT_TRUE(day.bar_volume.empty());
    EXPECT_TRUE(day.bar_t_start.empty());
    EXPECT_TRUE(day.bar_t_end.empty());
    EXPECT_TRUE(day.trade_prices.empty());
    EXPECT_TRUE(day.trade_sizes.empty());
    EXPECT_TRUE(day.bar_trade_offsets.empty());
    EXPECT_TRUE(day.label_values.empty());
    EXPECT_TRUE(day.label_tau.empty());
    EXPECT_TRUE(day.label_resolution_bar.empty());
    EXPECT_TRUE(day.features.empty());
    EXPECT_EQ(day.n_features, 0);
}

TEST(BarrierPrecomputedDay, NFeaturesMatchesConstant) {
    // When populated by barrier_precompute(), n_features should equal N_FEATURES
    // This test validates the struct field exists and can hold the value
    BarrierPrecomputedDay day{};
    day.n_features = N_FEATURES;
    EXPECT_EQ(day.n_features, 22);
}

// ===========================================================================
// Section 2: barrier_precompute() empty/insufficient data (~3)
// ===========================================================================

TEST(BarrierPrecompute, EmptyStreamReturnsZeroBars) {
    // Feed an empty ScriptedSource via barrier_precompute
    // The function takes a path, so we test via the struct contract:
    // With no messages, BarBuilder produces 0 bars → n_bars=0, n_usable=0
    ScriptedSource source({});
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(500, cfg);

    // Process empty stream
    Message msg;
    while (source.next(msg)) {
        builder.process(msg);
    }
    builder.flush();

    EXPECT_EQ(builder.bars().size(), 0u);
    // barrier_precompute should return n_bars=0, n_usable=0, empty features
    // Testing the function directly once it exists:
    // (This will fail to compile until barrier_precompute.h is created)
    BarrierPrecomputedDay day = barrier_precompute(source, cfg);
    EXPECT_EQ(day.n_bars, 0);
    EXPECT_EQ(day.n_usable, 0);
    EXPECT_TRUE(day.features.empty());
    EXPECT_TRUE(day.bar_open.empty());
    EXPECT_TRUE(day.label_values.empty());
}

TEST(BarrierPrecompute, NoRthTradesReturnsZeroBars) {
    // Only pre-market messages, no RTH trades → 0 bars
    std::vector<Message> msgs;
    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Pre-market book building only (Add messages, no Trades)
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            4000.00, 100, pre_ts));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            4000.25, 100, pre_ts + NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    BarrierPrecomputedDay day = barrier_precompute(source, cfg);

    EXPECT_EQ(day.n_bars, 0);
    EXPECT_EQ(day.n_usable, 0);
    EXPECT_TRUE(day.features.empty());
}

TEST(BarrierPrecompute, InsufficientBarsReturnsZeroUsable) {
    // Generate fewer bars than lookback + 1 → n_usable=0, empty features
    int bar_size = 5;
    int lookback = 10;
    int num_bars = 5;  // Less than lookback + 1 = 11

    auto msgs = make_precompute_stream(bar_size, num_bars);
    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    BarrierPrecomputedDay day = barrier_precompute(source, cfg, bar_size, lookback);

    EXPECT_EQ(day.n_bars, num_bars);
    EXPECT_EQ(day.n_usable, 0);
    EXPECT_TRUE(day.features.empty());
    // Bar data should still be populated even with insufficient bars for features
    EXPECT_EQ(static_cast<int>(day.bar_open.size()), num_bars);
    EXPECT_EQ(static_cast<int>(day.bar_close.size()), num_bars);
}

// ===========================================================================
// Section 3: Bar data population (~4)
// ===========================================================================

TEST(BarrierPrecompute, BarOHLCVArraysPopulated) {
    int bar_size = 5;
    int num_bars = 25;  // > lookback + REALIZED_VOL_WARMUP + 1

    BarrierPrecomputedDay day = run_precompute(bar_size, num_bars);

    EXPECT_EQ(day.n_bars, num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_open.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_high.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_low.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_close.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_volume.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_vwap.size()), num_bars);

    // All bar prices should be finite and positive
    for (int i = 0; i < num_bars; ++i) {
        EXPECT_TRUE(std::isfinite(day.bar_open[i])) << "bar_open[" << i << "]";
        EXPECT_TRUE(std::isfinite(day.bar_high[i])) << "bar_high[" << i << "]";
        EXPECT_TRUE(std::isfinite(day.bar_low[i])) << "bar_low[" << i << "]";
        EXPECT_TRUE(std::isfinite(day.bar_close[i])) << "bar_close[" << i << "]";
        EXPECT_TRUE(std::isfinite(day.bar_vwap[i])) << "bar_vwap[" << i << "]";
        EXPECT_GT(day.bar_open[i], 0.0) << "bar_open[" << i << "] must be positive";
        EXPECT_GT(day.bar_close[i], 0.0) << "bar_close[" << i << "] must be positive";
        EXPECT_GE(day.bar_high[i], day.bar_low[i])
            << "high >= low for bar " << i;
        EXPECT_GE(day.bar_high[i], day.bar_open[i])
            << "high >= open for bar " << i;
        EXPECT_GE(day.bar_high[i], day.bar_close[i])
            << "high >= close for bar " << i;
        EXPECT_LE(day.bar_low[i], day.bar_open[i])
            << "low <= open for bar " << i;
        EXPECT_LE(day.bar_low[i], day.bar_close[i])
            << "low <= close for bar " << i;
    }

    // Volume should be exactly bar_size per bar
    for (int i = 0; i < num_bars; ++i) {
        EXPECT_EQ(day.bar_volume[i], bar_size) << "Volume of bar " << i;
    }
}

TEST(BarrierPrecompute, TimestampsPopulated) {
    int num_bars = 25;

    BarrierPrecomputedDay day = run_precompute(5, num_bars);

    ASSERT_EQ(static_cast<int>(day.bar_t_start.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.bar_t_end.size()), num_bars);

    for (int i = 0; i < num_bars; ++i) {
        EXPECT_GT(day.bar_t_start[i], 0u) << "t_start[" << i << "] must be non-zero";
        EXPECT_GT(day.bar_t_end[i], 0u) << "t_end[" << i << "] must be non-zero";
        EXPECT_LE(day.bar_t_start[i], day.bar_t_end[i])
            << "t_start <= t_end for bar " << i;
    }

    // Timestamps should be monotonically non-decreasing across bars
    for (int i = 1; i < num_bars; ++i) {
        EXPECT_GE(day.bar_t_start[i], day.bar_t_start[i - 1])
            << "bar_t_start should be non-decreasing between bars " << i - 1 << " and " << i;
    }
}

TEST(BarrierPrecompute, TradePricesAndOffsetsCorrect) {
    int bar_size = 5;
    int num_bars = 25;

    BarrierPrecomputedDay day = run_precompute(bar_size, num_bars);

    // trade_prices and trade_sizes must have the same length
    EXPECT_EQ(day.trade_prices.size(), day.trade_sizes.size());

    // bar_trade_offsets must have size n_bars + 1
    ASSERT_EQ(static_cast<int>(day.bar_trade_offsets.size()), num_bars + 1);

    // Offsets must be monotonically non-decreasing
    for (int i = 1; i <= num_bars; ++i) {
        EXPECT_GE(day.bar_trade_offsets[i], day.bar_trade_offsets[i - 1])
            << "bar_trade_offsets must be monotonically non-decreasing at index " << i;
    }

    // First offset must be 0
    EXPECT_EQ(day.bar_trade_offsets[0], 0);

    // Last offset must equal total number of trades
    EXPECT_EQ(day.bar_trade_offsets[num_bars],
              static_cast<int64_t>(day.trade_prices.size()));

    // Each bar should have bar_size trades (since we made exact bars)
    for (int i = 0; i < num_bars; ++i) {
        int64_t bar_trades = day.bar_trade_offsets[i + 1] - day.bar_trade_offsets[i];
        EXPECT_EQ(bar_trades, bar_size)
            << "Bar " << i << " should have " << bar_size << " trades";
    }

    // All trade prices should be positive and finite
    for (size_t i = 0; i < day.trade_prices.size(); ++i) {
        EXPECT_TRUE(std::isfinite(day.trade_prices[i]))
            << "trade_prices[" << i << "]";
        EXPECT_GT(day.trade_prices[i], 0.0)
            << "trade_prices[" << i << "] must be positive";
    }

    // All trade sizes should be positive
    for (size_t i = 0; i < day.trade_sizes.size(); ++i) {
        EXPECT_GT(day.trade_sizes[i], 0)
            << "trade_sizes[" << i << "] must be positive";
    }
}

TEST(BarrierPrecompute, BarSizeStoredInResult) {
    int bar_size = 7;
    int lookback = 3;
    int num_bars = 25;

    auto msgs = make_precompute_stream(bar_size, num_bars);
    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    BarrierPrecomputedDay day = barrier_precompute(source, cfg, bar_size, lookback);

    EXPECT_EQ(day.bar_size, bar_size);
    EXPECT_EQ(day.lookback, lookback);
}

// ===========================================================================
// Section 4: Labels (~2)
// ===========================================================================

TEST(BarrierPrecompute, LabelsPopulatedCorrectLength) {
    int num_bars = 25;

    BarrierPrecomputedDay day = run_precompute(5, num_bars);

    // Labels arrays must all have length n_bars
    ASSERT_EQ(static_cast<int>(day.label_values.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.label_tau.size()), num_bars);
    ASSERT_EQ(static_cast<int>(day.label_resolution_bar.size()), num_bars);
}

TEST(BarrierPrecompute, LabelValuesAreValid) {
    int num_bars = 25;

    BarrierPrecomputedDay day = run_precompute(5, num_bars);

    for (int i = 0; i < num_bars; ++i) {
        int val = day.label_values[i];
        EXPECT_TRUE(val == +1 || val == -1 || val == 0)
            << "label_values[" << i << "] = " << val << " must be +1, -1, or 0";

        EXPECT_GE(day.label_tau[i], 1)
            << "label_tau[" << i << "] must be >= 1";

        EXPECT_GE(day.label_resolution_bar[i], i)
            << "label_resolution_bar[" << i << "] must be >= bar_index " << i;
    }
}

// ===========================================================================
// Section 5: Features (~4)
// ===========================================================================

TEST(BarrierPrecompute, FeaturesShapeCorrect) {
    int lookback = 3;
    int num_bars = 30;  // > REALIZED_VOL_WARMUP + lookback

    BarrierPrecomputedDay day = run_precompute(5, num_bars, lookback);

    EXPECT_GT(day.n_usable, 0)
        << "With " << num_bars << " bars and lookback=" << lookback
        << ", n_usable should be > 0";

    int expected_cols = N_FEATURES * lookback;
    int expected_size = day.n_usable * expected_cols;

    EXPECT_EQ(static_cast<int>(day.features.size()), expected_size)
        << "features should have n_usable * N_FEATURES * lookback elements"
        << " (n_usable=" << day.n_usable << ", cols=" << expected_cols << ")";
}

TEST(BarrierPrecompute, FeaturesAllFinite) {
    BarrierPrecomputedDay day = run_precompute(5, 30);

    ASSERT_GT(day.n_usable, 0);

    // All feature values must be finite (no NaN after normalization)
    EXPECT_ALL_FINITE(day.features);
}

TEST(BarrierPrecompute, NFeaturesFieldMatchesConstant) {
    BarrierPrecomputedDay day = run_precompute(5, 30);

    EXPECT_EQ(day.n_features, N_FEATURES)
        << "n_features in result must equal the N_FEATURES constant (" << N_FEATURES << ")";
}

TEST(BarrierPrecompute, DefaultParametersMatch) {
    // Verify the default parameter values from the spec:
    // bar_size=500, lookback=10, a=20, b=10, t_max=40
    // We can't easily test defaults without a file path, so test via ScriptedSource overload
    // The function signature should accept these defaults.

    // Build enough bars for defaults: need > REALIZED_VOL_WARMUP + lookback + 1 = 30
    // We'd need 500*35 = 17500 trades with bar_size=500, so use bar_size=5 for test
    auto msgs = make_precompute_stream(5, 35);
    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    // Call with only required args — defaults for lookback, a, b, t_max
    BarrierPrecomputedDay day = barrier_precompute(source, cfg, 5);

    // The default lookback should be 10
    EXPECT_EQ(day.lookback, 10);
    EXPECT_EQ(day.bar_size, 5);
}

// ===========================================================================
// Section 6: Integration tests with synthetic data (~3)
// ===========================================================================

TEST(BarrierPrecompute, KnownStreamProducesExpectedBarCount) {
    int bar_size = 5;
    int num_bars = 10;

    BarrierPrecomputedDay day = run_precompute(bar_size, num_bars);

    EXPECT_EQ(day.n_bars, num_bars)
        << "Should produce exactly " << num_bars << " bars from " << num_bars * bar_size << " trades";
}

TEST(BarrierPrecompute, BarClosePricesFormValidSeries) {
    BarrierPrecomputedDay day = run_precompute(5, 25);

    // All close prices should be finite and positive (no NaN in the price series)
    for (int i = 0; i < day.n_bars; ++i) {
        EXPECT_TRUE(std::isfinite(day.bar_close[i]))
            << "bar_close[" << i << "] must be finite";
        EXPECT_GT(day.bar_close[i], 0.0)
            << "bar_close[" << i << "] must be positive";
    }

    // VWAP should be between low and high for each bar
    for (int i = 0; i < day.n_bars; ++i) {
        EXPECT_GE(day.bar_vwap[i], day.bar_low[i])
            << "vwap should be >= low for bar " << i;
        EXPECT_LE(day.bar_vwap[i], day.bar_high[i])
            << "vwap should be <= high for bar " << i;
    }
}

TEST(BarrierPrecompute, FeatureDimensionsMatchExpected) {
    int lookback = 3;
    int num_bars = 30;

    BarrierPrecomputedDay day = run_precompute(5, num_bars, lookback);

    // n_usable should be:
    //   (num_bars - REALIZED_VOL_WARMUP) - lookback + 1
    // = num_bars - REALIZED_VOL_WARMUP - lookback + 1
    // For num_bars=30, warmup=19, lookback=3: 30 - 19 - 3 + 1 = 9
    int trimmed = num_bars - REALIZED_VOL_WARMUP;
    int expected_usable = trimmed - lookback + 1;

    EXPECT_EQ(day.n_usable, expected_usable)
        << "n_usable = (n_bars - warmup) - lookback + 1"
        << " = (" << num_bars << " - " << REALIZED_VOL_WARMUP
        << ") - " << lookback << " + 1 = " << expected_usable;

    // Feature row width = N_FEATURES * lookback
    int row_width = N_FEATURES * lookback;
    EXPECT_EQ(static_cast<int>(day.features.size()), day.n_usable * row_width);
}

TEST(BarrierPrecompute, NUsableIsZeroWhenBarsEqualLookbackPlusOne) {
    // Edge case: exactly lookback + 1 bars, but after warmup trimming
    // we might have 0 usable rows
    int lookback = 10;
    // Need bars = REALIZED_VOL_WARMUP + lookback to get exactly lookback rows
    // after warmup, which gives 1 usable row after lookback assembly.
    // But REALIZED_VOL_WARMUP + lookback = 29 bars.
    // After warmup: 29 - 19 = 10 rows. After lookback assembly: 10 - 10 + 1 = 1.
    int num_bars = REALIZED_VOL_WARMUP + lookback; // 29

    BarrierPrecomputedDay day = run_precompute(5, num_bars, lookback);

    EXPECT_EQ(day.n_bars, num_bars);
    // After warmup trim: 29 - 19 = 10 rows. lookback = 10. 10 - 10 + 1 = 1.
    EXPECT_EQ(day.n_usable, 1)
        << "Exactly REALIZED_VOL_WARMUP + lookback bars → 1 usable row";
}

TEST(BarrierPrecompute, PartialBarsFromFlushAreIncluded) {
    // Test that flush() captures the partial bar
    int bar_size = 10;
    int num_full_bars = 25;
    int partial_trades = 3;

    BarrierPrecomputedDay day = run_precompute(bar_size, num_full_bars,
                                               /*lookback=*/3, /*a=*/20, /*b=*/10, /*t_max=*/40,
                                               partial_trades);

    // flush() should emit the partial bar
    EXPECT_EQ(day.n_bars, num_full_bars + 1)
        << "flush() should emit partial bar, giving " << num_full_bars + 1 << " bars";
}

// ===========================================================================
// Section 7: String path overload (DbnFileSource) (~1)
// ===========================================================================

TEST(BarrierPrecompute, StringOverloadWithInvalidPathThrows) {
    // barrier_precompute(path, instrument_id, ...) should throw for non-existent file
    EXPECT_ANY_THROW(
        barrier_precompute("/nonexistent/path/to/file.dbn.zst", 0)
    );
}

// ===========================================================================
// Section 8: Parameter propagation (~2)
// ===========================================================================

TEST(BarrierPrecompute, CustomLabelParametersAffectLabels) {
    int bar_size = 5;
    int lookback = 3;
    int num_bars = 30;

    auto msgs = make_precompute_stream(bar_size, num_bars);

    // Run with default label params (a=20, b=10, t_max=40)
    ScriptedSource source1(msgs);
    SessionConfig cfg = SessionConfig::default_rth();
    BarrierPrecomputedDay day1 = barrier_precompute(
        source1, cfg, bar_size, lookback, /*a=*/20, /*b=*/10, /*t_max=*/40);

    // Run with very small t_max to force more timeouts
    ScriptedSource source2(msgs);
    BarrierPrecomputedDay day2 = barrier_precompute(
        source2, cfg, bar_size, lookback, /*a=*/20, /*b=*/10, /*t_max=*/1);

    ASSERT_EQ(day1.n_bars, day2.n_bars) << "Same data → same bar count";

    // With t_max=1, more labels should be timeout (0) compared to t_max=40
    int timeouts1 = 0, timeouts2 = 0;
    for (int i = 0; i < day1.n_bars; ++i) {
        if (day1.label_values[i] == 0) ++timeouts1;
        if (day2.label_values[i] == 0) ++timeouts2;
    }
    EXPECT_GE(timeouts2, timeouts1)
        << "t_max=1 should produce at least as many timeouts as t_max=40";
}

TEST(BarrierPrecompute, CustomLookbackAffectsFeatureDimensions) {
    int bar_size = 5;
    int num_bars = 35;

    auto msgs = make_precompute_stream(bar_size, num_bars);

    // lookback = 3
    ScriptedSource source1(msgs);
    SessionConfig cfg = SessionConfig::default_rth();
    BarrierPrecomputedDay day1 = barrier_precompute(source1, cfg, bar_size, /*lookback=*/3);

    // lookback = 5
    ScriptedSource source2(msgs);
    BarrierPrecomputedDay day2 = barrier_precompute(source2, cfg, bar_size, /*lookback=*/5);

    // Both should have the same n_bars
    EXPECT_EQ(day1.n_bars, day2.n_bars);

    // Feature row width differs: N_FEATURES * lookback
    int row_width_1 = N_FEATURES * 3;
    int row_width_2 = N_FEATURES * 5;

    if (day1.n_usable > 0) {
        EXPECT_EQ(static_cast<int>(day1.features.size()), day1.n_usable * row_width_1);
    }
    if (day2.n_usable > 0) {
        EXPECT_EQ(static_cast<int>(day2.features.size()), day2.n_usable * row_width_2);
    }

    // Larger lookback → fewer usable rows
    EXPECT_GE(day1.n_usable, day2.n_usable)
        << "lookback=3 should produce >= usable rows than lookback=5";
}
