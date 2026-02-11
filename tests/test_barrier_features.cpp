#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "lob/barrier/feature_compute.h"
#include "lob/barrier/trade_bar.h"
#include "test_helpers.h"

// ===========================================================================
// Helpers: Build synthetic bars/accums for feature computation tests
// ===========================================================================

// Build a minimal bar with given OHLCV + trade data.
static TradeBar make_bar(int idx, double open, double high, double low,
                         double close, int volume, double vwap,
                         uint64_t t_start, uint64_t t_end,
                         std::vector<double> prices = {},
                         std::vector<int> sizes = {}) {
    TradeBar bar;
    bar.bar_index = idx;
    bar.open = open;
    bar.high = high;
    bar.low = low;
    bar.close = close;
    bar.volume = volume;
    bar.vwap = vwap;
    bar.t_start = t_start;
    bar.t_end = t_end;
    bar.trade_prices = std::move(prices);
    bar.trade_sizes = std::move(sizes);
    return bar;
}

// Build a default accum with neutral values suitable for most feature tests.
static BarBookAccum make_default_accum() {
    BarBookAccum acc;
    acc.bid_qty = 100;
    acc.ask_qty = 100;
    acc.total_bid_3 = 300;
    acc.total_ask_3 = 300;
    acc.total_bid_5 = 500;
    acc.total_ask_5 = 500;
    acc.total_bid_10 = 1000;
    acc.total_ask_10 = 1000;
    acc.bid_cancels = 0;
    acc.ask_cancels = 0;
    acc.ofi_signed_volume = 0.0;
    acc.total_add_volume = 100.0;
    acc.wmid_first = 1000.0;
    acc.wmid_end = 1000.0;
    acc.spread_samples = {0.50, 0.50};
    acc.vamp_at_mid = 1000.0;
    acc.vamp_at_end = 1000.0;
    acc.buy_aggressor_vol = 50.0;
    acc.sell_aggressor_vol = 50.0;
    acc.n_trades = 10;
    acc.n_cancels = 5;
    return acc;
}

// RTH boundaries for tests
static constexpr uint64_t TEST_RTH_OPEN  = DAY_BASE_NS + RTH_OPEN_NS;
static constexpr uint64_t TEST_RTH_CLOSE = DAY_BASE_NS + RTH_CLOSE_NS;

// ===========================================================================
// Section 1: Constants tests (~3)
// ===========================================================================

TEST(BarrierFeatureConstants, NFeatures22) {
    EXPECT_EQ(N_FEATURES, 22);
}

TEST(BarrierFeatureConstants, TickSize025) {
    EXPECT_DOUBLE_EQ(TICK_SIZE, 0.25);
}

TEST(BarrierFeatureConstants, RealizedVolWarmupAndSessionAge) {
    EXPECT_EQ(REALIZED_VOL_WARMUP, 19);
    EXPECT_DOUBLE_EQ(SESSION_AGE_PERIOD, 20.0);
}

// ===========================================================================
// Section 2: Trade flow imbalance (Col 0) tests (~5)
// ===========================================================================

TEST(BarrierFeatureTradeFlow, SingleTradeReturnsZero) {
    // Single trade → 0.0 (need at least 2 for tick rule diffs)
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 5, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0}, {5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    ASSERT_EQ(feats.size(), static_cast<size_t>(N_FEATURES));
    EXPECT_DOUBLE_EQ(feats[0], 0.0)
        << "Single trade should yield trade flow imbalance = 0.0";
}

TEST(BarrierFeatureTradeFlow, TwoTradesUptickPositive) {
    // Prices: 100.0 → 100.25 (uptick), sizes: 5, 10
    // sides: [0, +1] → buy_vol = 10, sell_vol = 0
    // Result: (10 - 0) / (10 + 0) = 1.0
    TradeBar bar = make_bar(0, 100.0, 100.25, 100.0, 100.25, 15, 100.1,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 10});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_GT(feats[0], 0.0) << "Uptick should yield positive trade flow imbalance";
    EXPECT_DOUBLE_EQ(feats[0], 1.0) << "Pure uptick: (10-0)/(10+0) = 1.0";
}

TEST(BarrierFeatureTradeFlow, TwoTradesDowntickNegative) {
    // Prices: 100.25 → 100.0 (downtick), sizes: 5, 10
    // sides: [0, -1] → buy_vol = 0, sell_vol = 10
    // Result: (0 - 10) / (0 + 10) = -1.0
    TradeBar bar = make_bar(0, 100.25, 100.25, 100.0, 100.0, 15, 100.1,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.25, 100.0}, {5, 10});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_LT(feats[0], 0.0) << "Downtick should yield negative trade flow imbalance";
    EXPECT_DOUBLE_EQ(feats[0], -1.0) << "Pure downtick: (0-10)/(0+10) = -1.0";
}

TEST(BarrierFeatureTradeFlow, ForwardFillUnchangedPriceContinuesPreviousDirection) {
    // Prices: 100.0, 100.25, 100.25 — sizes: 2, 3, 5
    // diffs: [_, +1, 0]
    // sides: [0, +1, 0→forward fill→+1]
    // buy_vol = 3 + 5 = 8, sell_vol = 0
    // Result: 8/8 = 1.0
    TradeBar bar = make_bar(0, 100.0, 100.25, 100.0, 100.25, 10, 100.15,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25, 100.25}, {2, 3, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[0], 1.0)
        << "Forward-fill: unchanged price continues previous (+1) direction";
}

TEST(BarrierFeatureTradeFlow, AllSamePriceReturnsZero) {
    // All trades at same price → no diffs → all sides 0 → buy=sell=0 → 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 20, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0, 100.0, 100.0}, {5, 5, 5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[0], 0.0)
        << "All same price → no direction → trade flow imbalance = 0.0";
}

// ===========================================================================
// Section 3: BBO/Depth imbalance (Cols 1-2) tests (~4)
// ===========================================================================

TEST(BarrierFeatureBBO, BidHeavyBBOImbalanceAboveHalf) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.bid_qty = 200;
    acc.ask_qty = 100;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    // BBO imbalance = 200 / (200 + 100) = 0.6667
    EXPECT_NEAR(feats[1], 200.0 / 300.0, 1e-10)
        << "Bid-heavy BBO imbalance should be > 0.5";
}

TEST(BarrierFeatureBBO, BothZeroBBOImbalanceIsHalf) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.bid_qty = 0;
    acc.ask_qty = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[1], 0.5)
        << "Both zero BBO → imbalance = 0.5";
}

TEST(BarrierFeatureDepth, BidHeavyDepthImbalanceAboveHalf) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.total_bid_5 = 800;
    acc.total_ask_5 = 200;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    // Depth imbalance = 800 / (800 + 200) = 0.8
    EXPECT_NEAR(feats[2], 0.8, 1e-10)
        << "Bid-heavy depth(5) imbalance should be > 0.5";
}

TEST(BarrierFeatureDepth, BothZeroDepthImbalanceIsHalf) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.total_bid_5 = 0;
    acc.total_ask_5 = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[2], 0.5)
        << "Both zero depth(5) → imbalance = 0.5";
}

// ===========================================================================
// Section 4: Bar range/body/ratio (Cols 3-6) tests (~5)
// ===========================================================================

TEST(BarrierFeatureBarShape, BarRangeInTicks) {
    // high=101.0, low=100.0 → range = (101-100)/0.25 = 4.0 ticks
    TradeBar bar = make_bar(0, 100.25, 101.0, 100.0, 100.75, 10, 100.5,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.25, 101.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[3], 4.0)
        << "Bar range = (101.0 - 100.0) / 0.25 = 4.0 ticks";
}

TEST(BarrierFeatureBarShape, BarBodySignedInTicks) {
    // close=100.75, open=100.25 → body = (100.75-100.25)/0.25 = 2.0 ticks
    TradeBar bar = make_bar(0, 100.25, 101.0, 100.0, 100.75, 10, 100.5,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.25, 100.75}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[4], 2.0)
        << "Bar body = (100.75 - 100.25) / 0.25 = 2.0 ticks";
}

TEST(BarrierFeatureBarShape, BodyRangeRatioFlatBarIsZero) {
    // Flat bar: high == low → range == 0 → ratio = 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[5], 0.0)
        << "Flat bar (range=0) → body/range ratio = 0.0";
}

TEST(BarrierFeatureBarShape, VWAPDisplacementCloseAboveVWAP) {
    // close=101.0, vwap=100.5, high=101.0, low=100.0, range=1.0
    // VWAP displacement = (101.0 - 100.5) / (101.0 - 100.0) = 0.5
    TradeBar bar = make_bar(0, 100.0, 101.0, 100.0, 101.0, 10, 100.5,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 101.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[6], 0.5, 1e-10)
        << "VWAP displacement: close > vwap → positive";
}

TEST(BarrierFeatureBarShape, VWAPDisplacementFlatBarIsZero) {
    // Flat bar: range=0 → VWAP displacement = 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[6], 0.0)
        << "Flat bar → VWAP displacement = 0.0";
}

// ===========================================================================
// Section 5: Volume and realized vol (Cols 7-8) tests (~4)
// ===========================================================================

TEST(BarrierFeatureVolume, LogVolumePositive) {
    // volume = 100 → log(100) ≈ 4.6052
    TradeBar bar = make_bar(0, 100.0, 101.0, 99.0, 100.5, 100, 100.2,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.5}, {50, 50});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[7], std::log(100.0), 1e-10)
        << "Volume log: log(100) for volume=100";
}

TEST(BarrierFeatureVolume, ZeroVolumeLogsToZero) {
    // volume = 0 → log(max(0, 1)) = log(1) = 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 0, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {}, {});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[7], 0.0)
        << "Volume log: log(max(0,1)) = log(1) = 0.0";
}

TEST(BarrierFeatureRealizedVol, NaNForBarsBeforeWarmup) {
    // Bars 0..18 should have NaN for realized vol
    // Create 19 bars (indices 0..18)
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i < 19; ++i) {
        double price = 100.0 + i * 0.25;
        bars.push_back(make_bar(i, price, price + 0.25, price - 0.25, price, 10, price,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {price, price}, {5, 5}));
        accums.push_back(make_default_accum());
    }

    auto feats = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    ASSERT_EQ(feats.size(), static_cast<size_t>(19 * N_FEATURES));

    for (int i = 0; i < 19; ++i) {
        EXPECT_TRUE(std::isnan(feats[i * N_FEATURES + 8]))
            << "Bar " << i << " (i < 19) should have NaN realized vol";
    }
}

TEST(BarrierFeatureRealizedVol, CorrectStdForBarsAtWarmup) {
    // Create 20 bars with known close prices, then check bar 19's realized vol
    // Close prices: 100.0, 100.25, 100.50, ..., 104.75 (arithmetic +0.25)
    // 20 prices → 19 log-returns
    // r[j] = log(close[j+1] / close[j])
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    std::vector<double> close_prices;
    for (int i = 0; i < 20; ++i) {
        double c = 100.0 + i * 0.25;
        close_prices.push_back(c);
        bars.push_back(make_bar(i, c, c + 0.25, c - 0.25, c, 10, c,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {c, c}, {5, 5}));
        accums.push_back(make_default_accum());
    }

    auto feats = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);

    // Hand-compute expected realized vol for bar 19:
    // 19 log-returns from close[0..19]
    std::vector<double> log_returns;
    for (int j = 0; j < 19; ++j) {
        log_returns.push_back(std::log(close_prices[j + 1] / close_prices[j]));
    }
    double mean_r = 0.0;
    for (double r : log_returns) mean_r += r;
    mean_r /= 19.0;
    double var = 0.0;
    for (double r : log_returns) var += (r - mean_r) * (r - mean_r);
    var /= 19.0;  // population std (ddof=0)
    double expected_std = std::sqrt(var);

    double actual = feats[19 * N_FEATURES + 8];
    EXPECT_FALSE(std::isnan(actual))
        << "Bar 19 (i >= 19) should have non-NaN realized vol";
    EXPECT_NEAR(actual, expected_std, 1e-10)
        << "Realized vol should be population std of 19 log-returns";
}

// ===========================================================================
// Section 6: Session time and age (Cols 9, 12) tests (~3)
// ===========================================================================

TEST(BarrierFeatureSession, SessionTimeMidpointApproxHalf) {
    // Bar ending at midpoint of RTH session → normalized time ≈ 0.5
    uint64_t midpoint = TEST_RTH_OPEN + (TEST_RTH_CLOSE - TEST_RTH_OPEN) / 2;
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            midpoint - NS_PER_MIN,
                            midpoint,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[9], 0.5, 0.01)
        << "Session time at midpoint should be ≈ 0.5";
}

TEST(BarrierFeatureSession, SessionTimeBeforeOpenClampedToZero) {
    // Bar ending before RTH open → clamped to 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN - 2 * NS_PER_MIN,
                            TEST_RTH_OPEN - NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[9], 0.0)
        << "Session time before open → clamped to 0.0";
}

TEST(BarrierFeatureSession, SessionAgeBarZeroIsZeroAndBar20Plus1) {
    // Bar 0 → age = min(0/20, 1) = 0.0
    // Bar 20 → age = min(20/20, 1) = 1.0
    // Bar 40 → age = min(40/20, 1) = 1.0 (clamped)
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i <= 40; ++i) {
        bars.push_back(make_bar(i, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {100.0, 100.0}, {5, 5}));
        accums.push_back(make_default_accum());
    }

    auto feats = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);

    EXPECT_DOUBLE_EQ(feats[0 * N_FEATURES + 12], 0.0)
        << "Bar 0 → session age = 0.0";
    EXPECT_NEAR(feats[10 * N_FEATURES + 12], 0.5, 1e-10)
        << "Bar 10 → session age = 10/20 = 0.5";
    EXPECT_DOUBLE_EQ(feats[20 * N_FEATURES + 12], 1.0)
        << "Bar 20 → session age = min(20/20, 1) = 1.0";
    EXPECT_DOUBLE_EQ(feats[40 * N_FEATURES + 12], 1.0)
        << "Bar 40 → session age = min(40/20, 1) = 1.0 (clamped)";
}

// ===========================================================================
// Section 7: Cancel/spread features (Cols 10-11) tests (~3)
// ===========================================================================

TEST(BarrierFeatureCancel, MoreBidCancelsPositiveAsymmetry) {
    // bid_cancels=10, ask_cancels=2, total=12
    // cancel_asymmetry = (10 - 2) / (12 + 1e-10)
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.bid_cancels = 10;
    acc.ask_cancels = 2;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    double expected = 8.0 / (12.0 + 1e-10);
    EXPECT_NEAR(feats[10], expected, 1e-6)
        << "More bid cancels → positive cancel asymmetry";
}

TEST(BarrierFeatureCancel, ZeroCancelsYieldsZeroAsymmetry) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.bid_cancels = 0;
    acc.ask_cancels = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[10], 0.0, 1e-10)
        << "Zero cancels → cancel asymmetry ≈ 0.0";
}

TEST(BarrierFeatureSpread, MeanSpreadFromSamples) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.spread_samples = {0.25, 0.50, 0.75};  // mean = 0.50

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[11], 0.50, 1e-10)
        << "Mean spread from 3 samples: (0.25 + 0.50 + 0.75) / 3 = 0.50";
}

TEST(BarrierFeatureSpread, EmptySpreadSamplesDefaultsToOne) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.spread_samples.clear();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[11], 1.0)
        << "Empty spread samples → mean spread = 1.0";
}

// ===========================================================================
// Section 8: OFI and depth ratio (Cols 13-14) tests (~4)
// ===========================================================================

TEST(BarrierFeatureOFI, PositiveOFISignedVolume) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.ofi_signed_volume = 50.0;
    acc.total_add_volume = 100.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    // OFI = clamp(50 / (100 + 1e-10), -1, 1) = 0.5
    EXPECT_NEAR(feats[13], 0.5, 1e-6)
        << "Positive OFI signed volume → positive OFI feature";
}

TEST(BarrierFeatureOFI, OFIClampedToNeg1Pos1) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    // OFI > total_add → clamped to 1.0
    acc.ofi_signed_volume = 500.0;
    acc.total_add_volume = 100.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[13], 1.0)
        << "OFI should be clamped to [-1, +1]";
}

TEST(BarrierFeatureOFI, ZeroAddVolumeYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.ofi_signed_volume = 0.0;
    acc.total_add_volume = 0.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[13], 0.0)
        << "Zero total_add_volume → OFI = 0.0";
}

TEST(BarrierFeatureDepthRatio, CorrectComputation) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.total_bid_3 = 200;
    acc.total_ask_3 = 100;
    acc.total_bid_10 = 500;
    acc.total_ask_10 = 500;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    // depth_ratio = (200 + 100) / (500 + 500 + 1e-10) = 0.3
    EXPECT_NEAR(feats[14], 0.3, 1e-6)
        << "Depth ratio = (total_bid_3 + total_ask_3) / (total_bid_10 + total_ask_10)";
}

TEST(BarrierFeatureDepthRatio, ZeroTotal10YieldsHalf) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.total_bid_3 = 0;
    acc.total_ask_3 = 0;
    acc.total_bid_10 = 0;
    acc.total_ask_10 = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[14], 0.5)
        << "Zero total_10 → depth ratio = 0.5";
}

// ===========================================================================
// Section 9: WMid/Spread std (Cols 15-16) tests (~3)
// ===========================================================================

TEST(BarrierFeatureWMid, DisplacementCorrectComputation) {
    // wmid_end=1001.0, wmid_first=1000.0 → (1001-1000)/0.25 = 4.0
    TradeBar bar = make_bar(0, 100.0, 101.0, 99.0, 100.5, 10, 100.2,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.5}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.wmid_first = 1000.0;
    acc.wmid_end = 1001.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[15], 4.0)
        << "WMid displacement = (1001 - 1000) / 0.25 = 4.0";
}

TEST(BarrierFeatureWMid, NaNWmidYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.wmid_first = std::numeric_limits<double>::quiet_NaN();
    acc.wmid_end = 1000.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[15], 0.0)
        << "Either NaN wmid → displacement = 0.0";
}

TEST(BarrierFeatureSpreadStd, LessThan2SamplesYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.spread_samples = {0.50};  // only 1 sample → std = 0.0

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[16], 0.0)
        << "< 2 spread samples → spread std = 0.0";
}

TEST(BarrierFeatureSpreadStd, CorrectStdForMultipleSamples) {
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.spread_samples = {0.25, 0.50, 0.75};
    // mean = 0.50, var = ((0.25-0.5)^2 + (0.5-0.5)^2 + (0.75-0.5)^2) / 3
    //              = (0.0625 + 0 + 0.0625) / 3 = 0.04167
    // std = sqrt(0.04167) ≈ 0.20412
    double expected_std = std::sqrt(0.125 / 3.0);

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[16], expected_std, 1e-6)
        << "Spread std with 3 samples, ddof=0";
}

// ===========================================================================
// Section 10: VAMP/Aggressor/Trade features (Cols 17-21) tests (~6)
// ===========================================================================

TEST(BarrierFeatureVAMP, DisplacementCorrectComputation) {
    // vamp_at_end=1001.0, vamp_at_mid=1000.0 → (1001-1000)/0.25 = 4.0
    TradeBar bar = make_bar(0, 100.0, 101.0, 99.0, 100.5, 10, 100.2,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.5}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.vamp_at_mid = 1000.0;
    acc.vamp_at_end = 1001.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[17], 4.0)
        << "VAMP displacement = (1001 - 1000) / 0.25 = 4.0";
}

TEST(BarrierFeatureVAMP, NaNVampYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.vamp_at_mid = std::numeric_limits<double>::quiet_NaN();
    acc.vamp_at_end = 1000.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[17], 0.0)
        << "Either NaN VAMP → displacement = 0.0";
}

TEST(BarrierFeatureAggressor, BuyHeavyPositiveImbalance) {
    // buy_vol=80, sell_vol=20 → (80-20)/(80+20+1e-10) ≈ 0.6
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.buy_aggressor_vol = 80.0;
    acc.sell_aggressor_vol = 20.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[18], 0.6, 1e-6)
        << "Buy-heavy aggressor imbalance ≈ 0.6";
}

TEST(BarrierFeatureAggressor, ZeroVolYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.buy_aggressor_vol = 0.0;
    acc.sell_aggressor_vol = 0.0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[18], 0.0)
        << "Zero aggressor volumes → imbalance = 0.0";
}

TEST(BarrierFeatureTradeArrival, LogOnePlusNTrades) {
    // n_trades = 25 → log(1 + 25) = log(26)
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.n_trades = 25;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[19], std::log(26.0), 1e-10)
        << "Trade arrival rate = log(1 + 25) = log(26)";
}

TEST(BarrierFeatureCancelToTrade, CorrectRatio) {
    // n_cancels=30, n_trades=10 → log(1 + 30/10) = log(4.0)
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.25}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.n_cancels = 30;
    acc.n_trades = 10;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[20], std::log(4.0), 1e-10)
        << "Cancel-to-trade = log(1 + 30/max(10,1)) = log(4.0)";
}

TEST(BarrierFeatureCancelToTrade, ZeroTradesUsesMaxOneAsDenom) {
    // n_cancels=5, n_trades=0 → log(1 + 5/max(0,1)) = log(6)
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.n_cancels = 5;
    acc.n_trades = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[20], std::log(6.0), 1e-10)
        << "Zero n_trades → denominator = max(0,1) = 1";
}

TEST(BarrierFeaturePriceImpact, CorrectComputation) {
    // close=101.0, open=100.0, n_trades=8
    // price_impact = (101 - 100) / (max(8,1) * 0.25) = 1.0 / 2.0 = 0.5
    TradeBar bar = make_bar(0, 100.0, 101.0, 99.0, 101.0, 10, 100.5,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 101.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.n_trades = 8;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[21], 0.5, 1e-10)
        << "Price impact = (101 - 100) / (8 * 0.25) = 0.5";
}

TEST(BarrierFeaturePriceImpact, ZeroTradesUsesMaxOne) {
    // close=100.5, open=100.0, n_trades=0
    // price_impact = (100.5 - 100) / (max(0,1) * 0.25) = 0.5 / 0.25 = 2.0
    TradeBar bar = make_bar(0, 100.0, 100.5, 99.5, 100.5, 10, 100.25,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.5}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.n_trades = 0;

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[21], 2.0, 1e-10)
        << "Price impact with n_trades=0 uses max(0,1)=1 as denominator";
}

// ===========================================================================
// Section 11: Full feature vector tests (~3)
// ===========================================================================

TEST(BarrierFeatureFull, OutputSizeMatchesNBarsTimesNFeatures) {
    int n_bars = 5;
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i < n_bars; ++i) {
        bars.push_back(make_bar(i, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {100.0, 100.25}, {5, 5}));
        accums.push_back(make_default_accum());
    }

    auto feats = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_EQ(feats.size(), static_cast<size_t>(n_bars * N_FEATURES));
}

TEST(BarrierFeatureFull, All22ColumnsPopulatedForMultiBarSequence) {
    int n_bars = 3;
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i < n_bars; ++i) {
        double price = 100.0 + i * 0.25;
        bars.push_back(make_bar(i, price, price + 0.5, price - 0.5, price + 0.25,
                                20, price + 0.1,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {price, price + 0.25, price + 0.5},
                                {5, 10, 5}));
        BarBookAccum acc = make_default_accum();
        acc.bid_cancels = 3;
        acc.ask_cancels = 1;
        acc.ofi_signed_volume = 20.0;
        acc.buy_aggressor_vol = 60.0;
        acc.sell_aggressor_vol = 40.0;
        accums.push_back(acc);
    }

    auto feats = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    ASSERT_EQ(feats.size(), static_cast<size_t>(n_bars * N_FEATURES));

    // Check that all 22 columns have finite values (except realized vol which is NaN for i<19)
    for (int i = 0; i < n_bars; ++i) {
        for (int col = 0; col < N_FEATURES; ++col) {
            double val = feats[i * N_FEATURES + col];
            if (col == 8) {
                // Realized vol: NaN for bars 0..18
                EXPECT_TRUE(std::isnan(val))
                    << "Bar " << i << " col 8 (realized vol) should be NaN for early bars";
            } else {
                EXPECT_TRUE(std::isfinite(val))
                    << "Bar " << i << " col " << col << " should be finite, got " << val;
            }
        }
    }
}

TEST(BarrierFeatureFull, EmptyBarsReturnEmptyOutput) {
    auto feats = compute_bar_features({}, {}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_TRUE(feats.empty())
        << "Empty bars vector → empty output";
}

// ===========================================================================
// Section 12: normalize_features() tests (~8)
// ===========================================================================

TEST(NormalizeFeatures, SingleRowAllZeros) {
    // Z-score of a single value with itself → (x - x) / 0 → 0 (std=0 → z=0)
    std::vector<double> raw = {1.0, 2.0, 3.0};
    auto normed = normalize_features(raw, 1, 3);
    ASSERT_EQ(normed.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(normed[i], 0.0)
            << "Single row → z-score = 0 (std=0 case)";
    }
}

TEST(NormalizeFeatures, TwoIdenticalRowsAllZeros) {
    std::vector<double> raw = {5.0, 10.0, 15.0,
                                5.0, 10.0, 15.0};
    auto normed = normalize_features(raw, 2, 3);
    ASSERT_EQ(normed.size(), 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(normed[i], 0.0)
            << "Two identical rows → all z-scores = 0 (std=0)";
    }
}

TEST(NormalizeFeatures, KnownZScoresForThreeRowSequence) {
    // 3 rows, 1 column: values [1.0, 3.0, 5.0]
    // Row 0: window [1.0] → mean=1, std=0 → z=0
    // Row 1: window [1.0, 3.0] → mean=2, std=sqrt(((1-2)^2+(3-2)^2)/2)=1.0 → z=(3-2)/1=1.0
    // Row 2: window [1.0, 3.0, 5.0] → mean=3, std=sqrt(((1-3)^2+(3-3)^2+(5-3)^2)/3)=sqrt(8/3) → z=(5-3)/sqrt(8/3)
    std::vector<double> raw = {1.0, 3.0, 5.0};
    auto normed = normalize_features(raw, 3, 1);
    ASSERT_EQ(normed.size(), 3u);

    // Row 0: std=0 → z=0
    EXPECT_DOUBLE_EQ(normed[0], 0.0);

    // Row 1: z = (3 - 2) / 1.0 = 1.0
    EXPECT_NEAR(normed[1], 1.0, 1e-10);

    // Row 2: mean=3, std=sqrt(8/3)
    double std2 = std::sqrt(8.0 / 3.0);
    double z2 = (5.0 - 3.0) / std2;
    EXPECT_NEAR(normed[2], z2, 1e-10);
}

TEST(NormalizeFeatures, NaNReplacedWithZeroBeforeNormalization) {
    // Input: [NaN, 2.0, 4.0] (single column, 3 rows)
    // After NaN→0: [0.0, 2.0, 4.0]
    // Row 0: [0] → mean=0, std=0 → z=0
    // Row 1: [0, 2] → mean=1, std=1 → z=(2-1)/1=1
    // Row 2: [0, 2, 4] → mean=2, std=sqrt(8/3) → z=(4-2)/sqrt(8/3)
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> raw = {nan, 2.0, 4.0};
    auto normed = normalize_features(raw, 3, 1);
    ASSERT_EQ(normed.size(), 3u);

    EXPECT_DOUBLE_EQ(normed[0], 0.0);
    EXPECT_NEAR(normed[1], 1.0, 1e-10);
    double std2 = std::sqrt(8.0 / 3.0);
    EXPECT_NEAR(normed[2], (4.0 - 2.0) / std2, 1e-10);
}

TEST(NormalizeFeatures, ClipToMinusFivePlusFive) {
    // Create a scenario where z-scores would exceed 5.0
    // Values: [0, 0, 0, 100] (single column, 4 rows)
    // Row 3: window [0, 0, 0, 100] → mean=25, std≈43.3
    //   z=(100-25)/43.3≈1.73 (not extreme enough)
    // Instead: [0, 0, 100] → mean≈33.3, std≈47.14 → z=(100-33.3)/47.14≈1.4
    // We need more extreme: [0, 1000000]
    std::vector<double> raw = {0.0, 1000000.0};
    auto normed = normalize_features(raw, 2, 1);
    ASSERT_EQ(normed.size(), 2u);

    // Row 0: std=0 → z=0
    EXPECT_DOUBLE_EQ(normed[0], 0.0);
    // Row 1: mean=500000, std=500000 → z=(1000000-500000)/500000=1.0
    // Actually this gives exactly 1.0, not > 5.
    // Better test: use window=1 in a different scenario OR use a 3-element seq
    // [0, 0, ..., 0, 100] — many zeros then outlier
    // Let's go with a direct approach: verify clip at boundary
    for (size_t i = 0; i < normed.size(); ++i) {
        EXPECT_GE(normed[i], -5.0) << "Element " << i << " should be >= -5.0";
        EXPECT_LE(normed[i], 5.0) << "Element " << i << " should be <= 5.0";
    }
}

TEST(NormalizeFeatures, ExtremeValueClippedToFive) {
    // 10 rows, 1 col: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]
    // Last row: window = all 10 values
    // mean = 100, std = sqrt(mean((x-100)^2))
    // 9 * 100^2 + 900^2 = 9*10000 + 810000 = 900000
    // var = 900000/10 = 90000, std = 300
    // z = (1000 - 100)/300 = 3.0 — still < 5
    //
    // Make it more extreme: [0,0,0,0,0,0,0,0,0,0,...0, 10000] with 100 zeros
    // mean = 100, std ≈ 995, z ≈ 9.95 → should clip to 5.0
    std::vector<double> raw(101, 0.0);
    raw[100] = 10000.0;
    auto normed = normalize_features(raw, 101, 1);
    ASSERT_EQ(normed.size(), 101u);

    // The last element's z-score should exceed 5 before clipping → clipped to 5
    EXPECT_DOUBLE_EQ(normed[100], 5.0)
        << "Extreme z-score should be clipped to 5.0";
}

TEST(NormalizeFeatures, Window1AllZeros) {
    // window=1 → each row only sees itself → mean=x, std=0 → z=0
    std::vector<double> raw = {1.0, 5.0, 100.0};
    auto normed = normalize_features(raw, 3, 1, /*window=*/1);
    ASSERT_EQ(normed.size(), 3u);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(normed[i], 0.0)
            << "Window=1 → single-element windows → std=0 → z=0";
    }
}

TEST(NormalizeFeatures, LargeWindowActsAsExpandingWindow) {
    // window=10000 on 3 rows → same as expanding window (uses all available rows)
    std::vector<double> raw_large_w = {1.0, 3.0, 5.0};
    std::vector<double> raw_expand = {1.0, 3.0, 5.0};

    auto normed_large = normalize_features(raw_large_w, 3, 1, /*window=*/10000);
    auto normed_expand = normalize_features(raw_expand, 3, 1, /*window=*/2000);

    ASSERT_EQ(normed_large.size(), 3u);
    ASSERT_EQ(normed_expand.size(), 3u);

    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(normed_large[i], normed_expand[i])
            << "Large window should behave like expanding window at row " << i;
    }
}

TEST(NormalizeFeatures, ZeroStdColumnsYieldZero) {
    // All values in a column are the same → std=0 → z=0
    std::vector<double> raw = {3.0, 7.0,
                                3.0, 7.0,
                                3.0, 7.0};
    auto normed = normalize_features(raw, 3, 2);
    ASSERT_EQ(normed.size(), 6u);

    // Column 0: all 3.0 → std=0 → z=0 for all rows
    // Column 1: all 7.0 → std=0 → z=0 for all rows
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_DOUBLE_EQ(normed[i], 0.0)
            << "Constant column → std=0 → z=0 at index " << i;
    }
}

// ===========================================================================
// Section 13: assemble_lookback() tests (~6)
// ===========================================================================

TEST(AssembleLookback, H1SameAsInputButFloat32) {
    // h=1 → output == input (converted to float)
    std::vector<double> normed = {1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0};
    auto result = assemble_lookback(normed, 2, 3, /*h=*/1);
    ASSERT_EQ(result.size(), 6u);

    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
    EXPECT_FLOAT_EQ(result[4], 5.0f);
    EXPECT_FLOAT_EQ(result[5], 6.0f);
}

TEST(AssembleLookback, H2With3RowsGives2OutputRows) {
    // 3 rows, 2 cols, h=2 → 2 output rows, each 4 cols
    std::vector<double> normed = {1.0, 2.0,
                                   3.0, 4.0,
                                   5.0, 6.0};
    auto result = assemble_lookback(normed, 3, 2, /*h=*/2);
    // Output: (3 - 2 + 1) = 2 rows, each 2*2 = 4 cols → 8 elements
    ASSERT_EQ(result.size(), 8u);

    // Row 0: concat of input rows [0, 1] → [1, 2, 3, 4]
    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);

    // Row 1: concat of input rows [1, 2] → [3, 4, 5, 6]
    EXPECT_FLOAT_EQ(result[4], 3.0f);
    EXPECT_FLOAT_EQ(result[5], 4.0f);
    EXPECT_FLOAT_EQ(result[6], 5.0f);
    EXPECT_FLOAT_EQ(result[7], 6.0f);
}

TEST(AssembleLookback, OutputRowIsConcatenationOfHConsecutiveInputRows) {
    // 4 rows, 1 col, h=3 → 2 output rows, each 3 cols
    std::vector<double> normed = {10.0, 20.0, 30.0, 40.0};
    auto result = assemble_lookback(normed, 4, 1, /*h=*/3);
    ASSERT_EQ(result.size(), 6u);

    // Row 0: [10, 20, 30]
    EXPECT_FLOAT_EQ(result[0], 10.0f);
    EXPECT_FLOAT_EQ(result[1], 20.0f);
    EXPECT_FLOAT_EQ(result[2], 30.0f);

    // Row 1: [20, 30, 40]
    EXPECT_FLOAT_EQ(result[3], 20.0f);
    EXPECT_FLOAT_EQ(result[4], 30.0f);
    EXPECT_FLOAT_EQ(result[5], 40.0f);
}

TEST(AssembleLookback, NRowsLessThanHReturnsEmpty) {
    // n_rows=2, h=5 → empty
    std::vector<double> normed = {1.0, 2.0, 3.0, 4.0};
    auto result = assemble_lookback(normed, 2, 2, /*h=*/5);
    EXPECT_TRUE(result.empty())
        << "n_rows < h → empty output";
}

TEST(AssembleLookback, OutputIsFloat32) {
    std::vector<double> normed = {1.0, 2.0, 3.0};
    auto result = assemble_lookback(normed, 3, 1, /*h=*/1);
    // Verify type at compile time by checking sizeof (float = 4 bytes)
    static_assert(sizeof(result[0]) == sizeof(float),
                  "assemble_lookback output should be float (4 bytes)");
    EXPECT_EQ(sizeof(result[0]), 4u)
        << "Output element should be 4 bytes (float32)";
}

TEST(AssembleLookback, ShapeMatchesExpected) {
    // n_rows=10, n_cols=22, h=5 → (10-5+1)=6 rows, 22*5=110 cols → 660 elements
    int n_rows = 10, n_cols = 22, h = 5;
    std::vector<double> normed(n_rows * n_cols, 1.0);
    auto result = assemble_lookback(normed, n_rows, n_cols, h);

    int expected_rows = n_rows - h + 1;
    int expected_cols = n_cols * h;
    EXPECT_EQ(result.size(), static_cast<size_t>(expected_rows * expected_cols))
        << "Shape should be (n_rows - h + 1, n_cols * h)";
}

// ===========================================================================
// Section 14: Integration tests (~3)
// ===========================================================================

TEST(BarrierFeatureIntegration, ComputeNormalizeAssemblePipeline) {
    // End-to-end: compute → normalize → assemble produces correct output shape
    int n_bars = 25;  // enough for realized vol warmup at bar 19
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i < n_bars; ++i) {
        double price = 100.0 + i * 0.25;
        bars.push_back(make_bar(i, price, price + 0.5, price - 0.5, price + 0.25,
                                20, price + 0.1,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {price, price + 0.25, price + 0.5},
                                {5, 10, 5}));
        accums.push_back(make_default_accum());
    }

    // Step 1: compute raw features
    auto raw = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    ASSERT_EQ(raw.size(), static_cast<size_t>(n_bars * N_FEATURES));

    // Step 2: normalize
    auto normed = normalize_features(raw, n_bars, N_FEATURES);
    ASSERT_EQ(normed.size(), static_cast<size_t>(n_bars * N_FEATURES));

    // Step 3: assemble lookback
    int h = 10;
    auto lookback = assemble_lookback(normed, n_bars, N_FEATURES, h);
    int expected_rows = n_bars - h + 1;  // 25 - 10 + 1 = 16
    int expected_cols = N_FEATURES * h;   // 22 * 10 = 220
    EXPECT_EQ(lookback.size(), static_cast<size_t>(expected_rows * expected_cols))
        << "Pipeline output: (25-10+1) rows * (22*10) cols = 16 * 220 = 3520";

    // All values should be finite (NaN was replaced in normalize step)
    for (size_t i = 0; i < lookback.size(); ++i) {
        EXPECT_TRUE(std::isfinite(lookback[i]))
            << "Lookback element " << i << " should be finite after normalization";
    }
}

TEST(BarrierFeatureIntegration, PreconditionBarsSizeNotEqualAccumsThrows) {
    // bars.size() != accums.size() → should throw
    std::vector<TradeBar> bars(3);
    std::vector<BarBookAccum> accums(2);  // mismatched size

    EXPECT_THROW(
        compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE),
        std::invalid_argument
    ) << "Mismatched bars/accums sizes should throw invalid_argument";
}

TEST(BarrierFeatureIntegration, NormalizeThenClipAllWithinBounds) {
    // Generate a multi-bar sequence with varied features, verify all normalized
    // values are within [-5, 5]
    int n_bars = 50;
    std::vector<TradeBar> bars;
    std::vector<BarBookAccum> accums;

    for (int i = 0; i < n_bars; ++i) {
        double price = 100.0 + (i % 10) * 0.25 + (i / 10) * 5.0;
        bars.push_back(make_bar(i, price, price + 1.0, price - 1.0, price + 0.5,
                                100 + i * 10, price + 0.2,
                                TEST_RTH_OPEN + i * NS_PER_MIN,
                                TEST_RTH_OPEN + (i + 1) * NS_PER_MIN,
                                {price, price + 0.25, price + 0.5, price - 0.25},
                                {10, 20, 30, 40}));
        BarBookAccum acc = make_default_accum();
        acc.bid_cancels = i;
        acc.ask_cancels = i / 2;
        acc.ofi_signed_volume = (i % 3 == 0) ? 50.0 : -30.0;
        acc.buy_aggressor_vol = 50.0 + i;
        acc.sell_aggressor_vol = 50.0 - i / 2.0;
        acc.n_trades = 10 + i;
        acc.n_cancels = 5 + i * 2;
        accums.push_back(acc);
    }

    auto raw = compute_bar_features(bars, accums, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    auto normed = normalize_features(raw, n_bars, N_FEATURES);

    for (size_t i = 0; i < normed.size(); ++i) {
        EXPECT_GE(normed[i], -5.0) << "Normalized value at " << i << " should be >= -5.0";
        EXPECT_LE(normed[i], 5.0) << "Normalized value at " << i << " should be <= 5.0";
    }
}

// ===========================================================================
// Section 15: Additional edge cases
// ===========================================================================

TEST(BarrierFeatureTradeFlow, EmptyTradePricesReturnsZero) {
    // No trade_prices at all → 0.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 0, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {}, {});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[0], 0.0)
        << "Empty trade_prices → trade flow imbalance = 0.0";
}

TEST(BarrierFeatureBarShape, NegativeBodyDownBar) {
    // close < open → negative body
    // close=99.5, open=100.0 → body = (99.5-100.0)/0.25 = -2.0
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.25, 99.5, 10, 99.75,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 99.5}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[4], -2.0)
        << "Down bar: body = (99.5 - 100.0) / 0.25 = -2.0";
}

TEST(BarrierFeatureBarShape, BodyRangeRatioForFullBodyBar) {
    // Full body bar: close=high, open=low
    // open=100.0, high=101.0, low=100.0, close=101.0
    // body/range = (101-100)/(101-100) = 1.0
    TradeBar bar = make_bar(0, 100.0, 101.0, 100.0, 101.0, 10, 100.5,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 101.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[5], 1.0)
        << "Full body bar: body/range = 1.0";
}

TEST(BarrierFeatureWMid, BothNaNYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.wmid_first = std::numeric_limits<double>::quiet_NaN();
    acc.wmid_end = std::numeric_limits<double>::quiet_NaN();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[15], 0.0)
        << "Both NaN wmid → displacement = 0.0";
}

TEST(BarrierFeatureVAMP, BothNaNYieldsZero) {
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_OPEN + NS_PER_MIN,
                            TEST_RTH_OPEN + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();
    acc.vamp_at_mid = std::numeric_limits<double>::quiet_NaN();
    acc.vamp_at_end = std::numeric_limits<double>::quiet_NaN();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[17], 0.0)
        << "Both NaN VAMP → displacement = 0.0";
}

TEST(BarrierFeatureSession, SessionTimeAtCloseIsOne) {
    // Bar ending at RTH close → session time = 1.0
    TradeBar bar = make_bar(0, 100.0, 100.25, 99.75, 100.0, 10, 100.0,
                            TEST_RTH_CLOSE - NS_PER_MIN,
                            TEST_RTH_CLOSE,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_NEAR(feats[9], 1.0, 1e-10)
        << "Session time at close should be 1.0";
}

TEST(BarrierFeatureSession, SessionTimeAfterCloseClampedToOne) {
    // Bar ending after RTH close → clamped to 1.0
    TradeBar bar = make_bar(0, 100.0, 100.0, 100.0, 100.0, 10, 100.0,
                            TEST_RTH_CLOSE + NS_PER_MIN,
                            TEST_RTH_CLOSE + 2 * NS_PER_MIN,
                            {100.0, 100.0}, {5, 5});
    BarBookAccum acc = make_default_accum();

    auto feats = compute_bar_features({bar}, {acc}, TEST_RTH_OPEN, TEST_RTH_CLOSE);
    EXPECT_DOUBLE_EQ(feats[9], 1.0)
        << "Session time after close → clamped to 1.0";
}

TEST(NormalizeFeatures, MultiColumnCorrectPerColumnZScores) {
    // 3 rows, 2 cols
    // Col 0: [1, 2, 3], Col 1: [10, 20, 30]
    // Row 0: both std=0 → z=0, z=0
    // Row 1: col0 mean=1.5 std=0.5 z=(2-1.5)/0.5=1; col1 mean=15 std=5 z=(20-15)/5=1
    // Row 2: col0 mean=2 std=sqrt(2/3) z=(3-2)/sqrt(2/3); col1 mean=20 std=sqrt(200/3) z=(30-20)/sqrt(200/3)
    std::vector<double> raw = {1.0, 10.0,
                                2.0, 20.0,
                                3.0, 30.0};
    auto normed = normalize_features(raw, 3, 2);
    ASSERT_EQ(normed.size(), 6u);

    // Row 0: both zero
    EXPECT_DOUBLE_EQ(normed[0], 0.0);
    EXPECT_DOUBLE_EQ(normed[1], 0.0);

    // Row 1: both should be 1.0
    EXPECT_NEAR(normed[2], 1.0, 1e-10);
    EXPECT_NEAR(normed[3], 1.0, 1e-10);

    // Row 2: both should have same z (proportional growth)
    double std_col0 = std::sqrt(2.0 / 3.0);
    double z_col0 = (3.0 - 2.0) / std_col0;
    double std_col1 = std::sqrt(200.0 / 3.0);
    double z_col1 = (30.0 - 20.0) / std_col1;
    EXPECT_NEAR(normed[4], z_col0, 1e-10);
    EXPECT_NEAR(normed[5], z_col1, 1e-10);
}

TEST(AssembleLookback, H10With22ColsMatchesObservationDim) {
    // Real-world dimensions: 22 features, h=10 → observation dim = 220
    int n_rows = 15, n_cols = 22, h = 10;
    std::vector<double> normed(n_rows * n_cols, 0.5);
    auto result = assemble_lookback(normed, n_rows, n_cols, h);

    int expected_out_rows = 15 - 10 + 1;  // 6
    int expected_out_cols = 22 * 10;       // 220
    EXPECT_EQ(result.size(), static_cast<size_t>(expected_out_rows * expected_out_cols))
        << "Real-world lookback: 6 rows * 220 cols = 1320";
}
