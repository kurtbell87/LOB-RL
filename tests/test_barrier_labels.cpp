// Tests for barrier label computation (C++ port of label_pipeline.py).
// Spec: docs/barrier-labels-cpp.md
//
// Tests the compute_labels() function which assigns triple-barrier labels
// (+1 upper hit, -1 lower hit, 0 timeout) to trade bars, with intrabar
// tiebreaking via trade sequence scanning.

#include <gtest/gtest.h>
#include <vector>
#include "lob/barrier/barrier_label.h"
#include "lob/barrier/trade_bar.h"
#include "test_helpers.h"

// ===========================================================================
// Helpers: Build synthetic bars for label tests
// ===========================================================================

// Minimal bar with OHLCV and optional trade data.
// Uses timestamps spaced 1 minute apart from RTH open.
static TradeBar make_bar(int idx, double open, double high, double low,
                         double close, int volume = 100, double vwap = 0.0,
                         std::vector<double> prices = {},
                         std::vector<int> sizes = {}) {
    TradeBar bar;
    bar.bar_index = idx;
    bar.open = open;
    bar.high = high;
    bar.low = low;
    bar.close = close;
    bar.volume = volume;
    bar.vwap = (vwap == 0.0) ? close : vwap;
    bar.t_start = DAY_BASE_NS + RTH_OPEN_NS + idx * NS_PER_MIN;
    bar.t_end = bar.t_start + NS_PER_MIN;
    bar.trade_prices = std::move(prices);
    bar.trade_sizes = std::move(sizes);
    return bar;
}

// Build N bars at a constant price (flat market).
static std::vector<TradeBar> make_flat_bars(int n, double price = 1000.0) {
    std::vector<TradeBar> bars;
    for (int i = 0; i < n; ++i) {
        bars.push_back(make_bar(i, price, price, price, price));
    }
    return bars;
}

// Build N bars with monotonically rising closes.
// Each bar rises by `step` ticks from the previous close.
static std::vector<TradeBar> make_rising_bars(int n, double start = 1000.0,
                                               double step = 1.0) {
    std::vector<TradeBar> bars;
    for (int i = 0; i < n; ++i) {
        double price = start + i * step;
        double high = price + step * 0.5;
        double low = price - step * 0.5;
        double open_p = (i == 0) ? price : start + (i - 1) * step;
        bars.push_back(make_bar(i, open_p, high, low, price));
    }
    return bars;
}

// ===========================================================================
// Section 1: BarrierLabel struct tests (~2)
// ===========================================================================

TEST(BarrierLabelStruct, DefaultConstruction) {
    BarrierLabel lbl{};
    EXPECT_EQ(lbl.bar_index, 0);
    EXPECT_EQ(lbl.label, 0);
    EXPECT_EQ(lbl.tau, 0);
    EXPECT_EQ(lbl.resolution_bar, 0);
}

TEST(BarrierLabelStruct, FieldsSetCorrectly) {
    BarrierLabel lbl{};
    lbl.bar_index = 5;
    lbl.label = 1;
    lbl.tau = 3;
    lbl.resolution_bar = 8;
    EXPECT_EQ(lbl.bar_index, 5);
    EXPECT_EQ(lbl.label, 1);
    EXPECT_EQ(lbl.tau, 3);
    EXPECT_EQ(lbl.resolution_bar, 8);
}

// ===========================================================================
// Section 2: Basic label tests (~6)
// ===========================================================================

TEST(BarrierLabelBasic, EmptyBarsReturnsEmpty) {
    std::vector<TradeBar> bars;
    auto labels = compute_labels(bars);
    EXPECT_TRUE(labels.empty());
}

TEST(BarrierLabelBasic, SingleBarTimesOutWithTauOne) {
    // A single bar can't look ahead, so timeout with tau=1 minimum.
    auto bars = make_flat_bars(1, 1000.0);
    auto labels = compute_labels(bars);
    ASSERT_EQ(labels.size(), 1u);
    EXPECT_EQ(labels[0].label, 0);      // timeout
    EXPECT_EQ(labels[0].tau, 1);         // tau >= 1 minimum
    EXPECT_EQ(labels[0].bar_index, 0);
}

TEST(BarrierLabelBasic, UpperBarrierHit) {
    // Bar 0: close=1000. Upper barrier = 1000 + 20*0.25 = 1005.
    // Bar 1: high=1006 >= 1005 → label=+1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 999, 1004));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[0].label, 1);   // upper hit
    EXPECT_EQ(labels[0].tau, 1);
}

TEST(BarrierLabelBasic, LowerBarrierHit) {
    // Bar 0: close=1000. Lower barrier = 1000 - 10*0.25 = 997.5.
    // Bar 1: low=997 <= 997.5 → label=-1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 999, 1001, 997, 998));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[0].label, -1);  // lower hit
    EXPECT_EQ(labels[0].tau, 1);
}

TEST(BarrierLabelBasic, TimeoutWhenNoBarrierHit) {
    // All bars at same price → no barrier ever hit.
    auto bars = make_flat_bars(10, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    ASSERT_EQ(labels.size(), 10u);
    // Bar 0: has 9 bars to scan, within t_max=40. No barrier hit.
    EXPECT_EQ(labels[0].label, 0);   // timeout
}

TEST(BarrierLabelBasic, LabelCountEqualsBarCount) {
    auto bars = make_flat_bars(25, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels.size(), bars.size());
}

// ===========================================================================
// Section 3: Tau and resolution bar tests (~4)
// ===========================================================================

TEST(BarrierLabelTau, ImmediateNextBarHit) {
    // Upper barrier hit on bar j=k+1 → tau=1, resolution_bar=k+1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 999, 1004));
    bars.push_back(make_bar(2, 1004, 1005, 1003, 1004));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].tau, 1);
    EXPECT_EQ(labels[0].resolution_bar, 1);   // bar_index(0) + tau(1)
}

TEST(BarrierLabelTau, DistantBarrierHit) {
    // Entry bar 0 at 1000. Upper barrier = 1005. Bars 1-3 stay below.
    // Bar 4 high = 1006 → tau = 4, resolution_bar = 4.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1000, 1001, 999, 1000.5));
    bars.push_back(make_bar(2, 1000, 1002, 999, 1001));
    bars.push_back(make_bar(3, 1001, 1003, 999, 1002));
    bars.push_back(make_bar(4, 1002, 1006, 1001, 1004));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].tau, 4);
    EXPECT_EQ(labels[0].resolution_bar, 4);
}

TEST(BarrierLabelTau, ResolutionBarEqualsBarIndexPlusTau) {
    // For every label, resolution_bar == bar_index + tau.
    auto bars = make_flat_bars(15, 1000.0);
    // Give bars some movement so some hit barriers
    bars[3].high = 1006.0;  // triggers upper for bar 0 at 1000
    bars[7].low = 993.0;    // triggers lower for some bars

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    for (const auto& lbl : labels) {
        EXPECT_EQ(lbl.resolution_bar, lbl.bar_index + lbl.tau)
            << "bar_index=" << lbl.bar_index;
    }
}

TEST(BarrierLabelTau, TimeoutTauCappedByRemainingBars) {
    // Entry at bar 8 of 10 bars. Remaining = 1 bar. t_max=40.
    // tau should be min(t_max=40, remaining=1) = 1.
    auto bars = make_flat_bars(10, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);

    // Second-to-last bar: remaining = 1
    EXPECT_EQ(labels[8].label, 0);   // timeout
    EXPECT_EQ(labels[8].tau, 1);

    // Last bar: remaining = 0, but tau >= 1 minimum
    EXPECT_EQ(labels[9].label, 0);
    EXPECT_GE(labels[9].tau, 1);
}

// ===========================================================================
// Section 4: Tiebreak tests (~6)
// ===========================================================================

TEST(BarrierLabelTiebreak, DualBreachUpperFirst) {
    // Bar 0: close=1000.
    // Upper barrier = 1000 + 20*0.25 = 1005.
    // Lower barrier = 1000 - 10*0.25 = 997.5.
    // Bar 1: high=1006 >= 1005, low=997 <= 997.5 — dual breach.
    // Trade prices: [1001, 1003, 1005, 997] — 1005 >= upper first → +1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 997, 1002, 100, 1002,
                            {1001.0, 1003.0, 1005.0, 997.0},
                            {10, 10, 10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, 1);   // upper hit first via trade scan
    EXPECT_EQ(labels[0].tau, 1);
}

TEST(BarrierLabelTiebreak, DualBreachLowerFirst) {
    // Bar 0: close=1000.
    // Bar 1: dual breach. Trade prices: [999, 997, 1005] — 997 <= 997.5 first → -1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 999, 1006, 997, 1002, 100, 1002,
                            {999.0, 997.0, 1005.0},
                            {10, 10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // lower hit first via trade scan
}

TEST(BarrierLabelTiebreak, DualBreachEmptyTradesGapUp) {
    // Dual breach with empty trade_prices. Gap direction: bar[j].open >= prev close → +1.
    // Bar 0: close=1000.
    // Bar 1: high=1006, low=997, open=1001 (no trade_prices).
    // Gap direction: first_trade = open = 1001 >= prev close 1000 → +1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 997, 1002));  // no trades

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, 1);   // gap direction: open >= prev close
}

TEST(BarrierLabelTiebreak, DualBreachEmptyTradesGapDown) {
    // Gap direction with open < prev close → -1.
    // Bar 0: close=1000. Bar 1: open=999, high=1006, low=997.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 999, 1006, 997, 1002));  // no trades

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // gap direction: open < prev close
}

TEST(BarrierLabelTiebreak, FirstTradePastBarrierFallsBackToGap) {
    // Dual breach, first trade already >= upper_barrier → fall back to gap.
    // Bar 0: close=1000. Upper=1005, Lower=997.5.
    // Bar 1: trade_prices = [1006, 997, ...]. first trade=1006 >= 1005.
    // Gap direction: first_trade=1006 >= prev_close=1000 → +1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 997, 1002, 100, 1002,
                            {1006.0, 997.0, 1000.0},
                            {10, 10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, 1);   // gap fallback: first trade >= prev close
}

TEST(BarrierLabelTiebreak, FirstTradePastLowerFallsBackToGap) {
    // Dual breach, first trade already <= lower_barrier → fall back to gap.
    // Bar 0: close=1000. Lower=997.5.
    // Bar 1: trade_prices = [996, 1006, ...]. first trade=996 <= 997.5.
    // Gap direction: first_trade=996 < prev_close=1000 → -1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 996, 1006, 996, 1002, 100, 1002,
                            {996.0, 1006.0, 1000.0},
                            {10, 10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // gap fallback: first trade < prev close
}

// ===========================================================================
// Section 5: Barrier distance / parameter tests (~4)
// ===========================================================================

TEST(BarrierLabelParams, TightBarriersFrequentHits) {
    // a=1, b=1, tick=0.25. Upper=entry+0.25, lower=entry-0.25.
    // Any price movement triggers a hit. Most bars should resolve as ±1.
    auto bars = make_rising_bars(20, 1000.0, 0.50);

    auto labels = compute_labels(bars, 1, 1, 40, 0.25);
    ASSERT_EQ(labels.size(), 20u);

    int hit_count = 0;
    for (const auto& lbl : labels) {
        if (lbl.label != 0) ++hit_count;
    }
    // With tight barriers and rising prices, most should be hits (not timeouts).
    EXPECT_GT(hit_count, 10) << "Tight barriers should produce frequent hits";
}

TEST(BarrierLabelParams, WideBarriersMostlyTimeouts) {
    // a=100, b=100, tick=0.25. Upper=entry+25, lower=entry-25.
    // Small price movements → most are timeouts.
    auto bars = make_flat_bars(20, 1000.0);
    // Add small noise: bars drift by 0.5 each
    for (int i = 0; i < 20; ++i) {
        double p = 1000.0 + i * 0.5;
        bars[i].open = p;
        bars[i].high = p + 0.5;
        bars[i].low = p - 0.5;
        bars[i].close = p;
    }

    auto labels = compute_labels(bars, 100, 100, 40, 0.25);
    int timeout_count = 0;
    for (const auto& lbl : labels) {
        if (lbl.label == 0) ++timeout_count;
    }
    EXPECT_GT(timeout_count, 15) << "Wide barriers should produce mostly timeouts";
}

TEST(BarrierLabelParams, AsymmetricBarriers) {
    // a=4, b=1, tick=0.25. Upper=entry+1.0, lower=entry-0.25.
    // With equal-magnitude moves, lower should be hit more often.
    std::vector<TradeBar> bars;
    // Bar 0: close=1000.
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    // Bar 1: drops 0.25 → hits lower barrier (1000 - 1*0.25 = 999.75).
    bars.push_back(make_bar(1, 1000, 1000, 999.70, 999.80));

    auto labels = compute_labels(bars, 4, 1, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // lower hit with tight b=1 stop
}

TEST(BarrierLabelParams, TickSizeUsedCorrectly) {
    // Verify tick_size parameter is actually used, not hardcoded 0.25.
    // With tick_size=1.0 and a=5: upper = entry + 5*1.0 = entry+5.
    // With tick_size=0.25 and a=5: upper = entry + 5*0.25 = entry+1.25.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1003, 999, 1002));  // high=1003

    // tick_size=1.0, a=5: upper=1005. high=1003 < 1005 → no upper hit.
    auto labels_wide = compute_labels(bars, 5, 5, 40, 1.0);
    EXPECT_NE(labels_wide[0].label, 1) << "tick_size=1.0 should NOT trigger upper at 1003";

    // tick_size=0.25, a=5: upper=1001.25. high=1003 >= 1001.25 → upper hit.
    auto labels_narrow = compute_labels(bars, 5, 5, 40, 0.25);
    EXPECT_EQ(labels_narrow[0].label, 1) << "tick_size=0.25 SHOULD trigger upper at 1003";
}

// ===========================================================================
// Section 6: Edge case tests (~5)
// ===========================================================================

TEST(BarrierLabelEdge, LastBarCanOnlyTimeout) {
    auto bars = make_flat_bars(5, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    // Last bar has no future bars to scan → timeout.
    EXPECT_EQ(labels[4].label, 0);
    EXPECT_GE(labels[4].tau, 1);
}

TEST(BarrierLabelEdge, SecondToLastBarTauCappedAtOne) {
    // Second-to-last bar can only look 1 bar ahead.
    auto bars = make_flat_bars(5, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    // Bar 3 has 1 remaining bar (bar 4). If no hit → timeout with tau=1.
    EXPECT_EQ(labels[3].label, 0);
    EXPECT_EQ(labels[3].tau, 1);
}

TEST(BarrierLabelEdge, TMaxOneOnlyLooksAtNextBar) {
    // t_max=1: for each entry bar, only check the immediately next bar.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1000, 1001, 999, 1000));  // no hit
    bars.push_back(make_bar(2, 1000, 1006, 999, 1004));   // upper hit IF reachable

    // With t_max=1: bar 0 can only look at bar 1. Bar 1 doesn't hit → timeout.
    auto labels = compute_labels(bars, 20, 10, 1, 0.25);
    EXPECT_EQ(labels[0].label, 0);  // timeout because t_max=1 limits scan to bar 1 only
    EXPECT_EQ(labels[0].tau, 1);
}

TEST(BarrierLabelEdge, AllBarsSamePriceAllTimeout) {
    auto bars = make_flat_bars(10, 1000.0);
    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    for (const auto& lbl : labels) {
        EXPECT_EQ(lbl.label, 0) << "All-flat market should produce all timeouts, bar " << lbl.bar_index;
    }
}

TEST(BarrierLabelEdge, MonotonicallyRisingAllUpperHitsExceptNearEnd) {
    // Rising by 1.0 per bar. a=2, tick=0.25 → upper = entry + 0.50.
    // Each bar's close is 1.0 above previous, high is 1.5 above previous close.
    // So next bar's high should exceed entry + 0.50.
    auto bars = make_rising_bars(10, 1000.0, 1.0);
    auto labels = compute_labels(bars, 2, 100, 40, 0.25);

    // All bars except the last one should get label=+1 (upper hit from rising prices).
    // The last bar has no future bar → timeout.
    for (int i = 0; i < 9; ++i) {
        EXPECT_EQ(labels[i].label, 1)
            << "Rising market bar " << i << " should hit upper barrier";
    }
    // Last bar: timeout (no future bars).
    EXPECT_EQ(labels[9].label, 0);
}

// ===========================================================================
// Section 7: Integration / validation tests (~3)
// ===========================================================================

TEST(BarrierLabelIntegration, HandComputedSequence) {
    // Hand-computed sequence:
    // Bar 0: close=1000. Upper=1005, Lower=997.5.
    // Bar 1: H=1002, L=999 → no hit.
    // Bar 2: H=1004, L=998 → no hit.
    // Bar 3: H=1006, L=999 → upper hit (1006 >= 1005). label[0] = +1, tau=3.
    //
    // Bar 1: close=1001. Upper=1006, Lower=998.5.
    // Bar 2: H=1004, L=998 → lower hit (998 <= 998.5). label[1] = -1, tau=1.
    //
    // Bar 2: close=1003. Upper=1008, Lower=1000.5.
    // Bar 3: H=1006, L=999 → lower hit (999 <= 1000.5). label[2] = -1, tau=1.
    //
    // Bar 3: close=1004. Upper=1009, Lower=1001.5.
    // Bar 4: H=1003, L=1000 → lower hit (1000 <= 1001.5). label[3] = -1, tau=1.
    //
    // Bar 4: close=1000. No more bars. label[4] = 0, tau=1.

    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1000, 1002, 999,  1001));
    bars.push_back(make_bar(2, 1001, 1004, 998,  1003));
    bars.push_back(make_bar(3, 1003, 1006, 999,  1004));
    bars.push_back(make_bar(4, 1004, 1003, 1000, 1000));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    ASSERT_EQ(labels.size(), 5u);

    // Bar 0: upper hit at bar 3
    EXPECT_EQ(labels[0].label, 1);
    EXPECT_EQ(labels[0].tau, 3);
    EXPECT_EQ(labels[0].resolution_bar, 3);

    // Bar 1: lower hit at bar 2
    EXPECT_EQ(labels[1].label, -1);
    EXPECT_EQ(labels[1].tau, 1);
    EXPECT_EQ(labels[1].resolution_bar, 2);

    // Bar 2: lower hit at bar 3
    EXPECT_EQ(labels[2].label, -1);
    EXPECT_EQ(labels[2].tau, 1);
    EXPECT_EQ(labels[2].resolution_bar, 3);

    // Bar 3: lower hit at bar 4
    EXPECT_EQ(labels[3].label, -1);
    EXPECT_EQ(labels[3].tau, 1);
    EXPECT_EQ(labels[3].resolution_bar, 4);

    // Bar 4: timeout (last bar)
    EXPECT_EQ(labels[4].label, 0);
    EXPECT_EQ(labels[4].tau, 1);
}

TEST(BarrierLabelIntegration, LabelCountEqualsBarCountAlways) {
    // Test with various sizes.
    for (int n : {1, 2, 5, 10, 50, 100}) {
        auto bars = make_flat_bars(n, 1000.0);
        auto labels = compute_labels(bars, 20, 10, 40, 0.25);
        EXPECT_EQ(static_cast<int>(labels.size()), n)
            << "Label count should equal bar count for n=" << n;
    }
}

TEST(BarrierLabelIntegration, AllLabelsHaveValidFields) {
    // Build a realistic sequence with mixed outcomes.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0,  1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1,  1000, 1002, 998,  1001));
    bars.push_back(make_bar(2,  1001, 1003, 997,  1002));
    bars.push_back(make_bar(3,  1002, 1006, 996,  1003, 100, 1003,
                            {1002.0, 1003.0, 1006.0, 996.0},
                            {10, 10, 10, 10}));
    bars.push_back(make_bar(4,  1003, 1004, 999,  1001));
    bars.push_back(make_bar(5,  1001, 1002, 998,  999));
    bars.push_back(make_bar(6,  999,  1000, 997,  998));
    bars.push_back(make_bar(7,  998,  1001, 996,  1000));
    bars.push_back(make_bar(8,  1000, 1003, 998,  1002));
    bars.push_back(make_bar(9,  1002, 1004, 1000, 1003));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    ASSERT_EQ(labels.size(), 10u);

    for (size_t i = 0; i < labels.size(); ++i) {
        const auto& lbl = labels[i];
        // bar_index matches position
        EXPECT_EQ(lbl.bar_index, static_cast<int>(i))
            << "bar_index should equal position";
        // label in {-1, 0, +1}
        EXPECT_TRUE(lbl.label == -1 || lbl.label == 0 || lbl.label == 1)
            << "Label must be -1, 0, or +1 at bar " << i;
        // tau >= 1
        EXPECT_GE(lbl.tau, 1) << "tau must be >= 1 at bar " << i;
        // resolution_bar >= bar_index
        EXPECT_GE(lbl.resolution_bar, lbl.bar_index)
            << "resolution_bar must be >= bar_index at bar " << i;
        // resolution_bar = bar_index + tau
        EXPECT_EQ(lbl.resolution_bar, lbl.bar_index + lbl.tau)
            << "resolution_bar must equal bar_index + tau at bar " << i;
    }
}

// ===========================================================================
// Section 8: Additional tiebreak & gap direction edge cases
// ===========================================================================

TEST(BarrierLabelTiebreak, DualBreachResolvedBeforeScanningAllTrades) {
    // Dual breach: upper barrier crossed by second trade. Don't need to scan all.
    // Bar 0: close=1000. Upper=1005, Lower=997.5.
    // Bar 1: dual breach. Trades: [1002, 1005, 997, 996].
    // Trade[1]=1005 >= 1005 → upper, should stop scanning.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1006, 997, 1002, 100, 1002,
                            {1002.0, 1005.0, 997.0, 996.0},
                            {10, 10, 10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, 1);  // upper hit resolved at trade[1]
}

TEST(BarrierLabelTiebreak, GapDirectionUsesFirstTradeIfAvailable) {
    // When falling back to gap direction, use trade_prices[0] not open.
    // Bar 0: close=1000.
    // Bar 1: dual breach. First trade at 1006 (>= upper=1005).
    // Fall back to gap direction. first_trade=1006 >= prev_close=1000 → +1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    // open=998 (below prev close), but first trade=1006 (above prev close).
    // Gap direction should use first_trade, not open.
    bars.push_back(make_bar(1, 998, 1006, 997, 1002, 100, 1002,
                            {1006.0, 997.0},
                            {10, 10}));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    // First trade=1006 >= upper=1005, so gap fallback.
    // first_trade=1006 >= prev_close=1000 → +1.
    EXPECT_EQ(labels[0].label, 1);
}

TEST(BarrierLabelTiebreak, GapDirectionUsesOpenWhenNoTrades) {
    // Empty trade_prices → gap direction uses bar[j].open.
    // Bar 0: close=1000. Bar 1: open=998, no trades.
    // Gap direction: 998 < 1000 → -1.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 998, 1006, 997, 1002));  // empty trades

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // gap direction: open < prev close
}

// ===========================================================================
// Section 9: Default parameter tests
// ===========================================================================

TEST(BarrierLabelDefaults, DefaultParametersMatch) {
    // Default: a=20, b=10, t_max=40, tick_size=0.25.
    // Upper = entry + 20*0.25 = entry + 5.0.
    // Lower = entry - 10*0.25 = entry - 2.5.
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1005, 997, 1004));  // high=1005 == upper exact

    // With defaults: upper = 1005. high >= 1005 → upper hit.
    auto labels = compute_labels(bars);
    EXPECT_EQ(labels[0].label, 1);
}

TEST(BarrierLabelDefaults, UpperBarrierExactlyOnBoundary) {
    // Upper = 1005 exactly. High = 1005 exactly. Should still be a hit (>=).
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 1001, 1005, 998, 1004));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, 1);  // >= not just >
}

TEST(BarrierLabelDefaults, LowerBarrierExactlyOnBoundary) {
    // Lower = 997.5 exactly. Low = 997.5 exactly. Should still be a hit (<=).
    std::vector<TradeBar> bars;
    bars.push_back(make_bar(0, 1000, 1000, 1000, 1000));
    bars.push_back(make_bar(1, 999, 1001, 997.5, 998));

    auto labels = compute_labels(bars, 20, 10, 40, 0.25);
    EXPECT_EQ(labels[0].label, -1);  // <= not just <
}
