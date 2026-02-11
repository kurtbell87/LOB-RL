#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include "lob/barrier/trade_bar.h"
#include "lob/barrier/bar_builder.h"
#include "lob/book.h"
#include "lob/message.h"
#include "lob/session.h"
#include "lob/source.h"
#include "dbn_file_source.h"
#include "test_helpers.h"

// ===========================================================================
// Helpers: Build trade messages for a known price/size sequence
// ===========================================================================

static constexpr double TICK = 0.25;

// Build a complete RTH message stream for bar-building tests.
// Pre-market warmup + RTH trades forming exact bars.
// Returns {messages, expected_bar_count}.
struct StreamSpec {
    std::vector<Message> messages;
    int expected_bars;
};

static StreamSpec make_bar_stream(int bar_size, int num_bars,
                                  int partial_trades = 0) {
    std::vector<Message> msgs;
    uint64_t next_id = 1;

    // Pre-market: build book with bids and asks
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    double mid = 1000.0;

    // Add 2 bid levels + 2 ask levels during pre-market
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - TICK, 100, pre_ts));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + TICK, 100, pre_ts + NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - 2 * TICK, 200, pre_ts + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + 2 * TICK, 200, pre_ts + 3 * NS_PER_MIN));

    // RTH: trades to fill bars
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    int total_trades = num_bars * bar_size + partial_trades;

    double prices[] = {1000.25, 1000.50, 1000.00, 1000.75, 1000.25};
    int n_prices = 5;

    for (int i = 0; i < total_trades; ++i) {
        double p = prices[i % n_prices];
        // Alternate sides for aggressor diversity
        Message::Side side = (i % 3 == 0) ? Message::Side::Bid : Message::Side::Ask;
        msgs.push_back(make_trade_msg(next_id++, p, 1, rth_ts + i * 1'000'000ULL, side));
    }

    return {msgs, num_bars};
}

// ===========================================================================
// Section 1: TradeBar struct tests (~5)
// ===========================================================================

TEST(TradeBar, DefaultConstructionHasZeroValues) {
    TradeBar bar{};
    EXPECT_EQ(bar.bar_index, 0);
    EXPECT_DOUBLE_EQ(bar.open, 0.0);
    EXPECT_DOUBLE_EQ(bar.high, 0.0);
    EXPECT_DOUBLE_EQ(bar.low, 0.0);
    EXPECT_DOUBLE_EQ(bar.close, 0.0);
    EXPECT_EQ(bar.volume, 0);
    EXPECT_DOUBLE_EQ(bar.vwap, 0.0);
    EXPECT_EQ(bar.t_start, 0);
    EXPECT_EQ(bar.t_end, 0);
    EXPECT_TRUE(bar.trade_prices.empty());
    EXPECT_TRUE(bar.trade_sizes.empty());
}

TEST(TradeBar, TradePricesAndSizesPopulatedCorrectly) {
    TradeBar bar{};
    bar.trade_prices = {100.0, 100.25, 100.50};
    bar.trade_sizes = {5, 10, 3};

    EXPECT_EQ(bar.trade_prices.size(), 3u);
    EXPECT_EQ(bar.trade_sizes.size(), 3u);
    EXPECT_DOUBLE_EQ(bar.trade_prices[0], 100.0);
    EXPECT_DOUBLE_EQ(bar.trade_prices[2], 100.50);
    EXPECT_EQ(bar.trade_sizes[1], 10);
}

TEST(TradeBar, OHLCVFromKnownTradeSequence) {
    // Given 4 trades: 100.25 x2, 100.50 x3, 100.00 x1, 100.75 x4
    // Open = 100.25, High = 100.75, Low = 100.00, Close = 100.75, Volume = 10
    TradeBar bar{};
    bar.open = 100.25;
    bar.high = 100.75;
    bar.low = 100.00;
    bar.close = 100.75;
    bar.volume = 10;

    EXPECT_DOUBLE_EQ(bar.open, 100.25);
    EXPECT_DOUBLE_EQ(bar.high, 100.75);
    EXPECT_DOUBLE_EQ(bar.low, 100.00);
    EXPECT_DOUBLE_EQ(bar.close, 100.75);
    EXPECT_EQ(bar.volume, 10);
}

TEST(TradeBar, VWAPHandComputedFromKnownTrades) {
    // Trades: 100.0 x 5, 100.50 x 10, 101.0 x 5
    // VWAP = (100*5 + 100.5*10 + 101*5) / (5+10+5) = (500 + 1005 + 505) / 20 = 100.50
    TradeBar bar{};
    bar.trade_prices = {100.0, 100.50, 101.0};
    bar.trade_sizes = {5, 10, 5};
    bar.vwap = 100.50;
    bar.volume = 20;

    EXPECT_DOUBLE_EQ(bar.vwap, 100.50);
}

TEST(TradeBar, TimestampsMatchFirstAndLastTrade) {
    TradeBar bar{};
    bar.t_start = 1000000;
    bar.t_end = 2000000;

    EXPECT_EQ(bar.t_start, 1000000);
    EXPECT_EQ(bar.t_end, 2000000);
    EXPECT_LT(bar.t_start, bar.t_end);
}

// ===========================================================================
// Section 2: BarBuilder basic tests (~10)
// ===========================================================================

TEST(BarBuilder, ConstructsWithBarSizeAndConfig) {
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(10, cfg);

    EXPECT_TRUE(builder.bars().empty());
    EXPECT_TRUE(builder.accums().empty());
}

TEST(BarBuilder, EmptyStreamProducesNoBars) {
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(10, cfg);
    // No messages processed
    EXPECT_EQ(builder.bars().size(), 0u);
    EXPECT_EQ(builder.accums().size(), 0u);
}

TEST(BarBuilder, SingleBarOfExactBarSizeTrades) {
    int bar_size = 5;
    auto [msgs, expected] = make_bar_stream(bar_size, 1);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    EXPECT_EQ(builder.bars().size(), 1u);
    EXPECT_EQ(builder.accums().size(), 1u);
}

TEST(BarBuilder, TwoCompleteBarsWithCorrectIndices) {
    int bar_size = 5;
    auto [msgs, expected] = make_bar_stream(bar_size, 2);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    ASSERT_EQ(builder.bars().size(), 2u);
    EXPECT_EQ(builder.bars()[0].bar_index, 0);
    EXPECT_EQ(builder.bars()[1].bar_index, 1);
}

TEST(BarBuilder, IncompleteFinalBarNotEmittedWithoutFlush) {
    int bar_size = 10;
    // 1 full bar + 3 partial trades
    auto [msgs, expected] = make_bar_stream(bar_size, 1, 3);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    EXPECT_EQ(builder.bars().size(), 1u)
        << "Only the complete bar should be emitted without flush()";
}

TEST(BarBuilder, FlushEmitsPartialBar) {
    int bar_size = 10;
    // 1 full bar + 3 partial trades
    auto [msgs, expected] = make_bar_stream(bar_size, 1, 3);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    EXPECT_EQ(builder.bars().size(), 1u);
    bool flushed = builder.flush();
    EXPECT_TRUE(flushed);
    EXPECT_EQ(builder.bars().size(), 2u)
        << "flush() should emit the partial bar";
}

TEST(BarBuilder, FlushReturnsFalseWhenNoPendingTrades) {
    int bar_size = 5;
    // Exactly 1 bar, no leftovers
    auto [msgs, expected] = make_bar_stream(bar_size, 1);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    bool flushed = builder.flush();
    EXPECT_FALSE(flushed)
        << "flush() should return false when there are no pending trades";
}

TEST(BarBuilder, BarOHLCVValuesCorrect) {
    int bar_size = 4;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    // Pre-market: build book
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // RTH: 4 trades at known prices
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 1000.25, 1, rth_ts));       // open
    builder.process(make_trade_msg(next_id++, 1001.00, 1, rth_ts + 1'000'000));  // high
    builder.process(make_trade_msg(next_id++,  999.50, 1, rth_ts + 2'000'000));  // low
    builder.process(make_trade_msg(next_id++, 1000.50, 1, rth_ts + 3'000'000));  // close

    ASSERT_EQ(builder.bars().size(), 1u);
    const TradeBar& bar = builder.bars()[0];

    EXPECT_DOUBLE_EQ(bar.open, 1000.25);
    EXPECT_DOUBLE_EQ(bar.high, 1001.00);
    EXPECT_DOUBLE_EQ(bar.low, 999.50);
    EXPECT_DOUBLE_EQ(bar.close, 1000.50);
}

TEST(BarBuilder, BarVWAPHandComputed) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // 3 trades: 100.0 x 2, 100.50 x 3, 101.0 x 5
    // VWAP = (100*2 + 100.5*3 + 101*5) / (2+3+5) = (200 + 301.5 + 505) / 10 = 100.65
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.0, 2, rth_ts));
    builder.process(make_trade_msg(next_id++, 100.50, 3, rth_ts + 1'000'000));
    builder.process(make_trade_msg(next_id++, 101.0, 5, rth_ts + 2'000'000));

    ASSERT_EQ(builder.bars().size(), 1u);
    EXPECT_NEAR(builder.bars()[0].vwap, 100.65, 1e-9);
}

TEST(BarBuilder, TradePricesAndSizesStoredInBar) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.0, 2, rth_ts));
    builder.process(make_trade_msg(next_id++, 100.50, 3, rth_ts + 1'000'000));
    builder.process(make_trade_msg(next_id++, 101.0, 5, rth_ts + 2'000'000));

    ASSERT_EQ(builder.bars().size(), 1u);
    const TradeBar& bar = builder.bars()[0];

    ASSERT_EQ(bar.trade_prices.size(), 3u);
    ASSERT_EQ(bar.trade_sizes.size(), 3u);
    EXPECT_DOUBLE_EQ(bar.trade_prices[0], 100.0);
    EXPECT_DOUBLE_EQ(bar.trade_prices[1], 100.50);
    EXPECT_DOUBLE_EQ(bar.trade_prices[2], 101.0);
    EXPECT_EQ(bar.trade_sizes[0], 2);
    EXPECT_EQ(bar.trade_sizes[1], 3);
    EXPECT_EQ(bar.trade_sizes[2], 5);
}

TEST(BarBuilder, BarTimestampsMatchFirstAndLastTrade) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t t1 = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    uint64_t t2 = t1 + 1'000'000;
    uint64_t t3 = t1 + 2'000'000;
    builder.process(make_trade_msg(next_id++, 100.0, 1, t1));
    builder.process(make_trade_msg(next_id++, 100.25, 1, t2));
    builder.process(make_trade_msg(next_id++, 100.50, 1, t3));

    ASSERT_EQ(builder.bars().size(), 1u);
    EXPECT_EQ(builder.bars()[0].t_start, t1);
    EXPECT_EQ(builder.bars()[0].t_end, t3);
}

TEST(BarBuilder, VolumeEqualsSumOfTradeSizes) {
    int bar_size = 4;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.0, 3, rth_ts));
    builder.process(make_trade_msg(next_id++, 100.25, 7, rth_ts + 1'000'000));
    builder.process(make_trade_msg(next_id++, 100.50, 2, rth_ts + 2'000'000));
    builder.process(make_trade_msg(next_id++, 100.75, 8, rth_ts + 3'000'000));

    ASSERT_EQ(builder.bars().size(), 1u);
    const TradeBar& bar = builder.bars()[0];

    int expected_volume = 3 + 7 + 2 + 8;
    EXPECT_EQ(bar.volume, expected_volume);

    // Cross-check: sum of trade_sizes == volume
    int sum_sizes = 0;
    for (auto s : bar.trade_sizes) sum_sizes += s;
    EXPECT_EQ(bar.volume, sum_sizes);
}

// ===========================================================================
// Section 3: RTH filtering tests (~5)
// ===========================================================================

TEST(BarBuilder, PreMarketMessagesAreBookWarmupOnly) {
    int bar_size = 5;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Send many pre-market trades (should NOT form bars)
    for (int i = 0; i < 20; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0 + i * TICK, 1,
                                   pre_ts + i * 1'000'000));
    }

    EXPECT_EQ(builder.bars().size(), 0u)
        << "Pre-market trades should not form bars";
}

TEST(BarBuilder, PostMarketMessagesIgnored) {
    int bar_size = 5;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    // Pre-market book setup
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // Post-market trades
    uint64_t post_ts = DAY_BASE_NS + RTH_CLOSE_NS + NS_PER_MIN;
    for (int i = 0; i < 20; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   post_ts + i * 1'000'000));
    }

    EXPECT_EQ(builder.bars().size(), 0u)
        << "Post-market trades should not form bars";
}

TEST(BarBuilder, MixedPreMarketAndRTHOnlyRTHFormsBars) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Pre-market: adds + trades (shouldn't count toward bars)
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));
    // Pre-market trades
    for (int i = 0; i < 10; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   pre_ts + (i + 2) * NS_PER_MIN));
    }

    // RTH: exactly bar_size trades → 1 bar
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < bar_size; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.25, 1,
                                   rth_ts + i * 1'000'000));
    }

    EXPECT_EQ(builder.bars().size(), 1u)
        << "Only RTH trades should form bars";
}

TEST(BarBuilder, PreMarketBookStateCarriesIntoRTH) {
    // Pre-market builds the book; RTH bar's accum should see that book state
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build significant book depth during pre-market
    for (int i = 0; i < 5; ++i) {
        builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                 999.75 - i * TICK, 100 * (i + 1), pre_ts + i * NS_PER_MIN));
        builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                 1000.25 + i * TICK, 100 * (i + 1), pre_ts + (i + 5) * NS_PER_MIN));
    }

    // RTH: 2 trades to form a bar
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 1000.25, 1, rth_ts));
    builder.process(make_trade_msg(next_id++, 1000.50, 1, rth_ts + 1'000'000));

    ASSERT_EQ(builder.bars().size(), 1u);
    ASSERT_EQ(builder.accums().size(), 1u);

    // The accum should have non-zero BBO qty (from pre-market book)
    const BarBookAccum& acc = builder.accums()[0];
    EXPECT_GT(acc.bid_qty, 0u) << "Pre-market book state should carry into RTH";
    EXPECT_GT(acc.ask_qty, 0u) << "Pre-market book state should carry into RTH";
}

TEST(BarBuilder, RthOpenAndCloseNsReturnCorrectValues) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));

    // Send an RTH message to establish the day
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 1000.0, 1, rth_ts));

    // rth_open_ns and rth_close_ns should be epoch-based (not time-of-day)
    uint64_t open_ns = builder.rth_open_ns();
    uint64_t close_ns = builder.rth_close_ns();

    EXPECT_EQ(open_ns, DAY_BASE_NS + RTH_OPEN_NS);
    EXPECT_EQ(close_ns, DAY_BASE_NS + RTH_CLOSE_NS);
    EXPECT_GT(close_ns, open_ns);
}

// ===========================================================================
// Section 4: BarBookAccum tests (~15)
// ===========================================================================

TEST(BarBookAccum, DefaultConstructionHasNeutralValues) {
    BarBookAccum acc{};
    EXPECT_EQ(acc.bid_qty, 0u);
    EXPECT_EQ(acc.ask_qty, 0u);
    EXPECT_EQ(acc.total_bid_3, 0u);
    EXPECT_EQ(acc.total_ask_3, 0u);
    EXPECT_EQ(acc.total_bid_5, 0u);
    EXPECT_EQ(acc.total_ask_5, 0u);
    EXPECT_EQ(acc.total_bid_10, 0u);
    EXPECT_EQ(acc.total_ask_10, 0u);
    EXPECT_EQ(acc.bid_cancels, 0);
    EXPECT_EQ(acc.ask_cancels, 0);
    EXPECT_DOUBLE_EQ(acc.ofi_signed_volume, 0.0);
    EXPECT_DOUBLE_EQ(acc.total_add_volume, 0.0);
    EXPECT_TRUE(std::isnan(acc.wmid_first));
    EXPECT_TRUE(std::isnan(acc.wmid_end));
    EXPECT_TRUE(acc.spread_samples.empty());
    EXPECT_TRUE(std::isnan(acc.vamp_at_mid));
    EXPECT_TRUE(std::isnan(acc.vamp_at_end));
    EXPECT_DOUBLE_EQ(acc.buy_aggressor_vol, 0.0);
    EXPECT_DOUBLE_EQ(acc.sell_aggressor_vol, 0.0);
    EXPECT_EQ(acc.n_trades, 0);
    EXPECT_EQ(acc.n_cancels, 0);
}

TEST(BarBuilder, AccumBBOQtySnapshotAtBarClose) {
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build book: bid 999.75 x 50, ask 1000.25 x 75
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 50, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 75, pre_ts + NS_PER_MIN));

    // RTH: 2 trades → bar complete
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 1000.00, 1, rth_ts));
    builder.process(make_trade_msg(next_id++, 1000.25, 1, rth_ts + 1'000'000));

    ASSERT_EQ(builder.accums().size(), 1u);
    const BarBookAccum& acc = builder.accums()[0];

    // BBO qty should snapshot the book at bar close time
    EXPECT_EQ(acc.bid_qty, 50u);
    EXPECT_EQ(acc.ask_qty, 75u);
}

TEST(BarBuilder, AccumDepthSnapshotAtBarClose) {
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build 10 levels each side during pre-market
    for (int i = 0; i < 10; ++i) {
        builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                 100.0 - i * TICK, 10 * (i + 1), pre_ts + i * NS_PER_MIN));
        builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                 100.25 + i * TICK, 10 * (i + 1), pre_ts + (i + 10) * NS_PER_MIN));
    }

    // RTH: 2 trades → bar
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.25, 1, rth_ts));
    builder.process(make_trade_msg(next_id++, 100.50, 1, rth_ts + 1'000'000));

    ASSERT_EQ(builder.accums().size(), 1u);
    const BarBookAccum& acc = builder.accums()[0];

    // Depth(3) should be sum of top 3 levels' qty
    EXPECT_GT(acc.total_bid_3, 0u);
    EXPECT_GT(acc.total_ask_3, 0u);
    // Depth(5) >= Depth(3)
    EXPECT_GE(acc.total_bid_5, acc.total_bid_3);
    EXPECT_GE(acc.total_ask_5, acc.total_ask_3);
    // Depth(10) >= Depth(5)
    EXPECT_GE(acc.total_bid_10, acc.total_bid_5);
    EXPECT_GE(acc.total_ask_10, acc.total_ask_5);
}

TEST(BarBuilder, AccumCancelCountsPerSide) {
    int bar_size = 4;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Pre-market: build book
    uint64_t bid_oid = next_id++;
    uint64_t ask_oid1 = next_id++;
    uint64_t ask_oid2 = next_id++;
    builder.process(make_msg(bid_oid, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(ask_oid1, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));
    builder.process(make_msg(ask_oid2, Message::Side::Ask, Message::Action::Add,
                             1000.50, 100, pre_ts + 2 * NS_PER_MIN));

    // RTH: cancels + trades
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;

    // Cancel 1 bid, 2 asks during bar
    builder.process(make_msg(bid_oid, Message::Side::Bid, Message::Action::Cancel,
                             999.75, 50, rth_ts));
    builder.process(make_msg(ask_oid1, Message::Side::Ask, Message::Action::Cancel,
                             1000.25, 50, rth_ts + 1'000'000));
    builder.process(make_msg(ask_oid2, Message::Side::Ask, Message::Action::Cancel,
                             1000.50, 50, rth_ts + 2'000'000));

    // 4 trades to complete bar
    for (int i = 0; i < 4; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (3 + i) * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_EQ(builder.accums()[0].bid_cancels, 1);
    EXPECT_EQ(builder.accums()[0].ask_cancels, 2);
}

TEST(BarBuilder, AccumOFIAddAtBBOIncreasesSigned) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build book: best bid at 999.75, best ask at 1000.25
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // RTH: Add at BBO level on bid side, then trades
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 20, rth_ts));  // Add at best bid

    // 3 trades to complete bar
    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (i + 1) * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    // OFI should be positive (bid add contributes positive signed volume)
    // or at least non-zero
    EXPECT_NE(builder.accums()[0].ofi_signed_volume, 0.0)
        << "Add at BBO should affect OFI signed volume";
    EXPECT_GT(builder.accums()[0].total_add_volume, 0.0)
        << "Add should contribute to total add volume";
}

TEST(BarBuilder, AccumOFIAddAwayFromBBODoesNotAffectOFI) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build book: best bid at 999.75, best ask at 1000.25
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // RTH: Add away from BBO (deep level), then trades
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             998.00, 50, rth_ts));  // Add far from best bid

    // 3 trades to complete bar
    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (i + 1) * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_DOUBLE_EQ(builder.accums()[0].ofi_signed_volume, 0.0)
        << "Add away from BBO should NOT affect OFI signed volume";
}

TEST(BarBuilder, AccumWmidFirstCapturedAfterFirstEvent) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build book
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // RTH: 3 trades to form a bar
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + i * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_FALSE(std::isnan(builder.accums()[0].wmid_first))
        << "wmid_first should be set after first in-bar event";
    EXPECT_TRUE(std::isfinite(builder.accums()[0].wmid_first));
}

TEST(BarBuilder, AccumWmidEndCapturedAtBarClose) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + i * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_FALSE(std::isnan(builder.accums()[0].wmid_end))
        << "wmid_end should be set at bar close";
    EXPECT_TRUE(std::isfinite(builder.accums()[0].wmid_end));
}

TEST(BarBuilder, AccumSpreadSamplesCollected) {
    int bar_size = 5;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // RTH: some adds + trades to populate spread samples
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.50, 50, rth_ts));  // event → sample spread
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.50, 50, rth_ts + 1'000'000));  // event → sample spread

    for (int i = 0; i < 5; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (2 + i) * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_GT(builder.accums()[0].spread_samples.size(), 0u)
        << "Spread samples should be collected during bar";
    for (double s : builder.accums()[0].spread_samples) {
        EXPECT_GE(s, 0.0) << "Spread should be non-negative";
    }
}

TEST(BarBuilder, AccumVAMPAtBarMidpoint) {
    // VAMP should be sampled at the bar's temporal midpoint
    int bar_size = 4;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build deep book for valid VAMP
    for (int i = 0; i < 5; ++i) {
        builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                 100.0 - i * TICK, 100, pre_ts + i * NS_PER_MIN));
        builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                 100.25 + i * TICK, 100, pre_ts + (i + 5) * NS_PER_MIN));
    }

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < 4; ++i) {
        builder.process(make_trade_msg(next_id++, 100.25, 1,
                                   rth_ts + i * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    // vamp_at_mid should be sampled when we cross the midpoint of the bar
    // It may be NaN if no valid VAMP can be computed, but if the book has depth it should be finite
    EXPECT_FALSE(std::isnan(builder.accums()[0].vamp_at_mid))
        << "VAMP at bar midpoint should be set when book has depth";
    EXPECT_TRUE(std::isfinite(builder.accums()[0].vamp_at_mid));
}

TEST(BarBuilder, AccumVAMPAtBarEnd) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Build deep book
    for (int i = 0; i < 5; ++i) {
        builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                 100.0 - i * TICK, 100, pre_ts + i * NS_PER_MIN));
        builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                 100.25 + i * TICK, 100, pre_ts + (i + 5) * NS_PER_MIN));
    }

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 100.25, 1,
                                   rth_ts + i * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_FALSE(std::isnan(builder.accums()[0].vamp_at_end))
        << "VAMP at bar end should be set when book has depth";
    EXPECT_TRUE(std::isfinite(builder.accums()[0].vamp_at_end));
}

TEST(BarBuilder, AccumAggressorVolumesBuyPassiveAsk) {
    // When a trade has side=Ask (passive side is ask), it's a buy aggressor
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    // 2 trades, both with passive Ask side → buy aggressor
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Trade,
                             1000.25, 5, rth_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Trade,
                             1000.25, 3, rth_ts + 1'000'000));

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_DOUBLE_EQ(builder.accums()[0].buy_aggressor_vol, 8.0)
        << "Passive ask → buy aggressor volume should be 5 + 3 = 8";
    EXPECT_DOUBLE_EQ(builder.accums()[0].sell_aggressor_vol, 0.0);
}

TEST(BarBuilder, AccumAggressorVolumesSellPassiveBid) {
    // When a trade has side=Bid (passive side is bid), it's a sell aggressor
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    // 2 trades, both with passive Bid side → sell aggressor
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Trade,
                             999.75, 4, rth_ts));
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Trade,
                             999.75, 6, rth_ts + 1'000'000));

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_DOUBLE_EQ(builder.accums()[0].sell_aggressor_vol, 10.0)
        << "Passive bid → sell aggressor volume should be 4 + 6 = 10";
    EXPECT_DOUBLE_EQ(builder.accums()[0].buy_aggressor_vol, 0.0);
}

TEST(BarBuilder, AccumNTradesCount) {
    int bar_size = 5;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < 5; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + i * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_EQ(builder.accums()[0].n_trades, 5);
}

TEST(BarBuilder, AccumNCancelsCount) {
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    uint64_t bid_oid1 = next_id++;
    uint64_t bid_oid2 = next_id++;
    builder.process(make_msg(bid_oid1, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(bid_oid2, Message::Side::Bid, Message::Action::Add,
                             999.50, 100, pre_ts + NS_PER_MIN));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + 2 * NS_PER_MIN));

    // RTH: 2 cancels + 3 trades
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_msg(bid_oid1, Message::Side::Bid, Message::Action::Cancel,
                             999.75, 50, rth_ts));
    builder.process(make_msg(bid_oid2, Message::Side::Bid, Message::Action::Cancel,
                             999.50, 50, rth_ts + 1'000'000));

    for (int i = 0; i < 3; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (2 + i) * 1'000'000));
    }

    ASSERT_EQ(builder.accums().size(), 1u);
    EXPECT_EQ(builder.accums()[0].n_cancels, 2);
}

TEST(BarBuilder, AccumResetsBetwenBars) {
    int bar_size = 2;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;

    // Pre-market: build book with cancel-able orders
    uint64_t bid_oid = next_id++;
    builder.process(make_msg(bid_oid, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;

    // Bar 0: 1 cancel + 2 trades (cancel happens to bid)
    builder.process(make_msg(bid_oid, Message::Side::Bid, Message::Action::Cancel,
                             999.75, 50, rth_ts));
    builder.process(make_trade_msg(next_id++, 1000.0, 3, rth_ts + 1'000'000));
    builder.process(make_trade_msg(next_id++, 1000.25, 2, rth_ts + 2'000'000));

    // Bar 1: just 2 trades (no cancels)
    builder.process(make_trade_msg(next_id++, 1000.50, 1, rth_ts + 3'000'000));
    builder.process(make_trade_msg(next_id++, 1000.75, 1, rth_ts + 4'000'000));

    ASSERT_EQ(builder.accums().size(), 2u);

    // Bar 0 should have 1 bid cancel
    EXPECT_EQ(builder.accums()[0].bid_cancels, 1);
    EXPECT_EQ(builder.accums()[0].n_cancels, 1);

    // Bar 1 should have 0 cancels (reset between bars)
    EXPECT_EQ(builder.accums()[1].bid_cancels, 0);
    EXPECT_EQ(builder.accums()[1].ask_cancels, 0);
    EXPECT_EQ(builder.accums()[1].n_cancels, 0);

    // n_trades should be correct for each bar independently
    EXPECT_EQ(builder.accums()[0].n_trades, 2);
    EXPECT_EQ(builder.accums()[1].n_trades, 2);
}

// ===========================================================================
// Section 5: DbnFileSource 'R' action mapping tests (~2)
// ===========================================================================

TEST(DbnFileSourceRAction, ClearMapsToCancel) {
    // After the spec change, 'R' should map to Cancel in DbnFileSource.
    // We test via the lower-level map_mbo_to_message function.
    // The spec says: Map 'R' to Message::Action::Cancel.
    //
    // NOTE: This test will initially FAIL because map_mbo_to_message currently
    // returns false for action='R'. The GREEN phase must modify map_action()
    // in dbn_file_source.cpp (or dbn_message_map.cpp) to map 'R' → Cancel.
    databento::MboMsg mbo{};
    mbo.hd.instrument_id = 12345;
    mbo.hd.ts_event = 1000;
    mbo.order_id = 99;
    mbo.price = 999750000000LL;
    mbo.size = 10;
    mbo.action = 'R';
    mbo.side = 'B';
    mbo.flags = 0;

    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345))
        << "'R' (Clear) action should now be accepted, not skipped";
    EXPECT_EQ(msg.action, Message::Action::Cancel)
        << "'R' (Clear) should map to Cancel";
}

TEST(DbnFileSourceRAction, BookHandlesClearMappedCancelCorrectly) {
    // After 'R' maps to Cancel, the book should process it like a normal cancel.
    // Add an order, then send a Cancel (simulating 'R' mapped to Cancel).
    Book book;
    uint64_t oid = 42;

    // Add an order
    book.apply(make_msg(oid, Message::Side::Bid, Message::Action::Add,
                        1000.0, 50));
    EXPECT_EQ(book.best_bid_qty(), 50u);

    // Cancel it (as 'R' would after mapping)
    book.apply(make_msg(oid, Message::Side::Bid, Message::Action::Cancel,
                        1000.0, 50));
    // After cancel, that level should be gone
    EXPECT_TRUE(std::isnan(book.best_bid()) || book.best_bid_qty() == 0u)
        << "Book should process 'R'-mapped Cancel like a normal cancel";
}

// ===========================================================================
// Section 6: Integration tests (~3)
// ===========================================================================

TEST(BarBuilder, IntegrationFullStreamCorrectBarCount) {
    int bar_size = 10;
    int expected_bars = 5;
    auto [msgs, exp] = make_bar_stream(bar_size, expected_bars);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    EXPECT_EQ(static_cast<int>(builder.bars().size()), expected_bars);
    EXPECT_EQ(builder.bars().size(), builder.accums().size())
        << "bars and accums vectors should have the same size";
}

TEST(BarBuilder, IntegrationBarClosePricesFormValidSeries) {
    int bar_size = 5;
    int num_bars = 4;
    auto [msgs, exp] = make_bar_stream(bar_size, num_bars);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    ASSERT_EQ(static_cast<int>(builder.bars().size()), num_bars);

    for (int i = 0; i < num_bars; ++i) {
        const TradeBar& bar = builder.bars()[i];
        EXPECT_TRUE(std::isfinite(bar.close))
            << "Bar " << i << " close price should be finite";
        EXPECT_GT(bar.close, 0.0)
            << "Bar " << i << " close price should be positive";
        EXPECT_GE(bar.high, bar.low)
            << "Bar " << i << " high should be >= low";
        EXPECT_GE(bar.high, bar.open)
            << "Bar " << i << " high should be >= open";
        EXPECT_GE(bar.high, bar.close)
            << "Bar " << i << " high should be >= close";
        EXPECT_LE(bar.low, bar.open)
            << "Bar " << i << " low should be <= open";
        EXPECT_LE(bar.low, bar.close)
            << "Bar " << i << " low should be <= close";
        EXPECT_GT(bar.volume, 0)
            << "Bar " << i << " should have positive volume";
        EXPECT_TRUE(std::isfinite(bar.vwap))
            << "Bar " << i << " VWAP should be finite";
        EXPECT_GE(bar.vwap, bar.low)
            << "Bar " << i << " VWAP should be >= low";
        EXPECT_LE(bar.vwap, bar.high)
            << "Bar " << i << " VWAP should be <= high";
    }
}

TEST(BarBuilder, IntegrationAccumValuesNonDefaultForPopulatedBars) {
    int bar_size = 5;
    int num_bars = 3;
    auto [msgs, exp] = make_bar_stream(bar_size, num_bars);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    ASSERT_EQ(static_cast<int>(builder.accums().size()), num_bars);

    for (int i = 0; i < num_bars; ++i) {
        const BarBookAccum& acc = builder.accums()[i];

        // These should be set from the book (populated via pre-market warmup)
        EXPECT_GT(acc.bid_qty, 0u)
            << "Accum " << i << " bid_qty should be > 0 (book was populated)";
        EXPECT_GT(acc.ask_qty, 0u)
            << "Accum " << i << " ask_qty should be > 0";

        // n_trades should match bar_size
        EXPECT_EQ(acc.n_trades, bar_size)
            << "Accum " << i << " should have n_trades == bar_size";

        // wmid should be set
        EXPECT_FALSE(std::isnan(acc.wmid_end))
            << "Accum " << i << " wmid_end should be set";
        EXPECT_FALSE(std::isnan(acc.wmid_first))
            << "Accum " << i << " wmid_first should be set";
    }
}

// ===========================================================================
// Section 7: Edge case / additional tests
// ===========================================================================

TEST(BarBuilder, BarsAndAccumsAlwaysSameSize) {
    // For any stream, bars() and accums() must always be the same length
    int bar_size = 3;
    auto [msgs, exp] = make_bar_stream(bar_size, 4, 1);  // 4 bars + 1 partial

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    EXPECT_EQ(builder.bars().size(), builder.accums().size());

    builder.flush();
    EXPECT_EQ(builder.bars().size(), builder.accums().size())
        << "bars and accums must stay in sync after flush()";
}

TEST(BarBuilder, FlushOnEmptyProducesNothing) {
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(10, cfg);

    bool flushed = builder.flush();
    EXPECT_FALSE(flushed);
    EXPECT_EQ(builder.bars().size(), 0u);
    EXPECT_EQ(builder.accums().size(), 0u);
}

TEST(BarBuilder, BarSizeOneCreatesBarPerTrade) {
    int bar_size = 1;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    int n_trades = 7;
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    for (int i = 0; i < n_trades; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0 + i * TICK, 1,
                                   rth_ts + i * 1'000'000));
    }

    EXPECT_EQ(static_cast<int>(builder.bars().size()), n_trades)
        << "bar_size=1 should create one bar per trade";
}

TEST(BarBuilder, ConsecutiveBarTimestampsDoNotOverlap) {
    int bar_size = 3;
    int num_bars = 4;
    auto [msgs, exp] = make_bar_stream(bar_size, num_bars);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    ASSERT_EQ(static_cast<int>(builder.bars().size()), num_bars);
    for (int i = 1; i < num_bars; ++i) {
        EXPECT_GT(builder.bars()[i].t_start, builder.bars()[i - 1].t_end)
            << "Bar " << i << " t_start should be after bar " << (i - 1) << " t_end";
    }
}

TEST(BarBuilder, NonTradeRTHEventsDoNotCountTowardBarSize) {
    // Only trades count toward bar_size; adds/cancels during RTH should not
    int bar_size = 3;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;

    // Send 10 RTH Add messages (should NOT count toward bar)
    for (int i = 0; i < 10; ++i) {
        builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                 999.50 - i * TICK, 50, rth_ts + i * 1'000'000));
    }

    // No bars yet
    EXPECT_EQ(builder.bars().size(), 0u)
        << "Non-trade RTH events should not form bars";

    // Now send exactly bar_size trades
    for (int i = 0; i < bar_size; ++i) {
        builder.process(make_trade_msg(next_id++, 1000.0, 1,
                                   rth_ts + (10 + i) * 1'000'000));
    }

    EXPECT_EQ(builder.bars().size(), 1u)
        << "Only trades should count toward bar_size";
}

TEST(BarBuilder, LargeBarSizeCorrectBehavior) {
    int bar_size = 100;
    auto [msgs, expected] = make_bar_stream(bar_size, 2);

    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    for (const auto& m : msgs) {
        builder.process(m);
    }

    ASSERT_EQ(static_cast<int>(builder.bars().size()), 2);
    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(builder.bars()[i].volume, bar_size)
            << "Each bar should have volume == bar_size (each trade qty=1)";
        EXPECT_EQ(static_cast<int>(builder.bars()[i].trade_prices.size()), bar_size);
        EXPECT_EQ(static_cast<int>(builder.bars()[i].trade_sizes.size()), bar_size);
    }
}

// ===========================================================================
// Section 8: Flush partial bar correctness
// ===========================================================================

TEST(BarBuilder, FlushedPartialBarHasCorrectOHLCV) {
    int bar_size = 10;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    // Only 3 trades (bar_size=10, so this is partial)
    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.50, 2, rth_ts));       // open
    builder.process(make_trade_msg(next_id++, 101.00, 3, rth_ts + 1'000'000));  // high
    builder.process(make_trade_msg(next_id++, 100.00, 1, rth_ts + 2'000'000));  // low, close

    EXPECT_EQ(builder.bars().size(), 0u);
    builder.flush();

    ASSERT_EQ(builder.bars().size(), 1u);
    const TradeBar& bar = builder.bars()[0];

    EXPECT_DOUBLE_EQ(bar.open, 100.50);
    EXPECT_DOUBLE_EQ(bar.high, 101.00);
    EXPECT_DOUBLE_EQ(bar.low, 100.00);
    EXPECT_DOUBLE_EQ(bar.close, 100.00);
    EXPECT_EQ(bar.volume, 6);  // 2 + 3 + 1
}

TEST(BarBuilder, FlushCalledTwiceDoesNotDuplicateBars) {
    int bar_size = 10;
    SessionConfig cfg = SessionConfig::default_rth();
    BarBuilder builder(bar_size, cfg);

    uint64_t next_id = 1;
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    builder.process(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                             999.75, 100, pre_ts));
    builder.process(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                             1000.25, 100, pre_ts + NS_PER_MIN));

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    builder.process(make_trade_msg(next_id++, 100.0, 1, rth_ts));
    builder.process(make_trade_msg(next_id++, 100.25, 1, rth_ts + 1'000'000));

    bool first_flush = builder.flush();
    bool second_flush = builder.flush();

    EXPECT_TRUE(first_flush);
    EXPECT_FALSE(second_flush)
        << "Second flush should return false (no pending trades)";
    EXPECT_EQ(builder.bars().size(), 1u)
        << "Flush called twice should not duplicate bars";
}
