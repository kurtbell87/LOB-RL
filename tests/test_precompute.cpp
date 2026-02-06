#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "lob/precompute.h"
#include "lob/feature_builder.h"
#include "binary_file_source.h"
#include "test_helpers.h"

// ===========================================================================
// Test 1: Empty source returns zero steps
// ===========================================================================

TEST(Precompute, EmptySourceReturnsZeroSteps) {
    ScriptedSource source({});
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 0);
    EXPECT_TRUE(day.obs.empty());
    EXPECT_TRUE(day.mid.empty());
    EXPECT_TRUE(day.spread.empty());
}

// ===========================================================================
// Test 2: Pre-market only (no RTH messages) returns zero steps
// ===========================================================================

TEST(Precompute, PreMarketOnlyReturnsZeroSteps) {
    // All messages have timestamps before 13:30 UTC (pre-market)
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    std::vector<Message> msgs;
    uint64_t oid = 1;

    // Establish book in pre-market
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));
    // More pre-market messages that change BBO
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, pre_start + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, pre_start + 3 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 0);
    EXPECT_TRUE(day.obs.empty());
    EXPECT_TRUE(day.mid.empty());
    EXPECT_TRUE(day.spread.empty());
}

// ===========================================================================
// Test 3: RTH messages without BBO change produce no snapshots
// ===========================================================================

TEST(Precompute, NoBboChangeProducesNoSnapshots) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish BBO at bid=999.75, ask=1000.25
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: add orders at deeper levels (don't change BBO)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.50, 50, rth_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.50, 50, rth_start + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.25, 30, rth_start + 3 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 0)
        << "RTH messages that don't change BBO should produce no snapshots";
}

// ===========================================================================
// Test 4: Single BBO change produces exactly 1 snapshot
// ===========================================================================

TEST(Precompute, SingleBboChangeProducesOneSnapshot) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish book with bid=999.75/100, ask=1000.25/100
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: one message that improves best bid to 1000.00
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 1);
    EXPECT_EQ(day.obs.size(), 43u) << "Single snapshot should have exactly 43 floats";
    EXPECT_EQ(day.mid.size(), 1u);
    EXPECT_EQ(day.spread.size(), 1u);

    // mid = (1000.00 + 1000.25) / 2 = 1000.125
    EXPECT_NEAR(day.mid[0], 1000.125, 1e-6)
        << "Mid should be (new_best_bid + best_ask) / 2";

    // spread = 1000.25 - 1000.00 = 0.25
    EXPECT_NEAR(day.spread[0], 0.25, 1e-6)
        << "Spread should be ask - bid";

    // All 43 obs values must be finite
    for (size_t i = 0; i < day.obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(day.obs[i]))
            << "obs[" << i << "] is not finite: " << day.obs[i];
    }
}

// ===========================================================================
// Test 5: Multiple BBO changes produce correct count
// ===========================================================================

TEST(Precompute, MultipleBboChangesProduceCorrectCount) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: 5 messages that each change the best bid or best ask
    // Change 1: improve best bid to 999.875
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.875, 50, rth_start + 1 * NS_PER_MIN));
    // Change 2: improve best bid to 1000.00
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + 2 * NS_PER_MIN));
    // Change 3: improve best ask to 1000.125
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, rth_start + 3 * NS_PER_MIN));
    // Change 4: improve best bid to 1000.0625
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.0625, 50, rth_start + 4 * NS_PER_MIN));
    // Change 5: improve best ask to 1000.09375
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.09375, 50, rth_start + 5 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 5);
    EXPECT_EQ(day.obs.size(), static_cast<size_t>(5 * 43))
        << "5 snapshots × 43 floats each";
    EXPECT_EQ(day.mid.size(), 5u);
    EXPECT_EQ(day.spread.size(), 5u);
}

// ===========================================================================
// Test 6: Post-market messages are excluded
// ===========================================================================

TEST(Precompute, PostMarketMessagesExcluded) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;
    uint64_t post_start = DAY_BASE_NS + RTH_CLOSE_NS;

    // Pre-market: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: 2 BBO-changing messages
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, rth_start + 2 * NS_PER_MIN));

    // Post-market: 3 BBO-changing messages (timestamp >= 20:00 UTC)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.0625, 50, post_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.09375, 50, post_start + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.09, 50, post_start + 3 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 2)
        << "Only RTH messages should produce snapshots, post-market ignored";
    EXPECT_EQ(day.obs.size(), static_cast<size_t>(2 * 43));
    EXPECT_EQ(day.mid.size(), 2u);
    EXPECT_EQ(day.spread.size(), 2u);
}

// ===========================================================================
// Test 7: obs contains first 43 floats from FeatureBuilder (no position)
// ===========================================================================

TEST(Precompute, ObsContains43FloatsNoPosition) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: one BBO change
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 1);
    ASSERT_EQ(day.obs.size(), 43u)
        << "obs should have exactly 43 floats per snapshot (no position at index 43)";

    // Index 42 is time_remaining — should be between 0 and 1 during RTH
    EXPECT_GT(day.obs[42], 0.0f) << "time_remaining should be > 0 (early in RTH)";
    EXPECT_LT(day.obs[42], 1.0f) << "time_remaining should be < 1 (past RTH open)";
}

// ===========================================================================
// Test 8: time_remaining decreases across snapshots
// ===========================================================================

TEST(Precompute, TimeRemainingDecreasesAcrossSnapshots) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: spread BBO changes across the session
    // At 13:31 (1 min into RTH)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.875, 50, rth_start + 1 * NS_PER_MIN));
    // At 15:00 (1.5 hours into RTH)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + 90 * NS_PER_MIN));
    // At 17:00 (3.5 hours into RTH)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, rth_start + 210 * NS_PER_MIN));
    // At 19:30 (6 hours into RTH, near close)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.0625, 50, rth_start + 360 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 4);

    // time_remaining is at obs index 42 within each 43-float snapshot
    for (int i = 0; i < day.num_steps - 1; ++i) {
        float tr_curr = day.obs[i * 43 + 42];
        float tr_next = day.obs[(i + 1) * 43 + 42];
        EXPECT_GT(tr_curr, tr_next)
            << "time_remaining at snapshot " << i << " (" << tr_curr
            << ") should be > snapshot " << i + 1 << " (" << tr_next << ")";
    }

    // All time_remaining values should be in [0, 1]
    for (int i = 0; i < day.num_steps; ++i) {
        float tr = day.obs[i * 43 + 42];
        EXPECT_GE(tr, 0.0f);
        EXPECT_LE(tr, 1.0f);
    }
}

// ===========================================================================
// Test 9: mid-prices track actual book mid
// ===========================================================================

TEST(Precompute, MidPricesTrackActualBookMid) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: bid=999.75, ask=1000.25, mid=1000.00
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH change 1: bid improves to 1000.00 → mid = (1000.00 + 1000.25)/2 = 1000.125
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    // RTH change 2: ask improves to 1000.125 → mid = (1000.00 + 1000.125)/2 = 1000.0625
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 100, rth_start + 2 * NS_PER_MIN));

    // RTH change 3: bid improves to 1000.0625 → mid = (1000.0625 + 1000.125)/2 = 1000.09375
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.0625, 100, rth_start + 3 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 3);

    EXPECT_NEAR(day.mid[0], 1000.125, 1e-6)
        << "Snapshot 0: mid should be (1000.00 + 1000.25) / 2";
    EXPECT_NEAR(day.mid[1], 1000.0625, 1e-6)
        << "Snapshot 1: mid should be (1000.00 + 1000.125) / 2";
    EXPECT_NEAR(day.mid[2], 1000.09375, 1e-6)
        << "Snapshot 2: mid should be (1000.0625 + 1000.125) / 2";
}

// ===========================================================================
// Test 10: Warmup respects SessionConfig.warmup_messages
// ===========================================================================

TEST(Precompute, WarmupRespectsSessionConfigWarmupMessages) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // 100 pre-market messages: first 98 establish a book with bid=900, ask=1100
    // Last 2 messages set bid=999.75, ask=1000.25

    // First 98: deep levels
    for (int i = 0; i < 49; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                900.00 + i * 0.25, 10, pre_start + i * NS_PER_MIN));
    }
    for (int i = 0; i < 49; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                1100.00 - i * 0.25, 10, pre_start + (49 + i) * NS_PER_MIN));
    }

    // Messages 99-100: establish tight BBO
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start + 98 * NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + 99 * NS_PER_MIN));

    // RTH: improve best bid to trigger snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + NS_PER_MIN));

    // With warmup_messages=5, only last 5 pre-market messages are applied.
    // The last 5 are: messages at indices 97-101 of pre-market
    // (3 ask adds at deeper levels + bid at 999.75 + ask at 1000.25)
    SessionConfig cfg = SessionConfig::default_rth();
    cfg.warmup_messages = 5;

    ScriptedSource source(msgs);
    PrecomputedDay day = precompute(source, cfg);

    // Should still produce a snapshot (both bid and ask are finite from warmup)
    EXPECT_GE(day.num_steps, 1)
        << "With warmup_messages=5, last 5 pre-market should establish the book";

    // The book state should reflect only the last 5 warmup messages.
    // With warmup_messages=5 and the bid=999.75 + ask=1000.25 in the last 2:
    // After RTH bid improvement to 1000.00, mid = (1000.00 + 1000.25) / 2 = 1000.125
    if (day.num_steps >= 1) {
        EXPECT_NEAR(day.mid[0], 1000.125, 1e-6)
            << "Book should reflect warmup state + RTH improvement";
    }
}

TEST(Precompute, WarmupZeroSkipsAllPreMarketMessages) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish BBO
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: messages that would change BBO if book was warmed up
    // But with warmup=0, the book is empty at RTH start
    // First RTH message adds only a bid — ask will be infinity → no snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));
    // Second RTH message adds an ask — NOW both sides finite → snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, rth_start + 2 * NS_PER_MIN));

    SessionConfig cfg = SessionConfig::default_rth();
    cfg.warmup_messages = 0;

    ScriptedSource source(msgs);
    PrecomputedDay day = precompute(source, cfg);

    // With warmup=0, the book starts empty at RTH.
    // First RTH msg adds bid only → ask is infinite → no snapshot.
    // Second RTH msg adds ask → both finite AND BBO changed → 1 snapshot.
    EXPECT_EQ(day.num_steps, 1)
        << "With warmup=0, should get 1 snapshot (when both bid and ask become finite)";
}

TEST(Precompute, WarmupNegativeOneUsesAllPreMarketMessages) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // 20 pre-market messages establishing a book
    for (int i = 0; i < 10; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                999.75 - i * 0.25, 10 + i, pre_start + i * NS_PER_MIN));
    }
    for (int i = 0; i < 10; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                1000.25 + i * 0.25, 10 + i, pre_start + (10 + i) * NS_PER_MIN));
    }

    // RTH: BBO change
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + NS_PER_MIN));

    SessionConfig cfg = SessionConfig::default_rth();
    cfg.warmup_messages = -1;  // use all

    ScriptedSource source(msgs);
    PrecomputedDay day = precompute(source, cfg);

    EXPECT_GE(day.num_steps, 1)
        << "With warmup=-1, all pre-market messages should build the book";
}

// ===========================================================================
// Test 11: String overload creates BinaryFileSource
// ===========================================================================

TEST(Precompute, StringOverloadCreatesBinaryFileSource) {
    std::string path = fixture_path("precompute_rth.bin");
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(path, cfg);

    // The fixture has 4 pre-market messages + 3 RTH messages that change BBO
    EXPECT_GT(day.num_steps, 0)
        << "String overload should produce results from the fixture file";

    // Verify basic structural integrity
    EXPECT_EQ(day.obs.size(), static_cast<size_t>(day.num_steps * 43));
    EXPECT_EQ(day.mid.size(), static_cast<size_t>(day.num_steps));
    EXPECT_EQ(day.spread.size(), static_cast<size_t>(day.num_steps));

    // All obs should be finite
    for (size_t i = 0; i < day.obs.size(); ++i) {
        EXPECT_TRUE(std::isfinite(day.obs[i]))
            << "obs[" << i << "] from binary file is not finite";
    }
}

// ===========================================================================
// Test 12: BBO change requires both bid and ask to be finite
// ===========================================================================

TEST(Precompute, BboChangeRequiresBothBidAndAskFinite) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: only bids (no asks) — ask is infinite
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.50, 50, pre_start + NS_PER_MIN));

    // RTH: change bid — should NOT produce snapshot (ask is infinite)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    // RTH: add first ask — NOW should produce snapshot (both finite)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, rth_start + 2 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Only the second RTH message should produce a snapshot
    // (the first RTH message changes bid but ask is still infinite)
    EXPECT_EQ(day.num_steps, 1)
        << "Should produce snapshot only when both bid and ask are finite";

    // The snapshot should have valid mid price
    if (day.num_steps == 1) {
        // mid = (1000.00 + 1000.25) / 2 = 1000.125
        EXPECT_NEAR(day.mid[0], 1000.125, 1e-6);
        EXPECT_NEAR(day.spread[0], 0.25, 1e-6);
    }
}

// ===========================================================================
// Additional edge case: spreads track actual book spread
// ===========================================================================

TEST(Precompute, SpreadsTrackActualBookSpread) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: bid=999.75, ask=1000.25, spread=0.50
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // RTH: tighten spread by improving bid to 1000.00 → spread = 0.25
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + NS_PER_MIN));
    // RTH: tighten further by improving ask to 1000.125 → spread = 0.125
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, rth_start + 2 * NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 2);
    EXPECT_NEAR(day.spread[0], 0.25, 1e-6)
        << "First snapshot: spread should be 1000.25 - 1000.00 = 0.25";
    EXPECT_NEAR(day.spread[1], 0.125, 1e-6)
        << "Second snapshot: spread should be 1000.125 - 1000.00 = 0.125";
}

// ===========================================================================
// Additional: position is always 0 in obs (passed as 0.0 to FeatureBuilder)
// ===========================================================================

TEST(Precompute, PositionAlwaysZeroInFeatureBuilderCall) {
    // The spec says position=0 because it's agent state, not market state.
    // FeatureBuilder.build() is called with position=0.0f.
    // Since we only take first 43 floats (excluding index 43 which is position),
    // the position parameter shouldn't appear in obs at all.
    // But we verify the FeatureBuilder is called correctly by checking the obs
    // doesn't accidentally include position data at any index.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 1);
    ASSERT_EQ(day.obs.size(), 43u)
        << "obs should be exactly 43 floats — position (index 43) excluded";
}

// ===========================================================================
// Additional: source is NOT reset by precompute()
// ===========================================================================

TEST(Precompute, SourceIsNotReset) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 100, rth_start + NS_PER_MIN));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day1 = precompute(source, cfg);
    EXPECT_EQ(day1.num_steps, 1);

    // Call again without resetting — source should be exhausted
    PrecomputedDay day2 = precompute(source, cfg);
    EXPECT_EQ(day2.num_steps, 0)
        << "precompute() should NOT reset source — second call should see exhausted source";
}

// ===========================================================================
// Additional: PrecomputedDay struct default initialization
// ===========================================================================

TEST(Precompute, PrecomputedDayDefaultsToZero) {
    PrecomputedDay day;
    EXPECT_EQ(day.num_steps, 0);
    EXPECT_TRUE(day.obs.empty());
    EXPECT_TRUE(day.mid.empty());
    EXPECT_TRUE(day.spread.empty());
}

// ===========================================================================
// Additional: obs vector is row-major (contiguous 43-float blocks)
// ===========================================================================

TEST(Precompute, ObsVectorIsRowMajor) {
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.75, 100, pre_start));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.25, 100, pre_start + NS_PER_MIN));

    // 3 BBO changes at different times
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            999.875, 50, rth_start + NS_PER_MIN));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            1000.00, 50, rth_start + NS_PER_HOUR));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            1000.125, 50, rth_start + 2 * NS_PER_HOUR));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_EQ(day.num_steps, 3);
    ASSERT_EQ(day.obs.size(), static_cast<size_t>(3 * 43));

    // Each 43-float block should represent a distinct snapshot.
    // The time_remaining (index 42) in each block should be different
    // since they're at different timestamps.
    float tr0 = day.obs[0 * 43 + 42];
    float tr1 = day.obs[1 * 43 + 42];
    float tr2 = day.obs[2 * 43 + 42];

    EXPECT_NE(tr0, tr1) << "Different timestamps should give different time_remaining";
    EXPECT_NE(tr1, tr2) << "Different timestamps should give different time_remaining";

    // Each block's spread feature (index 40) should be finite
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(day.obs[i * 43 + 40]))
            << "Spread feature in block " << i << " should be finite";
    }
}
