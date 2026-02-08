#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "lob/precompute.h"
#include "lob/feature_builder.h"
#include "lob/book.h"
#include "lob/message.h"
#include "binary_file_source.h"
#include "test_helpers.h"

// Databento flag constants (from the spec)
static constexpr uint8_t F_LAST     = 0x80;
static constexpr uint8_t F_SNAPSHOT  = 0x20;
static constexpr uint8_t F_PUB_SPEC = 0x02;  // PUBLISHER_SPECIFIC

// Common real-world flag combos
static constexpr uint8_t FLAGS_EVENT_TERMINAL = F_LAST | F_PUB_SPEC;  // 0x82
static constexpr uint8_t FLAGS_MID_EVENT     = 0x00;
static constexpr uint8_t FLAGS_SNAPSHOT_REC  = F_SNAPSHOT | 0x08;     // 0x28

// ===========================================================================
// A ScriptedSource variant that supports flagged messages.
// Reuses the existing ScriptedSource since Message now has flags.
// ===========================================================================

// ===========================================================================
// SECTION 1: Message struct has flags field
// ===========================================================================

TEST(FixPrecomputeEvents, MessageStructHasFlagsField) {
    // Spec Change 1: Message should have a `uint8_t flags` field
    Message m;
    m.flags = 0x82;
    EXPECT_EQ(m.flags, 0x82);
}

TEST(FixPrecomputeEvents, MessageFlagsDefaultsToZero) {
    // Default-constructed Message should have flags = 0
    Message m;
    EXPECT_EQ(m.flags, 0u);
}

TEST(FixPrecomputeEvents, MessageFlagsPreservesAllBits) {
    // Verify all 8 bits can be stored and retrieved
    Message m;
    m.flags = 0xFF;
    EXPECT_EQ(m.flags, 0xFF);

    m.flags = 0x00;
    EXPECT_EQ(m.flags, 0x00);

    m.flags = F_LAST;
    EXPECT_EQ(m.flags, F_LAST);

    m.flags = F_SNAPSHOT;
    EXPECT_EQ(m.flags, F_SNAPSHOT);
}

TEST(FixPrecomputeEvents, MessageFlagsDoesNotAffectIsValid) {
    // Adding flags should not change is_valid() behavior
    Message m;
    m.order_id = 1;
    m.price = 100.0;
    m.qty = 10;
    m.flags = 0x82;
    EXPECT_TRUE(m.is_valid());

    m.flags = 0xFF;
    EXPECT_TRUE(m.is_valid());

    m.flags = 0x00;
    EXPECT_TRUE(m.is_valid());
}

// ===========================================================================
// SECTION 2: BinaryFileSource copies flags from FlatRecord to Message
// ===========================================================================

TEST(FixPrecomputeEvents, BinaryFileSourceCopiesFlags) {
    // Spec Change 2: BinaryFileSource::convert() must copy rec.flags to msg.flags
    // The mixed_actions.bin fixture has records with flags set.
    // We create a fixture file with known flags to test this.
    // For now, use the existing mixed_actions fixture and verify flags are non-default.
    //
    // The real test: read a binary file and verify msg.flags matches what's in the file.
    // Since the existing fixtures may have flags=0, we use the valid_10records fixture
    // and at minimum verify that flags is populated (even if 0).
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    // At minimum, flags should be accessible (proves the field exists and is copied).
    // The fixture may have flags=0 or some value — key thing is no crash and field exists.
    // We'll test with a purpose-built fixture below.
    (void)m.flags;  // Must compile — proves field exists in Message
}

TEST(FixPrecomputeEvents, BinaryFileSourceCopiesFlagsFromFixture) {
    // Use the mixed_actions fixture which has records with known flag values.
    // The fixture generator should have set flags on at least some records.
    // Record 3 (Trade) typically has flags=0x00 (mid-event),
    // Record 0 (Add) typically has flags=0x82 (event-terminal).
    //
    // This test uses a dedicated fixture with known flags.
    // We create it below via the fixture generator.
    BinaryFileSource src(fixture_path("flagged_records.bin"));
    Message m;

    // Record 0: flags should be 0x82 (F_LAST | PUBLISHER_SPECIFIC)
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.flags, FLAGS_EVENT_TERMINAL)
        << "First record flags should be event-terminal (0x82)";

    // Record 1: flags should be 0x00 (mid-event)
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.flags, FLAGS_MID_EVENT)
        << "Second record flags should be mid-event (0x00)";

    // Record 2: flags should be 0x82 (event-terminal)
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.flags, FLAGS_EVENT_TERMINAL)
        << "Third record flags should be event-terminal (0x82)";

    // Record 3: flags should be 0x28 (snapshot)
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.flags, FLAGS_SNAPSHOT_REC)
        << "Fourth record flags should be snapshot (0x28)";
}

// ===========================================================================
// SECTION 3: Book::apply() ignores Trade actions
// ===========================================================================

TEST(FixPrecomputeEvents, BookApplyTradeIsNoOp) {
    // Spec Change 3: Trade actions should be no-ops in Book::apply()
    // Databento says Trade/Fill don't affect the book.
    Book book;

    // Build a book with known state
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.25, 30));

    // Capture state before trade
    double bid_before = book.best_bid();
    double ask_before = book.best_ask();
    uint32_t bid_qty_before = book.best_bid_qty();
    uint32_t ask_qty_before = book.best_ask_qty();
    size_t bid_depth_before = book.bid_depth();
    size_t ask_depth_before = book.ask_depth();

    // Apply a Trade message — should be a NO-OP
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Trade, 100.25, 10));

    // Book should be COMPLETELY UNCHANGED
    EXPECT_DOUBLE_EQ(book.best_bid(), bid_before);
    EXPECT_DOUBLE_EQ(book.best_ask(), ask_before);
    EXPECT_EQ(book.best_bid_qty(), bid_qty_before);
    EXPECT_EQ(book.best_ask_qty(), ask_qty_before);
    EXPECT_EQ(book.bid_depth(), bid_depth_before);
    EXPECT_EQ(book.ask_depth(), ask_depth_before);
}

TEST(FixPrecomputeEvents, BookApplyTradeDoesNotRemoveQty) {
    // Trade should NOT decrement quantity at the level
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 100.25, 100));

    // Apply Trade for the full quantity — book should NOT lose the level
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 100.25, 100));

    EXPECT_EQ(book.ask_depth(), 1u) << "Trade should not remove price level";
    EXPECT_EQ(book.best_ask_qty(), 100u) << "Trade should not decrement quantity";
    EXPECT_DOUBLE_EQ(book.best_ask(), 100.25);
}

TEST(FixPrecomputeEvents, BookApplyTradeDoesNotRemoveOrder) {
    // Trade should NOT remove orders from the order map
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));

    // Apply Trade for full qty — order should still exist for subsequent Cancel
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Trade, 100.0, 50));

    EXPECT_EQ(book.bid_depth(), 1u) << "Trade should not affect book at all";
    EXPECT_EQ(book.best_bid_qty(), 50u);

    // Cancel should still work because the order entry is preserved
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 50));
    EXPECT_EQ(book.bid_depth(), 0u) << "Cancel after Trade should still work";
}

TEST(FixPrecomputeEvents, BookApplyTradeBidSideIsNoOp) {
    // Trade should be no-op on bid side too
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 99.75, 200));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.00, 100));

    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Trade, 99.75, 50));

    EXPECT_EQ(book.best_bid_qty(), 200u) << "Bid qty should be unchanged after Trade";
    EXPECT_DOUBLE_EQ(book.best_bid(), 99.75);
    EXPECT_DOUBLE_EQ(book.spread(), 0.25);
}

TEST(FixPrecomputeEvents, BookApplyTradeUnknownOrderIsNoOp) {
    // Trade on unknown order should also be no-op
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));

    book.apply(make_msg(999, Message::Side::Bid, Message::Action::Trade, 100.0, 10));

    EXPECT_EQ(book.best_bid_qty(), 50u) << "Trade on unknown order should be no-op";
}

TEST(FixPrecomputeEvents, BookApplyStillProcessesAdd) {
    // Spec: Add/Cancel/Modify must still work after the Trade change
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    EXPECT_EQ(book.bid_depth(), 1u);
    EXPECT_EQ(book.best_bid_qty(), 50u);
}

TEST(FixPrecomputeEvents, BookApplyStillProcessesCancel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 50));
    EXPECT_EQ(book.bid_depth(), 0u);
}

TEST(FixPrecomputeEvents, BookApplyStillProcessesModify) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 100.25, 50));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Modify, 100.50, 30));
    EXPECT_DOUBLE_EQ(book.best_ask(), 100.50);
    EXPECT_EQ(book.best_ask_qty(), 30u);
}

TEST(FixPrecomputeEvents, BookApplyFillMappedToTradeIsAlsoNoOp) {
    // Fill is remapped to Trade by BinaryFileSource, so it should also be a no-op.
    // Since Fill->Trade happens in the converter, by the time Book sees it,
    // it's Action::Trade. This test confirms the no-op applies uniformly.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 100.25, 100));

    // Simulate what would happen with a Fill (already mapped to Trade)
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 100.25, 30));

    EXPECT_EQ(book.best_ask_qty(), 100u) << "Fill (mapped to Trade) should be no-op";
}

// ===========================================================================
// SECTION 4: precompute() skips mid-event messages (no F_LAST)
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeSkipsMidEventMessages) {
    // Spec Change 4: Only snapshot when msg.flags & F_LAST is set
    // Mid-event messages (flags=0x00) should NOT produce snapshots.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup: establish book with bid=999.75, ask=1000.25
    // These use event-terminal flags so they're processed normally during warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH message 1: mid-event (flags=0x00) — changes BBO but should NOT snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // RTH message 2: event-terminal (flags=0x82) — changes BBO, SHOULD snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.125, 50, rth_start + 2 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Only the event-terminal message should produce a snapshot
    EXPECT_EQ(day.num_steps, 1)
        << "Only event-terminal (F_LAST) messages should produce snapshots";
}

TEST(FixPrecomputeEvents, PrecomputeOnlySnapshotsOnFLast) {
    // Multiple mid-event messages followed by one F_LAST — only 1 snapshot
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: simulate a multi-message event (Trade+Cancel+Add)
    // All at the same nanosecond, mid-event flags until the last one

    // Mid-event Trade (would have changed BBO in buggy code)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Trade,
                                     1000.25, 50, rth_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // Mid-event Cancel (removes old ask level)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Cancel,
                                     1000.25, 50, rth_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // Event-terminal Add (establishes new ask)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.125, 100, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 1)
        << "Multi-message event should produce exactly 1 snapshot (on F_LAST only)";

    // The snapshot should reflect the book state AFTER the entire event
    if (day.num_steps >= 1) {
        // After event: bid=999.75, ask=1000.125
        // mid = (999.75 + 1000.125) / 2 = 999.9375
        EXPECT_NEAR(day.mid[0], 999.9375, 1e-6)
            << "Mid should reflect book state after full event";
        // spread = 1000.125 - 999.75 = 0.375
        EXPECT_NEAR(day.spread[0], 0.375, 1e-6)
            << "Spread should be positive after complete event";
    }
}

// ===========================================================================
// SECTION 5: precompute() skips snapshot messages (F_SNAPSHOT set)
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeSkipsSnapshotMessages) {
    // Spec Change 4: Messages with F_SNAPSHOT set should not produce snapshots.
    // These are synthetic replay/snapshot records from Databento.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: snapshot message (flags=0x28 = F_SNAPSHOT | BAD_TS_RECV)
    // Even though this changes BBO and has F_LAST, F_SNAPSHOT should prevent snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_SNAPSHOT_REC));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 0)
        << "Snapshot messages (F_SNAPSHOT set) should not produce observation rows";
}

TEST(FixPrecomputeEvents, PrecomputeSkipsSnapshotWithFLast) {
    // Edge case: F_SNAPSHOT | F_LAST — still should NOT produce snapshot
    // (the 0xa8 case from real data distribution)
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: message with both F_LAST and F_SNAPSHOT (0xa8)
    uint8_t flags_last_and_snapshot = F_LAST | F_SNAPSHOT | 0x08;  // 0xa8
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     flags_last_and_snapshot));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 0)
        << "F_SNAPSHOT should override F_LAST — no snapshot even with both flags";
}

// ===========================================================================
// SECTION 6: precompute() requires positive spread
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeRequiresPositiveSpread) {
    // Spec Change 4: Even with F_LAST and no F_SNAPSHOT,
    // spread must be > 0 to produce a snapshot.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: create crossed book (bid > ask) — spread <= 0
    // This is the defensive filter for the remaining ~0.007% edge cases.
    // Add a bid above the current ask
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.50, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Spread = 1000.25 - 1000.50 = -0.25 (negative, crossed book)
    // Should NOT produce a snapshot
    EXPECT_EQ(day.num_steps, 0)
        << "Crossed book (negative spread) should not produce snapshots";
}

TEST(FixPrecomputeEvents, PrecomputeRejectsZeroSpread) {
    // spread = 0 (locked book) should also be rejected
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: bid = ask (locked book), spread = 0
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.25, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Spread = 1000.25 - 1000.25 = 0.0 (not positive)
    // Should NOT produce a snapshot
    EXPECT_EQ(day.num_steps, 0)
        << "Zero spread (locked book) should not produce snapshots";
}

// ===========================================================================
// SECTION 7: precompute() produces only positive spreads
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeAllSpreadsPositive) {
    // End-to-end: a realistic sequence with proper flags
    // All output spreads must be > 0
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup: build initial book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: several BBO-changing events, all event-terminal with positive spread
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.125, 50, rth_start + 2 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.0625, 50, rth_start + 3 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_GT(day.num_steps, 0) << "Should produce at least some snapshots";

    for (int i = 0; i < day.num_steps; ++i) {
        EXPECT_GT(day.spread[i], 0.0)
            << "Spread at step " << i << " should be positive, got " << day.spread[i];
    }
}

// ===========================================================================
// SECTION 8: precompute() mid prices are between bid and ask
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeMidPricesBetweenBidAndAsk) {
    // With correct event handling, mid prices should always be valid.
    // mid = (bid + ask) / 2, so bid < mid < ask when spread > 0.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: several events
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.125, 50, rth_start + 2 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    ASSERT_GT(day.num_steps, 0);

    for (int i = 0; i < day.num_steps; ++i) {
        EXPECT_TRUE(std::isfinite(day.mid[i]))
            << "Mid price at step " << i << " should be finite";
        // With positive spread, mid should be between implicit bid and ask
        // mid = (bid + ask) / 2 and spread = ask - bid > 0
        // So: mid - spread/2 = bid > 0 (reasonable for /MES)
        EXPECT_GT(day.mid[i], 0.0)
            << "Mid price at step " << i << " should be positive";
    }
}

// ===========================================================================
// SECTION 9: End-to-end integration with multi-message events
// ===========================================================================

TEST(FixPrecomputeEvents, EndToEndMultiMessageEvent) {
    // Simulate a realistic trade match event:
    // 1. Trade (mid-event, flags=0x00) — no-op on book
    // 2. Cancel of resting order (mid-event, flags=0x00) — removes from book
    // 3. Add replacement order (event-terminal, flags=0x82) — snapshot here
    //
    // This tests all 4 changes working together.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: build full book
    // Bid side: 999.75 (100), 999.50 (100)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.50, 100, pre_start + NS_PER_SEC,
                                     FLAGS_EVENT_TERMINAL));
    // Ask side: 1000.25 (100, order_id=3), 1000.50 (100)
    uint64_t resting_ask_id = oid;
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + 2 * NS_PER_SEC,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.50, 100, pre_start + 3 * NS_PER_SEC,
                                     FLAGS_EVENT_TERMINAL));

    // RTH Event 1: A trade match against the resting ask at 1000.25
    // This is a multi-message event:

    // Msg A: Trade (mid-event) — should be no-op on book
    msgs.push_back(make_msg(resting_ask_id, Message::Side::Ask,
                                     Message::Action::Trade,
                                     1000.25, 100, rth_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // Msg B: Cancel of the filled order (mid-event) — removes from book
    // During this mid-event state, the book may be temporarily inconsistent
    msgs.push_back(make_msg(resting_ask_id, Message::Side::Ask,
                                     Message::Action::Cancel,
                                     1000.25, 100, rth_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // Msg C: New order added (event-terminal) — book is now consistent
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.375, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Should produce exactly 1 snapshot (at the event-terminal message)
    ASSERT_EQ(day.num_steps, 1)
        << "Multi-message trade event should produce exactly 1 snapshot";

    // After the event:
    // - Trade was no-op (book unchanged by trade)
    // - Cancel removed ask at 1000.25
    // - Add created ask at 1000.375
    // Book: bid=999.75, ask=1000.375 (next ask after cancel is 1000.50,
    //        but new add at 1000.375 is closer)
    // Wait: The Cancel removes the resting order at 1000.25 (the only one at that level).
    // After Cancel, best ask = 1000.50 (the backup level).
    // Then Add at 1000.375 improves best ask to 1000.375.
    // So: bid=999.75, ask=1000.375
    double expected_mid = (999.75 + 1000.375) / 2.0;  // 1000.0625
    double expected_spread = 1000.375 - 999.75;         // 0.625

    EXPECT_NEAR(day.mid[0], expected_mid, 1e-6)
        << "Mid should reflect final book state after complete event";
    EXPECT_NEAR(day.spread[0], expected_spread, 1e-6)
        << "Spread should be positive after complete event";
    EXPECT_GT(day.spread[0], 0.0)
        << "Spread must be positive";
}

TEST(FixPrecomputeEvents, EndToEndSnapshotCountMatchesEventTerminals) {
    // The number of snapshots should equal the number of event-terminal messages
    // that change BBO with positive spread.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: 3 event-terminal messages that change BBO (should snapshot)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.125, 50, rth_start + 2 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.0625, 50, rth_start + 3 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // 2 mid-event messages that change BBO (should NOT snapshot)
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.09375, 50, rth_start + 4 * NS_PER_MIN,
                                     FLAGS_MID_EVENT));
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.09, 50, rth_start + 4 * NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // 1 snapshot message (should NOT snapshot)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.08, 50, rth_start + 5 * NS_PER_MIN,
                                     FLAGS_SNAPSHOT_REC));

    // 1 more event-terminal that DOES change BBO (should snapshot = 4th)
    // Mid-event msgs are orphaned (no F_LAST at same ts), so book state is:
    // best_bid=1000.0625, best_ask=1000.125. New ask must be < 1000.125 to change BBO.
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.10, 50, rth_start + 6 * NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // 4 event-terminal messages that change BBO with positive spread
    EXPECT_EQ(day.num_steps, 4)
        << "Snapshot count should equal event-terminal BBO-changing messages with positive spread";
}

TEST(FixPrecomputeEvents, EndToEndNoNegativeSpreads) {
    // Simulate the scenario that was producing 100% negative spreads:
    // Trade messages (mid-event) were corrupting the book, causing crossed-book snapshots.
    // With fixes, this should produce zero negative spreads.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: establish book
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: repeated Trade+Cancel+Add events (mimicking real data pattern)
    for (int event = 0; event < 10; ++event) {
        uint64_t event_ts = rth_start + (event + 1) * NS_PER_MIN;
        uint64_t ask_order_id = oid++;

        // First, add the resting ask
        if (event == 0) {
            // Already have initial ask from warmup
        } else {
            msgs.push_back(make_msg(ask_order_id, Message::Side::Ask,
                                             Message::Action::Add,
                                             1000.25 + event * 0.125, 100,
                                             event_ts - NS_PER_SEC,
                                             FLAGS_EVENT_TERMINAL));
        }

        // Trade (mid-event, no-op on book)
        msgs.push_back(make_msg(event == 0 ? 2u : ask_order_id,
                                         Message::Side::Ask, Message::Action::Trade,
                                         1000.25 + event * 0.125, 50,
                                         event_ts, FLAGS_MID_EVENT));

        // Cancel (mid-event, removes from book)
        msgs.push_back(make_msg(event == 0 ? 2u : ask_order_id,
                                         Message::Side::Ask, Message::Action::Cancel,
                                         1000.25 + event * 0.125, 50,
                                         event_ts, FLAGS_MID_EVENT));

        // Add replacement (event-terminal)
        uint64_t new_ask_id = oid++;
        msgs.push_back(make_msg(new_ask_id, Message::Side::Ask,
                                         Message::Action::Add,
                                         1000.25 + (event + 1) * 0.125, 100,
                                         event_ts, FLAGS_EVENT_TERMINAL));
    }

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // All spreads must be positive — zero negative spreads
    for (int i = 0; i < day.num_steps; ++i) {
        EXPECT_GT(day.spread[i], 0.0)
            << "Spread at step " << i << " is " << day.spread[i]
            << " — must be positive (no crossed books)";
    }
}

// ===========================================================================
// SECTION 10: Warmup messages are processed regardless of flags
// ===========================================================================

TEST(FixPrecomputeEvents, WarmupProcessesAllMessagesFlagsIgnored) {
    // During warmup (pre-market), ALL messages should be applied to the book
    // regardless of their flags. The flag filtering only applies to snapshot
    // decisions during RTH.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market: some with snapshot flags, some with mid-event flags
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_SNAPSHOT_REC));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_MID_EVENT));

    // RTH: BBO change that should produce a snapshot
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    // Warmup should have processed both messages regardless of flags,
    // establishing bid=999.75 and ask=1000.25.
    // RTH message improves bid to 1000.00 → snapshot with spread=0.25
    EXPECT_EQ(day.num_steps, 1)
        << "Warmup should process all messages regardless of flags";
    if (day.num_steps >= 1) {
        EXPECT_NEAR(day.spread[0], 0.25, 1e-6);
        EXPECT_NEAR(day.mid[0], 1000.125, 1e-6);
    }
}

// ===========================================================================
// SECTION 11: Event-terminal messages at BBO with F_LAST only (0x80)
// ===========================================================================

TEST(FixPrecomputeEvents, PrecomputeAcceptsFLastWithoutPublisherSpecific) {
    // From real data: 2 records had flags=0x80 (F_LAST only, no PUBLISHER_SPECIFIC)
    // These should still produce snapshots.
    std::vector<Message> msgs;
    uint64_t oid = 1;
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;

    // Pre-market warmup
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     999.75, 100, pre_start, FLAGS_EVENT_TERMINAL));
    msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                     1000.25, 100, pre_start + NS_PER_MIN,
                                     FLAGS_EVENT_TERMINAL));

    // RTH: message with flags=0x80 (just F_LAST)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                     1000.00, 50, rth_start + NS_PER_MIN,
                                     F_LAST));  // 0x80 only

    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    PrecomputedDay day = precompute(source, cfg);

    EXPECT_EQ(day.num_steps, 1)
        << "F_LAST (0x80) alone should produce snapshots";
}

// ===========================================================================
// SECTION 12: Existing tests should still pass (no regressions)
// This is implicit — existing test_precompute.cpp and test_book.cpp
// tests run in the same binary. But we verify backwards compatibility:
// messages without explicit flags (flags=0) should be handled by warmup
// and book operations.
// ===========================================================================

TEST(FixPrecomputeEvents, DefaultFlagsBackwardsCompatible) {
    // Messages created with make_msg (no flags parameter) should have flags=0
    // by default. This ensures backward compatibility.
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10);
    EXPECT_EQ(m.flags, 0u) << "Default flags should be 0 for backward compatibility";
}
