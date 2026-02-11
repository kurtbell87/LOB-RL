#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include "lob/message.h"
#include "dbn_message_map.h"
#include "test_helpers.h"

// ===========================================================================
// Helper: Build a fake MboMsg-like struct for testing map_mbo_to_message().
//
// We construct databento::MboMsg objects directly. The fields we care about:
//   mbo.hd.instrument_id  (uint32_t)
//   mbo.hd.ts_event       (uint64_t)
//   mbo.order_id          (uint64_t)
//   mbo.price             (int64_t, fixed-point 1e-9)
//   mbo.size              (uint32_t)
//   mbo.action            (char)
//   mbo.side              (char)
//   mbo.flags             (uint8_t / FlagSet)
// ===========================================================================

static databento::MboMsg make_mbo(uint32_t instrument_id, uint64_t ts_event,
                                   uint64_t order_id, int64_t price,
                                   uint32_t size, char action, char side,
                                   uint8_t flags = 0) {
    databento::MboMsg mbo{};
    mbo.hd.instrument_id = instrument_id;
    mbo.hd.ts_event = ts_event;
    mbo.order_id = order_id;
    mbo.price = price;
    mbo.size = size;
    mbo.action = action;
    mbo.side = side;
    mbo.flags = flags;
    return mbo;
}

// ===========================================================================
// Action Mapping: Valid actions that should be mapped
// ===========================================================================

TEST(MapMboToMessage, ActionAddMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Add);
}

TEST(MapMboToMessage, ActionCancelMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 2, 999750000000LL, 5, 'C', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Cancel);
}

TEST(MapMboToMessage, ActionModifyMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 3, 999750000000LL, 15, 'M', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Modify);
}

TEST(MapMboToMessage, ActionTradeMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 4, 1000250000000LL, 3, 'T', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Trade);
}

TEST(MapMboToMessage, ActionFillMapsToTrade) {
    // Spec: Action::Fill ('F') remaps to Message::Action::Trade
    databento::MboMsg mbo = make_mbo(12345, 1000, 5, 1000250000000LL, 2, 'F', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Trade)
        << "Fill ('F') must map to Trade, not be a separate action";
}

// ===========================================================================
// Action Mapping: Actions that should be skipped (return false)
// ===========================================================================

TEST(MapMboToMessage, ActionClearMapsToCancel) {
    // Clear ('R') maps to Cancel for book order removal
    databento::MboMsg mbo = make_mbo(12345, 1000, 6, 999750000000LL, 10, 'R', 'N');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345))
        << "Action 'R' (Clear) should map to Cancel";
    EXPECT_EQ(msg.action, Message::Action::Cancel);
}

TEST(MapMboToMessage, ActionNoneIsSkipped) {
    // Spec: Action::None ('N') should be skipped
    databento::MboMsg mbo = make_mbo(12345, 1000, 7, 999750000000LL, 10, 'N', 'B');
    Message msg;
    EXPECT_FALSE(map_mbo_to_message(mbo, msg, 12345))
        << "Action 'N' (None) should be skipped";
}

// ===========================================================================
// Side Mapping
// ===========================================================================

TEST(MapMboToMessage, SideBidMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 10, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.side, Message::Side::Bid);
}

TEST(MapMboToMessage, SideAskMapsCorrectly) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 11, 1000250000000LL, 10, 'A', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.side, Message::Side::Ask);
}

TEST(MapMboToMessage, SideNoneMapsToSideBid) {
    // Spec: Side::None → Side::Bid (default for trades with unspecified aggressor)
    databento::MboMsg mbo = make_mbo(12345, 1000, 12, 999750000000LL, 5, 'T', 'N');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.side, Message::Side::Bid)
        << "Side 'N' (None) should default to Bid";
}

// ===========================================================================
// Price Conversion: i64 fixed-point (1e-9) → double
// ===========================================================================

TEST(MapMboToMessage, PriceConvertedFromFixedPointToDouble) {
    // price_raw = 999750000000 → 999.75
    databento::MboMsg mbo = make_mbo(12345, 1000, 20, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_DOUBLE_EQ(msg.price, 999.75);
}

TEST(MapMboToMessage, PriceConvertedFromFixedPointWholeNumber) {
    // price_raw = 1000000000000 → 1000.00
    databento::MboMsg mbo = make_mbo(12345, 1000, 21, 1000000000000LL, 10, 'A', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_DOUBLE_EQ(msg.price, 1000.0);
}

TEST(MapMboToMessage, PriceConvertedFromFixedPointQuarterTick) {
    // price_raw = 1000250000000 → 1000.25
    databento::MboMsg mbo = make_mbo(12345, 1000, 22, 1000250000000LL, 10, 'A', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_DOUBLE_EQ(msg.price, 1000.25);
}

TEST(MapMboToMessage, PriceConvertedForTypicalMESPrice) {
    // Typical /MES price: $5000.123456789
    // price_raw = 5000123456789
    databento::MboMsg mbo = make_mbo(12345, 1000, 23, 5000123456789LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_NEAR(msg.price, 5000.123456789, 1e-9);
}

TEST(MapMboToMessage, PriceConvertedForSmallPrice) {
    // price_raw = 250000000 → 0.25
    databento::MboMsg mbo = make_mbo(12345, 1000, 24, 250000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_DOUBLE_EQ(msg.price, 0.25);
}

// ===========================================================================
// Field Passthrough: order_id, qty, timestamp
// ===========================================================================

TEST(MapMboToMessage, OrderIdIsPreserved) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 42, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.order_id, 42u);
}

TEST(MapMboToMessage, SizeMapToQty) {
    // MboMsg uses 'size', our Message uses 'qty'
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 25, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.qty, 25u);
}

TEST(MapMboToMessage, TimestampIsPreserved) {
    // ts_event from hd → ts_ns in Message
    uint64_t ts = 1704067200000000000ULL;  // 2024-01-01 00:00:00 UTC in ns
    databento::MboMsg mbo = make_mbo(12345, ts, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.ts_ns, ts);
}

TEST(MapMboToMessage, FlagsArePreserved) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 10, 'A', 'B', 0x80);
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.flags, 0x80);
}

// ===========================================================================
// Instrument ID Filtering
// ===========================================================================

TEST(MapMboToMessage, MatchingInstrumentIdReturnsTrue) {
    databento::MboMsg mbo = make_mbo(42005347, 1000, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    EXPECT_TRUE(map_mbo_to_message(mbo, msg, 42005347));
}

TEST(MapMboToMessage, NonMatchingInstrumentIdReturnsFalse) {
    databento::MboMsg mbo = make_mbo(99999999, 1000, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    EXPECT_FALSE(map_mbo_to_message(mbo, msg, 42005347))
        << "Record with non-matching instrument_id should be skipped";
}

TEST(MapMboToMessage, InstrumentIdZeroMatchesAll) {
    // Spec: instrument_id = 0 means no filtering (accept all)
    databento::MboMsg mbo = make_mbo(42005347, 1000, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    EXPECT_TRUE(map_mbo_to_message(mbo, msg, 0))
        << "instrument_id=0 should match all records";
}

TEST(MapMboToMessage, InstrumentIdZeroAcceptsDifferentIds) {
    databento::MboMsg mbo1 = make_mbo(11111, 1000, 1, 999750000000LL, 10, 'A', 'B');
    databento::MboMsg mbo2 = make_mbo(22222, 1001, 2, 999750000000LL, 10, 'A', 'A');
    Message msg;

    EXPECT_TRUE(map_mbo_to_message(mbo1, msg, 0));
    EXPECT_TRUE(map_mbo_to_message(mbo2, msg, 0));
}

// ===========================================================================
// Combined Filtering: instrument_id + action skip
// ===========================================================================

TEST(MapMboToMessage, ClearActionMapsToCancel) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 10, 'R', 'N', 0);
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Cancel);
}

TEST(MapMboToMessage, ValidActionSkippedIfInstrumentDoesNotMatch) {
    databento::MboMsg mbo = make_mbo(99999, 1000, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    EXPECT_FALSE(map_mbo_to_message(mbo, msg, 12345));
}

TEST(MapMboToMessage, FillActionWithMatchingInstrumentReturnsTrue) {
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 10, 'F', 'A');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Trade);
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST(MapMboToMessage, ZeroPriceRawMapsToZeroDouble) {
    // Edge case: price=0 in the raw data
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 0LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_DOUBLE_EQ(msg.price, 0.0);
}

TEST(MapMboToMessage, LargeOrderIdPreserved) {
    databento::MboMsg mbo = make_mbo(12345, 1000, UINT64_MAX - 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.order_id, UINT64_MAX - 1);
}

TEST(MapMboToMessage, LargeTimestampPreserved) {
    uint64_t large_ts = UINT64_MAX - 100;
    databento::MboMsg mbo = make_mbo(12345, large_ts, 1, 999750000000LL, 10, 'A', 'B');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.ts_ns, large_ts);
}

TEST(MapMboToMessage, SideNoneWithFillActionStillMapsBidAndTrade) {
    // A fill with Side::None should get side=Bid and action=Trade
    databento::MboMsg mbo = make_mbo(12345, 1000, 1, 999750000000LL, 5, 'F', 'N');
    Message msg;
    ASSERT_TRUE(map_mbo_to_message(mbo, msg, 12345));
    EXPECT_EQ(msg.action, Message::Action::Trade);
    EXPECT_EQ(msg.side, Message::Side::Bid);
}

// ===========================================================================
// Comprehensive Integration: Multiple records in sequence
// ===========================================================================

TEST(MapMboToMessage, MixedRecordsFilteredCorrectly) {
    const uint32_t INST = 42005347;
    Message msg;
    int accepted = 0;
    int rejected = 0;

    // Record 1: matching Add Bid — accepted
    auto mbo1 = make_mbo(INST, 1000, 1, 999750000000LL, 10, 'A', 'B');
    if (map_mbo_to_message(mbo1, msg, INST)) ++accepted; else ++rejected;

    // Record 2: matching Clear — accepted (maps to Cancel)
    auto mbo2 = make_mbo(INST, 1001, 2, 999750000000LL, 10, 'R', 'N');
    if (map_mbo_to_message(mbo2, msg, INST)) ++accepted; else ++rejected;

    // Record 3: wrong instrument — rejected
    auto mbo3 = make_mbo(99999, 1002, 3, 999750000000LL, 10, 'A', 'B');
    if (map_mbo_to_message(mbo3, msg, INST)) ++accepted; else ++rejected;

    // Record 4: matching Fill Ask — accepted (as Trade)
    auto mbo4 = make_mbo(INST, 1003, 4, 1000250000000LL, 5, 'F', 'A');
    if (map_mbo_to_message(mbo4, msg, INST)) ++accepted; else ++rejected;

    // Record 5: matching None action — rejected
    auto mbo5 = make_mbo(INST, 1004, 5, 999750000000LL, 10, 'N', 'B');
    if (map_mbo_to_message(mbo5, msg, INST)) ++accepted; else ++rejected;

    // Record 6: matching Trade with Side::None — accepted (side→Bid)
    auto mbo6 = make_mbo(INST, 1005, 6, 999750000000LL, 3, 'T', 'N');
    if (map_mbo_to_message(mbo6, msg, INST)) ++accepted; else ++rejected;

    EXPECT_EQ(accepted, 4) << "Expected 4 accepted records (Add, Clear→Cancel, Fill→Trade, Trade)";
    EXPECT_EQ(rejected, 2) << "Expected 2 rejected records (wrong inst, None)";
}
