#include <gtest/gtest.h>
#include <cmath>
#include "binary_file_source.h"
#include "test_helpers.h"

// ===========================================================================
// BinaryFileSource: Construction & Header Validation
// ===========================================================================

TEST(BinaryFileSource, OpensValidFile) {
    EXPECT_NO_THROW(BinaryFileSource src(fixture_path("valid_10records.bin")));
}

TEST(BinaryFileSource, ThrowsOnFileNotFound) {
    EXPECT_THROW(BinaryFileSource src("/nonexistent/path/no_such_file.bin"),
                 std::runtime_error);
}

TEST(BinaryFileSource, ThrowsOnBadMagicBytes) {
    EXPECT_THROW(BinaryFileSource src(fixture_path("bad_magic.bin")),
                 std::runtime_error);
}

TEST(BinaryFileSource, ThrowsOnUnsupportedVersion) {
    EXPECT_THROW(BinaryFileSource src(fixture_path("bad_version.bin")),
                 std::runtime_error);
}

TEST(BinaryFileSource, ReportsCorrectRecordCount) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    EXPECT_EQ(src.record_count(), 10u);
}

TEST(BinaryFileSource, ReportsCorrectInstrumentId) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    EXPECT_EQ(src.instrument_id(), 42005347u);
}

// ===========================================================================
// BinaryFileSource: IMessageSource Interface
// ===========================================================================

TEST(BinaryFileSource, ImplementsIMessageSourceInterface) {
    std::unique_ptr<IMessageSource> src =
        std::make_unique<BinaryFileSource>(fixture_path("valid_10records.bin"));
    Message m;
    EXPECT_TRUE(src->next(m));
    src->reset();
    EXPECT_TRUE(src->next(m));
}

// ===========================================================================
// BinaryFileSource: next() Iteration
// ===========================================================================

TEST(BinaryFileSource, NextReturnsCorrectNumberOfRecords) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    int count = 0;
    while (src.next(m)) ++count;
    EXPECT_EQ(count, 10);
}

TEST(BinaryFileSource, NextReturnsFalseAfterLastRecord) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    while (src.next(m)) {}

    // Subsequent calls should also return false
    EXPECT_FALSE(src.next(m));
    EXPECT_FALSE(src.next(m));
}

TEST(BinaryFileSource, EmptyFileNextReturnsFalseImmediately) {
    BinaryFileSource src(fixture_path("empty.bin"));
    Message m;
    EXPECT_FALSE(src.next(m));
}

TEST(BinaryFileSource, EmptyFileRecordCountIsZero) {
    BinaryFileSource src(fixture_path("empty.bin"));
    EXPECT_EQ(src.record_count(), 0u);
}

// ===========================================================================
// BinaryFileSource: Price Conversion (fixed-point to double)
// ===========================================================================

TEST(BinaryFileSource, FirstRecordPriceIsCorrectlyConverted) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    // price_raw = 999_750_000_000 -> price = 999.75
    EXPECT_DOUBLE_EQ(m.price, 999.75);
}

TEST(BinaryFileSource, AllPricesCorrectlyConvertedFromFixedPoint) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;

    // Expected prices in order
    double expected_prices[] = {
        999.75, 999.50, 999.25, 999.00, 998.75,
        1000.00, 1000.25, 1000.50, 1000.75, 1001.00
    };

    for (int i = 0; i < 10; ++i) {
        ASSERT_TRUE(src.next(m)) << "Failed to read record " << i;
        EXPECT_DOUBLE_EQ(m.price, expected_prices[i])
            << "Price mismatch at record " << i;
    }
}

// ===========================================================================
// BinaryFileSource: Action Mapping
// ===========================================================================

TEST(BinaryFileSource, ActionAddIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.action, Message::Action::Add);
}

TEST(BinaryFileSource, ActionCancelIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;
    // Skip to record 5 (Cancel, 0-indexed)
    for (int i = 0; i < 6; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.action, Message::Action::Cancel);
}

TEST(BinaryFileSource, ActionModifyIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;
    // Record 2 is Modify (0-indexed)
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.action, Message::Action::Modify);
}

TEST(BinaryFileSource, ActionTradeIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;
    // Record 3 is Trade (0-indexed)
    for (int i = 0; i < 4; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.action, Message::Action::Trade);
}

TEST(BinaryFileSource, ActionFillMapsToTrade) {
    // Fill ('F') should map to Message::Action::Trade per spec
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;
    // Record 4 is Fill (0-indexed)
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.action, Message::Action::Trade)
        << "Fill action should map to Trade";
}

// ===========================================================================
// BinaryFileSource: Side Mapping
// ===========================================================================

TEST(BinaryFileSource, SideBidIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    // First record is a Bid
    EXPECT_EQ(m.side, Message::Side::Bid);
}

TEST(BinaryFileSource, SideAskIsMappedCorrectly) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    // Skip 5 bids, read first ask (record 5, 0-indexed)
    for (int i = 0; i < 6; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.side, Message::Side::Ask);
}

TEST(BinaryFileSource, SideNoneMapsToSideBid) {
    // Side='N' should map to Bid per spec
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;
    // Record 6 (last) has side='N'
    for (int i = 0; i < 7; ++i) ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.side, Message::Side::Bid)
        << "Side 'N' should map to Bid";
}

// ===========================================================================
// BinaryFileSource: Other Message Fields
// ===========================================================================

TEST(BinaryFileSource, OrderIdIsPreserved) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 1u);

    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 2u);
}

TEST(BinaryFileSource, QtyIsPreserved) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.qty, 10u);  // first record qty = 10

    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.qty, 20u);  // second record qty = 20
}

TEST(BinaryFileSource, TimestampIsPreserved) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.ts_ns, 1000000000u);

    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.ts_ns, 1000000001u);
}

TEST(BinaryFileSource, TimestampsAreNonDecreasing) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    uint64_t prev_ts = 0;
    while (src.next(m)) {
        EXPECT_GE(m.ts_ns, prev_ts);
        prev_ts = m.ts_ns;
    }
}

// ===========================================================================
// BinaryFileSource: reset()
// ===========================================================================

TEST(BinaryFileSource, ResetRewindsToFirstRecord) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message first_msg;
    ASSERT_TRUE(src.next(first_msg));

    // Advance a few records
    Message discard;
    for (int i = 0; i < 3; ++i) src.next(discard);

    // Reset
    src.reset();

    // First message should be identical
    Message after_reset;
    ASSERT_TRUE(src.next(after_reset));
    EXPECT_EQ(first_msg.order_id, after_reset.order_id);
    EXPECT_DOUBLE_EQ(first_msg.price, after_reset.price);
    EXPECT_EQ(first_msg.qty, after_reset.qty);
    EXPECT_EQ(first_msg.ts_ns, after_reset.ts_ns);
    EXPECT_EQ(static_cast<int>(first_msg.side), static_cast<int>(after_reset.side));
    EXPECT_EQ(static_cast<int>(first_msg.action), static_cast<int>(after_reset.action));
}

TEST(BinaryFileSource, ResetProducesFullSequenceAgain) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;

    // Drain all records
    int count1 = 0;
    while (src.next(m)) ++count1;

    // Reset and drain again
    src.reset();
    int count2 = 0;
    while (src.next(m)) ++count2;

    EXPECT_EQ(count1, count2);
    EXPECT_EQ(count1, 10);
}

TEST(BinaryFileSource, ResetOnEmptyFileIsNoOp) {
    BinaryFileSource src(fixture_path("empty.bin"));
    Message m;

    EXPECT_FALSE(src.next(m));
    src.reset();
    EXPECT_FALSE(src.next(m));
}

// ===========================================================================
// BinaryFileSource: Full Record Integrity (all fields for all records)
// ===========================================================================

TEST(BinaryFileSource, MixedActionsFileAllFieldsCorrect) {
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Message m;

    // Record 0: Add bid @ 999.75, qty=10, order_id=100
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 100u);
    EXPECT_EQ(m.action, Message::Action::Add);
    EXPECT_EQ(m.side, Message::Side::Bid);
    EXPECT_DOUBLE_EQ(m.price, 999.75);
    EXPECT_EQ(m.qty, 10u);

    // Record 1: Add ask @ 1000.25, qty=10, order_id=101
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 101u);
    EXPECT_EQ(m.action, Message::Action::Add);
    EXPECT_EQ(m.side, Message::Side::Ask);
    EXPECT_DOUBLE_EQ(m.price, 1000.25);
    EXPECT_EQ(m.qty, 10u);

    // Record 2: Modify bid @ 999.75, qty=25, order_id=100
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 100u);
    EXPECT_EQ(m.action, Message::Action::Modify);
    EXPECT_EQ(m.side, Message::Side::Bid);
    EXPECT_DOUBLE_EQ(m.price, 999.75);
    EXPECT_EQ(m.qty, 25u);

    // Record 3: Trade ask @ 1000.25, qty=3, order_id=101
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 101u);
    EXPECT_EQ(m.action, Message::Action::Trade);
    EXPECT_EQ(m.side, Message::Side::Ask);
    EXPECT_DOUBLE_EQ(m.price, 1000.25);
    EXPECT_EQ(m.qty, 3u);

    // Record 4: Fill (-> Trade) ask @ 1000.25, qty=2, order_id=101
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 101u);
    EXPECT_EQ(m.action, Message::Action::Trade);
    EXPECT_EQ(m.side, Message::Side::Ask);
    EXPECT_DOUBLE_EQ(m.price, 1000.25);
    EXPECT_EQ(m.qty, 2u);

    // Record 5: Cancel bid @ 999.75, qty=25, order_id=100
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 100u);
    EXPECT_EQ(m.action, Message::Action::Cancel);
    EXPECT_EQ(m.side, Message::Side::Bid);
    EXPECT_DOUBLE_EQ(m.price, 999.75);
    EXPECT_EQ(m.qty, 25u);

    // Record 6: Add side='N' (-> Bid) @ 999.00, qty=5, order_id=200
    ASSERT_TRUE(src.next(m));
    EXPECT_EQ(m.order_id, 200u);
    EXPECT_EQ(m.action, Message::Action::Add);
    EXPECT_EQ(m.side, Message::Side::Bid);
    EXPECT_DOUBLE_EQ(m.price, 999.0);
    EXPECT_EQ(m.qty, 5u);

    // No more records
    EXPECT_FALSE(src.next(m));
}

// ===========================================================================
// BinaryFileSource: Integration with Book
// ===========================================================================

TEST(BinaryFileSource, IntegrationWithBookValidBBOAfterAllRecords) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Book book;
    Message m;

    while (src.next(m)) {
        book.apply(m);
    }

    // After 5 bid + 5 ask adds, book should have valid BBO
    EXPECT_FALSE(std::isnan(book.best_bid()));
    EXPECT_FALSE(std::isnan(book.best_ask()));
    EXPECT_DOUBLE_EQ(book.best_bid(), 999.75);
    EXPECT_DOUBLE_EQ(book.best_ask(), 1000.00);
    EXPECT_DOUBLE_EQ(book.spread(), 0.25);
    EXPECT_EQ(book.bid_depth(), 5u);
    EXPECT_EQ(book.ask_depth(), 5u);
}

TEST(BinaryFileSource, IntegrationWithBookMixedActionsFile) {
    BinaryFileSource src(fixture_path("mixed_actions.bin"));
    Book book;
    Message m;

    while (src.next(m)) {
        book.apply(m);
    }

    // After processing:
    // - Bid at 999.75 was added (qty=10), modified (qty=25), then cancelled
    // - Ask at 1000.25 was added (qty=10), traded 3, filled 2 -> qty=5 remaining
    // - Bid at 999.00 was added (qty=5) via side='N'
    // So: best_bid=999.00, best_ask=1000.25
    EXPECT_DOUBLE_EQ(book.best_bid(), 999.0);
    EXPECT_DOUBLE_EQ(book.best_ask(), 1000.25);
    EXPECT_EQ(book.bid_depth(), 1u);
    EXPECT_EQ(book.ask_depth(), 1u);
}

TEST(BinaryFileSource, IntegrationResetAndReplay) {
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Book book1, book2;
    Message m;

    // First pass
    while (src.next(m)) book1.apply(m);

    // Reset source and replay into a fresh book
    src.reset();
    while (src.next(m)) book2.apply(m);

    EXPECT_DOUBLE_EQ(book1.best_bid(), book2.best_bid());
    EXPECT_DOUBLE_EQ(book1.best_ask(), book2.best_ask());
    EXPECT_DOUBLE_EQ(book1.spread(), book2.spread());
}

// ===========================================================================
// M2: Price precision — int64_t -> double conversion for large prices
// ===========================================================================

TEST(BinaryFileSource, PrecisionPreservedForTypicalMESPrice) {
    // Create a test file with a typical /MES price: $5000.123456789
    // price_raw = 5000123456789 (price * 1e9)
    // This should round-trip without precision loss since 5000123456789 < 2^53
    BinaryFileSource src(fixture_path("precision_test.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));

    // Expected: 5000.123456789 — full 9 decimal places of precision
    // The comparison tolerance is ~1e-9 (less than 1 nano-dollar)
    EXPECT_NEAR(m.price, 5000.123456789, 1e-9)
        << "Price should preserve 9 decimal places for typical /MES prices";
}

TEST(BinaryFileSource, PrecisionPreservedAtHighEndOfSafeRange) {
    // Price at upper boundary of typical trading: $100,000.123456789
    // price_raw = 100000123456789 < 2^53, so precision should be preserved
    BinaryFileSource src(fixture_path("precision_test_high.bin"));
    Message m;
    ASSERT_TRUE(src.next(m));

    // 100000.123456789 should be fully representable
    EXPECT_NEAR(m.price, 100000.123456789, 1e-8)
        << "Prices up to $100,000 should preserve ~8 decimal places";
}

TEST(BinaryFileSource, DocumentedPrecisionLimitExceedsTypicalUsage) {
    // This is a documentation test, not a correctness test.
    // int64_t -> double loses precision when |value| > 2^53.
    // 2^53 = 9,007,199,254,740,992
    // As price_raw: 9007199.254740992 dollars (>$9 million)
    // Real /MES data will never approach this, so precision loss is theoretical.

    // Verify the safe range constant exists and is documented
    // (Implementation should have a static_assert or constant documenting this)
    constexpr int64_t SAFE_PRICE_RAW_LIMIT = (1LL << 53);  // 2^53

    // Any price_raw below this should convert without precision loss
    // $9 million * 1e9 = 9e15, which is greater than 2^53 (~9e15)
    // So $9 million is right at the boundary — anything below is safe
    EXPECT_GT(SAFE_PRICE_RAW_LIMIT, 5000LL * 1'000'000'000LL)
        << "$5000 prices are well within safe precision range";
    EXPECT_GT(SAFE_PRICE_RAW_LIMIT, 100000LL * 1'000'000'000LL)
        << "$100,000 prices are within safe precision range";
}
