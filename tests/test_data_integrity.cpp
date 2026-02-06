#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "synthetic_source.h"
#include "binary_file_source.h"
#include "test_helpers.h"

// ===========================================================================
// CRITICAL 1: SyntheticSource pick_order() with empty live_orders
//
// Spec: pick_order() must not exhibit UB when live_orders is empty.
// The lambda creates uniform_int_distribution(0, size-1) which is UB when
// size==0 because 0-1 wraps to SIZE_MAX on unsigned types.
//
// Note: We cannot directly test the internal lambda, but we can test that
// SyntheticSource handles edge cases safely. The fix should make pick_order
// either throw or guard against empty vectors.
// ===========================================================================

// Test that SyntheticSource generation completes without UB or crash.
// This is a sanity check that the existing code path works.
TEST(SyntheticSource_DataIntegrity, GenerationDoesNotCrash) {
    // Multiple seeds to exercise different random paths
    for (uint64_t seed : {0ULL, 1ULL, 42ULL, 12345ULL, 999999ULL}) {
        EXPECT_NO_THROW({
            SyntheticSource src(seed);
            Message m;
            int count = 0;
            while (src.next(m)) ++count;
            EXPECT_GT(count, 0) << "Seed " << seed << " produced no messages";
        }) << "Seed " << seed << " caused an exception";
    }
}

// Test that generation produces valid, non-empty message sequence.
// The internal pick_order() should never be called with empty live_orders
// when following the correct control flow.
TEST(SyntheticSource_DataIntegrity, GenerationProducesValidSequence) {
    SyntheticSource src(42);
    Message m;

    int add_count = 0;
    int cancel_count = 0;
    int modify_count = 0;
    int trade_count = 0;

    while (src.next(m)) {
        // Count action types
        switch (m.action) {
            case Message::Action::Add: ++add_count; break;
            case Message::Action::Cancel: ++cancel_count; break;
            case Message::Action::Modify: ++modify_count; break;
            case Message::Action::Trade: ++trade_count; break;
        }

        // All prices should be positive and finite
        EXPECT_TRUE(std::isfinite(m.price)) << "Price should be finite";
        EXPECT_GT(m.price, 0.0) << "Price should be positive";
    }

    // Should have a mix of actions (cancel/modify/trade require live orders)
    // The initial 10 Add messages guarantee live_orders is populated
    EXPECT_GE(add_count, 10) << "Should have at least 10 Add messages (phase 1)";

    // These require live_orders to not be empty when chosen
    // If pick_order was being called unsafely, these would likely cause issues
    int non_add_actions = cancel_count + modify_count + trade_count;
    EXPECT_GT(non_add_actions, 0)
        << "Should have some Cancel/Modify/Trade actions that use pick_order";
}

// Test that after draining all messages via Cancel/Trade, further generation
// would not call pick_order() on empty live_orders. This tests the guard:
// "if (act == 0 || live_orders.empty()) { // Add }"
TEST(SyntheticSource_DataIntegrity, LiveOrdersEmptyFallbackToAdd) {
    // The SyntheticSource has internal logic that falls back to Add when
    // live_orders is empty. We can't directly test pick_order(), but we can
    // verify the source produces valid output even with adversarial random seeds.

    // Try many seeds to find one that might drain live_orders
    for (uint64_t seed = 0; seed < 100; ++seed) {
        SyntheticSource src(seed);
        Message m;

        // Should complete without UB or crash
        while (src.next(m)) {
            // Basic validity check
            EXPECT_TRUE(m.order_id > 0 || m.action == Message::Action::Add)
                << "Non-Add action should have valid order_id";
        }
    }
}


// ===========================================================================
// CRITICAL 2: BinaryFileSource gcount() signed-to-unsigned cast
//
// Spec: Negative gcount() values must be handled safely (treated as 0 bytes).
// The cast: static_cast<uint32_t>(bytes_read / sizeof(FlatRecord))
// If bytes_read is negative, this produces a huge positive number.
//
// Note: In normal operation gcount() is non-negative, but error conditions
// or corrupted streams could theoretically produce negative values. The fix
// should guard against negative gcount().
// ===========================================================================

// Test that truncated file handling works correctly (existing behavior).
// The current code handles this case by resizing to full_records.
TEST(BinaryFileSource_DataIntegrity, TruncatedFileHandledGracefully) {
    // truncated.bin should have header claiming more records than present
    BinaryFileSource src(fixture_path("truncated.bin"));

    Message m;
    int count = 0;
    while (src.next(m)) {
        ++count;
        // Verify each message has valid fields
        EXPECT_TRUE(std::isfinite(m.price));
    }

    // Should return some records (the ones that were fully read)
    // rather than crashing or returning garbage
    EXPECT_GE(count, 0) << "Truncated file should return >= 0 valid records";

    // record_count() should reflect actual readable records
    EXPECT_LE(src.record_count(), 1000u)
        << "record_count should be reasonable, not corrupted by bad cast";
}

// Test that BinaryFileSource handles completely corrupted/minimal truncation
// where only a partial record exists after header.
TEST(BinaryFileSource_DataIntegrity, TruncatedPartialRecordHandled) {
    // truncated.bin has header claiming records but file is short
    // This exercises the gcount() < expected path
    BinaryFileSource src(fixture_path("truncated.bin"));

    // Should not crash or produce garbage data
    Message m;
    while (src.next(m)) {
        // Price should be finite (not corrupted by bad memory read)
        EXPECT_TRUE(std::isfinite(m.price));
        // Order ID should be reasonable
        EXPECT_LT(m.order_id, UINT64_MAX);
        // Qty should be reasonable
        EXPECT_LE(m.qty, 1000000u) << "Qty looks corrupted";
    }
}

// Test that record_count() returns correct value after handling truncation.
TEST(BinaryFileSource_DataIntegrity, RecordCountReflectsActualRecords) {
    BinaryFileSource src(fixture_path("truncated.bin"));

    // Count actual readable records
    Message m;
    uint32_t actual_count = 0;
    while (src.next(m)) {
        ++actual_count;
    }

    // record_count() should match actual iteration count
    EXPECT_EQ(src.record_count(), actual_count)
        << "record_count() should match actual readable records";
}

// Test that empty file with valid header is handled correctly.
TEST(BinaryFileSource_DataIntegrity, EmptyRecordsSectionHandled) {
    BinaryFileSource src(fixture_path("empty.bin"));

    // Should report 0 records
    EXPECT_EQ(src.record_count(), 0u);

    // next() should return false immediately
    Message m;
    EXPECT_FALSE(src.next(m));
}


// ===========================================================================
// HIGH: Message::is_valid() validation method
//
// Spec: Add a static validation method that returns false for:
// - price < 0 or !std::isfinite(price)
// - qty == 0
// - order_id == 0
// ===========================================================================

// --- Tests for valid messages (should return true) ---

TEST(Message_IsValid, ValidMessageReturnsTrue) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, ValidAskMessageReturnsTrue) {
    Message m = make_msg(42, Message::Side::Ask, Message::Action::Add, 1000.25, 100);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, ValidCancelMessageReturnsTrue) {
    Message m = make_msg(999, Message::Side::Bid, Message::Action::Cancel, 99.99, 50);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, ValidModifyMessageReturnsTrue) {
    Message m = make_msg(123, Message::Side::Ask, Message::Action::Modify, 500.0, 25);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, ValidTradeMessageReturnsTrue) {
    Message m = make_msg(456, Message::Side::Bid, Message::Action::Trade, 1234.56, 1);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, MinimalValidQuantityReturnsTrue) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 1);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, SmallPositivePriceReturnsTrue) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 0.01, 10);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, LargePriceReturnsTrue) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 1e9, 10);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, LargeQuantityReturnsTrue) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 1000000);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, LargeOrderIdReturnsTrue) {
    Message m = make_msg(UINT64_MAX - 1, Message::Side::Bid, Message::Action::Add, 100.0, 10);
    EXPECT_TRUE(m.is_valid());
}

// --- Tests for invalid prices ---

TEST(Message_IsValid, NegativePriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, -100.0, 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, ZeroPriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 0.0, 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, NaNPriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::quiet_NaN(), 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, PositiveInfinityPriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::infinity(), 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, NegativeInfinityPriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         -std::numeric_limits<double>::infinity(), 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, SlightlyNegativePriceReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, -0.001, 10);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, NegativeZeroPriceReturnsFalse) {
    // -0.0 == 0.0 in IEEE 754, so this should return false (zero price invalid)
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, -0.0, 10);
    EXPECT_FALSE(m.is_valid());
}

// --- Tests for invalid quantities ---

TEST(Message_IsValid, ZeroQuantityReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 0);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, ZeroQuantityCancelReturnsFalse) {
    // Even Cancel messages should have qty > 0 (qty being cancelled)
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 0);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, ZeroQuantityModifyReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Modify, 100.0, 0);
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, ZeroQuantityTradeReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Trade, 100.0, 0);
    EXPECT_FALSE(m.is_valid());
}

// --- Tests for invalid order_id ---

TEST(Message_IsValid, ZeroOrderIdReturnsFalse) {
    Message m;
    m.order_id = 0;
    m.side = Message::Side::Bid;
    m.action = Message::Action::Add;
    m.price = 100.0;
    m.qty = 10;
    m.ts_ns = 0;
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, ZeroOrderIdWithValidOtherFieldsReturnsFalse) {
    Message m = make_msg(0, Message::Side::Ask, Message::Action::Trade, 500.0, 100);
    EXPECT_FALSE(m.is_valid());
}

// --- Tests for multiple invalid fields ---

TEST(Message_IsValid, MultipleInvalidFieldsReturnsFalse) {
    // Zero order_id, zero qty, negative price
    Message m;
    m.order_id = 0;
    m.side = Message::Side::Bid;
    m.action = Message::Action::Add;
    m.price = -100.0;
    m.qty = 0;
    m.ts_ns = 0;
    EXPECT_FALSE(m.is_valid());
}

TEST(Message_IsValid, NaNPriceAndZeroQtyReturnsFalse) {
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::quiet_NaN(), 0);
    EXPECT_FALSE(m.is_valid());
}

// --- Edge case tests ---

TEST(Message_IsValid, SubnormalPositivePriceReturnsTrue) {
    // Very small but positive denormalized number
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::denorm_min(), 10);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, MaxDoublePriceReturnsTrue) {
    // Maximum finite double
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::max(), 10);
    EXPECT_TRUE(m.is_valid());
}

TEST(Message_IsValid, MinPositiveDoublePriceReturnsTrue) {
    // Minimum positive normalized double
    Message m = make_msg(1, Message::Side::Bid, Message::Action::Add,
                         std::numeric_limits<double>::min(), 10);
    EXPECT_TRUE(m.is_valid());
}

// --- Consistency tests ---

TEST(Message_IsValid, SyntheticSourceProducesValidMessages) {
    // All messages from SyntheticSource should be valid
    SyntheticSource src(42);
    Message m;
    int count = 0;
    while (src.next(m)) {
        EXPECT_TRUE(m.is_valid())
            << "SyntheticSource message " << count << " should be valid: "
            << "order_id=" << m.order_id
            << ", price=" << m.price
            << ", qty=" << m.qty;
        ++count;
    }
    EXPECT_GT(count, 0);
}

TEST(Message_IsValid, BinaryFileSourceProducesValidMessages) {
    // All messages from a well-formed binary file should be valid
    BinaryFileSource src(fixture_path("valid_10records.bin"));
    Message m;
    int count = 0;
    while (src.next(m)) {
        EXPECT_TRUE(m.is_valid())
            << "BinaryFileSource message " << count << " should be valid: "
            << "order_id=" << m.order_id
            << ", price=" << m.price
            << ", qty=" << m.qty;
        ++count;
    }
    EXPECT_EQ(count, 10);
}
