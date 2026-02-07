#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <climits>
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// Book: Empty state
// ===========================================================================

TEST(Book, EmptyBookBestBidIsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.best_bid()));
}

TEST(Book, EmptyBookBestAskIsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.best_ask()));
}

TEST(Book, EmptyBookMidPriceIsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.mid_price()));
}

TEST(Book, EmptyBookSpreadIsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.spread()));
}

TEST(Book, EmptyBookHasZeroBidDepth) {
    Book book;
    EXPECT_EQ(book.bid_depth(), 0);
}

TEST(Book, EmptyBookHasZeroAskDepth) {
    Book book;
    EXPECT_EQ(book.ask_depth(), 0);
}

// ===========================================================================
// Book: Add messages
// ===========================================================================

TEST(Book, AddBidCreatesLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));

    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
    EXPECT_EQ(book.bid_depth(), 1);
}

TEST(Book, AddAskCreatesLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    EXPECT_DOUBLE_EQ(book.best_ask(), 101.0);
    EXPECT_EQ(book.ask_depth(), 1);
}

TEST(Book, AddMultipleBidLevels) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 101.0, 5));

    EXPECT_DOUBLE_EQ(book.best_bid(), 101.0);  // Highest bid
    EXPECT_EQ(book.bid_depth(), 3);
}

TEST(Book, AddMultipleAskLevels) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 102.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 103.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 5));

    EXPECT_DOUBLE_EQ(book.best_ask(), 101.0);  // Lowest ask
    EXPECT_EQ(book.ask_depth(), 3);
}

TEST(Book, AddToExistingLevelAggregatesQty) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));

    // Should still be one level, with aggregated quantity
    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
}

// ===========================================================================
// Book: Cancel messages
// ===========================================================================

TEST(Book, CancelReducesQtyAtLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));

    // Cancel order 1 (qty 10) — level should still exist with remaining qty
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));
    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
}

TEST(Book, CancelRemovesLevelWhenQtyReachesZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_TRUE(std::isnan(book.best_bid()));
}

TEST(Book, CancelUnknownOrderIsNoOp) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));

    // Cancel an order that doesn't exist — should not crash
    book.apply(make_msg(999, Message::Side::Bid, Message::Action::Cancel, 100.0, 5));

    // Original order should be unaffected
    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
}

TEST(Book, CancelBestBidRevealsNextLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    // Cancel the best bid
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    EXPECT_DOUBLE_EQ(book.best_bid(), 99.0);
    EXPECT_EQ(book.bid_depth(), 1);
}

TEST(Book, CancelBestAskRevealsNextLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));

    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Cancel, 101.0, 10));

    EXPECT_DOUBLE_EQ(book.best_ask(), 102.0);
    EXPECT_EQ(book.ask_depth(), 1);
}

// ===========================================================================
// Book: Modify messages
// ===========================================================================

TEST(Book, ModifyUpdatesQtyAtLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 100.0, 25));

    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
}

TEST(Book, ModifySamePriceUpdatesLevelQtyCorrectly) {
    // Req 1: Modify with same price — level qty should reflect the change
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 100.0, 25));

    EXPECT_EQ(book.best_bid_qty(), 25);
}

TEST(Book, ModifySamePriceWithOtherOrdersOnLevel) {
    // Req 1: Only the modified order's qty changes; other orders unaffected
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));
    // Total at 100.0 = 30. Modify order 1: 10 -> 5. New total should be 25.
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 100.0, 5));

    EXPECT_EQ(book.best_bid_qty(), 25);
    EXPECT_EQ(book.bid_depth(), 1);
}

TEST(Book, ModifyBidPriceChangeMovesOrderToNewLevel) {
    // Req 2: Modify with different price — old qty removed, new qty at new price
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 15));

    // Old price level should be gone (was only order there)
    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 101.0);
    EXPECT_EQ(book.best_bid_qty(), 15);
}

TEST(Book, ModifyAskPriceChangeMovesOrderToNewLevel) {
    // Req 2: Same behavior on ask side
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Modify, 100.0, 20));

    EXPECT_EQ(book.ask_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_ask(), 100.0);
    EXPECT_EQ(book.best_ask_qty(), 20);
}

TEST(Book, ModifyPriceChangeRemovesEmptyOldLevel) {
    // Req 2: Old price level removed if it reaches zero after the move
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 102.0, 10));

    // 100.0 should no longer exist as a level
    auto levels = book.top_bids(10);
    EXPECT_DOUBLE_EQ(levels[0].price, 102.0);
    EXPECT_EQ(levels[0].qty, 10u);
    EXPECT_EQ(book.bid_depth(), 1);
}

TEST(Book, ModifyPriceChangePreservesOtherOrdersAtOldLevel) {
    // Req 4: Two orders at same price, modify one to a new price.
    // The other order's qty remains at the old level.
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));
    // Move order 1 from 100.0 to 101.0
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 15));

    // 100.0 should still have order 2's qty
    EXPECT_EQ(book.bid_depth(), 2);  // two price levels now
    EXPECT_DOUBLE_EQ(book.best_bid(), 101.0);
    EXPECT_EQ(book.best_bid_qty(), 15);  // new level

    // Check old level still has order 2's qty
    auto levels = book.top_bids(2);
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_EQ(levels[0].qty, 15u);
    EXPECT_DOUBLE_EQ(levels[1].price, 100.0);
    EXPECT_EQ(levels[1].qty, 20u);
}

TEST(Book, ModifyPriceChangeUpdatesBestBid) {
    // AC: Modify with price change updates best_bid() correctly
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 5));

    // Move order 1 up to 101.0 — best_bid should update
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 10));
    EXPECT_DOUBLE_EQ(book.best_bid(), 101.0);

    // Move order 1 down below order 2 — best_bid should revert to 99.0
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 98.0, 10));
    EXPECT_DOUBLE_EQ(book.best_bid(), 99.0);
}

TEST(Book, ModifyPriceChangeUpdatesBestAsk) {
    // AC: Modify with price change updates best_ask() correctly
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 5));

    // Move order 1 down to 100.0 — best_ask should update
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Modify, 100.0, 10));
    EXPECT_DOUBLE_EQ(book.best_ask(), 100.0);

    // Move order 1 above order 2 — best_ask should revert to 102.0
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Modify, 103.0, 10));
    EXPECT_DOUBLE_EQ(book.best_ask(), 102.0);
}

TEST(Book, ModifyPriceChangeUpdatesSpreadAndMidPrice) {
    // AC: spread and mid_price stay consistent after price-changing modify
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    // Spread = 2.0, mid = 101.0
    EXPECT_DOUBLE_EQ(book.spread(), 2.0);
    EXPECT_DOUBLE_EQ(book.mid_price(), 101.0);

    // Move bid up to 101.0
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 10));
    EXPECT_DOUBLE_EQ(book.spread(), 1.0);
    EXPECT_DOUBLE_EQ(book.mid_price(), 101.5);
}

TEST(Book, ModifyUnknownOrderTreatedAsAdd) {
    // Req 3: Modify unknown order should act as Add
    Book book;
    book.apply(make_msg(999, Message::Side::Bid, Message::Action::Modify, 100.0, 10));

    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 100.0);
    EXPECT_EQ(book.best_bid_qty(), 10);
}

TEST(Book, ModifyPriceChangeThenCancelWorksCorrectly) {
    // After a price-changing modify, cancel should remove from the new level
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 15));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 101.0, 15));

    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_TRUE(std::isnan(book.best_bid()));
}

TEST(Book, ModifyPriceChangeThenTradeIsNoOp) {
    // Databento spec: Trade is a no-op. After modify, trade should not change qty.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Modify, 100.0, 20));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 100.0, 5));

    EXPECT_EQ(book.ask_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_ask(), 100.0);
    EXPECT_EQ(book.best_ask_qty(), 20u);  // Trade is no-op, qty unchanged from modify
}

TEST(Book, ModifyPriceChangeToExistingLevel) {
    // Move an order to a price level that already has other orders
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 101.0, 20));

    // Move order 1 from 100.0 to 101.0 with qty 15
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 15));

    // 100.0 level should be gone, 101.0 should have 20 + 15 = 35
    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 101.0);
    EXPECT_EQ(book.best_bid_qty(), 35);
}

TEST(Book, ModifyPriceChangeUpdatesOrderEntryPrice) {
    // After price-changing modify, the order entry must track the new price.
    // Subsequent modify at yet another price should work correctly.
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 101.0, 15));
    // Now modify again to 102.0 — should remove from 101.0, not 100.0
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Modify, 102.0, 20));

    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_bid(), 102.0);
    EXPECT_EQ(book.best_bid_qty(), 20);
    // 100.0 and 101.0 should both be gone
}

// ===========================================================================
// Book: Trade messages
// ===========================================================================

TEST(Book, TradeIsNoOp) {
    // Databento spec: Trade/Fill messages do not affect the book.
    // Book changes are communicated entirely through Add, Cancel, and Modify.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 101.0, 3));

    EXPECT_EQ(book.ask_depth(), 1);
    EXPECT_DOUBLE_EQ(book.best_ask(), 101.0);
    EXPECT_EQ(book.best_ask_qty(), 10u);  // unchanged
}

TEST(Book, TradeDoesNotRemoveLevelWhenFullyFilled) {
    // Databento spec: Trade does not affect the book.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 101.0, 10));

    EXPECT_EQ(book.ask_depth(), 1);  // level stays
    EXPECT_DOUBLE_EQ(book.best_ask(), 101.0);  // price unchanged
    EXPECT_EQ(book.best_ask_qty(), 10u);  // qty unchanged
}

// ===========================================================================
// Book: Spread and Mid Price
// ===========================================================================

TEST(Book, SpreadIsAskMinusBid) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 99.50, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.50, 10));

    EXPECT_DOUBLE_EQ(book.spread(), 1.0);
}

TEST(Book, MidPriceIsAverageOfBidAndAsk) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 99.50, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.50, 10));

    EXPECT_DOUBLE_EQ(book.mid_price(), 100.0);
}

TEST(Book, SpreadIsNaNWhenOnlyBids) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    EXPECT_TRUE(std::isnan(book.spread()));
}

TEST(Book, SpreadIsNaNWhenOnlyAsks) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    EXPECT_TRUE(std::isnan(book.spread()));
}

TEST(Book, MidPriceIsNaNWhenOnlyBids) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    EXPECT_TRUE(std::isnan(book.mid_price()));
}

// ===========================================================================
// Book: Reset
// ===========================================================================

TEST(Book, ResetClearsAllState) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    book.reset();

    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.ask_depth(), 0);
    EXPECT_TRUE(std::isnan(book.best_bid()));
    EXPECT_TRUE(std::isnan(book.best_ask()));
    EXPECT_TRUE(std::isnan(book.mid_price()));
    EXPECT_TRUE(std::isnan(book.spread()));
}

TEST(Book, ResetAllowsRebuildingBook) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.reset();
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 200.0, 5));

    EXPECT_DOUBLE_EQ(book.best_bid(), 200.0);
    EXPECT_EQ(book.bid_depth(), 1);
}

// ===========================================================================
// Book: Integration with SyntheticSource
// ===========================================================================

TEST(Book, ProcessesAllSyntheticSourceMessagesWithoutCrash) {
    // Forward-declared to avoid circular dependency — just include headers
    // This tests that the Book can process all messages from SyntheticSource
    Book book;
    SyntheticSource src;
    Message m;
    while (src.next(m)) {
        book.apply(m);
    }
    // After processing all messages, book should have some state
    // (not necessarily valid BBO — depends on the messages)
}

TEST(Book, HasValidBBOAfterPhase1) {
    Book book;
    SyntheticSource src;
    Message m;

    // Process first 10 messages (Phase 1: build initial book)
    for (int i = 0; i < 10 && src.next(m); ++i) {
        book.apply(m);
    }

    // After Phase 1, book should have valid BBO
    EXPECT_FALSE(std::isnan(book.best_bid()));
    EXPECT_FALSE(std::isnan(book.best_ask()));
    EXPECT_GT(book.best_bid(), 0.0);
    EXPECT_GT(book.best_ask(), 0.0);
    EXPECT_GT(book.best_ask(), book.best_bid()) << "Ask should be above bid";
    EXPECT_GT(book.spread(), 0.0);
    EXPECT_EQ(book.bid_depth(), 5);
    EXPECT_EQ(book.ask_depth(), 5);
}

// ===========================================================================
// M3: Quantity overflow — saturating addition to prevent wraparound
// ===========================================================================

TEST(Book, NormalQuantityAdditionWorks) {
    // Normal case: adding typical quantities should work as before
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 1000));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 2000));

    EXPECT_EQ(book.best_bid_qty(), 3000u);
}

TEST(Book, AddSaturatesAtUint32Max) {
    // Adding UINT32_MAX to an existing quantity should saturate, not wrap
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 100));

    // Now add UINT32_MAX — should saturate at UINT32_MAX, not wrap to ~99
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, UINT32_MAX));

    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX)
        << "Quantity should saturate at UINT32_MAX, not wrap around";
}

TEST(Book, AddStartingAtUint32MaxStaysSaturated) {
    // If level is already at max, adding more should keep it at max
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, UINT32_MAX));

    // Add another quantity — should stay at UINT32_MAX
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 1000));

    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX)
        << "Adding to already-saturated level should stay at UINT32_MAX";
}

TEST(Book, MultipleAddsThatWouldOverflowSaturate) {
    // Multiple large additions that would overflow should saturate
    Book book;
    constexpr uint32_t LARGE_QTY = UINT32_MAX / 2 + 1;  // Just over half max

    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, LARGE_QTY));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, LARGE_QTY));

    // LARGE_QTY + LARGE_QTY would overflow, should saturate
    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX)
        << "Sum of two large quantities should saturate, not wrap";
}

TEST(Book, ModifySaturatesOnOverflow) {
    // Modify that increases qty beyond UINT32_MAX should saturate
    Book book;
    constexpr uint32_t LARGE_QTY = UINT32_MAX - 100;

    // Add two orders: one with LARGE_QTY, one with 50
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, LARGE_QTY));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 50));

    // Total at level: LARGE_QTY + 50 = UINT32_MAX - 50 (no overflow yet)
    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX - 50);

    // Modify order 2 from 50 to 200: would add 150, causing overflow
    // New total would be (LARGE_QTY - 50) + 200, which exceeds UINT32_MAX
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Modify, 100.0, 200));

    // Should saturate at UINT32_MAX
    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX)
        << "Modify that causes overflow should saturate";
}

TEST(Book, SubtractFromSaturatedValueWorks) {
    // After saturation, subtracting (via Cancel or Trade) should work correctly
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, UINT32_MAX));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 1000));

    // Level is now saturated at UINT32_MAX
    EXPECT_EQ(book.best_bid_qty(), UINT32_MAX);

    // Cancel order 1 — should subtract UINT32_MAX, leaving just order 2's qty (1000)
    // But wait — the level was capped at UINT32_MAX, so we don't know order 1's true contribution
    // The expected behavior: Cancel removes the order's recorded qty from the level
    // Since order 1's qty is UINT32_MAX, this should leave the level at 0 or small value
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, UINT32_MAX));

    // Order 2 (qty=1000) should remain
    EXPECT_EQ(book.best_bid_qty(), 1000u)
        << "After canceling saturated order, remaining order's qty should be visible";
}

TEST(Book, TradeDoesNotReduceSaturatedLevel) {
    // Databento spec: Trade is a no-op. Saturated level stays at UINT32_MAX.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, UINT32_MAX));

    // Trade for 1000 — should NOT reduce (Trade is no-op)
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 101.0, 1000));

    EXPECT_EQ(book.best_ask_qty(), UINT32_MAX)
        << "Trade should not reduce level quantity (Databento spec: Trade is no-op)";
}

TEST(Book, OverflowOnAskSideSaturates) {
    // Same overflow protection should apply to ask side
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, UINT32_MAX - 1));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 100));

    EXPECT_EQ(book.best_ask_qty(), UINT32_MAX)
        << "Ask side should also saturate on overflow";
}
