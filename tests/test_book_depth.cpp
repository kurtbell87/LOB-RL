#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// PriceLevel struct
// ===========================================================================

TEST(BookDepth, PriceLevelStructHasPriceAndQty) {
    Book::PriceLevel lvl{100.0, 42};
    EXPECT_DOUBLE_EQ(lvl.price, 100.0);
    EXPECT_EQ(lvl.qty, 42u);
}

// ===========================================================================
// top_bids: Empty book
// ===========================================================================

TEST(BookDepth, TopBidsEmptyBookReturnsKEntries) {
    Book book;
    auto levels = book.top_bids(10);
    EXPECT_EQ(levels.size(), 10u);
}

TEST(BookDepth, TopBidsEmptyBookAllPricesAreNaN) {
    Book book;
    auto levels = book.top_bids(10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price)) << "Level " << i << " price should be NaN";
    }
}

TEST(BookDepth, TopBidsEmptyBookAllQtysAreZero) {
    Book book;
    auto levels = book.top_bids(10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(levels[i].qty, 0u) << "Level " << i << " qty should be 0";
    }
}

// ===========================================================================
// top_bids: Partial fill (fewer levels than K)
// ===========================================================================

// Fixture: Book with 3 bid levels at 100.0/10, 99.0/20, 98.0/30.
class ThreeBidLevels : public ::testing::Test {
protected:
    void SetUp() override {
        book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
        book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
        book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));
    }

    Book book;
};

TEST_F(ThreeBidLevels, TopBidsReturns10Entries) {
    auto levels = book.top_bids(10);
    EXPECT_EQ(levels.size(), 10u);
}

TEST_F(ThreeBidLevels, TopBidsFirst3Populated) {
    auto levels = book.top_bids(10);

    // First 3 should be populated with real data
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_EQ(levels[0].qty, 10u);
    EXPECT_DOUBLE_EQ(levels[1].price, 99.0);
    EXPECT_EQ(levels[1].qty, 20u);
    EXPECT_DOUBLE_EQ(levels[2].price, 98.0);
    EXPECT_EQ(levels[2].qty, 30u);
}

TEST_F(ThreeBidLevels, TopBidsRemaining7AreNaNZero) {
    auto levels = book.top_bids(10);

    for (int i = 3; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price)) << "Level " << i << " price should be NaN";
        EXPECT_EQ(levels[i].qty, 0u) << "Level " << i << " qty should be 0";
    }
}

// ===========================================================================
// top_bids: Ordering (descending price, best first)
// ===========================================================================

TEST(BookDepth, TopBidsOrderedDescendingPrice) {
    Book book;
    // Insert in random order
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 97.0, 5));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 99.0, 15));
    book.apply(make_msg(4, Message::Side::Bid, Message::Action::Add, 98.0, 20));
    book.apply(make_msg(5, Message::Side::Bid, Message::Action::Add, 96.0, 25));

    auto levels = book.top_bids(5);

    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_DOUBLE_EQ(levels[1].price, 99.0);
    EXPECT_DOUBLE_EQ(levels[2].price, 98.0);
    EXPECT_DOUBLE_EQ(levels[3].price, 97.0);
    EXPECT_DOUBLE_EQ(levels[4].price, 96.0);
}

// ===========================================================================
// top_bids: More levels than K
// ===========================================================================

TEST(BookDepth, TopBids12LevelsRequestingDefault10ReturnsOnly10) {
    Book book;
    for (int i = 0; i < 12; ++i) {
        book.apply(make_msg(i + 1, Message::Side::Bid, Message::Action::Add,
                            100.0 - i * 0.25, 10));
    }

    auto levels = book.top_bids();  // default k=10
    EXPECT_EQ(levels.size(), 10u);
}

TEST(BookDepth, TopBids12LevelsReturnsTopOnesOnly) {
    Book book;
    for (int i = 0; i < 12; ++i) {
        book.apply(make_msg(i + 1, Message::Side::Bid, Message::Action::Add,
                            100.0 - i * 0.25, 10));
    }

    auto levels = book.top_bids(10);

    // Best bid is 100.0, worst of top-10 is 100.0 - 9*0.25 = 97.75
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_DOUBLE_EQ(levels[9].price, 97.75);

    // The levels at 97.50 and 97.25 should NOT appear
    for (const auto& lvl : levels) {
        EXPECT_GE(lvl.price, 97.75);
    }
}

// ===========================================================================
// top_bids: Aggregation (multiple orders at same price)
// ===========================================================================

TEST(BookDepth, TopBidsAggregatesOrdersAtSamePrice) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 25));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 100.0, 5));

    auto levels = book.top_bids(1);
    EXPECT_EQ(levels.size(), 1u);
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_EQ(levels[0].qty, 40u);
}

// ===========================================================================
// top_bids: After cancel removes best level
// ===========================================================================

TEST_F(ThreeBidLevels, TopBidsCancelBestLevelShiftsUp) {
    // Cancel best bid
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    auto levels = book.top_bids(3);
    EXPECT_DOUBLE_EQ(levels[0].price, 99.0);
    EXPECT_EQ(levels[0].qty, 20u);
    EXPECT_DOUBLE_EQ(levels[1].price, 98.0);
    EXPECT_EQ(levels[1].qty, 30u);
    EXPECT_TRUE(std::isnan(levels[2].price));
    EXPECT_EQ(levels[2].qty, 0u);
}

// ===========================================================================
// top_bids: k boundary cases
// ===========================================================================

TEST(BookDepth, TopBidsKZeroReturnsEmptyVector) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));

    auto levels = book.top_bids(0);
    EXPECT_EQ(levels.size(), 0u);
}

TEST(BookDepth, TopBidsK1ReturnsBestBidOnly) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    auto levels = book.top_bids(1);
    EXPECT_EQ(levels.size(), 1u);
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_EQ(levels[0].qty, 10u);
}

TEST(BookDepth, TopBidsDefaultKIs10) {
    Book book;
    auto levels = book.top_bids();  // no argument
    EXPECT_EQ(levels.size(), 10u);
}

// ===========================================================================
// top_asks: Empty book
// ===========================================================================

TEST(BookDepth, TopAsksEmptyBookReturnsKEntries) {
    Book book;
    auto levels = book.top_asks(10);
    EXPECT_EQ(levels.size(), 10u);
}

TEST(BookDepth, TopAsksEmptyBookAllPricesAreNaN) {
    Book book;
    auto levels = book.top_asks(10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price)) << "Level " << i << " price should be NaN";
    }
}

TEST(BookDepth, TopAsksEmptyBookAllQtysAreZero) {
    Book book;
    auto levels = book.top_asks(10);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(levels[i].qty, 0u) << "Level " << i << " qty should be 0";
    }
}

// ===========================================================================
// top_asks: Partial fill (fewer levels than K)
// ===========================================================================

// Fixture: Book with 3 ask levels at 101.0/10, 102.0/20, 103.0/30.
class ThreeAskLevels : public ::testing::Test {
protected:
    void SetUp() override {
        book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
        book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));
        book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 30));
    }

    Book book;
};

TEST_F(ThreeAskLevels, TopAsksFirst3Populated) {
    auto levels = book.top_asks(10);
    EXPECT_EQ(levels.size(), 10u);

    // First 3 populated (ascending price — best ask first)
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_EQ(levels[0].qty, 10u);
    EXPECT_DOUBLE_EQ(levels[1].price, 102.0);
    EXPECT_EQ(levels[1].qty, 20u);
    EXPECT_DOUBLE_EQ(levels[2].price, 103.0);
    EXPECT_EQ(levels[2].qty, 30u);
}

TEST_F(ThreeAskLevels, TopAsksRemaining7AreNaNZero) {
    auto levels = book.top_asks(10);

    for (int i = 3; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price)) << "Level " << i << " price should be NaN";
        EXPECT_EQ(levels[i].qty, 0u) << "Level " << i << " qty should be 0";
    }
}

// ===========================================================================
// top_asks: Ordering (ascending price, best first)
// ===========================================================================

TEST(BookDepth, TopAsksOrderedAscendingPrice) {
    Book book;
    // Insert in random order
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 104.0, 5));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 15));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(5, Message::Side::Ask, Message::Action::Add, 105.0, 25));

    auto levels = book.top_asks(5);

    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_DOUBLE_EQ(levels[1].price, 102.0);
    EXPECT_DOUBLE_EQ(levels[2].price, 103.0);
    EXPECT_DOUBLE_EQ(levels[3].price, 104.0);
    EXPECT_DOUBLE_EQ(levels[4].price, 105.0);
}

// ===========================================================================
// top_asks: More levels than K
// ===========================================================================

TEST(BookDepth, TopAsks12LevelsReturnsOnly10) {
    Book book;
    for (int i = 0; i < 12; ++i) {
        book.apply(make_msg(i + 1, Message::Side::Ask, Message::Action::Add,
                            101.0 + i * 0.25, 10));
    }

    auto levels = book.top_asks();  // default k=10
    EXPECT_EQ(levels.size(), 10u);

    // Best ask is 101.0, worst of top-10 is 101.0 + 9*0.25 = 103.25
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_DOUBLE_EQ(levels[9].price, 103.25);
}

// ===========================================================================
// top_asks: Aggregation
// ===========================================================================

TEST(BookDepth, TopAsksAggregatesOrdersAtSamePrice) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 25));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 5));

    auto levels = book.top_asks(1);
    EXPECT_EQ(levels.size(), 1u);
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_EQ(levels[0].qty, 40u);
}

// ===========================================================================
// top_asks: After cancel removes best level
// ===========================================================================

TEST_F(ThreeAskLevels, TopAsksCancelBestLevelShiftsUp) {
    // Cancel best ask
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Cancel, 101.0, 10));

    auto levels = book.top_asks(3);
    EXPECT_DOUBLE_EQ(levels[0].price, 102.0);
    EXPECT_EQ(levels[0].qty, 20u);
    EXPECT_DOUBLE_EQ(levels[1].price, 103.0);
    EXPECT_EQ(levels[1].qty, 30u);
    EXPECT_TRUE(std::isnan(levels[2].price));
    EXPECT_EQ(levels[2].qty, 0u);
}

// ===========================================================================
// top_asks: k boundary cases
// ===========================================================================

TEST(BookDepth, TopAsksKZeroReturnsEmptyVector) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    auto levels = book.top_asks(0);
    EXPECT_EQ(levels.size(), 0u);
}

TEST(BookDepth, TopAsksK1ReturnsBestAskOnly) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));

    auto levels = book.top_asks(1);
    EXPECT_EQ(levels.size(), 1u);
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);
    EXPECT_EQ(levels[0].qty, 10u);
}

TEST(BookDepth, TopAsksDefaultKIs10) {
    Book book;
    auto levels = book.top_asks();  // no argument
    EXPECT_EQ(levels.size(), 10u);
}

// ===========================================================================
// best_bid_qty: Empty book
// ===========================================================================

TEST(BookDepth, BestBidQtyEmptyBookReturnsZero) {
    Book book;
    EXPECT_EQ(book.best_bid_qty(), 0u);
}

// ===========================================================================
// best_bid_qty: Single order
// ===========================================================================

TEST(BookDepth, BestBidQtySingleOrder) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 42));
    EXPECT_EQ(book.best_bid_qty(), 42u);
}

// ===========================================================================
// best_bid_qty: Multiple orders at best price (aggregated)
// ===========================================================================

TEST(BookDepth, BestBidQtyAggregatesMultipleOrdersAtBest) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 100.0, 15));

    EXPECT_EQ(book.best_bid_qty(), 45u);
}

// ===========================================================================
// best_bid_qty: Does not include non-best levels
// ===========================================================================

TEST(BookDepth, BestBidQtyOnlyCountsBestLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 50));

    // Should only report qty at 100.0
    EXPECT_EQ(book.best_bid_qty(), 10u);
}

// ===========================================================================
// best_bid_qty: After cancel at best level
// ===========================================================================

TEST(BookDepth, BestBidQtyAfterPartialCancel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 20));
    // Cancel one order at best
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    EXPECT_EQ(book.best_bid_qty(), 20u);
}

TEST(BookDepth, BestBidQtyAfterFullCancelFallsToNextLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 30));
    // Cancel all at best
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    EXPECT_EQ(book.best_bid_qty(), 30u);
}

// ===========================================================================
// best_ask_qty: Empty book
// ===========================================================================

TEST(BookDepth, BestAskQtyEmptyBookReturnsZero) {
    Book book;
    EXPECT_EQ(book.best_ask_qty(), 0u);
}

// ===========================================================================
// best_ask_qty: Single order
// ===========================================================================

TEST(BookDepth, BestAskQtySingleOrder) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 37));
    EXPECT_EQ(book.best_ask_qty(), 37u);
}

// ===========================================================================
// best_ask_qty: Multiple orders at best price (aggregated)
// ===========================================================================

TEST(BookDepth, BestAskQtyAggregatesMultipleOrdersAtBest) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 15));

    EXPECT_EQ(book.best_ask_qty(), 45u);
}

// ===========================================================================
// best_ask_qty: Does not include non-best levels
// ===========================================================================

TEST(BookDepth, BestAskQtyOnlyCountsBestLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 50));

    EXPECT_EQ(book.best_ask_qty(), 10u);
}

// ===========================================================================
// best_ask_qty: After cancel at best level
// ===========================================================================

TEST(BookDepth, BestAskQtyAfterPartialCancel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 20));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Cancel, 101.0, 10));

    EXPECT_EQ(book.best_ask_qty(), 20u);
}

TEST(BookDepth, BestAskQtyAfterFullCancelFallsToNextLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 30));
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Cancel, 101.0, 10));

    EXPECT_EQ(book.best_ask_qty(), 30u);
}

// ===========================================================================
// Integration: SyntheticSource depth snapshot
// ===========================================================================

// Fixture: Book populated with Phase 1 of SyntheticSource (5 bid + 5 ask levels).
class BookDepthPhase1 : public ::testing::Test {
protected:
    void SetUp() override {
        SyntheticSource src;
        Message m;
        for (int i = 0; i < 10 && src.next(m); ++i) {
            book.apply(m);
        }
    }

    Book book;
};

TEST_F(BookDepthPhase1, SyntheticSourceAfterPhase1Has5BidLevels) {
    auto bids = book.top_bids(10);

    // First 5 should be populated
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::isnan(bids[i].price)) << "Bid level " << i << " should be populated";
        EXPECT_GT(bids[i].qty, 0u) << "Bid level " << i << " qty should be > 0";
    }

    // Remaining 5 should be NaN/0
    for (int i = 5; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(bids[i].price)) << "Bid level " << i << " should be NaN";
        EXPECT_EQ(bids[i].qty, 0u) << "Bid level " << i << " qty should be 0";
    }
}

TEST_F(BookDepthPhase1, SyntheticSourceAfterPhase1Has5AskLevels) {
    auto asks = book.top_asks(10);

    // First 5 should be populated
    for (int i = 0; i < 5; ++i) {
        EXPECT_FALSE(std::isnan(asks[i].price)) << "Ask level " << i << " should be populated";
        EXPECT_GT(asks[i].qty, 0u) << "Ask level " << i << " qty should be > 0";
    }

    // Remaining 5 should be NaN/0
    for (int i = 5; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(asks[i].price)) << "Ask level " << i << " should be NaN";
        EXPECT_EQ(asks[i].qty, 0u) << "Ask level " << i << " qty should be 0";
    }
}

TEST_F(BookDepthPhase1, SyntheticSourceBidsDescendingAfterPhase1) {
    auto bids = book.top_bids(5);

    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(bids[i].price, bids[i + 1].price)
            << "Bid level " << i << " should have higher price than level " << i + 1;
    }
}

TEST_F(BookDepthPhase1, SyntheticSourceAsksAscendingAfterPhase1) {
    auto asks = book.top_asks(5);

    for (int i = 0; i < 4; ++i) {
        EXPECT_LT(asks[i].price, asks[i + 1].price)
            << "Ask level " << i << " should have lower price than level " << i + 1;
    }
}

TEST_F(BookDepthPhase1, SyntheticSourceBestBidQtyMatchesTopBids) {
    auto bids = book.top_bids(1);
    EXPECT_EQ(book.best_bid_qty(), bids[0].qty);
}

TEST_F(BookDepthPhase1, SyntheticSourceBestAskQtyMatchesTopAsks) {
    auto asks = book.top_asks(1);
    EXPECT_EQ(book.best_ask_qty(), asks[0].qty);
}

// ===========================================================================
// top_bids/top_asks: After trade removes level
// ===========================================================================

TEST(BookDepth, TopBidsAfterTradeIsNoOp) {
    // Databento spec: Trade does not affect the book.
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    // Trade — should be no-op
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Trade, 100.0, 10));

    auto levels = book.top_bids(2);
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);  // best bid unchanged
    EXPECT_EQ(levels[0].qty, 10u);  // qty unchanged
    EXPECT_DOUBLE_EQ(levels[1].price, 99.0);
    EXPECT_EQ(levels[1].qty, 20u);
}

TEST(BookDepth, TopAsksAfterTradeIsNoOp) {
    // Databento spec: Trade does not affect the book.
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));

    // Trade — should be no-op
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 101.0, 10));

    auto levels = book.top_asks(2);
    EXPECT_DOUBLE_EQ(levels[0].price, 101.0);  // best ask unchanged
    EXPECT_EQ(levels[0].qty, 10u);  // qty unchanged
    EXPECT_DOUBLE_EQ(levels[1].price, 102.0);
    EXPECT_EQ(levels[1].qty, 20u);
}

// ===========================================================================
// top_bids/top_asks: After reset
// ===========================================================================

TEST(BookDepth, TopBidsAfterResetAllNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    book.reset();

    auto levels = book.top_bids(5);
    EXPECT_EQ(levels.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price));
        EXPECT_EQ(levels[i].qty, 0u);
    }
}

TEST(BookDepth, TopAsksAfterResetAllNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    book.reset();

    auto levels = book.top_asks(5);
    EXPECT_EQ(levels.size(), 5u);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(levels[i].price));
        EXPECT_EQ(levels[i].qty, 0u);
    }
}

TEST(BookDepth, BestBidQtyAfterResetIsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.reset();
    EXPECT_EQ(book.best_bid_qty(), 0u);
}

TEST(BookDepth, BestAskQtyAfterResetIsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.reset();
    EXPECT_EQ(book.best_ask_qty(), 0u);
}

// ===========================================================================
// top_bids: Only bids, no asks (and vice versa)
// ===========================================================================

TEST(BookDepth, TopBidsPopulatedWhileAsksEmpty) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));

    auto bids = book.top_bids(1);
    auto asks = book.top_asks(1);

    EXPECT_DOUBLE_EQ(bids[0].price, 100.0);
    EXPECT_EQ(bids[0].qty, 10u);
    EXPECT_TRUE(std::isnan(asks[0].price));
    EXPECT_EQ(asks[0].qty, 0u);
}

TEST(BookDepth, TopAsksPopulatedWhileBidsEmpty) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    auto bids = book.top_bids(1);
    auto asks = book.top_asks(1);

    EXPECT_TRUE(std::isnan(bids[0].price));
    EXPECT_EQ(bids[0].qty, 0u);
    EXPECT_DOUBLE_EQ(asks[0].price, 101.0);
    EXPECT_EQ(asks[0].qty, 10u);
}

// ===========================================================================
// total_bid_depth: Empty book
// ===========================================================================

TEST(BookDepth, TotalBidDepthEmptyBookReturnsZero) {
    Book book;
    EXPECT_EQ(book.total_bid_depth(5), 0u);
}

// ===========================================================================
// total_bid_depth: n <= 0 returns 0
// ===========================================================================

TEST(BookDepth, TotalBidDepthNZeroReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    EXPECT_EQ(book.total_bid_depth(0), 0u);
}

TEST(BookDepth, TotalBidDepthNNegativeReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    EXPECT_EQ(book.total_bid_depth(-1), 0u);
}

// ===========================================================================
// total_bid_depth: Single level
// ===========================================================================

TEST(BookDepth, TotalBidDepthSingleLevelReturnsThatQty) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 42));
    EXPECT_EQ(book.total_bid_depth(1), 42u);
}

TEST(BookDepth, TotalBidDepthSingleLevelAggregatedOrders) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 100.0, 25));
    EXPECT_EQ(book.total_bid_depth(1), 35u);
}

// ===========================================================================
// total_bid_depth: Multiple levels
// ===========================================================================

TEST(BookDepth, TotalBidDepthMultipleLevelsReturnsSumOfTopN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

    // n=2 should sum top 2 levels: 10 + 20 = 30
    EXPECT_EQ(book.total_bid_depth(2), 30u);
}

TEST(BookDepth, TotalBidDepthAllLevels) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

    // n=3 should sum all: 10 + 20 + 30 = 60
    EXPECT_EQ(book.total_bid_depth(3), 60u);
}

// ===========================================================================
// total_bid_depth: Fewer levels than n (sums all available)
// ===========================================================================

TEST(BookDepth, TotalBidDepthFewerLevelsThanNSumsAll) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    // Only 2 levels but asking for 10 — should return 10 + 20 = 30
    EXPECT_EQ(book.total_bid_depth(10), 30u);
}

// ===========================================================================
// total_bid_depth: With make_symmetric_book
// ===========================================================================

TEST(BookDepth, TotalBidDepthSymmetricBookKnownValues) {
    // make_symmetric_book(n=5, base_qty=10):
    // Bid levels: 100.0(10), 99.75(20), 99.50(30), 99.25(40), 99.0(50)
    Book book = make_symmetric_book(5, 10);

    EXPECT_EQ(book.total_bid_depth(1), 10u);
    EXPECT_EQ(book.total_bid_depth(2), 30u);   // 10 + 20
    EXPECT_EQ(book.total_bid_depth(3), 60u);   // 10 + 20 + 30
    EXPECT_EQ(book.total_bid_depth(5), 150u);  // 10 + 20 + 30 + 40 + 50
}

// ===========================================================================
// total_ask_depth: Empty book
// ===========================================================================

TEST(BookDepth, TotalAskDepthEmptyBookReturnsZero) {
    Book book;
    EXPECT_EQ(book.total_ask_depth(5), 0u);
}

// ===========================================================================
// total_ask_depth: n <= 0 returns 0
// ===========================================================================

TEST(BookDepth, TotalAskDepthNZeroReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 50));
    EXPECT_EQ(book.total_ask_depth(0), 0u);
}

TEST(BookDepth, TotalAskDepthNNegativeReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 50));
    EXPECT_EQ(book.total_ask_depth(-1), 0u);
}

// ===========================================================================
// total_ask_depth: Single level
// ===========================================================================

TEST(BookDepth, TotalAskDepthSingleLevelReturnsThatQty) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 37));
    EXPECT_EQ(book.total_ask_depth(1), 37u);
}

// ===========================================================================
// total_ask_depth: Multiple levels
// ===========================================================================

TEST(BookDepth, TotalAskDepthMultipleLevelsReturnsSumOfTopN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 30));

    // n=2 should sum top 2 asks (ascending price): 10 + 20 = 30
    EXPECT_EQ(book.total_ask_depth(2), 30u);
}

// ===========================================================================
// total_ask_depth: Fewer levels than n
// ===========================================================================

TEST(BookDepth, TotalAskDepthFewerLevelsThanNSumsAll) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 15));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 25));

    EXPECT_EQ(book.total_ask_depth(10), 40u);
}

// ===========================================================================
// total_ask_depth: With make_symmetric_book
// ===========================================================================

TEST(BookDepth, TotalAskDepthSymmetricBookKnownValues) {
    // make_symmetric_book(n=5, base_qty=10):
    // Ask levels: 100.25(10), 100.50(20), 100.75(30), 101.0(40), 101.25(50)
    Book book = make_symmetric_book(5, 10);

    EXPECT_EQ(book.total_ask_depth(1), 10u);
    EXPECT_EQ(book.total_ask_depth(2), 30u);   // 10 + 20
    EXPECT_EQ(book.total_ask_depth(3), 60u);   // 10 + 20 + 30
    EXPECT_EQ(book.total_ask_depth(5), 150u);  // 10 + 20 + 30 + 40 + 50
}

// ===========================================================================
// total_bid_depth / total_ask_depth: Integration with SyntheticSource
// ===========================================================================

TEST_F(BookDepthPhase1, TotalBidDepthMatchesSumOfTopBids) {
    auto bids = book.top_bids(5);
    uint32_t expected = 0;
    for (int i = 0; i < 5; ++i) {
        expected += bids[i].qty;
    }
    EXPECT_EQ(book.total_bid_depth(5), expected);
}

TEST_F(BookDepthPhase1, TotalAskDepthMatchesSumOfTopAsks) {
    auto asks = book.top_asks(5);
    uint32_t expected = 0;
    for (int i = 0; i < 5; ++i) {
        expected += asks[i].qty;
    }
    EXPECT_EQ(book.total_ask_depth(5), expected);
}

// ===========================================================================
// weighted_mid: Empty book returns NaN
// ===========================================================================

TEST(BookDepth, WeightedMidEmptyBookReturnsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.weighted_mid()));
}

// ===========================================================================
// weighted_mid: Only bids, no asks returns NaN
// ===========================================================================

TEST(BookDepth, WeightedMidOnlyBidsReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    EXPECT_TRUE(std::isnan(book.weighted_mid()));
}

// ===========================================================================
// weighted_mid: Only asks, no bids returns NaN
// ===========================================================================

TEST(BookDepth, WeightedMidOnlyAsksReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    EXPECT_TRUE(std::isnan(book.weighted_mid()));
}

// ===========================================================================
// weighted_mid: Symmetric BBO equals simple mid_price
// ===========================================================================

TEST(BookDepth, WeightedMidSymmetricBBOEqualsMidPrice) {
    // When bid_qty == ask_qty, weighted_mid should equal mid_price
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 50));

    // weighted_mid = (50*101 + 50*100) / (50+50) = (5050+5000)/100 = 100.5
    // mid_price = (100 + 101) / 2 = 100.5
    EXPECT_DOUBLE_EQ(book.weighted_mid(), book.mid_price());
    EXPECT_DOUBLE_EQ(book.weighted_mid(), 100.5);
}

// ===========================================================================
// weighted_mid: Asymmetric BBO weighted toward larger side
// ===========================================================================

TEST(BookDepth, WeightedMidAsymmetricBBOWeightedTowardLargerBidSide) {
    // bid_qty=90, ask_qty=10, bid=100.0, ask=101.0
    // weighted_mid = (90*101 + 10*100) / (90+10) = (9090+1000)/100 = 100.9
    // Closer to ask because bid has more qty (pulls mid toward ask price)
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 90));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    double expected = (90.0 * 101.0 + 10.0 * 100.0) / (90.0 + 10.0);
    EXPECT_DOUBLE_EQ(book.weighted_mid(), expected);
    // Verify it's above the simple mid (100.5)
    EXPECT_GT(book.weighted_mid(), book.mid_price());
}

TEST(BookDepth, WeightedMidAsymmetricBBOWeightedTowardLargerAskSide) {
    // bid_qty=10, ask_qty=90, bid=100.0, ask=101.0
    // weighted_mid = (10*101 + 90*100) / (10+90) = (1010+9000)/100 = 100.1
    // Closer to bid because ask has more qty
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 90));

    double expected = (10.0 * 101.0 + 90.0 * 100.0) / (10.0 + 90.0);
    EXPECT_DOUBLE_EQ(book.weighted_mid(), expected);
    // Verify it's below the simple mid (100.5)
    EXPECT_LT(book.weighted_mid(), book.mid_price());
}

// ===========================================================================
// weighted_mid: Integration with make_symmetric_book
// ===========================================================================

TEST(BookDepth, WeightedMidSymmetricBookEqualsMidPrice) {
    // make_symmetric_book: bid best=100.0 qty=10, ask best=100.25 qty=10
    // Equal qty at BBO => weighted_mid == mid_price
    Book book = make_symmetric_book(5, 10);

    // best_bid_qty=10, best_ask_qty=10 => equal weight
    EXPECT_DOUBLE_EQ(book.weighted_mid(), book.mid_price());
}

// ===========================================================================
// weighted_mid: Both BBO quantities are 0 returns NaN
// ===========================================================================

TEST(BookDepth, WeightedMidBothQtyZeroReturnsNaN) {
    // This tests the contract: if both best_bid_qty and best_ask_qty are 0,
    // return NaN (division by zero).
    // In practice this shouldn't happen with a normal book, but we test the edge.
    // We can't easily construct a Book with price levels but zero qty through
    // normal operations, so we test that an empty book returns NaN.
    Book book;
    EXPECT_TRUE(std::isnan(book.weighted_mid()));
}

// ===========================================================================
// weighted_mid: Hand-computed value with specific quantities
// ===========================================================================

TEST(BookDepth, WeightedMidHandComputedValue) {
    // bid=99.50 qty=30, ask=100.50 qty=70
    // weighted_mid = (30*100.50 + 70*99.50) / (30+70)
    //             = (3015.0 + 6965.0) / 100 = 9980.0 / 100 = 99.80
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 99.50, 30));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.50, 70));

    double expected = (30.0 * 100.50 + 70.0 * 99.50) / 100.0;
    EXPECT_DOUBLE_EQ(book.weighted_mid(), expected);
}

// ===========================================================================
// vamp: Empty book returns NaN
// ===========================================================================

TEST(BookDepth, VampEmptyBookReturnsNaN) {
    Book book;
    EXPECT_TRUE(std::isnan(book.vamp(5)));
}

// ===========================================================================
// vamp: n = 0 returns NaN
// ===========================================================================

TEST(BookDepth, VampNZeroReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    EXPECT_TRUE(std::isnan(book.vamp(0)));
}

// ===========================================================================
// vamp: n < 0 returns NaN
// ===========================================================================

TEST(BookDepth, VampNNegativeReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    EXPECT_TRUE(std::isnan(book.vamp(-1)));
}

// ===========================================================================
// vamp: Only bids, no asks returns NaN
// ===========================================================================

TEST(BookDepth, VampOnlyBidsReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    EXPECT_TRUE(std::isnan(book.vamp(5)));
}

// ===========================================================================
// vamp: Only asks, no bids returns NaN
// ===========================================================================

TEST(BookDepth, VampOnlyAsksReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    EXPECT_TRUE(std::isnan(book.vamp(5)));
}

// ===========================================================================
// vamp: Single level each side equals weighted_mid
// ===========================================================================

TEST(BookDepth, VampSingleLevelEqualsWeightedMid) {
    // With n=1, vamp uses only BBO — same as weighted_mid
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 30));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 70));

    // vamp(1) = (100*30 + 101*70) / (30+70) = (3000+7070)/100 = 100.70
    // weighted_mid = (30*101 + 70*100) / (30+70) = (3030+7000)/100 = 100.30
    // Note: vamp uses price*qty per level summed across both sides
    // While weighted_mid uses bid_qty*ask_price + ask_qty*bid_price
    // These are DIFFERENT formulas, so vamp(1) != weighted_mid in general

    double vamp_expected = (100.0 * 30 + 101.0 * 70) / (30.0 + 70.0);
    EXPECT_DOUBLE_EQ(book.vamp(1), vamp_expected);
}

// ===========================================================================
// vamp: Multiple levels with hand-computed value
// ===========================================================================

TEST(BookDepth, VampMultipleLevelsHandComputed) {
    // Bid levels: 100.0(10), 99.0(20)
    // Ask levels: 101.0(15), 102.0(25)
    // vamp(2) = (100*10 + 99*20 + 101*15 + 102*25) / (10+20+15+25)
    //         = (1000 + 1980 + 1515 + 2550) / 70
    //         = 7045 / 70 = 100.642857...
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 15));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 25));

    double expected = (100.0 * 10 + 99.0 * 20 + 101.0 * 15 + 102.0 * 25) / 70.0;
    EXPECT_DOUBLE_EQ(book.vamp(2), expected);
}

// ===========================================================================
// vamp: Fewer levels on one side uses all available
// ===========================================================================

TEST(BookDepth, VampFewerLevelsOnOneSideUsesAllAvailable) {
    // Bid: 1 level (100.0, qty=10)
    // Ask: 3 levels (101.0/10, 102.0/20, 103.0/30)
    // vamp(3): bid has only 1 level, use it. Ask has 3, use all 3.
    // = (100*10 + 101*10 + 102*20 + 103*30) / (10+10+20+30)
    // = (1000 + 1010 + 2040 + 3090) / 70
    // = 7140 / 70 = 102.0
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 103.0, 30));

    double expected = (100.0 * 10 + 101.0 * 10 + 102.0 * 20 + 103.0 * 30) / 70.0;
    EXPECT_DOUBLE_EQ(book.vamp(3), expected);
}

// ===========================================================================
// vamp: Symmetric book with known value
// ===========================================================================

TEST(BookDepth, VampSymmetricBookKnownValue) {
    // make_symmetric_book(n=3, base_qty=10):
    // Bids: 100.0(10), 99.75(20), 99.50(30)
    // Asks: 100.25(10), 100.50(20), 100.75(30)
    //
    // vamp(3) = (100.0*10 + 99.75*20 + 99.50*30 + 100.25*10 + 100.50*20 + 100.75*30)
    //           / (10+20+30+10+20+30)
    //         = (1000 + 1995 + 2985 + 1002.5 + 2010 + 3022.5) / 120
    //         = 12015 / 120 = 100.125
    Book book = make_symmetric_book(3, 10);

    double num = 100.0*10 + 99.75*20 + 99.50*30 + 100.25*10 + 100.50*20 + 100.75*30;
    double den = 10.0 + 20 + 30 + 10 + 20 + 30;
    double expected = num / den;
    EXPECT_DOUBLE_EQ(book.vamp(3), expected);
}

// ===========================================================================
// vamp: n greater than available levels on both sides
// ===========================================================================

TEST(BookDepth, VampNGreaterThanLevelsSumsAll) {
    // 2 bid levels, 2 ask levels, but requesting vamp(10)
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 15));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 25));

    // Should use all 2 levels each side (same as vamp(2))
    double expected = (100.0 * 10 + 99.0 * 20 + 101.0 * 15 + 102.0 * 25) / 70.0;
    EXPECT_DOUBLE_EQ(book.vamp(10), expected);
}

// ===========================================================================
// vamp: Symmetric quantities produce mid between bid and ask sides
// ===========================================================================

TEST(BookDepth, VampSymmetricQuantitiesIsMidOfVwapEachSide) {
    // When both sides have equal total qty, vamp = midpoint of bid-VWAP and ask-VWAP
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 10));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    // bid_vwap = (100*10 + 99*10) / 20 = 99.5
    // ask_vwap = (101*10 + 102*10) / 20 = 101.5
    // vamp(2) = (1990 + 2030) / 40 = 4020 / 40 = 100.5
    // Which is (bid_vwap + ask_vwap) / 2 = (99.5 + 101.5) / 2 = 100.5
    EXPECT_DOUBLE_EQ(book.vamp(2), 100.5);
}

// ===========================================================================
// vamp: After book modification, recalculates correctly
// ===========================================================================

TEST(BookDepth, VampAfterCancelRecalculates) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 15));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 25));

    // Cancel the best bid — now bid side has only 99.0(20)
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Cancel, 100.0, 10));

    // vamp(2) with remaining: bid 99.0(20), ask 101.0(15), 102.0(25)
    double expected = (99.0 * 20 + 101.0 * 15 + 102.0 * 25) / (20.0 + 15 + 25);
    EXPECT_DOUBLE_EQ(book.vamp(2), expected);
}

// ===========================================================================
// vamp: Integration with SyntheticSource
// ===========================================================================

TEST_F(BookDepthPhase1, VampWithSyntheticSourceIsFinite) {
    double v = book.vamp(5);
    EXPECT_TRUE(std::isfinite(v)) << "vamp should be finite for a populated book";
    // vamp should be between best bid and best ask
    EXPECT_GE(v, book.best_bid());
    EXPECT_LE(v, book.best_ask());
}

TEST_F(BookDepthPhase1, WeightedMidWithSyntheticSourceIsFinite) {
    double w = book.weighted_mid();
    EXPECT_TRUE(std::isfinite(w)) << "weighted_mid should be finite for a populated book";
    EXPECT_GE(w, book.best_bid());
    EXPECT_LE(w, book.best_ask());
}

// ===========================================================================
// vamp: n=1 produces value between best bid and best ask
// ===========================================================================

TEST(BookDepth, VampN1BetweenBBOPrices) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    double v = book.vamp(1);
    // (100*10 + 102*10) / 20 = 2020/20 = 101.0
    EXPECT_DOUBLE_EQ(v, 101.0);
    EXPECT_GE(v, 100.0);
    EXPECT_LE(v, 102.0);
}

// ===========================================================================
// weighted_mid: After reset returns NaN
// ===========================================================================

TEST(BookDepth, WeightedMidAfterResetReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.reset();
    EXPECT_TRUE(std::isnan(book.weighted_mid()));
}

// ===========================================================================
// vamp: After reset returns NaN
// ===========================================================================

TEST(BookDepth, VampAfterResetReturnsNaN) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.reset();
    EXPECT_TRUE(std::isnan(book.vamp(5)));
}

// ===========================================================================
// total_bid_depth / total_ask_depth: After reset returns 0
// ===========================================================================

TEST(BookDepth, TotalBidDepthAfterResetReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.reset();
    EXPECT_EQ(book.total_bid_depth(5), 0u);
}

TEST(BookDepth, TotalAskDepthAfterResetReturnsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 50));
    book.reset();
    EXPECT_EQ(book.total_ask_depth(5), 0u);
}
