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

TEST(BookDepth, TopBids3LevelsReturns10Entries) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

    auto levels = book.top_bids(10);
    EXPECT_EQ(levels.size(), 10u);
}

TEST(BookDepth, TopBids3LevelsFirst3Populated) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

    auto levels = book.top_bids(10);

    // First 3 should be populated with real data
    EXPECT_DOUBLE_EQ(levels[0].price, 100.0);
    EXPECT_EQ(levels[0].qty, 10u);
    EXPECT_DOUBLE_EQ(levels[1].price, 99.0);
    EXPECT_EQ(levels[1].qty, 20u);
    EXPECT_DOUBLE_EQ(levels[2].price, 98.0);
    EXPECT_EQ(levels[2].qty, 30u);
}

TEST(BookDepth, TopBids3LevelsRemaining7AreNaNZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

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

TEST(BookDepth, TopBidsCancelBestLevelShiftsUp) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 98.0, 30));

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

TEST(BookDepth, TopAsks3LevelsFirst3Populated) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 30));

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

TEST(BookDepth, TopAsks3LevelsRemaining7AreNaNZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 30));

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

TEST(BookDepth, TopAsksCancelBestLevelShiftsUp) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 103.0, 30));

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

TEST(BookDepth, TopBidsAfterTradeRemovesBestLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));

    // Trade fills the entire best bid
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Trade, 100.0, 10));

    auto levels = book.top_bids(2);
    EXPECT_DOUBLE_EQ(levels[0].price, 99.0);
    EXPECT_EQ(levels[0].qty, 20u);
    EXPECT_TRUE(std::isnan(levels[1].price));
    EXPECT_EQ(levels[1].qty, 0u);
}

TEST(BookDepth, TopAsksAfterTradeRemovesBestLevel) {
    Book book;
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 20));

    // Trade fills the entire best ask
    book.apply(make_msg(1, Message::Side::Ask, Message::Action::Trade, 101.0, 10));

    auto levels = book.top_asks(2);
    EXPECT_DOUBLE_EQ(levels[0].price, 102.0);
    EXPECT_EQ(levels[0].qty, 20u);
    EXPECT_TRUE(std::isnan(levels[1].price));
    EXPECT_EQ(levels[1].qty, 0u);
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
