#include <gtest/gtest.h>
#include "lob/book.h"

using namespace lob;

class BookTest : public ::testing::Test {
protected:
    Book book;

    void SetUp() override {
        book.clear();
    }

    MBOMessage make_add(uint64_t order_id, int64_t price, uint32_t qty, Side side) {
        return MBOMessage{
            .timestamp_ns = 0,
            .order_id = order_id,
            .price = price,
            .quantity = qty,
            .side = side,
            .action = Action::Add
        };
    }

    MBOMessage make_cancel(uint64_t order_id, Side side) {
        return MBOMessage{
            .timestamp_ns = 0,
            .order_id = order_id,
            .price = 0,
            .quantity = 0,
            .side = side,
            .action = Action::Cancel
        };
    }

    MBOMessage make_trade(uint64_t order_id, int64_t price, uint32_t qty, Side side) {
        return MBOMessage{
            .timestamp_ns = 0,
            .order_id = order_id,
            .price = price,
            .quantity = qty,
            .side = side,
            .action = Action::Trade
        };
    }
};

TEST_F(BookTest, EmptyBook) {
    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.ask_depth(), 0);
    EXPECT_EQ(book.mid_price(), 0);
    EXPECT_EQ(book.spread(), 0);
    EXPECT_DOUBLE_EQ(book.imbalance(1), 0.0);
}

TEST_F(BookTest, SingleBid) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));

    EXPECT_EQ(book.bid_depth(), 1);
    EXPECT_EQ(book.ask_depth(), 0);

    Level best_bid = book.bid(0);
    EXPECT_EQ(best_bid.price, 100'000'000'000);
    EXPECT_EQ(best_bid.quantity, 10);
}

TEST_F(BookTest, SingleAsk) {
    book.apply(make_add(1, 101'000'000'000, 20, Side::Ask));

    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.ask_depth(), 1);

    Level best_ask = book.ask(0);
    EXPECT_EQ(best_ask.price, 101'000'000'000);
    EXPECT_EQ(best_ask.quantity, 20);
}

TEST_F(BookTest, BidAskSpread) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));
    book.apply(make_add(2, 101'000'000'000, 10, Side::Ask));

    EXPECT_EQ(book.spread(), 1'000'000'000);
    EXPECT_EQ(book.mid_price(), 100'500'000'000);
}

TEST_F(BookTest, MultipleLevels) {
    // Add bids at different prices
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));
    book.apply(make_add(2, 99'000'000'000, 20, Side::Bid));
    book.apply(make_add(3, 98'000'000'000, 30, Side::Bid));

    EXPECT_EQ(book.bid_depth(), 3);

    // Best bid should be highest price
    EXPECT_EQ(book.bid(0).price, 100'000'000'000);
    EXPECT_EQ(book.bid(1).price, 99'000'000'000);
    EXPECT_EQ(book.bid(2).price, 98'000'000'000);
}

TEST_F(BookTest, MultipleOrdersSameLevel) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));
    book.apply(make_add(2, 100'000'000'000, 20, Side::Bid));

    EXPECT_EQ(book.bid_depth(), 1);  // Only one price level
    EXPECT_EQ(book.bid(0).quantity, 30);  // Combined quantity
}

TEST_F(BookTest, CancelOrder) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));
    book.apply(make_add(2, 100'000'000'000, 20, Side::Bid));

    EXPECT_EQ(book.bid(0).quantity, 30);

    book.apply(make_cancel(1, Side::Bid));
    EXPECT_EQ(book.bid(0).quantity, 20);

    book.apply(make_cancel(2, Side::Bid));
    EXPECT_EQ(book.bid_depth(), 0);
}

TEST_F(BookTest, Trade) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));

    // Trade removes quantity
    book.apply(make_trade(1, 100'000'000'000, 5, Side::Bid));
    EXPECT_EQ(book.bid(0).quantity, 5);

    // Full fill removes order
    book.apply(make_trade(1, 100'000'000'000, 5, Side::Bid));
    EXPECT_EQ(book.bid_depth(), 0);
}

TEST_F(BookTest, Clear) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));
    book.apply(make_add(2, 101'000'000'000, 10, Side::Ask));

    book.clear();

    EXPECT_EQ(book.bid_depth(), 0);
    EXPECT_EQ(book.ask_depth(), 0);
}

TEST_F(BookTest, Imbalance) {
    book.apply(make_add(1, 100'000'000'000, 100, Side::Bid));
    book.apply(make_add(2, 101'000'000'000, 100, Side::Ask));

    // Equal size: imbalance = 0
    EXPECT_DOUBLE_EQ(book.imbalance(1), 0.0);

    // Add more bids
    book.apply(make_add(3, 100'000'000'000, 100, Side::Bid));

    // Now 200 bids vs 100 asks: imbalance = (200-100)/(200+100) = 1/3
    EXPECT_NEAR(book.imbalance(1), 1.0/3.0, 0.001);
}

TEST_F(BookTest, DepthOutOfBounds) {
    book.apply(make_add(1, 100'000'000'000, 10, Side::Bid));

    Level empty = book.bid(99);  // Way out of bounds
    EXPECT_EQ(empty.price, 0);
    EXPECT_EQ(empty.quantity, 0);
}
