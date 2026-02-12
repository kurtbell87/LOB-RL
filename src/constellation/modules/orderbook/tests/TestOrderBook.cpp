#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <vector>
#include <optional>
#include <thread>
#include <future>
#include <algorithm>

#include "databento/record.hpp"
#include "databento/enums.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"
#include "orderbook/OrderBookFactory.hpp"
#include "interfaces/logging/NullLogger.hpp"

/**
 * @file   TestOrderBook.cpp
 * @brief  Basic tests for LimitOrderBook & MarketBook
 *
 * These tests remain internal to the orderbook module. No cross-module references.
 */

namespace constellation::modules::orderbook {

TEST_CASE("LimitOrderBook basic actions: Add, Cancel, Modify, Fill", "[orderbook]") {
    LimitOrderBook lob(1001);

    auto make_mbo = [&](uint64_t order_id, int64_t price, uint32_t size,
                        databento::Side side, databento::Action action)
    {
      databento::MboMsg mbo{};
      mbo.hd.instrument_id = 1001;
      mbo.order_id = order_id;
      mbo.price    = price;
      mbo.size     = size;
      mbo.side     = side;
      mbo.action   = action;
      return mbo;
    };

    SECTION("Add orders, check best quotes") {
        lob.OnMboUpdate(make_mbo(111, 399900, 10, databento::Side::Bid, databento::Action::Add));
        lob.OnMboUpdate(make_mbo(222, 400100, 8,  databento::Side::Ask, databento::Action::Add));

        auto best_bid = lob.BestBid();
        REQUIRE(best_bid.has_value());
        CHECK(best_bid->price == 399900);
        CHECK(best_bid->total_quantity == 10);

        auto best_ask = lob.BestAsk();
        REQUIRE(best_ask.has_value());
        CHECK(best_ask->price == 400100);
        CHECK(best_ask->total_quantity == 8);

        // add a better bid => check
        lob.OnMboUpdate(make_mbo(333, 400000, 5, databento::Side::Bid, databento::Action::Add));
        auto new_best_bid = lob.BestBid();
        REQUIRE(new_best_bid.has_value());
        CHECK(new_best_bid->price == 400000);
        CHECK(new_best_bid->total_quantity == 5);
    }

    SECTION("Cancel existing order => removed from LOB") {
        lob.OnMboUpdate(make_mbo(111, 400000, 10, databento::Side::Bid, databento::Action::Add));
        auto best_bid = lob.BestBid();
        REQUIRE(best_bid.has_value());
        CHECK(best_bid->price == 400000);

        // Cancel
        lob.OnMboUpdate(make_mbo(111, 400000, 0, databento::Side::Bid, databento::Action::Cancel));
        CHECK_FALSE(lob.BestBid().has_value());
    }

    SECTION("Partial fill => reduce quantity") {
        lob.OnMboUpdate(make_mbo(777, 410000, 10, databento::Side::Ask, databento::Action::Add));
        // fill 3
        auto trade_mbo = make_mbo(777, 410000, 3, databento::Side::Ask, databento::Action::Trade);
        lob.OnMboUpdate(trade_mbo);
        auto ask = lob.BestAsk();
        REQUIRE(ask.has_value());
        CHECK(ask->total_quantity == 7);

        // fill 4 => leftover=3
        trade_mbo.size = 4;
        lob.OnMboUpdate(trade_mbo);
        ask = lob.BestAsk();
        REQUIRE(ask.has_value());
        CHECK(ask->total_quantity == 3);

        // fill final 3 => gone
        trade_mbo.size = 3;
        lob.OnMboUpdate(trade_mbo);
        CHECK_FALSE(lob.BestAsk().has_value());
    }

    SECTION("Modify => re-insert at new price/size") {
        lob.OnMboUpdate(make_mbo(888, 405000, 10, databento::Side::Bid, databento::Action::Add));
        auto best_bid = lob.BestBid();
        CHECK(best_bid->total_quantity == 10);

        // modify => new size=15
        lob.OnMboUpdate(make_mbo(888, 405000, 15, databento::Side::Bid, databento::Action::Modify));
        best_bid = lob.BestBid();
        CHECK(best_bid->total_quantity == 15);

        // modify => new size=5
        lob.OnMboUpdate(make_mbo(888, 405000, 5, databento::Side::Bid, databento::Action::Modify));
        best_bid = lob.BestBid();
        CHECK(best_bid->total_quantity == 5);
    }

    SECTION("Ignore MBO for different instrument") {
        lob.OnMboUpdate(make_mbo(999, 500000, 20, databento::Side::Ask, databento::Action::Add));
        databento::MboMsg mismatch = make_mbo(1234, 501000, 5, databento::Side::Ask, databento::Action::Add);
        mismatch.hd.instrument_id = 9999;
        lob.OnMboUpdate(mismatch);

        auto best_ask = lob.BestAsk();
        REQUIRE(best_ask.has_value());
        CHECK(best_ask->price == 500000);
        CHECK(best_ask->total_quantity == 20);
    }

    SECTION("Clear action => remove all") {
        lob.OnMboUpdate(make_mbo(100, 400000, 10, databento::Side::Bid, databento::Action::Add));
        lob.OnMboUpdate(make_mbo(300, 400100, 7,  databento::Side::Ask, databento::Action::Add));
        REQUIRE(lob.BestBid().has_value());
        REQUIRE(lob.BestAsk().has_value());

        // Clear
        lob.OnMboUpdate(make_mbo(0, 0, 0, databento::Side::None, databento::Action::Clear));
        CHECK_FALSE(lob.BestBid().has_value());
        CHECK_FALSE(lob.BestAsk().has_value());
    }
}

TEST_CASE("LimitOrderBook retrieving full ladders", "[orderbook]") {
    LimitOrderBook lob(5000);

    auto make_mbo = [&](uint64_t oid, int64_t px, uint32_t sz, databento::Side s, databento::Action a){
      databento::MboMsg msg{};
      msg.hd.instrument_id = 5000;
      msg.order_id = oid;
      msg.price    = px;
      msg.size     = sz;
      msg.side     = s;
      msg.action   = a;
      return msg;
    };

    // multiple bids
    lob.OnMboUpdate(make_mbo(1, 100, 5, databento::Side::Bid, databento::Action::Add));
    lob.OnMboUpdate(make_mbo(2, 105, 3, databento::Side::Bid, databento::Action::Add));
    lob.OnMboUpdate(make_mbo(3, 103, 10, databento::Side::Bid, databento::Action::Add));

    // multiple asks
    lob.OnMboUpdate(make_mbo(10, 110, 2, databento::Side::Ask, databento::Action::Add));
    lob.OnMboUpdate(make_mbo(11, 108, 1, databento::Side::Ask, databento::Action::Add));
    lob.OnMboUpdate(make_mbo(12, 115, 5, databento::Side::Ask, databento::Action::Add));

    auto bids = lob.GetBids();
    REQUIRE(bids.size() == 3);
    CHECK(bids[0].price == 105);
    CHECK(bids[1].price == 103);
    CHECK(bids[2].price == 100);

    auto asks = lob.GetAsks();
    REQUIRE(asks.size() == 3);
    CHECK(asks[0].price == 108);
    CHECK(asks[1].price == 110);
    CHECK(asks[2].price == 115);
}

TEST_CASE("MarketBook multi-instrument", "[orderbook]") {
    // Get the singleton instance of MarketBook and reset its state for clean testing
    auto market_ptr = std::make_shared<MarketBook>();
    auto& market = *market_ptr;
    market.ResetGlobalCounters();

    // create LOBs for two instruments
    auto lob1 = std::make_unique<LimitOrderBook>(111);
    auto lob2 = std::make_unique<LimitOrderBook>(222);
    market.AddInstrument(111, std::move(lob1));
    market.AddInstrument(222, std::move(lob2));

    auto make_mbo_111 = [&](uint64_t oid, int64_t px, uint32_t sz,
                            databento::Side s, databento::Action a){
      databento::MboMsg msg{};
      msg.hd.instrument_id = 111;
      msg.order_id = oid;
      msg.price    = px;
      msg.size     = sz;
      msg.side     = s;
      msg.action   = a;
      return msg;
    };
    auto make_mbo_222 = [&](uint64_t oid, int64_t px, uint32_t sz,
                            databento::Side s, databento::Action a){
      databento::MboMsg msg{};
      msg.hd.instrument_id = 222;
      msg.order_id = oid;
      msg.price    = px;
      msg.size     = sz;
      msg.side     = s;
      msg.action   = a;
      return msg;
    };

    // Add a bid => instrument=111
    market.OnMboUpdate(make_mbo_111(101, 1000, 5, databento::Side::Bid, databento::Action::Add));
    // Add an ask => instrument=222
    market.OnMboUpdate(make_mbo_222(202, 2000, 8, databento::Side::Ask, databento::Action::Add));

    {
      auto lob_ptr = market.GetBook(111);
      REQUIRE(lob_ptr != nullptr);
      auto best_bid = lob_ptr->BestBid();
      REQUIRE(best_bid.has_value());
      CHECK(best_bid->price == 1000);
      CHECK(best_bid->total_quantity == 5);
    }
    {
      auto lob_ptr = market.GetBook(222);
      REQUIRE(lob_ptr != nullptr);
      auto best_ask = lob_ptr->BestAsk();
      REQUIRE(best_ask.has_value());
      CHECK(best_ask->price == 2000);
      CHECK(best_ask->total_quantity == 8);
    }

    auto all_quotes = market.AllBestQuotes();
    REQUIRE(all_quotes.size() == 2);
    std::sort(all_quotes.begin(), all_quotes.end(),
              [](auto& a, auto& b){return std::get<0>(a) < std::get<0>(b);});

    // for 111 => bestBid=1000
    auto& [inst1, bid1, ask1] = all_quotes[0];
    CHECK(inst1 == 111);
    REQUIRE(bid1.has_value());
    CHECK(bid1->price == 1000);
    CHECK_FALSE(ask1.has_value());

    // for 222 => bestAsk=2000
    auto& [inst2, bid2, ask2] = all_quotes[1];
    CHECK(inst2 == 222);
    REQUIRE(ask2.has_value());
    CHECK(ask2->price == 2000);
    CHECK_FALSE(bid2.has_value());
}

TEST_CASE("LimitOrderBook concurrency single-writer, multi-reader", "[orderbook][concurrency]") {
    LimitOrderBook lob(123);

    auto writer = [&]() {
      for (int i=0; i<100; ++i) {
        databento::MboMsg msg{};
        msg.hd.instrument_id = 123;
        msg.order_id = 100 + i;
        msg.price = 100000 + i;
        msg.size  = 10 + (i % 5);
        msg.side  = (i % 2 == 0) ? databento::Side::Bid : databento::Side::Ask;
        msg.action= databento::Action::Add;
        lob.OnMboUpdate(msg);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (i % 10 == 9) {
          msg.action = (i % 20 == 9) ? databento::Action::Cancel : databento::Action::Trade;
          msg.size   = (msg.action == databento::Action::Trade ? 5 : 0);
          lob.OnMboUpdate(msg);
        }
      }
    };
    auto reader_func = [&](int) {
      for (int i=0; i<50; ++i) {
        auto bid = lob.BestBid();
        auto ask = lob.BestAsk();
        (void)bid; (void)ask;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    };

    std::thread w(writer);
    std::thread r1([&](){reader_func(1);});
    std::thread r2([&](){reader_func(2);});

    w.join();
    r1.join();
    r2.join();

    SUCCEED("No crash under concurrency test.");
}

TEST_CASE("MarketBook Factory Functions", "[orderbook]") {
  // Create instances using direct construction
  auto instance1 = std::make_shared<MarketBook>();
  auto instance2 = std::make_shared<MarketBook>();
  
  // Verify that separate instances are created
  REQUIRE(instance1.get() != instance2.get());
  
  // Test with an explicit logger
  auto logger = std::make_shared<constellation::interfaces::logging::NullLogger>();
  auto instance3 = std::make_shared<MarketBook>(logger);
  
  // Verify a third instance is created
  REQUIRE(instance1.get() != instance3.get());
  
  // Test factory functions
  auto market_view = CreateMarketBook(logger);
  auto market_book = CreateIMarketBook(logger);
  
  // Verify factory functions create new instances
  REQUIRE(instance1.get() != std::static_pointer_cast<MarketBook>(market_view).get());
  REQUIRE(instance1.get() != std::static_pointer_cast<MarketBook>(market_book).get());
  REQUIRE(std::static_pointer_cast<MarketBook>(market_view).get() != 
          std::static_pointer_cast<MarketBook>(market_book).get());
}

TEST_CASE("MarketBook Thread Safety", "[orderbook][concurrency]") {
  // Test concurrent access to a shared MarketBook instance from multiple threads
  constexpr int NUM_THREADS = 10;
  std::vector<std::thread> threads;
  
  // Create a shared MarketBook instance to test thread safety of its operations
  auto sharedMarketBook = std::make_shared<MarketBook>();
  
  // Create multiple threads that all access the same market book instance
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([sharedMarketBook, i]() {
      // Each thread performs operations on the shared instance
      auto msg = databento::MboMsg{};
      msg.hd.instrument_id = static_cast<uint32_t>(i);
      msg.action = databento::Action::Add;
      msg.order_id = i + 1000;
      msg.side = databento::Side::Bid;
      msg.price = 100000000 + i;
      msg.size = 10;
      
      // Update the shared market book
      sharedMarketBook->OnMboUpdate(msg);
      
      // Add some work to increase potential for race conditions
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      
      // Try reading data
      auto bestBid = sharedMarketBook->GetBestBid(msg.hd.instrument_id);
      // We should be able to see our own update
      REQUIRE(bestBid.has_value());
    });
  }
  
  // Wait for all threads to complete
  for (auto& t : threads) {
    t.join();
  }
  
  // Verify we have the expected number of instruments
  auto instrumentIds = sharedMarketBook->GetInstrumentIds();
  REQUIRE(instrumentIds.size() == NUM_THREADS);
}

} // end namespace constellation::modules::orderbook
