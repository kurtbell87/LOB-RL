#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <optional>
#include <thread>
#include <chrono>
#include <atomic>
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"
#include "databento/constants.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"

/**
 * @file  TestOrderBookExtended.cpp
 * @brief Additional edge-case & advanced scenario tests
 */

namespace constellation::modules::orderbook {

static databento::MboMsg MakeMbo(std::uint32_t inst, std::uint64_t oid,
                                 std::int64_t px, std::uint32_t sz,
                                 databento::Side s, databento::Action a)
{
  databento::MboMsg m{};
  m.hd.instrument_id = inst;
  m.order_id = oid;
  m.price    = px;
  m.size     = sz;
  m.side     = s;
  m.action   = a;
  return m;
}

TEST_CASE("Multiple orders at the same price", "[orderbook][same-price]") {
  LimitOrderBook lob(1234);

  // 3 orders at price=100000 => total=17
  auto a1 = MakeMbo(1234, 1, 100000, 5,  databento::Side::Bid, databento::Action::Add);
  auto a2 = MakeMbo(1234, 2, 100000, 10, databento::Side::Bid, databento::Action::Add);
  auto a3 = MakeMbo(1234, 3, 100000, 2,  databento::Side::Bid, databento::Action::Add);

  lob.OnMboUpdate(a1);
  lob.OnMboUpdate(a2);
  lob.OnMboUpdate(a3);

  auto best_bid = lob.BestBid();
  REQUIRE(best_bid.has_value());
  CHECK(best_bid->total_quantity == 17);
  CHECK(best_bid->order_count == 3);

  // partial cancel of the second => reduce from 10 to 5 => total=12
  auto cxl2 = MakeMbo(1234, 2, 100000, 5, databento::Side::Bid, databento::Action::Cancel);
  lob.OnMboUpdate(cxl2);
  best_bid = lob.BestBid();
  REQUIRE(best_bid.has_value());
  CHECK(best_bid->total_quantity == 12);
  CHECK(best_bid->order_count == 3);

  // cancel the 3rd => remove 2 => total=10 => order_count=2
  auto cxl3 = MakeMbo(1234, 3, 100000, 2, databento::Side::Bid, databento::Action::Cancel);
  lob.OnMboUpdate(cxl3);
  best_bid = lob.BestBid();
  REQUIRE(best_bid.has_value());
  CHECK(best_bid->total_quantity == 10);
  CHECK(best_bid->order_count == 2);
}

TEST_CASE("Modify can change price & size", "[orderbook][price-mod]") {
  LimitOrderBook lob(999);

  // Add at p=100, size=10
  auto add = MakeMbo(999, 1, 100, 10, databento::Side::Ask, databento::Action::Add);
  lob.OnMboUpdate(add);

  // Modify => p=105, size=15
  auto mod = MakeMbo(999, 1, 105, 15, databento::Side::Ask, databento::Action::Modify);
  lob.OnMboUpdate(mod);

  auto asks = lob.GetAsks();
  REQUIRE(asks.size() == 1);
  CHECK(asks[0].price == 105);
  CHECK(asks[0].total_quantity == 15);

  // no leftover at old price=100
  CHECK_FALSE(lob.GetLevel(constellation::interfaces::orderbook::BookSide::Ask,1).has_value());
}

TEST_CASE("Edge cases zero/negative price, re-add same ID, etc.", "[orderbook][edge-cases]") {
  LimitOrderBook lob(100);

  SECTION("Zero or negative price") {
    auto m1 = MakeMbo(100, 1, 0, 10, databento::Side::Bid, databento::Action::Add);
    lob.OnMboUpdate(m1);
    auto best = lob.BestBid();
    REQUIRE(best.has_value());
    CHECK(best->price == 0);
    CHECK(best->total_quantity == 10);

    // negative price ask
    auto m2 = MakeMbo(100, 2, -100, 5, databento::Side::Ask, databento::Action::Add);
    lob.OnMboUpdate(m2);
    auto best_ask = lob.BestAsk();
    // Could store or ignore negative; just ensure no crash
    if (best_ask) {
      CHECK(best_ask->price == -100);
    }
  }

  SECTION("Re-add same ID") {
    auto a1 = MakeMbo(100, 1000, 300, 10, databento::Side::Bid, databento::Action::Add);
    lob.OnMboUpdate(a1);
    // re-add same ID with new size=5
    auto a2 = MakeMbo(100, 1000, 300, 5, databento::Side::Bid, databento::Action::Add);
    lob.OnMboUpdate(a2);

    SUCCEED("No crash re-adding same ID");
  }
}

TEST_CASE("Actions side=None, action=None => ignored", "[orderbook][none]") {
  LimitOrderBook lob(777);

  // normal add
  {
    auto msg = MakeMbo(777, 1, 10000, 10, databento::Side::Bid, databento::Action::Add);
    lob.OnMboUpdate(msg);
  }
  auto best = lob.BestBid();
  REQUIRE(best.has_value());
  CHECK(best->price == 10000);

  // side=None => ignored
  {
    auto msg = MakeMbo(777, 2, 9999, 5, databento::Side::None, databento::Action::Add);
    lob.OnMboUpdate(msg);
    CHECK(lob.BestBid()->total_quantity == 10);
  }

  // action=None => ignored
  {
    auto msg = MakeMbo(777, 3, 9998, 5, databento::Side::Ask, databento::Action::None);
    lob.OnMboUpdate(msg);
    CHECK(lob.BestBid()->total_quantity == 10);
  }
}

TEST_CASE("Multi-step partial cancel", "[orderbook][partial-cancels]") {
  LimitOrderBook lob(5555);

  // big ask => price=20000, size=50
  auto add = MakeMbo(5555, 99, 20000, 50, databento::Side::Ask, databento::Action::Add);
  lob.OnMboUpdate(add);

  // cancel 10 => leftover=40
  auto cxl = MakeMbo(5555, 99, 20000, 10, databento::Side::Ask, databento::Action::Cancel);
  lob.OnMboUpdate(cxl);
  CHECK(lob.BestAsk()->total_quantity == 40);

  // cancel 10 => leftover=30
  cxl.size = 10;
  lob.OnMboUpdate(cxl);
  CHECK(lob.BestAsk()->total_quantity == 30);

  // cancel 5 => leftover=25
  cxl.size = 5;
  lob.OnMboUpdate(cxl);
  CHECK(lob.BestAsk()->total_quantity == 25);

  // cancel 15 => leftover=10
  cxl.size = 15;
  lob.OnMboUpdate(cxl);
  CHECK(lob.BestAsk()->total_quantity == 10);

  // cancel final 10 => removed
  cxl.size = 10;
  lob.OnMboUpdate(cxl);
  CHECK_FALSE(lob.BestAsk().has_value());
}

TEST_CASE("Concurrent updates across multiple instruments in MarketBook", "[orderbook][concurrency][multi-instrument]") {
  // Create a new MarketBook instance
  auto market_ptr = std::make_shared<MarketBook>();
  auto& market = *market_ptr;
  market.AddInstrument(100, std::make_unique<LimitOrderBook>(100));
  market.AddInstrument(200, std::make_unique<LimitOrderBook>(200));
  market.AddInstrument(300, std::make_unique<LimitOrderBook>(300));

  auto w1 = [&]() {
    for(int i=0; i<50; ++i) {
      auto msg = MakeMbo(100, i, 1000+i, 10+i, (i%2==0? databento::Side::Bid: databento::Side::Ask), databento::Action::Add);
      market.OnMboUpdate(msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  auto w2 = [&]() {
    for(int i=0; i<50; ++i) {
      auto msg = MakeMbo(200, i+100, 2000+i, 5+i, (i%2==0? databento::Side::Ask: databento::Side::Bid), databento::Action::Add);
      market.OnMboUpdate(msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  auto w3 = [&]() {
    for(int i=0; i<50; ++i) {
      auto msg = MakeMbo(300, i+200, 3000+i, 20, (i%2==0? databento::Side::Bid: databento::Side::Ask), databento::Action::Add);
      market.OnMboUpdate(msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  auto reader = [&](int){
    for(int i=0; i<20; ++i){
      auto quotes = market.AllBestQuotes();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  };

  std::thread tw1(w1);
  std::thread tw2(w2);
  std::thread tw3(w3);
  std::thread tr1([&](){reader(1);});
  std::thread tr2([&](){reader(2);});

  tw1.join();
  tw2.join();
  tw3.join();
  tr1.join();
  tr2.join();

  SUCCEED("No concurrency crash in multi-instrument usage.");
}

TEST_CASE("Snapshot iteration concurrency for LOB", "[orderbook][snapshot][concurrency]") {
  LimitOrderBook lob(700);

  auto writer = [&]() {
    for(int i=0; i<50; ++i){
      auto msg = MakeMbo(700, i, 10000+i, 10+i, (i%2==0? databento::Side::Bid: databento::Side::Ask), databento::Action::Add);
      lob.OnMboUpdate(msg);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  auto snapshot_checker = [&]() {
    for(int i=0; i<50; ++i){
      auto bids = lob.GetBids();
      auto asks = lob.GetAsks();
      for(std::size_t j=1; j<bids.size(); ++j){
        REQUIRE(bids[j-1].price >= bids[j].price);
      }
      for(std::size_t j=1; j<asks.size(); ++j){
        REQUIRE(asks[j-1].price <= asks[j].price);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  };

  std::thread w(writer);
  std::thread s(snapshot_checker);

  w.join();
  s.join();
  SUCCEED("Snapshots remained consistent under concurrency.");
}

TEST_CASE("Cancel/Trade unknown order => no effect", "[orderbook][unknown-order]") {
  LimitOrderBook lob(9999);

  // add known
  auto known = MakeMbo(9999, 123, 5000, 10, databento::Side::Ask, databento::Action::Add);
  lob.OnMboUpdate(known);

  // fill or cancel unknown => no effect
  auto unknown_trade = MakeMbo(9999, 999, 5000, 5, databento::Side::Ask, databento::Action::Trade);
  lob.OnMboUpdate(unknown_trade);
  auto unknown_cxl   = MakeMbo(9999, 999, 5000, 0, databento::Side::Ask, databento::Action::Cancel);
  lob.OnMboUpdate(unknown_cxl);

  auto best_ask = lob.BestAsk();
  REQUIRE(best_ask.has_value());
  CHECK(best_ask->price == 5000);
  CHECK(best_ask->total_quantity == 10);
}

TEST_CASE("Stress demonstration (~100k updates)", "[orderbook][stress]") {
  LimitOrderBook lob(42);
  constexpr int count = 100000;
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<count; ++i){
    databento::MboMsg msg{};
    msg.hd.instrument_id=42;
    msg.order_id=i;
    msg.price=10000+(i%500);
    msg.size=1+(i%5);
    msg.side=(i%2==0? databento::Side::Bid: databento::Side::Ask);
    msg.action=(i%10==0? databento::Action::Trade: databento::Action::Add);
    lob.OnMboUpdate(msg);
  }
  auto end=std::chrono::steady_clock::now();
  auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  WARN("Stress test " << count << " updates in " << ms << " ms");
  SUCCEED("No crash");
}

TEST_CASE("Extended TDS: verifying new LOB counters, queries", "[orderbook][tds]") {
  LimitOrderBook lob(999);

  auto mk = [&](uint64_t oid, int64_t px, uint32_t sz, databento::Side s, databento::Action a){
    databento::MboMsg msg{};
    msg.hd.instrument_id=999;
    msg.order_id=oid;
    msg.price=px;
    msg.size=sz;
    msg.side=s;
    msg.action=a;
    return msg;
  };

  CHECK(lob.GetAddCount()==0ULL);
  CHECK(lob.GetCancelCount()==0ULL);
  CHECK(lob.GetModifyCount()==0ULL);
  CHECK(lob.GetTradeCount()==0ULL);
  CHECK(lob.GetClearCount()==0ULL);
  CHECK(lob.GetTotalEventCount()==0ULL);

  // 2 adds
  lob.OnMboUpdate(mk(1,1000,10, databento::Side::Bid, databento::Action::Add));
  lob.OnMboUpdate(mk(2,1000,5,  databento::Side::Bid, databento::Action::Add));
  CHECK(lob.GetAddCount()==2ULL);
  CHECK(lob.VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Bid,1000)==15ULL);
  CHECK(lob.NumOrdersAtPrice(constellation::interfaces::orderbook::BookSide::Bid,1000)==2U);

  // 1 modify => size=7
  lob.OnMboUpdate(mk(2,1000,7, databento::Side::Bid, databento::Action::Modify));
  CHECK(lob.GetModifyCount()==1ULL);
  CHECK(lob.VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Bid,1000)==17ULL);

  // partial cancel => order_id=1 => cancel 5 => leftover=5
  lob.OnMboUpdate(mk(1,1000,5, databento::Side::Bid, databento::Action::Cancel));
  CHECK(lob.GetCancelCount()==1ULL);
  CHECK(lob.VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Bid,1000)==12ULL);

  // cancel => order_id=2 => remove 7 => leftover=5
  lob.OnMboUpdate(mk(2,1000,7, databento::Side::Bid, databento::Action::Cancel));
  CHECK(lob.GetCancelCount()==2ULL);
  CHECK(lob.VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Bid,1000)==5ULL);

  // clear => remove all
  lob.OnMboUpdate(mk(0,0,0, databento::Side::None, databento::Action::Clear));
  CHECK(lob.GetClearCount()==1ULL);
  CHECK_FALSE(lob.BestBid().has_value());

  // test GetLevel
  lob.OnMboUpdate(mk(100,2000,10, databento::Side::Ask, databento::Action::Add));
  lob.OnMboUpdate(mk(101,1990,5,  databento::Side::Ask, databento::Action::Add));
  // best ask=1990 => 5, next=2000 => 10
  auto lvl0 = lob.GetLevel(constellation::interfaces::orderbook::BookSide::Ask,0);
  auto lvl1 = lob.GetLevel(constellation::interfaces::orderbook::BookSide::Ask,1);
  auto lvl2 = lob.GetLevel(constellation::interfaces::orderbook::BookSide::Ask,2);
  REQUIRE(lvl0);
  CHECK(lvl0->price==1990);
  CHECK(lvl0->total_quantity==5);
  REQUIRE(lvl1);
  CHECK(lvl1->price==2000);
  CHECK(lvl1->total_quantity==10);
  CHECK_FALSE(lvl2);
}

TEST_CASE("Extended TDS: verifying MarketBook global counters", "[orderbook][tds]") {
  // Create a new MarketBook instance
  auto market_ptr = std::make_shared<MarketBook>();
  auto& market = *market_ptr;
  market.ResetGlobalCounters();
  auto lob1 = std::make_unique<LimitOrderBook>(1111);
  auto lob2 = std::make_unique<LimitOrderBook>(2222);
  market.AddInstrument(1111, std::move(lob1));
  market.AddInstrument(2222, std::move(lob2));

  auto mk = [&](std::uint32_t inst, std::uint64_t oid, std::int64_t px, std::uint32_t sz,
                databento::Side s, databento::Action a){
    databento::MboMsg m{};
    m.hd.instrument_id=inst;
    m.order_id=oid;
    m.price=px;
    m.size=sz;
    m.side=s;
    m.action=a;
    return m;
  };

  CHECK(market.GetGlobalAddCount()==0ULL);
  CHECK(market.GetGlobalCancelCount()==0ULL);
  CHECK(market.GetGlobalModifyCount()==0ULL);
  CHECK(market.GetGlobalTradeCount()==0ULL);
  CHECK(market.GetGlobalClearCount()==0ULL);
  CHECK(market.GetGlobalTotalEventCount()==0ULL);

  // instrument=1111 => 1 add
  market.OnMboUpdate(mk(1111,101,5000,10, databento::Side::Bid, databento::Action::Add));
  CHECK(market.GetGlobalAddCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==1ULL);

  // instrument=2222 => 1 add + 1 modify
  market.OnMboUpdate(mk(2222,201,6000,8, databento::Side::Ask, databento::Action::Add));
  market.OnMboUpdate(mk(2222,201,6000,12, databento::Side::Ask, databento::Action::Modify));
  CHECK(market.GetGlobalAddCount()==2ULL);
  CHECK(market.GetGlobalModifyCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==3ULL);

  // instrument=1111 => trade
  market.OnMboUpdate(mk(1111,101,5000,5, databento::Side::Bid, databento::Action::Trade));
  CHECK(market.GetGlobalTradeCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==4ULL);

  // instrument=2222 => cancel
  market.OnMboUpdate(mk(2222,201,6000,12, databento::Side::Ask, databento::Action::Cancel));
  CHECK(market.GetGlobalCancelCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==5ULL);

  // instrument=1111 => clear
  market.OnMboUpdate(mk(1111,0,0,0, databento::Side::None, databento::Action::Clear));
  CHECK(market.GetGlobalClearCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==6ULL);

  CHECK(market.GetGlobalAddCount()==2ULL);
  CHECK(market.GetGlobalModifyCount()==1ULL);
  CHECK(market.GetGlobalTradeCount()==1ULL);
  CHECK(market.GetGlobalCancelCount()==1ULL);
  CHECK(market.GetGlobalClearCount()==1ULL);
  CHECK(market.GetGlobalTotalEventCount()==6ULL);
}

TEST_CASE("IsTob Add clears bid side and creates synthetic level", "[orderbook][tob]") {
  LimitOrderBook lob(800);

  // Add regular orders on bid side
  lob.OnMboUpdate(MakeMbo(800, 1, 5000, 10, databento::Side::Bid, databento::Action::Add));
  lob.OnMboUpdate(MakeMbo(800, 2, 4900, 5,  databento::Side::Bid, databento::Action::Add));
  REQUIRE(lob.BestBid().has_value());
  CHECK(lob.BestBid()->price == 5000);
  CHECK(lob.GetBids().size() == 2);

  // TOB add: clears entire bid side, replaces with one synthetic level
  auto tob_msg = MakeMbo(800, 99, 5100, 20, databento::Side::Bid, databento::Action::Add);
  tob_msg.flags.SetTob();
  lob.OnMboUpdate(tob_msg);

  auto bids = lob.GetBids();
  REQUIRE(bids.size() == 1);
  CHECK(bids[0].price == 5100);
  CHECK(bids[0].total_quantity == 20);
  CHECK(bids[0].order_count == 0);  // TOB orders have count=0
}

TEST_CASE("IsTob Add with kUndefPrice clears side completely", "[orderbook][tob]") {
  LimitOrderBook lob(801);

  // Add regular ask orders
  lob.OnMboUpdate(MakeMbo(801, 1, 6000, 8, databento::Side::Ask, databento::Action::Add));
  REQUIRE(lob.BestAsk().has_value());

  // TOB add with kUndefPrice: clear ask side entirely, no replacement
  auto tob_msg = MakeMbo(801, 99, databento::kUndefPrice, 0, databento::Side::Ask, databento::Action::Add);
  tob_msg.flags.SetTob();
  lob.OnMboUpdate(tob_msg);

  CHECK_FALSE(lob.BestAsk().has_value());
  CHECK(lob.GetAsks().empty());
}

TEST_CASE("IsTob order not tracked in orders map", "[orderbook][tob]") {
  LimitOrderBook lob(802);

  // TOB add on bid side
  auto tob_msg = MakeMbo(802, 50, 7000, 15, databento::Side::Bid, databento::Action::Add);
  tob_msg.flags.SetTob();
  lob.OnMboUpdate(tob_msg);

  auto best = lob.BestBid();
  REQUIRE(best.has_value());
  CHECK(best->total_quantity == 15);

  // Cancel that order_id => no-op since TOB orders are not tracked
  lob.OnMboUpdate(MakeMbo(802, 50, 7000, 15, databento::Side::Bid, databento::Action::Cancel));
  best = lob.BestBid();
  REQUIRE(best.has_value());
  CHECK(best->total_quantity == 15);  // still there
}

TEST_CASE("Partial cancel reduces order size", "[orderbook][partial-cancel]") {
  LimitOrderBook lob(803);

  lob.OnMboUpdate(MakeMbo(803, 1, 1000, 10, databento::Side::Ask, databento::Action::Add));
  CHECK(lob.BestAsk()->total_quantity == 10);

  // Cancel 3 => leftover 7
  lob.OnMboUpdate(MakeMbo(803, 1, 1000, 3, databento::Side::Ask, databento::Action::Cancel));
  REQUIRE(lob.BestAsk().has_value());
  CHECK(lob.BestAsk()->total_quantity == 7);
  CHECK(lob.BestAsk()->order_count == 1);

  // Cancel remaining 7 => empty
  lob.OnMboUpdate(MakeMbo(803, 1, 1000, 7, databento::Side::Ask, databento::Action::Cancel));
  CHECK_FALSE(lob.BestAsk().has_value());
}

TEST_CASE("Modify of unknown order treated as Add", "[orderbook][modify-as-add]") {
  LimitOrderBook lob(804);

  // Modify a non-existent order => should be treated as Add
  lob.OnMboUpdate(MakeMbo(804, 999, 2000, 12, databento::Side::Bid, databento::Action::Modify));
  auto best = lob.BestBid();
  REQUIRE(best.has_value());
  CHECK(best->price == 2000);
  CHECK(best->total_quantity == 12);
  CHECK(best->order_count == 1);
}

TEST_CASE("Trade action is a no-op for book state", "[orderbook][trade-noop]") {
  LimitOrderBook lob(805);

  lob.OnMboUpdate(MakeMbo(805, 1, 3000, 10, databento::Side::Ask, databento::Action::Add));
  CHECK(lob.BestAsk()->total_quantity == 10);

  // Trade does NOT reduce order size (it's a no-op)
  lob.OnMboUpdate(MakeMbo(805, 1, 3000, 5, databento::Side::Ask, databento::Action::Trade));
  CHECK(lob.BestAsk()->total_quantity == 10);  // unchanged

  // But trade_count is still incremented
  CHECK(lob.GetTradeCount() == 1ULL);
}

} // end namespace constellation::modules::orderbook
