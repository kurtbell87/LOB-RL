#include <catch2/catch_test_macros.hpp>
#include <memory>
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"
#include "orderbook/LimitOrderBookSnapshot.hpp"
#include "orderbook/MarketBookSnapshot.hpp"
#include "databento/record.hpp"

/**
 * @file   TestOrderBookSnapshot.cpp
 * @brief  Verifies LOB/MarketBook snapshot/memento features.
 */

namespace constellation::modules::orderbook {

TEST_CASE("LimitOrderBook Snapshot/Restore", "[orderbook][snapshot]") {
  LimitOrderBook lob(999);

  databento::MboMsg m1{}, m2{}, m3{};
  m1.hd.instrument_id=999; m1.order_id=1001; m1.price=12345; m1.size=10; m1.side=databento::Side::Bid; m1.action=databento::Action::Add;
  m2.hd.instrument_id=999; m2.order_id=1002; m2.price=12350; m2.size=5;  m2.side=databento::Side::Bid; m2.action=databento::Action::Add;
  lob.OnMboUpdate(m1);
  lob.OnMboUpdate(m2);

  auto snap = lob.CreateSnapshot();
  REQUIRE(snap);

  // do changes
  m3.hd.instrument_id=999; m3.order_id=2001; m3.price=12400; m3.size=7; m3.side=databento::Side::Ask; m3.action=databento::Action::Add;
  lob.OnMboUpdate(m3);

  // partial fill order_id=1001 => fill=5
  databento::MboMsg fill{};
  fill.hd.instrument_id=999; fill.order_id=1001; fill.price=12345; fill.size=5; fill.side=databento::Side::Bid; fill.action=databento::Action::Trade;
  lob.OnMboUpdate(fill);

  // now restore
  lob.RestoreSnapshot(*snap);

  CHECK(lob.GetAddCount()==2ULL);
  CHECK(lob.GetTradeCount()==0ULL);
  CHECK_FALSE(lob.BestAsk().has_value());
  auto b = lob.BestBid();
  REQUIRE(b);
  CHECK(b->price==12350);
  CHECK(b->total_quantity==5);
}

TEST_CASE("MarketBook Snapshot/Restore multi-instrument", "[orderbook][snapshot][market]") {
  // Create a new MarketBook instance
  auto market_ptr = std::make_shared<MarketBook>();
  auto& market = *market_ptr;
  market.ResetGlobalCounters();
  // instruments
  market.AddInstrument(111, std::make_unique<LimitOrderBook>(111));
  market.AddInstrument(222, std::make_unique<LimitOrderBook>(222));

  // MBO
  {
    databento::MboMsg m{};
    m.hd.instrument_id=111;
    m.order_id=101;
    m.price=5000;
    m.size=10;
    m.side=databento::Side::Bid;
    m.action=databento::Action::Add;
    market.OnMboUpdate(m);
  }
  {
    databento::MboMsg m{};
    m.hd.instrument_id=222;
    m.order_id=201;
    m.price=7000;
    m.size=8;
    m.side=databento::Side::Ask;
    m.action=databento::Action::Add;
    market.OnMboUpdate(m);
  }

  CHECK(market.GetGlobalAddCount()==2ULL);

  auto snap = market.CreateSnapshot();
  REQUIRE(snap);

  // more changes
  {
    databento::MboMsg m{};
    m.hd.instrument_id=111;
    m.order_id=102;
    m.price=4900;
    m.size=5;
    m.side=databento::Side::Bid;
    m.action=databento::Action::Add;
    market.OnMboUpdate(m);
  }
  CHECK(market.GetGlobalAddCount()==3ULL);

  market.RestoreSnapshot(*snap);

  // revert => 2
  CHECK(market.GetGlobalAddCount()==2ULL);

  auto lob111 = market.GetBook(111);
  auto lob222 = market.GetBook(222);
  REQUIRE(lob111);
  REQUIRE(lob222);

  CHECK(lob111->GetAddCount()==1ULL);
  CHECK(lob222->GetAddCount()==1ULL);

  auto best_bid_111 = lob111->BestBid();
  REQUIRE(best_bid_111);
  CHECK(best_bid_111->price==5000);
  CHECK(best_bid_111->total_quantity==10);
}

} // end namespace constellation::modules::orderbook
