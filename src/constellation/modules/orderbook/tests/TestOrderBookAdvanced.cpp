#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdexcept>
#include <random>
#include <iostream>

#include "orderbook/MarketBook.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"

/**
 * @file   TestOrderBookAdvanced.cpp
 * @brief  Additional concurrency/fault/performance tests for orderbook.
 */

namespace constellation::modules::orderbook {

class FaultyMarketBook {
public:
  std::atomic<bool> enable_faults{false};
  std::atomic<double> fault_probability{0.1};

  FaultyMarketBook() {
    // Create a new market book instance
    marketBook_ = std::make_shared<MarketBook>();
  }

  // Methods that can throw faults
  void OnMboUpdate(const databento::MboMsg& mbo) {
    MaybeThrow();
    marketBook_->OnMboUpdate(mbo);
  }

  void BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages) {
    MaybeThrow();
    marketBook_->BatchOnMboUpdate(messages);
  }

  // Delegated methods
  void AddInstrument(std::uint32_t instrument_id, std::unique_ptr<LimitOrderBook> lob) {
    marketBook_->AddInstrument(instrument_id, std::move(lob));
  }

  std::optional<interfaces::orderbook::PriceLevel> GetBestBid(std::uint32_t instrument_id) const {
    return marketBook_->GetBestBid(instrument_id);
  }
  
  std::optional<interfaces::orderbook::PriceLevel> GetBestAsk(std::uint32_t instrument_id) const {
    return marketBook_->GetBestAsk(instrument_id);
  }

  std::uint64_t GetGlobalAddCount() const noexcept {
    return marketBook_->GetGlobalAddCount();
  }

private:
  void MaybeThrow() {
    if (!enable_faults) return;
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<double> dist(0.0,1.0);
    if (dist(rng) < fault_probability.load()) {
      throw std::runtime_error("FaultyMarketBook injection error");
    }
  }
  
  std::shared_ptr<MarketBook> marketBook_;
};

static databento::MboMsg MakeMboMsg(std::uint32_t inst_id, std::uint64_t oid,
                                    std::int64_t px, std::uint32_t sz,
                                    databento::Side side, databento::Action action)
{
  databento::MboMsg msg{};
  msg.hd.instrument_id = inst_id;
  msg.order_id = oid;
  msg.price    = px;
  msg.size     = sz;
  msg.side     = side;
  msg.action   = action;
  return msg;
}

TEST_CASE("OrderBook concurrency stress test (multi-instrument)", "[orderbook][advanced][concurrency]") {
  // Create a new MarketBook instance
  auto market_ptr = std::make_shared<MarketBook>();
  auto& market = *market_ptr;
  market.AddInstrument(100, std::make_unique<LimitOrderBook>(100));
  market.AddInstrument(200, std::make_unique<LimitOrderBook>(200));
  market.AddInstrument(300, std::make_unique<LimitOrderBook>(300));

  std::atomic<bool> stop_flag{false};

  auto writer1 = [&]() {
    std::uint64_t counter = 0;
    while (!stop_flag) {
      auto order_id = counter++;
      auto msg = MakeMboMsg(100, order_id, 100000 + (order_id % 50), 5, databento::Side::Bid, databento::Action::Add);
      market.OnMboUpdate(msg);
    }
  };
  auto writer2 = [&]() {
    std::uint64_t counter = 0;
    while (!stop_flag) {
      auto order_id = counter++;
      auto msg = MakeMboMsg(200, order_id, 200000 + (order_id % 50), 5, databento::Side::Ask, databento::Action::Add);
      market.OnMboUpdate(msg);
    }
  };
  auto reader = [&]() {
    while (!stop_flag) {
      auto quotes = market.AllBestQuotes();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  std::thread w1(writer1);
  std::thread w2(writer2);
  std::thread r1(reader);

  std::this_thread::sleep_for(std::chrono::seconds(1));
  stop_flag.store(true);

  w1.join();
  w2.join();
  r1.join();

  SUCCEED("No crash in concurrency stress test.");
}

// TEST_CASE("OrderBook performance benchmark", "[orderbook][advanced][performance]") {
//   LimitOrderBook lob(9999);

//   constexpr int NUM_UPDATES = 200000;
//   auto start = std::chrono::steady_clock::now();

//   for(int i=0; i<NUM_UPDATES; ++i) {
//     auto msg = MakeMboMsg(9999, i, 5000 + (i % 100), 10, databento::Side::Bid, databento::Action::Add);
//     lob.OnMboUpdate(msg);
//   }

//   auto end = std::chrono::steady_clock::now();
//   auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//   WARN("Performed " << NUM_UPDATES << " updates in " << ms << " ms");
//   SUCCEED();
// }

TEST_CASE("OrderBook fault injection test", "[orderbook][advanced][fault]") {
  // Create a FaultyMarketBook that wraps the singleton MarketBook
  FaultyMarketBook faulty_mb;
  faulty_mb.AddInstrument(999, std::make_unique<LimitOrderBook>(999));
  faulty_mb.enable_faults.store(true);
  faulty_mb.fault_probability.store(0.2);

  int success_count=0, fail_count=0;
  for(int i=0; i<100; ++i) {
    auto msg = MakeMboMsg(999, i, 1000+i, 5, databento::Side::Bid, databento::Action::Add);
    try {
      faulty_mb.OnMboUpdate(msg);
      ++success_count;
    } catch(...) {
      ++fail_count;
    }
  }
  SUCCEED("Fault injection: success=" << success_count << ", fail=" << fail_count);
}

} // end namespace constellation::modules::orderbook
