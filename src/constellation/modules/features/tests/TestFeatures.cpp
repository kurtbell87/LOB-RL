#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <atomic>
#include <chrono>

#include "features/FeatureManager.hpp"
#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/derived/MidPriceFeature.hpp"
#include "MockMarketDataSource.hpp"

using namespace constellation::modules::features::primitives;
using namespace constellation::modules::features::derived;
using constellation::modules::features::FeatureManager;

TEST_CASE("Test Basic Features - Best Bid, Best Ask, MidPrice") {
  // Provide config for instrument_id=1234
  BestBidPriceFeature::Config bb_cfg{1234};
  BestAskPriceFeature::Config ba_cfg{1234};
  MidPriceFeature::Config     mp_cfg{1234};

  auto bid_feat = std::make_shared<BestBidPriceFeature>(bb_cfg);
  auto ask_feat = std::make_shared<BestAskPriceFeature>(ba_cfg);
  auto mid_feat = std::make_shared<MidPriceFeature>(mp_cfg);

  // Create the mock data source
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 105.0;

  // Force an update
  bid_feat->OnDataUpdate(source, nullptr);
  ask_feat->OnDataUpdate(source, nullptr);
  mid_feat->OnDataUpdate(source, nullptr);

  // Then test values => each "GetValue" will do (raw int64 / 1e9)
  REQUIRE(bid_feat->GetValue("best_bid_price") == 100.0);
  REQUIRE(ask_feat->GetValue("best_ask_price") == 105.0);
  REQUIRE(mid_feat->GetValue("mid_price") == 102.5);
}

TEST_CASE("Handling missing data: no best bid or no best ask", "[features]") {
  FeatureManager manager;
  using BBF = constellation::modules::features::primitives::BestBidPriceFeature;
  using BAF = constellation::modules::features::primitives::BestAskPriceFeature;
  using MPF = constellation::modules::features::derived::MidPriceFeature;

  auto bid_feat = std::make_shared<BBF>(BBF::Config{1234});
  auto ask_feat = std::make_shared<BAF>(BAF::Config{1234});
  auto mid_feat = std::make_shared<MPF>(MPF::Config{1234});

  manager.Register(bid_feat);
  manager.Register(ask_feat);
  manager.Register(mid_feat);

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 105.0;

  SECTION("No best bid => best_bid_price=0.0, mid_price=0.0") {
    source.bid_valid = false;  // no best bid
    source.best_ask  = 105.0;
    manager.OnDataUpdate(source, nullptr);

    CHECK(manager.GetValue("best_bid_price") == Catch::Approx(0.0));
    CHECK(manager.GetValue("best_ask_price") == Catch::Approx(105.0));
    CHECK(manager.GetValue("mid_price") == Catch::Approx(0.0));
  }

  SECTION("No best ask => best_ask_price=0.0, mid_price=0.0") {
    source.ask_valid = false;  // no best ask
    source.best_bid  = 95.0;
    manager.OnDataUpdate(source, nullptr);

    CHECK(manager.GetValue("best_bid_price") == Catch::Approx(95.0));
    CHECK(manager.GetValue("best_ask_price") == Catch::Approx(0.0));
    CHECK(manager.GetValue("mid_price") == Catch::Approx(0.0));
  }
}

TEST_CASE("FeatureManager concurrency: single-writer, multiple-readers", "[features][concurrency]") {
  FeatureManager manager;
  using BBF = constellation::modules::features::primitives::BestBidPriceFeature;
  using BAF = constellation::modules::features::primitives::BestAskPriceFeature;
  using MPF = constellation::modules::features::derived::MidPriceFeature;

  auto bid_feat = std::make_shared<BBF>(BBF::Config{1234});
  auto ask_feat = std::make_shared<BAF>(BAF::Config{1234});
  auto mid_feat = std::make_shared<MPF>(MPF::Config{1234});

  manager.Register(bid_feat);
  manager.Register(ask_feat);
  manager.Register(mid_feat);

  MockMarketDataSource source;
  source.best_bid = 1000.0;
  source.best_ask = 1010.0;

  auto writer = [&]() {
    for (int i = 0; i < 50; ++i) {
      source.best_bid = 1000.0 + i;
      source.best_ask = 1010.0 + i;
      manager.OnDataUpdate(source, nullptr);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  auto reader = [&](int iterations){
    for (int i = 0; i < iterations; ++i) {
      double bid = manager.GetValue("best_bid_price");
      double ask = manager.GetValue("best_ask_price");
      double mid = manager.GetValue("mid_price");
      (void)bid;
      (void)ask;
      (void)mid;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  std::thread w(writer);
  std::thread r1([&](){ reader(40); });
  std::thread r2([&](){ reader(40); });

  w.join();
  r1.join();
  r2.join();

  SUCCEED("Concurrent read/write executed safely.");
}

TEST_CASE("FeatureManager: unknown feature name throws exception", "[features]") {
  FeatureManager manager;
  // Register a BestBidPriceFeature with instrument_id=1234
  manager.Register(std::make_shared<BestBidPriceFeature>(BestBidPriceFeature::Config{1234}));

  MockMarketDataSource source;
  manager.OnDataUpdate(source, nullptr);

  CHECK_THROWS_AS(manager.GetValue("nonexistent_feature"), std::runtime_error);
}
