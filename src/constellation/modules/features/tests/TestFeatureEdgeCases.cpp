#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "features/FeatureManager.hpp"
#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/primitives/SpreadFeature.hpp"
#include "features/primitives/MicroPriceFeature.hpp"
#include "features/primitives/OrderImbalanceFeature.hpp"
#include "features/primitives/LogReturnFeature.hpp"
#include "features/primitives/VolumeAtPriceFeature.hpp"
#include "features/primitives/MicroDepthFeature.hpp"
#include "features/derived/MidPriceFeature.hpp"
#include "features/derived/CancelAddRatioFeature.hpp"
#include "features/derived/RollingVolatilityFeature.hpp"
#include "MockMarketDataSource.hpp"

using namespace constellation::modules::features;
using primitives::BestAskPriceFeature;
using primitives::BestBidPriceFeature;
using primitives::SpreadFeature;
using primitives::MicroPriceFeature;
using primitives::OrderImbalanceFeature;
using primitives::LogReturnFeature;
using primitives::VolumeAtPriceFeature;
using primitives::MicroDepthFeature;
using derived::MidPriceFeature;
using derived::CancelAddRatioFeature;
using derived::RollingVolatilityFeature;

TEST_CASE("RollingVolatilityFeature handles zero or invalid window size", "[features][rolling-vol][edge]") {
  RollingVolatilityFeature::Config bad_cfg{1234, 0};
  CHECK_THROWS_AS(std::make_shared<RollingVolatilityFeature>(bad_cfg), FeatureException);

  RollingVolatilityFeature::Config big_cfg{1234, 999999};
  auto large_roll = std::make_shared<RollingVolatilityFeature>(big_cfg);
  REQUIRE(large_roll != nullptr);

  FeatureManager manager;
  manager.Register(large_roll);

  MockMarketDataSource ds;
  ds.best_bid = 100.0;

  auto writer = [&]() {
    for (int i = 0; i < 200; ++i) {
      ds.best_bid.store(100.0 + ((i % 11)*0.1));
      manager.OnDataUpdate(ds, nullptr);
    }
  };
  auto reader = [&]() {
    for (int i = 0; i < 200; ++i) {
      double vol = manager.GetValue("rolling_volatility");
      (void)vol;
    }
  };

  std::thread w(writer);
  std::thread r1(reader);
  std::thread r2(reader);

  w.join();
  r1.join();
  r2.join();

  SUCCEED("Concurrently updating RollingVolatilityFeature with large window size works.");
}

TEST_CASE("FeatureManager sub-feature name conflict resolution", "[features][conflict]") {
  FeatureManager mgr;
  auto feat1 = std::make_shared<LogReturnFeature>(LogReturnFeature::Config{1234});
  auto feat2 = std::make_shared<LogReturnFeature>(LogReturnFeature::Config{1234});

  mgr.Register(feat1);
  mgr.Register(feat2);

  MockMarketDataSource ds;
  ds.best_bid = 100.0;
  mgr.OnDataUpdate(ds, nullptr);
  ds.best_bid = 110.0;
  mgr.OnDataUpdate(ds, nullptr);

  double lr = mgr.GetValue("log_return");
  (void)lr;
  SUCCEED("No crash on sub-feature name conflict.");
}

TEST_CASE("FeatureManager repeated registration of the exact same feature instance", "[features][duplicates]") {
  FeatureManager mgr;
  auto mid_price = std::make_shared<MidPriceFeature>(MidPriceFeature::Config{1234});

  mgr.Register(mid_price);
  mgr.Register(mid_price);

  MockMarketDataSource ds;
  ds.best_bid = 100;
  ds.best_ask = 102;
  mgr.OnDataUpdate(ds, nullptr);

  REQUIRE_NOTHROW(mgr.GetValue("mid_price"));
  CHECK(mgr.GetValue("mid_price") == Catch::Approx(101.0));
}

TEST_CASE("Features handle extremely large or invalid best bid/ask", "[features][extreme-values]") {
  FeatureManager mgr;
  mgr.Register(std::make_shared<BestBidPriceFeature>(BestBidPriceFeature::Config{1234}));
  mgr.Register(std::make_shared<BestAskPriceFeature>(BestAskPriceFeature::Config{1234}));
  mgr.Register(std::make_shared<SpreadFeature>(SpreadFeature::Config{1234}));
  mgr.Register(std::make_shared<MidPriceFeature>(MidPriceFeature::Config{1234}));

  MockMarketDataSource ds;
  ds.best_bid = 1e9;       // $1,000,000,000
  ds.best_ask = 1e9 + 100; // $1,000,000,100
  mgr.OnDataUpdate(ds, nullptr);

  CHECK(mgr.GetValue("best_bid_price") == Catch::Approx(1e9));
  CHECK(mgr.GetValue("best_ask_price") == Catch::Approx(1e9 + 100));
  CHECK(mgr.GetValue("bid_ask_spread") == Catch::Approx(100.0));
  CHECK(mgr.GetValue("mid_price") == Catch::Approx(1e9 + 50.0));

  // Next: best bid > best ask => features revert to 0
  ds.best_bid = 5000.0;
  ds.best_ask = 4999.0;
  mgr.OnDataUpdate(ds, nullptr);

  CHECK(mgr.GetValue("best_bid_price") == Catch::Approx(5000.0));
  CHECK(mgr.GetValue("best_ask_price") == Catch::Approx(4999.0));
  CHECK(mgr.GetValue("bid_ask_spread") == Catch::Approx(0.0));
  CHECK(mgr.GetValue("mid_price") == Catch::Approx(0.0));
}

TEST_CASE("VolumeAtPriceFeature and MicroDepthFeature handle missing market pointer", "[features][market-view]") {
  VolumeAtPriceFeature::Config vol_cfg{1234};
  auto vol_feat = std::make_shared<VolumeAtPriceFeature>(vol_cfg);

  MicroDepthFeature::Config depth_cfg{1234};
  auto depth_feat = std::make_shared<MicroDepthFeature>(depth_cfg);

  FeatureManager mgr;
  mgr.Register(vol_feat);
  mgr.Register(depth_feat);

  MockMarketDataSource ds;
  ds.best_bid = 100.0;
  mgr.OnDataUpdate(ds, nullptr);

  CHECK(mgr.GetValue("volume_at_price") == Catch::Approx(0.0));
  CHECK(mgr.GetValue("micro_depth_price") == Catch::Approx(0.0));
  CHECK(mgr.GetValue("micro_depth_size") == Catch::Approx(0.0));
}

TEST_CASE("MicroPriceFeature large volumes and concurrency", "[features][micro-price][concurrency]") {
  FeatureManager mgr;
  auto mp_feat = std::make_shared<MicroPriceFeature>(MicroPriceFeature::Config{1234});
  mgr.Register(mp_feat);

  MockMarketDataSource ds;
  ds.best_bid = 1000.0;
  ds.best_ask = 1002.0;
  ds.volumes[1000.0] = 5000000000ULL;
  ds.volumes[1002.0] = 3000000000ULL;

  mgr.OnDataUpdate(ds, nullptr);

  double mp = mgr.GetValue("micro_price");
  CHECK(mp == Catch::Approx(1001.25));

  auto writer = [&]() {
    for (int i = 0; i < 100; ++i) {
      ds.best_bid = 1000.0 + i;
      ds.best_ask = 1002.0 + i;
      ds.volumes[1000.0 + i] = 5000000000ULL;
      ds.volumes[1002.0 + i] = 3000000000ULL;
      mgr.OnDataUpdate(ds, nullptr);
    }
  };
  auto reader = [&]() {
    for (int i=0; i<100; ++i) {
      double val = mgr.GetValue("micro_price");
      (void)val;
    }
  };

  std::thread w(writer);
  std::thread r(reader);
  w.join();
  r.join();

  SUCCEED("MicroPriceFeature handles large volumes under concurrency with no overflow or error.");
}

TEST_CASE("OrderImbalanceFeature missing volumes, partial data", "[features][imbalance]") {
  FeatureManager mgr;
  auto imb = std::make_shared<OrderImbalanceFeature>(OrderImbalanceFeature::Config{1234});
  mgr.Register(imb);

  MockMarketDataSource ds;
  ds.best_bid = 101.0;
  ds.best_ask = 102.0;
  ds.volumes[102.0] = 50ULL; // no volume at 101 => zero => imbalance negative

  mgr.OnDataUpdate(ds, nullptr);
  double val = mgr.GetValue("order_imbalance");
  CHECK(val == Catch::Approx(-1.0));
}

TEST_CASE("CancelAddRatioFeature zero adds and concurrency", "[features][cancel-add-ratio]") {
  FeatureManager mgr;
  auto ratio_feat = std::make_shared<CancelAddRatioFeature>();
  mgr.Register(ratio_feat);

  // We'll mock a "mock market view" that returns globalAdd= i, globalCancel= i/2
  // For concurrency, let's define a minimal class implementing IMarketView
  class MockMarketView final : public constellation::interfaces::orderbook::IMarketView {
  public:
    std::atomic<std::uint64_t> add_cnt{0};
    std::atomic<std::uint64_t> cancel_cnt{0};

    // IMarketStatistics
    std::uint64_t GetGlobalAddCount() const noexcept override { return add_cnt.load(); }
    std::uint64_t GetGlobalCancelCount() const noexcept override { return cancel_cnt.load(); }
    std::uint64_t GetGlobalModifyCount() const noexcept override { return 0; }
    std::uint64_t GetGlobalTradeCount() const noexcept override { return 0; }
    std::uint64_t GetGlobalClearCount() const noexcept override { return 0; }
    std::uint64_t GetGlobalTotalEventCount() const noexcept override { return 0; }

    // IMarketStateView
    std::optional<constellation::interfaces::orderbook::PriceLevel> GetBestBid(std::uint32_t) const override {
      return std::nullopt;
    }
    std::optional<constellation::interfaces::orderbook::PriceLevel> GetBestAsk(std::uint32_t) const override {
      return std::nullopt;
    }

    // IInstrumentRegistry
    std::size_t InstrumentCount() const noexcept override { return 0; }
    std::vector<std::uint32_t> GetInstrumentIds() const override { return {}; }

    constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
      return {1,0};
    }
  };

  MockMarketView mock_mv;
  MockMarketDataSource ds;

  auto writer = [&](int steps) {
    for (int i = 1; i <= steps; ++i) {
      mock_mv.add_cnt.store(i);
      mock_mv.cancel_cnt.store(i/2);
      mgr.OnDataUpdate(ds, &mock_mv);
    }
  };

  auto reader = [&](int steps){
    for (int i=0; i<steps; ++i) {
      double r = mgr.GetValue("cancel_add_ratio");
      (void)r;
    }
  };

  std::thread w(writer, 200);
  std::thread r1(reader, 100);
  std::thread r2(reader, 100);
  w.join();
  r1.join();
  r2.join();

  mock_mv.add_cnt.store(200);
  mock_mv.cancel_cnt.store(100);
  mgr.OnDataUpdate(ds, &mock_mv);

  double final_val = mgr.GetValue("cancel_add_ratio");
  CHECK(final_val == Catch::Approx(0.5).margin(0.001));
}

TEST_CASE("FeatureManager concurrency with many different features registered", "[features][all]") {
  FeatureManager manager;
  manager.Register(std::make_shared<BestBidPriceFeature>(BestBidPriceFeature::Config{1234}));
  manager.Register(std::make_shared<BestAskPriceFeature>(BestAskPriceFeature::Config{1234}));
  manager.Register(std::make_shared<SpreadFeature>(SpreadFeature::Config{1234}));
  manager.Register(std::make_shared<MidPriceFeature>(MidPriceFeature::Config{1234}));
  manager.Register(std::make_shared<MicroPriceFeature>(MicroPriceFeature::Config{1234}));
  manager.Register(std::make_shared<OrderImbalanceFeature>(OrderImbalanceFeature::Config{1234}));
  manager.Register(std::make_shared<LogReturnFeature>(LogReturnFeature::Config{1234}));
  manager.Register(std::make_shared<CancelAddRatioFeature>());

  {
    RollingVolatilityFeature::Config cfg{1234, 5};
    manager.Register(std::make_shared<RollingVolatilityFeature>(cfg));
  }
  {
    VolumeAtPriceFeature::Config vcfg{1234};
    manager.Register(std::make_shared<VolumeAtPriceFeature>(vcfg));
  }
  {
    MicroDepthFeature::Config mcfg{1234};
    manager.Register(std::make_shared<MicroDepthFeature>(mcfg));
  }

  MockMarketDataSource ds;
  ds.best_bid = 100.0;
  ds.best_ask = 102.0;
  ds.volumes[100.0] = 10ULL;
  ds.volumes[102.0] = 20ULL;

  auto updater = [&]() {
    for (int i = 0; i < 50; ++i) {
      ds.best_bid.store(100.0 + i);
      ds.best_ask.store(102.0 + i);
      ds.volumes[100.0 + i] = 10ULL + i;
      ds.volumes[102.0 + i] = 20ULL + i;
      manager.OnDataUpdate(ds, nullptr);
    }
  };

  auto reader = [&]() {
    const std::vector<std::string> names{
      "best_bid_price","best_ask_price","bid_ask_spread","mid_price",
      "micro_price","order_imbalance","log_return","cancel_add_ratio",
      "rolling_volatility","volume_at_price","micro_depth_price","micro_depth_size"
    };
    for (int i=0; i<50; ++i) {
      for (auto& nm : names) {
        double val = manager.GetValue(nm);
        (void)val;
      }
    }
  };

  std::thread w1(updater);
  std::thread w2(updater);
  std::thread r1(reader);
  std::thread r2(reader);

  w1.join();
  w2.join();
  r1.join();
  r2.join();

  SUCCEED("FeatureManager concurrency with all known features passed without error.");
}
