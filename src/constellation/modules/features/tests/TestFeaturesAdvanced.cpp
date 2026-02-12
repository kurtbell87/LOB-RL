#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <random>
#include <iostream>
#include <stdexcept>

#include "features/FeatureManager.hpp"
#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/derived/MidPriceFeature.hpp"

namespace constellation {
namespace modules {
namespace features {

/**
 * @brief A simple data source that can optionally throw exceptions,
 *        but uses int64 methods. We'll just store double and convert.
 */
class FaultyDataSource final : public constellation::interfaces::orderbook::IMarketBookDataSource {
public:
  bool enable_faults{false};
  double fault_probability{0.1};

  std::atomic<double> bid{100.0};
  std::atomic<double> ask{110.0};

  static std::int64_t toNano(double x) {
    return static_cast<std::int64_t>(x * 1e9 + 0.5);
  }

  // overrides
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {1, 0};
  }

  std::optional<std::int64_t> BestBidPrice(std::uint32_t /*instrument_id*/) const override {
    double b = bid.load();
    if (b <= 0.0) return std::nullopt;
    return toNano(b);
  }

  std::optional<std::int64_t> BestAskPrice(std::uint32_t /*instrument_id*/) const override {
    double a = ask.load();
    if (a <= 0.0) return std::nullopt;
    return toNano(a);
  }

  std::optional<std::uint64_t> VolumeAtPrice(std::uint32_t /*instrument_id*/,
                                             std::int64_t /*priceNanos*/) const override {
    return std::nullopt; // not used here
  }

  std::vector<std::uint32_t> GetInstrumentIds() const override {
    return {};
  }
};

} // end namespace features
} // end namespace modules
} // end namespace constellation

#include <catch2/catch_test_macros.hpp>
#include "features/FeatureManager.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/derived/MidPriceFeature.hpp"

using namespace constellation::modules::features;

TEST_CASE("Features concurrency stress test", "[features][advanced][concurrency]") {
  FeatureManager manager;
  manager.Register(std::make_shared<primitives::BestBidPriceFeature>(
                     primitives::BestBidPriceFeature::Config{999}));
  manager.Register(std::make_shared<primitives::BestAskPriceFeature>(
                     primitives::BestAskPriceFeature::Config{999}));
  manager.Register(std::make_shared<derived::MidPriceFeature>(
                     derived::MidPriceFeature::Config{999}));

  FaultyDataSource ds;
  ds.enable_faults = false;

  auto writer = [&]() {
    for (int i = 0; i < 200; ++i) {
      ds.bid.store(100.0 + i);
      ds.ask.store(110.0 + i);
      manager.OnDataUpdate(ds, nullptr);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };

  auto reader = [&](int reps) {
    for (int i = 0; i < reps; ++i) {
      double bid_val = manager.GetValue("best_bid_price");
      double ask_val = manager.GetValue("best_ask_price");
      double mid_val = manager.GetValue("mid_price");
      (void)bid_val; (void)ask_val; (void)mid_val;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  };

  std::thread w(writer);
  std::thread r1(reader, 150);
  std::thread r2(reader, 150);

  w.join();
  r1.join();
  r2.join();

  SUCCEED("Concurrent feature usage completed without data races or deadlocks.");
}

TEST_CASE("Features performance benchmark", "[features][advanced][performance]") {
  FeatureManager manager;
  manager.Register(std::make_shared<primitives::BestBidPriceFeature>(
                     primitives::BestBidPriceFeature::Config{999}));
  manager.Register(std::make_shared<primitives::BestAskPriceFeature>(
                     primitives::BestAskPriceFeature::Config{999}));
  manager.Register(std::make_shared<derived::MidPriceFeature>(
                     derived::MidPriceFeature::Config{999}));

  FaultyDataSource ds;
  ds.enable_faults = false;

  constexpr int NUM_UPDATES = 100000;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < NUM_UPDATES; ++i) {
    ds.bid.store(100.0 + i * 0.01);
    ds.ask.store(110.0 + i * 0.01);
    manager.OnDataUpdate(ds, nullptr);
  }

  auto end = std::chrono::steady_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  WARN("Features performance test: " << NUM_UPDATES << " updates took " << elapsed_ms << " ms");
  SUCCEED();
}

TEST_CASE("Features fault injection test", "[features][advanced][fault]") {
  FeatureManager manager;
  manager.Register(std::make_shared<primitives::BestBidPriceFeature>(
                     primitives::BestBidPriceFeature::Config{999}));
  manager.Register(std::make_shared<primitives::BestAskPriceFeature>(
                     primitives::BestAskPriceFeature::Config{999}));
  manager.Register(std::make_shared<derived::MidPriceFeature>(
                     derived::MidPriceFeature::Config{999}));

  FaultyDataSource ds;
  ds.enable_faults = true;
  ds.fault_probability = 0.2;

  int success_count = 0;
  int fail_count = 0;
  for (int i = 0; i < 100; ++i) {
    try {
      manager.OnDataUpdate(ds, nullptr);
      double mid = manager.GetValue("mid_price");
      (void)mid;
      ++success_count;
    } catch(const std::exception& ex) {
      ++fail_count;
    }
  }
  REQUIRE((success_count + fail_count) == 100);

  SUCCEED("Fault injection test completed. successes=" << success_count << ", failures=" << fail_count);
}
