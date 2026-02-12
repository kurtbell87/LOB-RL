// File: constellation-cpp/modules/market_data/tests/TestMarketDataAdvanced.cpp

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <thread>
#include <chrono>
#include <atomic>
#include <stdexcept>
#include <fstream>
#include <random>
#include <memory>
#include <iostream>

#include "market_data/MarketDataFactory.hpp"  // We use the factory now
#include "interfaces/market_data/FeedConfigs.hpp"
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

using namespace constellation::modules::market_data;
using namespace constellation::interfaces::market_data;
using namespace constellation::interfaces::logging;

/**
 * @brief A derived test feed for fault injection, still local to this module's tests.
 */
class FaultyIngestionFeed : public IIngestionFeed {
public:
  std::atomic<bool> enable_faults{false};
  std::atomic<double> fault_probability{0.1};

  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {1, 0};
  }

  void SubscribeMboCallback(const std::function<void(const databento::MboMsg&)>& cb) override {
    mbo_callbacks_.push_back(cb);
  }
  void SubscribeRecordCallback(const std::function<void(const databento::Record&)>& cb) override {
    record_callbacks_.push_back(cb);
  }
  void Start() override {
    MaybeThrow();
    running_ = true;
  }
  void Stop() override {
    running_ = false;
  }

private:
  void MaybeThrow() {
    if (!enable_faults.load()) return;
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(rng) < fault_probability.load()) {
      throw std::runtime_error("[FaultyIngestionFeed] Injection error");
    }
  }

  bool running_{false};
  std::vector<std::function<void(const databento::MboMsg&)>> mbo_callbacks_;
  std::vector<std::function<void(const databento::Record&)>> record_callbacks_;
};

TEST_CASE("MarketData concurrency stress test", "[market_data][advanced][concurrency]") {
  // Example: we use the DBN file feed to repeatedly start/stop in multiple threads
  // We'll just create it from the factory:

  DbnFileFeedConfig cfg;
  cfg.file_path    = "SOME_SMALL_FILE.dbn"; // user must supply
  cfg.loop_forever = false;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  std::atomic<bool> stop_flag{false};
  auto starter = [&]() {
    while (!stop_flag.load()) {
      feed->Start();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      feed->Stop();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  };

  std::thread t1(starter), t2(starter);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  stop_flag.store(true);
  t1.join();
  t2.join();

  SUCCEED("Concurrent Start/Stop on a DBN file feed did not crash.");
}

TEST_CASE("MarketData performance test (file replay)", "[market_data][advanced][performance]") {
  // We'll measure how quickly we can read from a local DBN file
  static constexpr const char* test_file_path = TEST_DATA_DIR "glbx-mdp3-20250102.mbo.dbn.zst";
  std::ifstream fcheck(test_file_path);
  if (!fcheck.is_open()) {
    WARN("No performance DBN file found - skipping test");
    return;
  }

  DbnFileFeedConfig cfg;
  cfg.file_path    = test_file_path;
  cfg.loop_forever = false;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  std::atomic<int> record_count(0);
  feed->SubscribeRecordCallback([&](const databento::Record&) {
    record_count.fetch_add(1, std::memory_order_relaxed);
  });

  auto start_time = std::chrono::steady_clock::now();
  feed->Start();
  std::this_thread::sleep_for(std::chrono::seconds(3));
  feed->Stop();

  auto end_time = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  WARN("Performance test read " << record_count.load() << " records in " << ms << " ms");
  SUCCEED();
}

TEST_CASE("MarketData fault injection test", "[market_data][advanced][fault]") {
  auto feed = std::make_shared<FaultyIngestionFeed>();
  feed->enable_faults.store(true);
  feed->fault_probability.store(0.25);

  int success_count = 0;
  int fail_count = 0;
  for (int i = 0; i < 50; ++i) {
    try {
      feed->Start();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      feed->Stop();
      ++success_count;
    } catch(const std::exception&) {
      ++fail_count;
    }
  }
  SUCCEED("Fault injection feed: success=" << success_count
           << ", fail=" << fail_count);
}
