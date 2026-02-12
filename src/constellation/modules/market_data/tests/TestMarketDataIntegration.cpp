// File: constellation-cpp/modules/market_data/tests/TestMarketDataIntegration.cpp

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include "market_data/MarketDataFactory.hpp"  // the new factory
#include "interfaces/market_data/FeedConfigs.hpp"
#include "interfaces/market_data/IIngestionFeed.hpp"

using namespace constellation::modules::market_data;
using namespace constellation::interfaces::market_data;

static constexpr int      EXPECTED_RECORD_COUNT   = 4176;
static constexpr int64_t  EXPECTED_INSTRUMENT_ID  = 42005347;
static constexpr double   KNOWN_MIN_PRICE         = 5752000000000.0;
static constexpr double   KNOWN_MAX_PRICE         = 6359000000000.0;

/**
 * Provide your test DBN file path at build time, e.g.:
 *   -DTEST_DATA_DIR=/path/to/...
 */
static constexpr const char* kTestDbnFilePath = TEST_DATA_DIR "/glbx-mdp3-20250121-MESH5-NY-9:30-9:31.mbo.dbn.zst";

TEST_CASE("Local DBN file replay with detailed checks", "[market_data]") {
  std::ifstream file_check(kTestDbnFilePath);
  if (!file_check.is_open()) {
    FAIL("Local DBN test file not found: " << kTestDbnFilePath);
  }

  DbnFileFeedConfig cfg;
  cfg.file_path    = kTestDbnFilePath;
  cfg.loop_forever = false;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  std::atomic<int> mbo_count(0);
  std::atomic<int> record_count(0);
  std::atomic<int64_t> min_px(std::numeric_limits<int64_t>::max());
  std::atomic<int64_t> max_px(std::numeric_limits<int64_t>::lowest());

  feed->SubscribeMboCallback([&](const databento::MboMsg& mbo){
    int idx = mbo_count.fetch_add(1);
    if (idx < 5) {
      std::cout << "[DEBUG] Mbo #" << idx << " price=" << mbo.price << "\n";
    }
    if (mbo.hd.instrument_id != EXPECTED_INSTRUMENT_ID) {
      FAIL_CHECK("Unexpected instrument_id: " << mbo.hd.instrument_id);
    }
    auto old_min = min_px.load();
    while (mbo.price < old_min && !min_px.compare_exchange_weak(old_min, mbo.price)) {}
    auto old_max = max_px.load();
    while (mbo.price > old_max && !max_px.compare_exchange_weak(old_max, mbo.price)) {}
  });

  feed->SubscribeRecordCallback([&](const databento::Record&) {
    record_count.fetch_add(1, std::memory_order_relaxed);
  });

  feed->Start();
  auto start_time = std::chrono::steady_clock::now();

  while (true) {
    if (record_count.load() >= EXPECTED_RECORD_COUNT) {
      break;
    }
    if (std::chrono::steady_clock::now() - start_time > std::chrono::seconds(5)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  feed->Stop();

  int final_mbo     = mbo_count.load();
  int final_records = record_count.load();
  int64_t final_min = min_px.load();
  int64_t final_max = max_px.load();

  REQUIRE(final_records == EXPECTED_RECORD_COUNT);
  CHECK(final_mbo == final_records);

  double dmin = static_cast<double>(final_min);
  double dmax = static_cast<double>(final_max);
  CHECK(dmin == Catch::Approx(KNOWN_MIN_PRICE).margin(0.01));
  CHECK(dmax == Catch::Approx(KNOWN_MAX_PRICE).margin(0.01));
}

TEST_CASE("Local DBN file replay with loop forever", "[market_data]") {
  std::ifstream file_check(kTestDbnFilePath);
  if (!file_check.is_open()) {
    FAIL("Local DBN test file not found: " << kTestDbnFilePath);
  }

  DbnFileFeedConfig cfg;
  cfg.file_path    = kTestDbnFilePath;
  cfg.loop_forever = true;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  std::atomic<int> rec_count(0);
  feed->SubscribeRecordCallback([&](const databento::Record&){
    rec_count.fetch_add(1);
  });

  feed->Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  feed->Stop();

  CHECK(rec_count.load() >= 2 * EXPECTED_RECORD_COUNT);
}

TEST_CASE("Local DBN file replay with invalid path", "[market_data]") {
  DbnFileFeedConfig cfg;
  cfg.file_path    = "DOES_NOT_EXIST.dbn.zst";
  cfg.loop_forever = false;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  std::atomic<int> rec_count(0);
  feed->SubscribeRecordCallback([&](const databento::Record&){
    rec_count.fetch_add(1);
  });

  feed->Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  feed->Stop();

  CHECK(rec_count.load() == 0);
}

TEST_CASE("Multiple Start/Stop calls", "[market_data]") {
  std::ifstream file_check(kTestDbnFilePath);
  if (!file_check.is_open()) {
    FAIL("Local DBN test file not found: " << kTestDbnFilePath);
  }

  DbnFileFeedConfig cfg;
  cfg.file_path = kTestDbnFilePath;

  auto feed = MarketDataFactory::CreateDbnFileFeed(cfg);

  CHECK_NOTHROW(feed->Start());
  CHECK_NOTHROW(feed->Stop());
  CHECK_NOTHROW(feed->Start());
  CHECK_NOTHROW(feed->Stop());
}

TEST_CASE("Historical Databento Integration with forced check", "[market_data][live]") {
  // Check for environment variable
  const char* key = std::getenv("DATABENTO_API_KEY");
  if (!key) {
    FAIL("DATABENTO_API_KEY is not set in environment; skipping live test");
  }

  // EXACT same date/time, symbol, etc. from your Python code
  DataBentoFeedConfig cfg;
  cfg.api_key        = std::getenv("DATABENTO_API_KEY");
  cfg.dataset        = "GLBX.MDP3";
  cfg.symbols        = {"MESH5"};
  cfg.schema         = "mbo"; // important
  cfg.use_live       = false; 
  cfg.start_datetime = "2025-01-06T14:30:00";
  cfg.end_datetime   = "2025-01-06T14:31:00";

  // advanced fields that Python sets automatically:
  cfg.gateway        = databento::HistoricalGateway::Bo1;
  cfg.stype_in       = databento::SType::RawSymbol;    // for "MESH5"
  cfg.stype_out      = databento::SType::InstrumentId; 
  cfg.limit          = 0; // unlimited

  // optional: if you want a metadata callback
  cfg.metadata_callback = [](databento::Metadata&& md) {
    std::cout << "[DEBUG] got md.dataset=" << md.dataset
              << " #symbols=" << md.symbols.size() << "\n";
  };

  // Now do MarketDataFactory::CreateDataBentoMboFeed(cfg)
  auto feed = MarketDataFactory::CreateDataBentoMboFeed(cfg);

  // We call it 'record_count' (just be consistent)
  std::atomic<int> record_count{0};

  feed->SubscribeRecordCallback([&](const databento::Record&) {
    record_count.fetch_add(1, std::memory_order_relaxed);
  });

  // Start feed
  feed->Start();

  // Wait up to 15 seconds for data
  auto start_t = std::chrono::steady_clock::now();
  while (true) {
    if (record_count.load() > 0) {
      // We got data => break out
      break;
    }
    auto elapsed = std::chrono::steady_clock::now() - start_t;
    if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 15) {
      // Timed out
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // Stop feed
  feed->Stop();

  CHECK(record_count.load() > 0);  // if zero => test fails
  std::cout << "[INFO] Received " << record_count.load() << " records from DataBento.\n";
}

