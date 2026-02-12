#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

#include "market_data/MarketDataFactory.hpp"
#include "interfaces/market_data/FeedConfigs.hpp"

using namespace constellation::modules::market_data;
using namespace constellation::interfaces::market_data;

TEST_CASE("MarketDataFactory validates DbnFileFeedConfig", "[market_data][edge-cases]") {
    // Test that factory handles invalid config
    DbnFileFeedConfig empty_cfg;
    // Empty file path should make CreateDbnFileFeed return nullptr
    auto empty_feed = MarketDataFactory::CreateDbnFileFeed(empty_cfg);
    CHECK(empty_feed == nullptr);
    
    // Valid config should work
    DbnFileFeedConfig valid_cfg;
    valid_cfg.file_path = "/valid/path.dbn";
    auto feed = MarketDataFactory::CreateDbnFileFeed(valid_cfg);
    CHECK(feed != nullptr);
}