#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <chrono>
#include <thread>

#include "engine/Engine.hpp"
#include "engine/EngineConfig.hpp"
#include "orderbook/OrderBookFactory.hpp"

TEST_CASE("Engine can be initialized and started/stopped", "[engine][integration]") {
    // Create engine instance
    constellation::applications::engine::Engine engine;
    
    // Create config with required components
    constellation::applications::engine::EngineConfig config;
    
    // Create and add a valid market book
    auto market_view = constellation::modules::orderbook::CreateMarketBook();
    auto market_book = std::dynamic_pointer_cast<constellation::interfaces::orderbook::IMarketBook>(market_view);
    config.marketBook = market_book;
    
    // Initialize
    REQUIRE_NOTHROW(engine.Initialize(config));
    
    // Start
    REQUIRE_NOTHROW(engine.Start());
    
    // Should be running
    CHECK(engine.IsRunning());
    
    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Stop
    REQUIRE_NOTHROW(engine.Stop());
    
    // Should no longer be running
    CHECK_FALSE(engine.IsRunning());
    
    // Get stats (should at least not crash)
    auto stats = engine.GetStats();
    CHECK(stats.feed_msgs == 0); // No feed configured, so no messages processed
}