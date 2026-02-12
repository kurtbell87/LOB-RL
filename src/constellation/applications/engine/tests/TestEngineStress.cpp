#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <chrono>
#include <thread>

#include "engine/Engine.hpp"
#include "engine/EngineConfig.hpp"
#include "orderbook/OrderBookFactory.hpp"

TEST_CASE("Engine can handle rapid start/stop cycles", "[engine][stress]") {
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
    
    // Perform multiple rapid start/stop cycles
    for (int i = 0; i < 3; i++) {
        REQUIRE_NOTHROW(engine.Start());
        CHECK(engine.IsRunning());
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        REQUIRE_NOTHROW(engine.Stop());
        CHECK_FALSE(engine.IsRunning());
    }
    
    // Engine should still be in a valid state
    REQUIRE_NOTHROW(engine.Start());
    REQUIRE_NOTHROW(engine.Stop());
}