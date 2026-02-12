#include <catch2/catch_test_macros.hpp>
#include "strategy/SampleBatchStrategy.hpp"
#include "strategy/StrategyFactory.hpp"
#include "interfaces/batch/RecordBatch.hpp"

TEST_CASE("SampleBatchStrategy basic usage", "[strategy]") {
  // Create the strategy object
  constellation::modules::strategy::SampleBatchStrategy strat;

  // Make a dummy RecordBatch to pass into OnDataChunk
  constellation::interfaces::batch::RecordBatch dummyBatch;
  
  // We can call OnDataChunk(...) with no real data
  REQUIRE_NOTHROW(strat.OnDataChunk(dummyBatch, nullptr));

  // Finally, shutting down should also succeed
  REQUIRE_NOTHROW(strat.Shutdown());
}

// Additional test for the factory method
TEST_CASE("StrategyFactory creates proper strategy instances", "[strategy][factory]") {
  // Create strategy with default config
  auto strategy = constellation::modules::strategy::CreateSampleBatchStrategy();
  
  // Verify it's a valid instance
  REQUIRE(strategy != nullptr);
  
  // Test basic functionality
  constellation::interfaces::batch::RecordBatch batch;
  REQUIRE_NOTHROW(strategy->OnDataChunk(batch, nullptr));
  REQUIRE_NOTHROW(strategy->Shutdown());
  
  // Create strategy with custom config
  constellation::modules::strategy::SampleBatchStrategyConfig config;
  auto strategy2 = constellation::modules::strategy::CreateSampleBatchStrategy(config);
  
  // Verify it's also a valid instance
  REQUIRE(strategy2 != nullptr);
}