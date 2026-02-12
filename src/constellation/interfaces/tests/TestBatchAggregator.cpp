#include <catch2/catch_test_macros.hpp>
#include "interfaces/batch/BatchAggregator.hpp"
#include "databento/record.hpp"

// A simple test strategy to confirm we can receive chunk callbacks
class TestChunkStrategy : public constellation::interfaces::batch::IChunkStrategy {
public:
  void OnDataChunk(const std::vector<databento::MboMsg>& mbo_chunk) override {
    lastChunkSize = mbo_chunk.size();
  }
  std::size_t lastChunkSize{0};
};

TEST_CASE("StubBatchAggregator basic usage", "[batch]") {
  using namespace constellation::interfaces::batch;

  // Create aggregator
  StubBatchAggregator aggregator;
  REQUIRE_FALSE(aggregator.IsRunning());

  // Initialize
  BatchAggregatorConfig cfg;
  cfg.name = "TestAggregator";
  aggregator.Initialize(cfg);
  REQUIRE(aggregator.IsRunning());

  // Set a test strategy
  auto strategy = std::make_shared<TestChunkStrategy>();
  aggregator.SetStrategy(strategy);

  // Ingest a chunk
  std::vector<databento::MboMsg> chunk;
  chunk.emplace_back(); // default MboMsg
  chunk.emplace_back(); // another
  aggregator.BatchOnMboUpdate(chunk);

  // Confirm the aggregator and strategy saw it
  REQUIRE(aggregator.GetTotalMessagesIngested() == 2);
  REQUIRE(strategy->lastChunkSize == 2);

  // Stop
  aggregator.Stop();
  REQUIRE_FALSE(aggregator.IsRunning());

  // Attempting ingestion after stop should throw
  REQUIRE_THROWS_AS(aggregator.BatchOnMboUpdate(chunk), std::runtime_error);
}
