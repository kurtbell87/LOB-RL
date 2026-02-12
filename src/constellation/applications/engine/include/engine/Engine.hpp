#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "engine/EngineConfig.hpp"
#include "orchestrator/SpscRingBuffer.hpp"
#include "replay/BatchAggregator.hpp"
#include "databento/record.hpp"
#include "interfaces/batch/IRecordBatch.hpp"

namespace constellation {
namespace applications {
namespace engine {

/**
 * @brief A struct that represents a processed chunk to be passed to the Python strategy thread
 */
struct StrategyChunk {
  std::shared_ptr<const interfaces::batch::IRecordBatch> recordBatch;
  std::shared_ptr<const interfaces::orderbook::IMarketView> marketView;
  std::uint64_t timestamp;
  
  StrategyChunk() = default;
  
  StrategyChunk(std::shared_ptr<const interfaces::batch::IRecordBatch> rb,
                std::shared_ptr<const interfaces::orderbook::IMarketView> mv,
                std::uint64_t ts)
      : recordBatch(std::move(rb)), marketView(std::move(mv)), timestamp(ts) {}
};

/**
 * @brief A plain struct for returning the engine's current counters without atomic fields.
 */
struct EngineStatsSnapshot {
  std::uint64_t feed_msgs;
  std::uint64_t snapshot_msgs;
  std::uint64_t snapshot_msgs_applied;
  std::uint64_t snapshot_msgs_skipped;
  std::uint64_t files_processed;
  std::uint64_t current_file_index;
  std::uint64_t aggregator_chunks;
  std::uint64_t aggregator_exceptions;
  std::uint64_t python_chunks;
};

/**
 * @class Engine
 * @brief Phase 3: Multi-file processing with snapshot warm-up management
 */
class Engine {
public:
  Engine();
  ~Engine();

  /**
   * @brief Initialize the engine with the given configuration.
   */
  void Initialize(const EngineConfig& config);

  /**
   * @brief Start the engine’s four threads.
   */
  void Start();

  /**
   * @brief Stop the engine and join all threads.
   */
  void Stop();

  /**
   * @brief Returns true if engine is running.
   */
  bool IsRunning() const;

  /**
   * @brief Retrieves a copy of the engine's statistics (plain struct).
   */
  EngineStatsSnapshot GetStats() const;

private:
  void CreateFeedIfNeeded();
  void SwitchToNextFile();  // Phase 3: Process multiple files sequentially
  
  void FeedThreadLoop();
  void MarketThreadLoop();
  void AggregatorThreadLoop();
  void PythonStrategyThreadLoop();

  bool IsSnapshotMessage(const databento::MboMsg& msg) const;
  bool IsLastSnapshot(const databento::MboMsg& msg) const;
  bool ShouldProcessSnapshot(const databento::MboMsg& msg) const;  // Phase 3: Based on config.snapshotMode

private:
  EngineConfig config_;

  // Modules
  std::shared_ptr<constellation::interfaces::market_data::IIngestionFeed> feed_;
  std::shared_ptr<constellation::interfaces::orderbook::IMarketBook>      marketBook_;
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>       ordersEngine_;
  std::shared_ptr<constellation::interfaces::features::IFeatureManager>   featureManager_;
  std::shared_ptr<constellation::applications::replay::BatchAggregator>   aggregator_;
  std::shared_ptr<constellation::interfaces::strategy::IStrategy>         strategy_;

  // ring buffers
  std::unique_ptr<orchestrator::SpscRingBuffer<databento::MboMsg>> ringBufferFeed_;
  std::unique_ptr<orchestrator::SpscRingBuffer<databento::MboMsg>> ringBufferAggregator_;
  std::unique_ptr<orchestrator::SpscRingBuffer<StrategyChunk>> ringBufferStrategy_;

  // Threads
  std::thread feedThread_;
  std::thread marketThread_;
  std::thread aggregatorThread_;
  std::thread pythonStrategyThread_;

  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> isRunning_{false};

  /**
   * @brief Private struct with atomic counters that can't be copied.
   */
  struct EngineStats {
    std::atomic<std::uint64_t> feed_msgs{0};
    std::atomic<std::uint64_t> snapshot_msgs{0};
    std::atomic<std::uint64_t> snapshot_msgs_applied{0};
    std::atomic<std::uint64_t> snapshot_msgs_skipped{0};
    std::atomic<std::uint64_t> files_processed{0};
    std::atomic<std::uint64_t> current_file_index{0};
    std::atomic<std::uint64_t> aggregator_chunks{0};
    std::atomic<std::uint64_t> aggregator_exceptions{0};
    std::atomic<std::uint64_t> python_chunks{0};
    std::atomic<std::uint64_t> python_exceptions{0};
  };

  mutable std::mutex statsMutex_;
  EngineStats stats_;

  // Phase 3: Multi-file processing state
  std::mutex filesMutex_;
  std::condition_variable filesCV_;
  bool currentFileDone_{false};
  bool isFirstFile_{true};

  // Tracks which instruments have completed their MBO snapshot
  std::unordered_map<std::uint32_t, bool> snapshotComplete_;
};

} // end namespace engine
} // end namespace applications
} // end namespace constellation
