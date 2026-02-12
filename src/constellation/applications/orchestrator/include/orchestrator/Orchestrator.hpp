#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <stdexcept>
#include "orchestrator/OrchestratorConfig.hpp"
#include "orchestrator/SpscRingBuffer.hpp"
#include "databento/record.hpp"

namespace constellation {
namespace applications {
namespace orchestrator {

/**
 * @brief Orchestrator previously included per-message strategy logic. In Phase D1,
 *        we remove old references to the per-message IStrategy calls.
 *
 *        It still manages feed, ring buffers, and a multi-thread pipeline, but
 *        it no longer calls strategy->OnDataUpdate(...) per message.
 *        Batch chunk-based strategies are handled by the new aggregator approach.
 */
class Orchestrator {
public:
  Orchestrator();
  ~Orchestrator();

  /**
   * @brief Initialize with a given config. Possibly create a feed if needed.
   */
  void Initialize(const OrchestratorConfig& config);

  /**
   * @brief Start the orchestrator pipeline: feed thread, market-updater thread, consumer thread.
   */
  void Start();

  /**
   * @brief Stop the orchestrator gracefully. Safe to call multiple times.
   */
  void Stop();

  /**
   * @brief Same as Stop(), kept for forward compatibility.
   */
  void Shutdown();

  /**
   * @brief Check if the orchestrator is running.
   */
  bool IsRunning() const;

  /**
   * @brief Returns a copy of stats if trackStats == true; otherwise zeroed.
   */
  OrchestratorStats GetStats() const;

  /**
   * @brief Phase C addition: Access the underlying orders engine for Python bridging.
   *        May return nullptr if config_.ordersEngine is not set.
   */
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> GetOrdersEngine() const;

  /**
   * @brief Phase C addition: Access the underlying market view for reading best quotes, volume, etc.
   *        May return nullptr if config_.marketBook is not set.
   */
  std::shared_ptr<constellation::interfaces::orderbook::IMarketView> GetMarketView() const;

private:
  void FeedThreadLoop();
  void MarketUpdateThreadLoop();
  void ConsumerThreadLoop();
  void CreateFeedIfNeeded();
  void HandleFeedException(const std::exception& ex);

private:
  OrchestratorConfig config_;

  // SPSC ring buffers for feed->market and market->consumer
  std::unique_ptr<SpscRingBuffer<databento::MboMsg>> ringBuffer_;
  std::unique_ptr<SpscRingBuffer<databento::MboMsg>> ringBufferConsumer_;

  // Threads
  std::thread feedThread_;
  std::thread marketThread_;
  std::thread consumerThread_;

  // Flags
  std::atomic<bool> stopRequested_{false};
  std::atomic<bool> isRunning_{false};

  // Stats
  mutable std::mutex statsMutex_;
  OrchestratorStats stats_;
};

} // end namespace orchestrator
} // end namespace applications
} // end namespace constellation
