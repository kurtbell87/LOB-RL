#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <stdexcept>

#include "interfaces/logging/ILogger.hpp"
#include "orderbook/MarketBook.hpp"
#include "interfaces/batch/RecordBatch.hpp"
#include "interfaces/strategy/IStrategy.hpp"
#include "databento/record.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"

namespace constellation {
namespace applications {
namespace replay {

/**
 * @brief Configuration for Batch aggregator:
 *   - batch_size (pure count-based chunk by default)
 *   - memory_factor_limit
 *   - etc.
 *
 * **New in Sprint 3**: We add a flexible "event boundary" for chunks, e.g.:
 *   - boundary_event_type = "trade"
 *   - boundary_event_count = 2000
 *   will forcibly end a chunk whenever 2000 "trade" events are encountered.
 */
struct BatchAggregatorConfig {
  std::uint32_t batch_size{50000};  ///< typical chunk size
  std::string symbology_file;
  bool enable_logging{true};

  std::shared_ptr<constellation::interfaces::logging::ILogger> logger{nullptr};

  bool release_gil_during_aggregation{false};

  std::size_t memory_factor_limit{3};

  /**
   * @brief If true, aggregator forcibly ends a chunk whenever a specific instrument
   *        sees boundary_instrument_trades "Trade" or "Fill" messages. (Day1–2 concept)
   */
  bool enable_instrument_boundary{false};
  std::uint32_t boundary_instrument_id{0};
  std::uint64_t boundary_instrument_trades{0};

  /**
   * @brief **New in Sprint 3**: A more general "event boundary" for chunking.
   *   If enable_event_count_boundary == true, we watch boundary_event_type
   *   (e.g. "trade") and forcibly end a chunk every boundary_event_count occurrences.
   */
  bool enable_event_count_boundary{false};
  std::string boundary_event_type{"mbo"};  ///< e.g. "trade", "fill", "mbo", ...
  std::uint64_t boundary_event_count{0};

};

/**
 * @class BatchAggregatorStats
 * @brief Holds aggregator-level stats: total records, total MBO, total microseconds.
 */
class BatchAggregatorStats {
public:
  std::atomic<std::uint64_t> total_records{0};
  std::atomic<std::uint64_t> total_mbo_messages{0};
  std::atomic<std::uint64_t> total_microseconds{0};

  BatchAggregatorStats() = default;
  BatchAggregatorStats(const BatchAggregatorStats& other) {
    total_records.store(other.total_records.load());
    total_mbo_messages.store(other.total_mbo_messages.load());
    total_microseconds.store(other.total_microseconds.load());
  }
  BatchAggregatorStats& operator=(const BatchAggregatorStats& other) {
    if (this != &other) {
      total_records.store(other.total_records.load());
      total_mbo_messages.store(other.total_mbo_messages.load());
      total_microseconds.store(other.total_microseconds.load());
    }
    return *this;
  }
  void Reset() {
    total_records.store(0);
    total_mbo_messages.store(0);
    total_microseconds.store(0);
  }
};

/**
 * @class BatchAggregator
 * @brief Processes DBN data in large "chunks," updating a MarketBook, possibly
 *        calling an IOrdersEngine, and an IStrategy. Also can forcibly end a chunk
 *        based on config (fixed size, or certain # of trades, etc.).
 */
class BatchAggregator {
public:
  BatchAggregator();
  ~BatchAggregator();

  /**
   * @brief One-time initialization with config. Must be called before usage.
   */
  void Initialize(const BatchAggregatorConfig& config);

  /**
   * @brief Process multiple DBN files in chunk mode.
   */
  void ProcessFiles(const std::vector<std::string>& dbn_files);

  /**
   * @brief Process a single DBN file.
   */
  void ProcessSingleFile(const std::string& file_path);

  /**
   * @brief Return the aggregator's MarketBook for advanced queries.
   */
  std::shared_ptr<constellation::modules::orderbook::MarketBook> GetMarketBook() const;

  /**
   * @brief Retrieve aggregator stats.
   */
  BatchAggregatorStats GetStats() const;

  /**
   * @brief Reset aggregator stats, also resets lastChunkMaxTimestamp_.
   */
  void ResetStats();

  /**
   * @brief Access the recordBatch from the last chunk (thread safe).
   */
  const constellation::interfaces::batch::RecordBatch& GetLastRecordBatch() const;

  /**
   * @brief Optionally set a chunk-based strategy (IStrategy) pointer.
   */
  void SetStrategy(const std::shared_ptr<constellation::interfaces::strategy::IStrategy>& strategy);

  /**
   * @brief Testing hook to pass a batch of DBN records directly, skipping file reading.
   *        Forces a single chunk distribution.
   */
  void TestDistributeBatch(const std::vector<const databento::Record*>& batch);

  /**
   * @brief Associate an orders engine for partial fill simulation. If set,
   *        aggregator calls ordersEngine->OnMarketViewUpdate(...) after updating
   *        the MarketBook on each chunk, then calls the strategy, etc.
   */
  void SetOrdersEngine(const std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>& ordersEngine);

private:
  /**
   * @brief Internal method to process one file from start to EOF, chunking as needed.
   */
  void ProcessOneFileInternal(const std::string& file_path);

  /**
   * @brief Sort chunk by ascending timestamp, discard out-of-order, fill recordBatch_,
   *        call MarketBook->BatchOnMboUpdate, then call strategy, etc.
   */
  void DistributeBatch(const std::vector<const databento::Record*>& batch);

  /**
   * @brief Check if recordBatch_ soared beyond memory_factor_limit × batch_size. 
   *        If so, throw runtime_error to avoid memory blowups.
   */
  void EnforceMemoryConstraint();

  /**
   * @brief Possibly release the Python GIL if config_.release_gil_during_aggregation == true.
   */
  void* ReleaseGILIfConfigured();

  /**
   * @brief Reacquire Python GIL if we previously saved it.
   */
  void ReacquireGILIfNeeded(void* gil_state);

private:
  mutable std::mutex init_mutex_;
  bool initialized_{false};

  BatchAggregatorConfig config_;
  BatchAggregatorStats  stats_;

  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;
  std::shared_ptr<constellation::modules::orderbook::MarketBook> marketBook_;

  constellation::interfaces::batch::RecordBatch recordBatch_;
  std::shared_ptr<constellation::interfaces::strategy::IStrategy> strategy_;
  std::uint64_t lastChunkMaxTimestamp_{0};

  // orders engine for partial fill simulation
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> ordersEngine_;
};

} // namespace replay
} // namespace applications
} // namespace constellation
