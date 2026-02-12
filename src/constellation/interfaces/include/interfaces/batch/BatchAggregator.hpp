#pragma once

/**
 * @file BatchAggregator.hpp
 *
 * @brief Defines interfaces for high-performance chunk-based ingestion of MBO messages,
 *        ensuring single-writer concurrency and no look-ahead ordering.
 *
 * Phase A1 - Batch Engine Interface & Docs
 *
 * Guidelines & Key Points:
 *  1. Single-writer concurrency: Only one writer/ingestion thread modifies aggregator state
 *     at a time. Any read/strategy calls occur after each chunk is fully applied.
 *  2. No look-ahead: All MBO messages in a chunk are strictly time-ordered up to T. The aggregator
 *     must finalize these updates before any further chunk from time > T is processed.
 *  3. Batch bridging: The aggregator can call a Python or C++ strategy via a chunk-based interface,
 *     avoiding per-message overhead.
 */

#include <cstdint>
#include <vector>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "databento/record.hpp"  // So we can reference databento::MboMsg

namespace constellation {
namespace interfaces {
namespace batch {

/**
 * @brief IChunkStrategy is a placeholder interface representing
 *        how a Python or C++ strategy would receive Batch chunk callbacks
 *        (e.g., for ML or RL).
 *
 * In Phase A1, this is minimal: a single method for receiving
 * a vector of MBO messages that have just been ingested.
 */
class IChunkStrategy {
public:
  virtual ~IChunkStrategy() = default;

  /**
   * @brief Called after a full chunk of MBO messages has been applied.
   *        Strategies can analyze these messages or aggregator state.
   */
  virtual void OnDataChunk(const std::vector<databento::MboMsg>& mbo_chunk) = 0;
};

/**
 * @brief Configuration struct for Batch Aggregator (Phase A1).
 *        Future expansions could add chunk-size hints, concurrency modes, etc.
 */
struct BatchAggregatorConfig {
  /**
   * @brief Optionally set an identifier or concurrency flag. For demonstration only.
   */
  std::string name;
  bool enableLogs{true};
};

/**
 * @class IBatchAggregator
 * @brief Interface for a high-performance aggregator ingesting MBO messages in chunk fashion.
 *
 * This interface enforces:
 *  - Single-writer concurrency
 *  - Chronological, no look-ahead updates
 *  - Optional bridging to a strategy (IChunkStrategy)
 */
class IBatchAggregator {
public:
  virtual ~IBatchAggregator() = default;

  /**
   * @brief One-time initialization. Must be called before any chunk ingestion.
   * @param config The aggregator settings.
   * @throws std::runtime_error If already initialized or config is invalid.
   */
  virtual void Initialize(const BatchAggregatorConfig& config) = 0;

  /**
   * @brief Main Batch ingestion method. Applies all MBO messages in 'mbo_chunk' to the aggregator state.
   *        Must be called strictly in chronological order, ensuring no overlap or look-ahead.
   * @param mbo_chunk The vector of MBO messages in ascending time. May be up to tens of thousands of messages.
   * @throws std::runtime_error If aggregator not initialized or aggregator already stopped.
   */
  virtual void BatchOnMboUpdate(const std::vector<databento::MboMsg>& mbo_chunk) = 0;

  /**
   * @brief Set or replace the strategy that will receive Batch chunk callbacks.
   * @param strategy A shared_ptr to an IChunkStrategy implementation (could be Python or C++).
   *        If null, aggregator calls no strategy.
   */
  virtual void SetStrategy(const std::shared_ptr<IChunkStrategy>& strategy) = 0;

  /**
   * @brief Stop or finalize aggregator. After calling Stop, no more ingestion may occur.
   *        Safe to call multiple times.
   */
  virtual void Stop() = 0;

  /**
   * @brief Check if aggregator is active (initialized and not stopped).
   */
  virtual bool IsRunning() const = 0;
};

/**
 * @class StubBatchAggregator
 * @brief A minimal, concrete Batch aggregator that demonstrates compliance with IBatchAggregator.
 *        It does not maintain a real order book, but it fully implements all interface methods.
 *
 * Phase A1: This is a placeholder or "stub" aggregator that can be extended in subsequent phases.
 */
class StubBatchAggregator final : public IBatchAggregator {
public:
  StubBatchAggregator()
    : running_{false} {}

  ~StubBatchAggregator() override {
    // If needed, automatic cleanup
  }

  void Initialize(const BatchAggregatorConfig& config) override {
    std::lock_guard<std::mutex> lock(mtx_);
    if (running_) {
      throw std::runtime_error("StubBatchAggregator: Already initialized");
    }
    config_ = config;
    running_ = true;
  }

  void BatchOnMboUpdate(const std::vector<databento::MboMsg>& mbo_chunk) override {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!running_) {
      throw std::runtime_error("StubBatchAggregator: Not running or already stopped");
    }
    // Demonstration: record how many messages we ingested.
    total_messages_ingested_ += mbo_chunk.size();

    // If strategy is set, call OnDataChunk
    if (strategy_) {
      strategy_->OnDataChunk(mbo_chunk);
    }
  }

  void SetStrategy(const std::shared_ptr<IChunkStrategy>& strategy) override {
    std::lock_guard<std::mutex> lock(mtx_);
    strategy_ = strategy;
  }

  void Stop() override {
    std::lock_guard<std::mutex> lock(mtx_);
    running_ = false;
  }

  bool IsRunning() const override {
    std::lock_guard<std::mutex> lock(mtx_);
    return running_;
  }

  // Extra accessor for testing or debugging:
  std::uint64_t GetTotalMessagesIngested() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return total_messages_ingested_;
  }

private:
  mutable std::mutex mtx_;
  bool running_;
  BatchAggregatorConfig config_;
  std::shared_ptr<IChunkStrategy> strategy_;

  // Example aggregator state:
  std::uint64_t total_messages_ingested_{0};
};

} // end namespace batch
} // end namespace interfaces
} // end namespace constellation
