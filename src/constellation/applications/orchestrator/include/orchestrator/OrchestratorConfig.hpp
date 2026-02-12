#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/orderbook/IMarketBook.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"
#include "interfaces/features/IFeatureManager.hpp"
#include "interfaces/strategy/IStrategy.hpp"

namespace constellation {
namespace applications {
namespace orchestrator {

/**
 * @brief Enumerates whether the Orchestrator should create a backtest feed
 *        (e.g. from a DBN file) or a live feed (DataBento MBO), or none.
 */
enum class FeedMode {
  NoFeed = 0,
  Backtest,
  Live
};

/**
 * @brief Configuration for a backtest feed (DBN file).
 */
struct BacktestFeedSettings {
  std::string file_path;
  bool loop_forever{false};
};

/**
 * @brief Configuration for a live feed via DataBento MBO.
 */
struct LiveFeedSettings {
  std::string api_key;
  std::string dataset;
  std::vector<std::string> symbols;
  bool use_live{false};    ///< If false, it's a historical pull
  std::string start_datetime;
  std::string end_datetime;

  LiveFeedSettings()
    : api_key(""),
      dataset("GLBX.MDP3"),
      symbols({"ESH5"}),
      use_live(false),
      start_datetime("2025-01-06T14:30:00"),
      end_datetime("2025-01-06T14:31:00")
  {}
};

/**
 * @brief Holds performance metrics for the Orchestrator pipeline:
 *        how many messages were processed, how many errors occurred, etc.
 *        All fields are atomic so we can safely copy them.
 */
struct OrchestratorStats {
  std::atomic<std::uint64_t> feed_messages{0};
  std::atomic<std::uint64_t> market_updates{0};
  std::atomic<std::uint64_t> consumer_updates{0};

  std::atomic<std::uint64_t> feed_exceptions{0};
  std::atomic<std::uint64_t> market_exceptions{0};
  std::atomic<std::uint64_t> consumer_exceptions{0};

  OrchestratorStats() = default;

  OrchestratorStats(const OrchestratorStats& other) {
    feed_messages.store(other.feed_messages.load(), std::memory_order_relaxed);
    market_updates.store(other.market_updates.load(), std::memory_order_relaxed);
    consumer_updates.store(other.consumer_updates.load(), std::memory_order_relaxed);

    feed_exceptions.store(other.feed_exceptions.load(), std::memory_order_relaxed);
    market_exceptions.store(other.market_exceptions.load(), std::memory_order_relaxed);
    consumer_exceptions.store(other.consumer_exceptions.load(), std::memory_order_relaxed);
  }

  OrchestratorStats& operator=(const OrchestratorStats& other) {
    if (this != &other) {
      feed_messages.store(other.feed_messages.load(), std::memory_order_relaxed);
      market_updates.store(other.market_updates.load(), std::memory_order_relaxed);
      consumer_updates.store(other.consumer_updates.load(), std::memory_order_relaxed);

      feed_exceptions.store(other.feed_exceptions.load(), std::memory_order_relaxed);
      market_exceptions.store(other.market_exceptions.load(), std::memory_order_relaxed);
      consumer_exceptions.store(other.consumer_exceptions.load(), std::memory_order_relaxed);
    }
    return *this;
  }

  std::string ToString() const {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
      "[OrchStats] feed_msgs=%llu, market_msgs=%llu, consumer_msgs=%llu, "
      "feed_exc=%llu, market_exc=%llu, consumer_exc=%llu",
      (unsigned long long)feed_messages.load(),
      (unsigned long long)market_updates.load(),
      (unsigned long long)consumer_updates.load(),
      (unsigned long long)feed_exceptions.load(),
      (unsigned long long)market_exceptions.load(),
      (unsigned long long)consumer_exceptions.load()
    );
    return std::string(buf);
  }
};

/**
 * @brief OrchestratorConfig holds references or pointers to the main modules
 *        that the Orchestrator will manage, via their **interfaces** only.
 *        Also indicates whether to create a feed from scratch (FeedMode).
 */
struct OrchestratorConfig {
  // If feedMode != None, the Orchestrator might create a feed automatically
  FeedMode feedMode { FeedMode::NoFeed };

  BacktestFeedSettings backtestSettings;
  LiveFeedSettings liveSettings;

  // If not null, we skip feed creation logic and use this feed pointer
  std::shared_ptr<constellation::interfaces::market_data::IIngestionFeed> feed;

  // The multi-instrument aggregator that implements IMarketBook
  std::shared_ptr<constellation::interfaces::orderbook::IMarketBook> marketBook;

  // The CQRS engine for orders, implementing IOrdersEngine
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> ordersEngine;

  // Feature manager for real-time analytics
  std::shared_ptr<constellation::interfaces::features::IFeatureManager> featureManager;

  // Strategy interface
  std::shared_ptr<constellation::interfaces::strategy::IStrategy> strategy;

  // SPSC ring buffer capacities
  std::size_t ringBufferSize { 65536 };
  std::size_t ringBufferConsumerSize { 65536 };

  // Advanced error handling: if true, orchestrator stops on feed exceptions
  bool stopOnFeedException { true };

  // Enable batch consumption from Market->Consumer
  bool enableConsumerBatching { false };
  std::size_t consumerBatchSize { 10 };

  // If false, skip tracking stats
  bool trackStats { true };
};

} // end namespace orchestrator
} // end namespace applications
} // end namespace constellation
