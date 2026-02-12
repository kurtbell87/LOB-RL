#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/features/IFeatureManager.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"
#include "interfaces/orderbook/IMarketBook.hpp"
#include "interfaces/strategy/IStrategy.hpp"
#include "replay/BatchAggregator.hpp"

namespace constellation {
namespace applications {
namespace engine {

/**
 * @brief Enumeration indicating feed mode:
 *  - NoFeed : no feed created (user might supply a custom feed pointer)
 *  - Backtest : DBN file-based replay
 *  - Live : DataBento live/historical feed
 */
enum class EngineFeedMode {
  NoFeed = 0,
  Backtest,
  Live
};

/**
 * @brief Configuration struct for the unified Engine, merging replay + orchestrator logic.
 */
struct EngineConfig {
  // Which feed mode to use:
  EngineFeedMode feedMode{EngineFeedMode::NoFeed};

  // If Backtest, we can list multiple DBN files to read in sequence
  std::vector<std::string> dbnFilePaths;

  // Live feed settings (if feedMode=Live)
  std::string liveApiKey;
  std::string liveDataset;
  std::vector<std::string> liveSymbols;
  bool liveUseLive{false};  // if false => historical pull
  std::string liveStart;
  std::string liveEnd;

  // Aggregator chunk config
  std::size_t chunkSize{50000}; // default chunk size
  bool trackStats{true};

  // Strategy thread config
  std::size_t strategyRingBufferSize{128}; // How many chunks can be queued for Python
  bool blockAggregatorWhenStrategyBufferFull{false}; // Whether to throttle aggregator if Python lags

  // Snapshot handling settings (Phase 3)
  // If true, we handle MBO snapshots by applying them directly to the MarketBook
  // in the market thread, skipping aggregator chunk logic for them.
  // Once 'LAST|SNAPSHOT' for an instrument is seen, subsequent messages flow
  // into normal aggregator chunking for that instrument.
  bool handleSnapshots{true};
  
  // Controls how to handle snapshots in multi-file processing
  enum class SnapshotHandlingMode {
    ProcessAllSnapshots,     // Process snapshots in all files (may duplicate)
    ProcessFirstFileOnly,    // Only process snapshots from the first file
    ProcessPerInstrument,    // Process first snapshot encountered per instrument
    SkipAllSnapshots         // Skip all snapshots (assume pre-loaded LOB)
  };
  
  // How to handle snapshots when loading multiple DBN files
  SnapshotHandlingMode snapshotMode{SnapshotHandlingMode::ProcessPerInstrument};

  // Shared pointers to custom modules (optional); if null, we create standard ones
  std::shared_ptr<constellation::interfaces::market_data::IIngestionFeed> feed;
  std::shared_ptr<constellation::interfaces::orderbook::IMarketBook> marketBook;
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> ordersEngine;
  std::shared_ptr<constellation::interfaces::features::IFeatureManager> featureManager;
  std::shared_ptr<constellation::interfaces::strategy::IStrategy> strategy;
};

} // end namespace engine
} // end namespace applications
} // end namespace constellation
