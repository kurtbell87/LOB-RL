// File: constellation-cpp/modules/market_data/include/market_data/MarketDataFactory.hpp
#pragma once

#include <memory>
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/logging/ILogger.hpp"
#include "interfaces/market_data/FeedConfigs.hpp"

using constellation::interfaces::market_data::IIngestionFeed;

namespace constellation {
namespace modules {
namespace market_data {

/**
 * @brief Factory functions for creating IIngestionFeed implementations
 *        (e.g., DBN file replay, Databento MBO feed) without exposing
 *        internal classes like DbnFileFeed or DataBentoMboFeed.
 */
class MarketDataFactory {
public:
  /**
   * @brief Create a DBN file feed that replays local .dbn files.
   * @param config A DbnFileFeedConfig struct with file path, loop forever, etc.
   * @param logger Optional logger injection; if null, an internal NullLogger is used.
   * @return A shared_ptr<IIngestionFeed> that can be started/stopped by the caller.
   */
  static std::shared_ptr<IIngestionFeed>
  CreateDbnFileFeed(
      const constellation::interfaces::market_data::DbnFileFeedConfig& config,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr
  );

  /**
   * @brief Create a DataBento MBO feed (historical or live) using the
   *        fields in DataBentoFeedConfig.
   * @param config A DataBentoFeedConfig with API key, dataset, symbols, etc.
   * @param logger Optional logger injection.
   * @return A shared_ptr<IIngestionFeed>.
   */
  static std::shared_ptr<IIngestionFeed>
  CreateDataBentoMboFeed(
      const constellation::interfaces::market_data::DataBentoFeedConfig& config,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr
  );
};

} // end namespace market_data
} // end namespace modules
} // end namespace constellation
