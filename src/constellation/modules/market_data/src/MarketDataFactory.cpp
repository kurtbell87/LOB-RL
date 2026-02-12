// File: constellation-cpp/modules/market_data/src/MarketDataFactory.cpp
#include "market_data/MarketDataFactory.hpp"
#include "market_data/DbnFileFeed.hpp"        // internal
#include "market_data/DataBentoMboFeed.hpp"   // internal
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation {
namespace modules {
namespace market_data {

std::shared_ptr<IIngestionFeed>
MarketDataFactory::CreateDbnFileFeed(
    const constellation::interfaces::market_data::DbnFileFeedConfig& config,
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger
) {
  // Validate config: require a non-empty file_path
  if (config.file_path.empty()) {
    return nullptr;
  }
  
  // Convert the public config struct -> internal DbnFileFeedConfig
  DbnFileFeedConfig internalCfg;
  internalCfg.file_path    = config.file_path;
  internalCfg.loop_forever = config.loop_forever;

  auto feedImpl = std::make_shared<DbnFileFeed>(internalCfg, logger);
  return std::static_pointer_cast<IIngestionFeed>(feedImpl);
}

std::shared_ptr<IIngestionFeed>
MarketDataFactory::CreateDataBentoMboFeed(
    const constellation::interfaces::market_data::DataBentoFeedConfig& config,
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger
) {
  // If you have an "internal" config:
  DataBentoFeedConfig internal;
  internal.api_key        = config.api_key;
  internal.dataset        = config.dataset;
  internal.symbols        = config.symbols;
  internal.schema         = config.schema;
  internal.use_live       = config.use_live;
  internal.start_datetime = config.start_datetime;
  internal.end_datetime   = config.end_datetime;

  // ensure you also copy advanced fields:
  internal.gateway  = config.gateway;
  internal.stype_in = config.stype_in;
  internal.stype_out= config.stype_out;
  internal.limit    = config.limit;
  internal.metadata_callback = config.metadata_callback;

  auto feedImpl = std::make_shared<DataBentoMboFeed>(internal, logger);
  return std::static_pointer_cast<IIngestionFeed>(feedImpl);
}


} // end namespace market_data
} // end namespace modules
} // end namespace constellation
