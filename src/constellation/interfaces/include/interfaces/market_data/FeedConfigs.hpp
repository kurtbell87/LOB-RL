// File: constellation-cpp/interfaces/include/interfaces/market_data/FeedConfigs.hpp
#pragma once

#include <string>
#include <vector>
#include <functional>
#include "databento/enums.hpp"
#include "databento/dbn.hpp"  // so we can reference databento::Metadata

namespace constellation {
namespace interfaces {
namespace market_data {

/**
 * @brief Configuration for replaying from a local DBN file.
 */
struct DbnFileFeedConfig {
  std::string file_path;
  bool loop_forever{false};
};

/**
 * @brief Configuration for connecting to the DataBento MBO feed
 *        (either historical or live).
 */
struct DataBentoFeedConfig {
  std::string api_key;
  std::string dataset;
  std::vector<std::string> symbols;
  std::string schema{"mbo"};
  bool use_live{false};
  std::string start_datetime{"2025-01-06T14:30:00"};
  std::string end_datetime{"2025-01-06T14:31:00"};

  // The advanced fields you need:
  databento::HistoricalGateway gateway {databento::HistoricalGateway::Bo1};
  databento::SType stype_in {databento::SType::RawSymbol};
  databento::SType stype_out{databento::SType::InstrumentId};
  std::uint64_t limit{0};

  // Optional callback for receiving metadata once it's retrieved
  std::function<void(databento::Metadata&&)> metadata_callback;
};

} // end namespace market_data
} // end namespace interfaces
} // end namespace constellation
