// File: constellation-cpp/modules/market_data/include/market_data/DataBentoMboFeed.hpp
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "databento/enums.hpp"
#include "databento/historical.hpp"
#include "databento/live_threaded.hpp"
#include "databento/metadata.hpp" // so we can reference databento::Metadata
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/logging/ILogger.hpp"

using constellation::interfaces::market_data::IIngestionFeed;

namespace constellation {
namespace modules {
namespace market_data {

/**
 * @brief Internal config for DataBentoMboFeed, used only in this module.
 */
struct DataBentoFeedConfig {
  std::string api_key;
  std::string dataset;
  std::vector<std::string> symbols;
  std::string schema{"mbo"};
  bool use_live{false};
  std::string start_datetime;
  std::string end_datetime;

  // These fields were in the original code:
  databento::HistoricalGateway gateway{databento::HistoricalGateway::Bo1};
  databento::SType stype_in{databento::SType::InstrumentId};
  databento::SType stype_out{databento::SType::InstrumentId};
  std::uint64_t limit{0};

  /**
   * @brief Optional user callback for receiving the Metadata once it's available.
   */
  std::function<void(databento::Metadata&&)> metadata_callback;
};

/**
 * @brief DataBentoMboFeed is a concrete data ingestion class that
 *        streams MBO updates from Databento's historical or live environment.
 */
class DataBentoMboFeed : public IIngestionFeed {
public:
  explicit DataBentoMboFeed(
      const DataBentoFeedConfig& config,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr
  );
  ~DataBentoMboFeed() override;

  // IIngestionFeed interface
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override;
  void SubscribeMboCallback(const std::function<void(const databento::MboMsg&)>& callback) override;
  void SubscribeRecordCallback(const std::function<void(const databento::Record&)>& callback) override;
  void Start() override;
  void Stop() override;

private:
  void RunFeed();
  databento::KeepGoing OnRecord(const databento::Record& record);

  DataBentoFeedConfig config_;
  std::vector<std::function<void(const databento::MboMsg&)>> mbo_callbacks_;
  std::vector<std::function<void(const databento::Record&)>> record_callbacks_;

  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  std::unique_ptr<databento::Historical> historical_client_;
  std::unique_ptr<databento::LiveThreaded> live_client_;

  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  std::mutex start_stop_mtx_;
};

} // end namespace market_data
} // end namespace modules
} // end namespace constellation
