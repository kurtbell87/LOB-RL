// File: constellation-cpp/modules/market_data/include/market_data/DbnFileFeed.hpp
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "databento/dbn_file_store.hpp"
#include "interfaces/market_data/IIngestionFeed.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation {
namespace modules {
namespace market_data {

using constellation::interfaces::market_data::IIngestionFeed;
/**
 * @brief Internal config for DbnFileFeed, used only inside this module.
 */
struct DbnFileFeedConfig {
  std::string file_path;
  bool loop_forever{false};
};

/**
 * @brief A local DBN file feed that replays records from disk,
 *        implementing IIngestionFeed. Not exposed publicly.
 */
class DbnFileFeed : public IIngestionFeed {
public:
  explicit DbnFileFeed(
      const DbnFileFeedConfig& config,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr
  );
  ~DbnFileFeed() override;

  // IIngestionFeed interface
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override;
  void SubscribeMboCallback(const std::function<void(const databento::MboMsg&)>& callback) override;
  void SubscribeRecordCallback(const std::function<void(const databento::Record&)>& callback) override;
  void Start() override;
  void Stop() override;

private:
  void RunFileReplay();
  void DispatchRecord(const databento::Record& record);

  DbnFileFeedConfig config_;
  std::vector<std::function<void(const databento::MboMsg&)>> mbo_callbacks_;
  std::vector<std::function<void(const databento::Record&)>> record_callbacks_;

  std::atomic<bool> running_{false};
  std::thread worker_;

  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  std::mutex start_stop_mtx_;
};

} // end namespace market_data
} // end namespace modules
} // end namespace constellation
