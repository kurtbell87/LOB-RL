#include "market_data/DbnFileFeed.hpp"
#include "interfaces/logging/NullLogger.hpp"

#include <stdexcept>
#include <utility>
#include <fstream>

namespace constellation::modules::market_data {

DbnFileFeed::DbnFileFeed(const DbnFileFeedConfig& config,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
    : config_(config),
      logger_{ logger ? logger : std::make_shared<constellation::interfaces::logging::NullLogger>() }
{ }

DbnFileFeed::~DbnFileFeed() {
  Stop();
}

constellation::interfaces::common::InterfaceVersionInfo
DbnFileFeed::GetVersionInfo() const noexcept {
  return {1, 0};
}

void DbnFileFeed::SubscribeMboCallback(
    const std::function<void(const databento::MboMsg&)>& callback) {
  mbo_callbacks_.push_back(callback);
}

void DbnFileFeed::SubscribeRecordCallback(
    const std::function<void(const databento::Record&)>& callback) {
  record_callbacks_.push_back(callback);
}

void DbnFileFeed::Start() {
  std::unique_lock<std::mutex> lock(start_stop_mtx_);

  if (running_.exchange(true)) {
    // Already running
    return;
  }
  worker_ = std::thread([this] {
    try {
      RunFileReplay();
    } catch (const std::exception& ex) {
      logger_->Error("[DbnFileFeed::Start worker] Exception: %s", ex.what());
    } catch (...) {
      logger_->Error("[DbnFileFeed::Start worker] Unknown exception thrown");
    }
  });
}

void DbnFileFeed::Stop() {
  std::unique_lock<std::mutex> lock(start_stop_mtx_);

  if (!running_.exchange(false)) {
    // already stopped
    return;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void DbnFileFeed::RunFileReplay() {
  while (running_) {
    try {
      databento::DbnFileStore store(config_.file_path);
      auto& metadata = store.GetMetadata(); // prime the store

      while (running_) {
        const databento::Record* rec = store.NextRecord();
        if (!rec) {
          break; // EOF
        }
        DispatchRecord(*rec);
      }
    } catch (const std::exception& ex) {
      logger_->Error("[DbnFileFeed::RunFileReplay] Exception reading file '%s': %s",
                     config_.file_path.c_str(), ex.what());
      break; // stop replay if file read fails
    }
    if (!config_.loop_forever) {
      break;
    }
  }
}

void DbnFileFeed::DispatchRecord(const databento::Record& record) {
  for (auto& cb : record_callbacks_) {
    cb(record);
  }
  if (record.Holds<databento::MboMsg>()) {
    const auto& mbo = record.Get<databento::MboMsg>();
    for (auto& cb : mbo_callbacks_) {
      cb(mbo);
    }
  }
}

} //namespace constellation::modules::market_data
