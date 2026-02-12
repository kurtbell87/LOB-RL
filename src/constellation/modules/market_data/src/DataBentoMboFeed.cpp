#include "market_data/DataBentoMboFeed.hpp"

#include <stdexcept>
#include <utility>

#include "databento/live.hpp"
#include "databento/exceptions.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation::modules::market_data {

DataBentoMboFeed::DataBentoMboFeed(const DataBentoFeedConfig& config,
                                   std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
  : config_(config),
    logger_( logger ? logger : std::make_shared<constellation::interfaces::logging::NullLogger>() )
{
}

constellation::interfaces::common::InterfaceVersionInfo
DataBentoMboFeed::GetVersionInfo() const noexcept {
  return {1, 0};
}

DataBentoMboFeed::~DataBentoMboFeed() {
  Stop();
}

void DataBentoMboFeed::SubscribeMboCallback(
    const std::function<void(const databento::MboMsg&)>& callback) {
  mbo_callbacks_.push_back(callback);
}

void DataBentoMboFeed::SubscribeRecordCallback(
    const std::function<void(const databento::Record&)>& callback) {
  record_callbacks_.push_back(callback);
}

void DataBentoMboFeed::Start() {
  std::unique_lock<std::mutex> lock(start_stop_mtx_);
  if (running_.exchange(true)) {
    // Already running
    return;
  }

  try {
    if (!config_.use_live) {
      databento::HistoricalBuilder builder;
      builder.SetKey(config_.api_key)
             .SetGateway(config_.gateway);
      historical_client_ = std::make_unique<databento::Historical>(builder.Build());
    } else {
      databento::LiveBuilder builder;
      builder.SetKey(config_.api_key)
             .SetDataset(config_.dataset)
             .SetSendTsOut(false);
      live_client_ = std::make_unique<databento::LiveThreaded>(builder.BuildThreaded());
    }
  } catch (const std::exception& ex) {
    logger_->Error("[DataBentoMboFeed::Start] Exception building client: %s", ex.what());
    running_.store(false);
    return;
  }

  worker_thread_ = std::thread([this] { RunFeed(); });
}

void DataBentoMboFeed::Stop() {
  std::unique_lock<std::mutex> lock(start_stop_mtx_);
  if (!running_.exchange(false)) {
    // Already stopped
    return;
  }
  try {
    if (historical_client_) {
      // historical client ends once the range is exhausted
    } else if (live_client_) {
      live_client_->BlockForStop();
    }
  } catch (const std::exception& ex) {
    logger_->Error("[DataBentoMboFeed::Stop] Exception: %s", ex.what());
  }

  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void DataBentoMboFeed::RunFeed() {
  try {
    if (historical_client_) {
      historical_client_->TimeseriesGetRange(
        config_.dataset,
        databento::DateTimeRange<std::string>{config_.start_datetime, config_.end_datetime},
        config_.symbols,
        databento::Schema::Mbo,             // or parse config_.schema
        config_.stype_in,                   
        config_.stype_out,
        config_.limit,
        [this](databento::Metadata&& md) {
          if (config_.metadata_callback) {
            config_.metadata_callback(std::move(md));
          }
        },
        [this](const databento::Record& rec) {
          return OnRecord(rec);
        }
      );
    } else if (live_client_) {
      live_client_->Start(
        [this](databento::Metadata&& md) {
          if (config_.metadata_callback) {
            config_.metadata_callback(std::move(md));
          }
        },
        [this](const databento::Record& record){
          return OnRecord(record);
        }
      );
      live_client_->BlockForStop();
    }
  } catch (const databento::Exception& ex) {
    logger_->Error("[DataBentoMboFeed::RunFeed] Databento exception: %s", ex.what());
  } catch (const std::exception& ex) {
    logger_->Error("[DataBentoMboFeed::RunFeed] std::exception: %s", ex.what());
  } catch (...) {
    logger_->Error("[DataBentoMboFeed::RunFeed] Unknown exception caught");
  }
}

databento::KeepGoing DataBentoMboFeed::OnRecord(const databento::Record& record) {
  for (auto& cb : record_callbacks_) {
    cb(record);
  }
  if (record.Holds<databento::MboMsg>()) {
    const auto& mbo = record.Get<databento::MboMsg>();
    for (auto& cb : mbo_callbacks_) {
      cb(mbo);
    }
  }
  if (!running_) {
    return databento::KeepGoing::Stop;
  }
  return databento::KeepGoing::Continue;
}

} // end namespace constellation::modules::market_data
