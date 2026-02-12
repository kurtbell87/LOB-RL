#include "orchestrator/Orchestrator.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <stdexcept>

// Include each module's *factory* so we can create them if needed
#include "market_data/MarketDataFactory.hpp"
#include "orderbook/OrderBookFactory.hpp"
#include "features/FeatureFactory.hpp"
#include "orders/OrdersFactory.hpp"
#include "strategy/StrategyFactory.hpp"

namespace constellation {
namespace applications {
namespace orchestrator {

Orchestrator::Orchestrator()
  : stopRequested_(false),
    isRunning_(false)
{
}

Orchestrator::~Orchestrator() {
  if (isRunning_.load()) {
    Stop();
  }
}

void Orchestrator::Initialize(const OrchestratorConfig& config) {
  config_ = config;

  // Possibly create feed if feed is null but feedMode != None
  CreateFeedIfNeeded();

  // Create ring buffers
  {
    std::size_t capacityA = (config_.ringBufferSize > 0) ? config_.ringBufferSize : 65536;
    ringBuffer_ = std::make_unique<SpscRingBuffer<databento::MboMsg>>(capacityA);
  }
  {
    std::size_t capacityB = (config_.ringBufferConsumerSize > 0) ? config_.ringBufferConsumerSize : 65536;
    ringBufferConsumer_ = std::make_unique<SpscRingBuffer<databento::MboMsg>>(capacityB);
  }
}

void Orchestrator::CreateFeedIfNeeded() {
  using namespace constellation::modules::market_data;

  if (!config_.feed && config_.feedMode != FeedMode::NoFeed) {
    if (config_.feedMode == FeedMode::Backtest) {
      // Create a DBN File feed
      constellation::interfaces::market_data::DbnFileFeedConfig fcfg;
      fcfg.file_path = config_.backtestSettings.file_path;
      fcfg.loop_forever = config_.backtestSettings.loop_forever;

      auto feedPtr = MarketDataFactory::CreateDbnFileFeed(fcfg);
      config_.feed = feedPtr;
    } else if (config_.feedMode == FeedMode::Live) {
      constellation::interfaces::market_data::DataBentoFeedConfig lcfg;
      lcfg.api_key        = config_.liveSettings.api_key;
      lcfg.dataset        = config_.liveSettings.dataset;
      lcfg.symbols        = config_.liveSettings.symbols;
      lcfg.use_live       = config_.liveSettings.use_live;
      lcfg.start_datetime = config_.liveSettings.start_datetime;
      lcfg.end_datetime   = config_.liveSettings.end_datetime;

      auto feedPtr = MarketDataFactory::CreateDataBentoMboFeed(lcfg);
      config_.feed = feedPtr;
    }
  }
}

void Orchestrator::Start() {
  if (isRunning_.exchange(true)) {
    return; // already started
  }
  stopRequested_.store(false);

  // Feed thread
  feedThread_ = std::thread(&Orchestrator::FeedThreadLoop, this);
  // Market update thread
  marketThread_ = std::thread(&Orchestrator::MarketUpdateThreadLoop, this);
  // Consumer thread
  consumerThread_ = std::thread(&Orchestrator::ConsumerThreadLoop, this);
}

void Orchestrator::Stop() {
  if (!isRunning_.exchange(false)) {
    return; // already stopped
  }
  stopRequested_.store(true);

  // If feed was created or provided
  if (config_.feed) {
    config_.feed->Stop();
  }

  // Join all threads
  if (feedThread_.joinable()) {
    feedThread_.join();
  }
  if (marketThread_.joinable()) {
    marketThread_.join();
  }
  if (consumerThread_.joinable()) {
    consumerThread_.join();
  }
  // Removed old references to strategy usage. Batch chunk-based calls are done in Batch aggregator now.
}

void Orchestrator::Shutdown() {
  Stop();
}

bool Orchestrator::IsRunning() const {
  return isRunning_.load();
}

OrchestratorStats Orchestrator::GetStats() const {
  if (!config_.trackStats) {
    return OrchestratorStats{};
  }
  std::lock_guard<std::mutex> lock(statsMutex_);
  return stats_;
}

void Orchestrator::FeedThreadLoop() {
  if (!config_.feed) {
    // No feed => just exit
    return;
  }
  // Subscribe to MBO callback
  config_.feed->SubscribeMboCallback([this](const databento::MboMsg& msg){
    if (stopRequested_.load()) {
      return;
    }
    while (!stopRequested_.load()) {
      bool ok = ringBuffer_->TryPush(msg);
      if (ok) {
        if (config_.trackStats) {
          stats_.feed_messages.fetch_add(1, std::memory_order_relaxed);
        }
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });

  // Start feed (may throw)
  try {
    config_.feed->Start();
  } catch (const std::exception& ex) {
    if (config_.trackStats) {
      stats_.feed_exceptions.fetch_add(1, std::memory_order_relaxed);
    }
    HandleFeedException(ex);
  }
}

void Orchestrator::HandleFeedException(const std::exception& ex) {
  std::cerr << "[ERROR] FeedThreadLoop exception: " << ex.what() << std::endl;
  if (config_.stopOnFeedException) {
    stopRequested_.store(true);
  }
}

void Orchestrator::MarketUpdateThreadLoop() {
  auto marketBook = config_.marketBook;
  if (!marketBook) {
    return;
  }
  while (!stopRequested_.load()) {
    try {
      databento::MboMsg msg;
      if (ringBuffer_->TryPop(msg)) {
        marketBook->OnMboUpdate(msg);
        if (config_.trackStats) {
          stats_.market_updates.fetch_add(1, std::memory_order_relaxed);
        }
        // push same msg to consumer ring
        while (!stopRequested_.load()) {
          bool ok = ringBufferConsumer_->TryPush(msg);
          if (ok) break;
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    } catch (const std::exception& ex) {
      if (config_.trackStats) {
        stats_.market_exceptions.fetch_add(1, std::memory_order_relaxed);
      }
      std::cerr << "[ERROR] MarketUpdateThreadLoop exception: " << ex.what() << std::endl;
      stopRequested_.store(true);
    }
  }
}

void Orchestrator::ConsumerThreadLoop() {
  auto marketBook = config_.marketBook;
  if (!marketBook) {
    return;
  }
  auto ordersEngine   = config_.ordersEngine;
  auto featureManager = config_.featureManager;
  // Old strategy calls removed. Batch aggregator now handles chunk-based calls.

  std::vector<databento::MboMsg> batch;
  batch.reserve((config_.consumerBatchSize > 0) ? config_.consumerBatchSize : 1);

  while (!stopRequested_.load()) {
    try {
      databento::MboMsg msg;
      if (!ringBufferConsumer_->TryPop(msg)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      // If batching is enabled
      if (config_.enableConsumerBatching) {
        batch.clear();
        batch.push_back(msg);
        for (std::size_t i = 1; i < config_.consumerBatchSize; ++i) {
          databento::MboMsg nextMsg;
          if (!ringBufferConsumer_->TryPop(nextMsg)) {
            break;
          }
          batch.push_back(nextMsg);
        }
        if (featureManager) {
          featureManager->OnDataUpdate(*marketBook, marketBook.get());
        }
        if (ordersEngine) {
          ordersEngine->OnMarketViewUpdate(marketBook.get());
        }
        if (config_.trackStats) {
          stats_.consumer_updates.fetch_add(batch.size(), std::memory_order_relaxed);
        }
      } else {
        // Per-message approach (still no Batch strategy calls)
        if (featureManager) {
          featureManager->OnDataUpdate(*marketBook, marketBook.get());
        }
        if (ordersEngine) {
          ordersEngine->OnMarketViewUpdate(marketBook.get());
        }
        if (config_.trackStats) {
          stats_.consumer_updates.fetch_add(1, std::memory_order_relaxed);
        }
      }
    } catch (const std::exception& ex) {
      if (config_.trackStats) {
        stats_.consumer_exceptions.fetch_add(1, std::memory_order_relaxed);
      }
      std::cerr << "[ERROR] ConsumerThreadLoop exception: " << ex.what() << std::endl;
      stopRequested_.store(true);
    }
  }
}

std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>
Orchestrator::GetOrdersEngine() const {
  return config_.ordersEngine;
}

std::shared_ptr<constellation::interfaces::orderbook::IMarketView>
Orchestrator::GetMarketView() const {
  return std::dynamic_pointer_cast<constellation::interfaces::orderbook::IMarketView>(config_.marketBook);
}

} // end namespace orchestrator
} // end namespace applications
} // end namespace constellation
