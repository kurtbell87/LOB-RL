#include "engine/Engine.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "market_data/MarketDataFactory.hpp"
#include "orderbook/OrderBookFactory.hpp"
#include "features/FeatureFactory.hpp"
#include "orders/OrdersFactory.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "interfaces/strategy/IStrategy.hpp"
#include "databento/record.hpp"

namespace constellation {
namespace applications {
namespace engine {

Engine::Engine()
  : stopRequested_(false),
    isRunning_(false)
{}

Engine::~Engine() {
  if (isRunning_.load()) {
    Stop();
  }
}

void Engine::Initialize(const EngineConfig& config) {
  config_ = config;

  // Possibly create feed
  CreateFeedIfNeeded();

  // If no marketBook, create and dynamic_cast
  if (!config_.marketBook) {
    auto mv = modules::orderbook::CreateMarketBook();  // returns shared_ptr<IMarketView>
    auto mk = std::dynamic_pointer_cast<interfaces::orderbook::IMarketBook>(mv);
    if (!mk) {
      throw std::runtime_error("[Engine::Initialize] MarketBook cast failed.");
    }
    marketBook_ = mk;
  } else {
    marketBook_ = config_.marketBook;
  }

  // If no ordersEngine, create
  if (!config_.ordersEngine) {
    ordersEngine_ = modules::orders::CreateIOrdersEngine();
  } else {
    ordersEngine_ = config_.ordersEngine;
  }

  // If no featureManager, create
  if (!config_.featureManager) {
    auto nullLogger = std::make_shared<interfaces::logging::NullLogger>();
    featureManager_ = modules::features::CreateFeatureManager(nullLogger);
  } else {
    featureManager_ = config_.featureManager;
  }

  // Create aggregator
  aggregator_ = std::make_shared<applications::replay::BatchAggregator>();
  {
    replay::BatchAggregatorConfig cfg;
    cfg.batch_size           = config_.chunkSize;
    cfg.enable_logging       = false;
    cfg.memory_factor_limit  = 3;
    aggregator_->Initialize(cfg);
  }
  // Set the strategy for the Python thread (not directly to the aggregator anymore)
  if (config_.strategy) {
    strategy_ = config_.strategy;
  }
  aggregator_->SetOrdersEngine(ordersEngine_);

  // Create ring buffers
  {
    std::size_t cap1 = 65536;
    ringBufferFeed_ =
        std::make_unique<orchestrator::SpscRingBuffer<databento::MboMsg>>(cap1);
  }
  {
    std::size_t cap2 = 65536;
    ringBufferAggregator_ =
        std::make_unique<orchestrator::SpscRingBuffer<databento::MboMsg>>(cap2);
  }
  {
    // Third ring buffer for passing chunks to Python strategy thread
    std::size_t cap3 = config_.strategyRingBufferSize;
    ringBufferStrategy_ =
        std::make_unique<orchestrator::SpscRingBuffer<StrategyChunk>>(cap3);
  }

  // Reset stats
  {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.feed_msgs.store(0, std::memory_order_relaxed);
    stats_.snapshot_msgs.store(0, std::memory_order_relaxed);
    stats_.snapshot_msgs_applied.store(0, std::memory_order_relaxed);
    stats_.snapshot_msgs_skipped.store(0, std::memory_order_relaxed);
    stats_.files_processed.store(0, std::memory_order_relaxed);
    stats_.current_file_index.store(0, std::memory_order_relaxed);
    stats_.aggregator_chunks.store(0, std::memory_order_relaxed);
    stats_.aggregator_exceptions.store(0, std::memory_order_relaxed);
    stats_.python_chunks.store(0, std::memory_order_relaxed);
    stats_.python_exceptions.store(0, std::memory_order_relaxed);
    
    // Reset snapshot tracking
    snapshotComplete_.clear();
    
    // Reset multi-file state
    isFirstFile_ = true;
    currentFileDone_ = false;
  }
}

void Engine::CreateFeedIfNeeded() {
  if (config_.feed) {
    feed_ = config_.feed;
    return;
  }

  if (config_.feedMode == EngineFeedMode::NoFeed) {
    feed_ = nullptr;
  } else if (config_.feedMode == EngineFeedMode::Backtest) {
    if (config_.dbnFilePaths.empty()) {
      throw std::runtime_error("EngineConfig: feedMode=Backtest but no dbnFilePaths provided");
    }
    
    // Create feed for the first file
    const auto& firstFile = config_.dbnFilePaths[0];
    constellation::interfaces::market_data::DbnFileFeedConfig fcfg;
    fcfg.file_path = firstFile;
    fcfg.loop_forever = false;
    feed_ = modules::market_data::MarketDataFactory::CreateDbnFileFeed(fcfg);
    
    // Phase 3: Initialize multi-file tracking
    isFirstFile_ = true;
    currentFileDone_ = false;
    stats_.current_file_index.store(0);
    
    // Log the number of files we'll process
    std::cout << "[Engine::CreateFeedIfNeeded] Phase 3: Processing " 
              << config_.dbnFilePaths.size() << " DBN files in sequence" << std::endl;
    
  } else if (config_.feedMode == EngineFeedMode::Live) {
    constellation::interfaces::market_data::DataBentoFeedConfig lcfg;
    lcfg.api_key        = config_.liveApiKey;
    lcfg.dataset        = config_.liveDataset;
    lcfg.symbols        = config_.liveSymbols;
    lcfg.use_live       = config_.liveUseLive;
    lcfg.start_datetime = config_.liveStart;
    lcfg.end_datetime   = config_.liveEnd;
    feed_ = modules::market_data::MarketDataFactory::CreateDataBentoMboFeed(lcfg);
  }
}

void Engine::SwitchToNextFile() {
  if (config_.feedMode != EngineFeedMode::Backtest) {
    return;  // Only applicable in backtest mode
  }
  
  std::lock_guard<std::mutex> lock(filesMutex_);
  
  // Increment file index
  uint64_t currentIdx = stats_.current_file_index.load();
  currentIdx++;
  
  if (currentIdx >= config_.dbnFilePaths.size()) {
    std::cout << "[Engine::SwitchToNextFile] Finished processing all files" << std::endl;
    return;  // No more files to process
  }
  
  stats_.current_file_index.store(currentIdx);
  isFirstFile_ = false;
  currentFileDone_ = false;
  
  // Stop the current feed
  if (feed_) {
    feed_->Stop();
  }
  
  // Create a new feed for the next file
  const auto& nextFile = config_.dbnFilePaths[currentIdx];
  constellation::interfaces::market_data::DbnFileFeedConfig fcfg;
  fcfg.file_path = nextFile;
  fcfg.loop_forever = false;
  feed_ = modules::market_data::MarketDataFactory::CreateDbnFileFeed(fcfg);
  
  // Register the callback
  feed_->SubscribeMboCallback([this](const databento::MboMsg& msg){
    if (stopRequested_.load()) {
      return;
    }
    while (!stopRequested_.load()) {
      bool ok = ringBufferFeed_->TryPush(msg);
      if (ok) {
        stats_.feed_msgs.fetch_add(1, std::memory_order_relaxed);
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  });
  
  // Start the new feed
  feed_->Start();
  
  stats_.files_processed.fetch_add(1, std::memory_order_relaxed);
  std::cout << "[Engine::SwitchToNextFile] Switched to file "
            << currentIdx + 1 << "/" << config_.dbnFilePaths.size()
            << ": " << nextFile << std::endl;
}

bool Engine::IsSnapshotMessage(const databento::MboMsg& msg) const {
  return msg.flags.IsSnapshot();
}

bool Engine::IsLastSnapshot(const databento::MboMsg& msg) const {
  return (msg.flags.IsLast() && msg.flags.IsSnapshot());
}

bool Engine::ShouldProcessSnapshot(const databento::MboMsg& msg) const {
  if (!config_.handleSnapshots) {
    return false;  // Snapshots completely disabled
  }
  
  if (msg.flags.IsSnapshot()) {
    auto inst = msg.hd.instrument_id;
    auto it = snapshotComplete_.find(inst);
    bool instrumentSeenBefore = (it != snapshotComplete_.end());
    
    switch (config_.snapshotMode) {
      case EngineConfig::SnapshotHandlingMode::ProcessAllSnapshots:
        return true;  // Always process
        
      case EngineConfig::SnapshotHandlingMode::ProcessFirstFileOnly:
        return isFirstFile_;  // Only first file
        
      case EngineConfig::SnapshotHandlingMode::ProcessPerInstrument:
        return !instrumentSeenBefore || (instrumentSeenBefore && !it->second);
        
      case EngineConfig::SnapshotHandlingMode::SkipAllSnapshots:
        return false;  // Skip all
        
      default:
        return true;  // Default - process
    }
  }
  
  return false;  // Not a snapshot
}

void Engine::Start() {
  if (isRunning_.exchange(true)) {
    return; // already started
  }
  stopRequested_.store(false);

  if (feed_) {
    // Subscribe feed callback
    feed_->SubscribeMboCallback([this](const databento::MboMsg& msg){
      if (stopRequested_.load()) {
        return;
      }
      while (!stopRequested_.load()) {
        bool ok = ringBufferFeed_->TryPush(msg);
        if (ok) {
          stats_.feed_msgs.fetch_add(1, std::memory_order_relaxed);
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
  }

  // feed thread
  feedThread_ = std::thread(&Engine::FeedThreadLoop, this);
  // market thread
  marketThread_ = std::thread(&Engine::MarketThreadLoop, this);
  // aggregator thread
  aggregatorThread_ = std::thread(&Engine::AggregatorThreadLoop, this);
  // python strategy thread (new in Phase 2)
  pythonStrategyThread_ = std::thread(&Engine::PythonStrategyThreadLoop, this);
}

void Engine::Stop() {
  if (!isRunning_.exchange(false)) {
    return; // already stopped
  }
  stopRequested_.store(true);

  if (feed_) {
    feed_->Stop();
  }

  if (feedThread_.joinable()) {
    feedThread_.join();
  }
  if (marketThread_.joinable()) {
    marketThread_.join();
  }
  if (aggregatorThread_.joinable()) {
    aggregatorThread_.join();
  }
  if (pythonStrategyThread_.joinable()) {
    pythonStrategyThread_.join();
  }
}

bool Engine::IsRunning() const {
  return isRunning_.load();
}

/**
 * @brief We manually load each atomic counter into a plain struct, so no copy of
 *        the atomic struct itself is required.
 */
EngineStatsSnapshot Engine::GetStats() const {
  EngineStatsSnapshot snapshot;
  {
    std::lock_guard<std::mutex> lock(statsMutex_);
    snapshot.feed_msgs          = stats_.feed_msgs.load(std::memory_order_relaxed);
    snapshot.snapshot_msgs      = stats_.snapshot_msgs.load(std::memory_order_relaxed);
    snapshot.snapshot_msgs_applied = stats_.snapshot_msgs_applied.load(std::memory_order_relaxed);
    snapshot.snapshot_msgs_skipped = stats_.snapshot_msgs_skipped.load(std::memory_order_relaxed);
    snapshot.files_processed    = stats_.files_processed.load(std::memory_order_relaxed);
    snapshot.current_file_index = stats_.current_file_index.load(std::memory_order_relaxed);
    snapshot.aggregator_chunks  = stats_.aggregator_chunks.load(std::memory_order_relaxed);
    snapshot.aggregator_exceptions = stats_.aggregator_exceptions.load(std::memory_order_relaxed);
    snapshot.python_chunks      = stats_.python_chunks.load(std::memory_order_relaxed);
  }
  return snapshot;
}

// ---------------------------
// Thread loops
// ---------------------------
void Engine::FeedThreadLoop() {
  if (!feed_) {
    return;
  }
  
  try {
    if (config_.feedMode == EngineFeedMode::Backtest) {
      // Note: In Phase 3, feed_->Start() will process the current file and return when done
      feed_->Start();
      
      // When we reach this point, the current file has been fully processed
      std::cout << "[Engine::FeedThreadLoop] File " 
                << (stats_.current_file_index.load() + 1) << "/" << config_.dbnFilePaths.size()
                << " completed" << std::endl;
      
      // Mark the current file as done so MarketThread can switch to next file
      {
        std::lock_guard<std::mutex> lock(filesMutex_);
        currentFileDone_ = true;
      }
      
      // If this is the last file, we're done
      if (stats_.current_file_index.load() == config_.dbnFilePaths.size() - 1) {
        std::cout << "[Engine::FeedThreadLoop] All files processed" << std::endl;
      }
      
    } else {
      // For live feed or custom feed, just start it normally
      feed_->Start();
    }
    
  } catch (const std::exception& ex) {
    std::cerr << "[Engine::FeedThreadLoop] exception: " << ex.what() << std::endl;
    stopRequested_.store(true);
  }
}

void Engine::MarketThreadLoop() {
  if (!marketBook_) {
    return;
  }
  
  while (!stopRequested_.load()) {
    databento::MboMsg msg;
    if (!ringBufferFeed_->TryPop(msg)) {
      // Check if current file is done and we need to switch to next file
      {
        std::unique_lock<std::mutex> lock(filesMutex_);
        if (currentFileDone_ && stats_.current_file_index.load() < config_.dbnFilePaths.size() - 1) {
          lock.unlock();
          SwitchToNextFile();  // Process next file
          continue;
        }
      }
      
      std::this_thread::sleep_for(std::chrono::microseconds(50));
      continue;
    }
    
    // Handle snapshots using Phase 3 logic
    if (IsSnapshotMessage(msg)) {
      stats_.snapshot_msgs.fetch_add(1, std::memory_order_relaxed);

      auto inst = msg.hd.instrument_id;
      auto it = snapshotComplete_.find(inst);
      if (it == snapshotComplete_.end()) {
        // First time seeing this instrument
        snapshotComplete_[inst] = false;
        it = snapshotComplete_.find(inst);
      }
      bool done = it->second;

      // Determine if we should process this snapshot based on our policy
      if (!done && ShouldProcessSnapshot(msg)) {
        marketBook_->OnMboUpdate(msg);
        stats_.snapshot_msgs_applied.fetch_add(1, std::memory_order_relaxed);
        
        if (IsLastSnapshot(msg)) {
          it->second = true; // snapshot done for this instrument
        }
        continue;
      } else {
        // Skip this snapshot
        stats_.snapshot_msgs_skipped.fetch_add(1, std::memory_order_relaxed);
        // Don't continue here - we'll pass skipped snapshots to the aggregator
      }
    }
    
    // Normal message (or skipped snapshot) - update the market book and pass to aggregator
    marketBook_->OnMboUpdate(msg);
    
    // Push to aggregator
    while (!stopRequested_.load()) {
      bool ok = ringBufferAggregator_->TryPush(msg);
      if (ok) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }
}

void Engine::AggregatorThreadLoop() {
  std::vector<const databento::Record*> batch;
  batch.reserve(config_.chunkSize);

  while (!stopRequested_.load()) {
    try {
      databento::MboMsg msg;
      if (!ringBufferAggregator_->TryPop(msg)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      auto* msgPtr = new databento::MboMsg(msg);
      batch.push_back(reinterpret_cast<const databento::Record*>(msgPtr));

      if (batch.size() >= config_.chunkSize) {
        // Phase 2: Instead of directly calling strategy from aggregator thread,
        // we distribute the batch to update the market book and features
        aggregator_->TestDistributeBatch(batch);
        
        // Create a StrategyChunk to pass via ringBufferStrategy_
        StrategyChunk chunk;
        // Clone the record batch to ensure thread safety
        auto recordBatch = std::make_shared<interfaces::batch::RecordBatch>(
            aggregator_->GetLastRecordBatch());
        // Get access to market view for Python strategy
        auto marketView = std::dynamic_pointer_cast<interfaces::orderbook::IMarketView>(marketBook_);
        
        chunk.recordBatch = recordBatch;
        chunk.marketView = marketView;
        chunk.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Push to Python strategy thread with potential blocking
        bool pushed = false;
        while (!stopRequested_.load() && !pushed) {
          pushed = ringBufferStrategy_->TryPush(chunk);
          if (!pushed) {
            if (config_.blockAggregatorWhenStrategyBufferFull) {
              // Block if configured to do so
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
              // Otherwise just drop the chunk and log a warning
              std::cerr << "[Engine::AggregatorThreadLoop] Strategy ring buffer full, dropping chunk" << std::endl;
              break;
            }
          }
        }
        
        // Clean up batch resources
        for (auto* rec : batch) {
          auto* ptr = reinterpret_cast<const databento::MboMsg*>(rec);
          delete ptr;
        }
        batch.clear();

        stats_.aggregator_chunks.fetch_add(1, std::memory_order_relaxed);
      }
    } catch (const std::exception& ex) {
      stats_.aggregator_exceptions.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "[Engine::AggregatorThreadLoop] exception: " << ex.what() << std::endl;
      stopRequested_.store(true);
    }
  }

  // leftover
  if (!batch.empty()) {
    try {
      aggregator_->TestDistributeBatch(batch);
      
      // Create final StrategyChunk for any leftover data
      StrategyChunk chunk;
      auto recordBatch = std::make_shared<interfaces::batch::RecordBatch>(
          aggregator_->GetLastRecordBatch());
      auto marketView = std::dynamic_pointer_cast<interfaces::orderbook::IMarketView>(marketBook_);
      
      chunk.recordBatch = recordBatch;
      chunk.marketView = marketView;
      chunk.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      
      // Try to push one last time
      ringBufferStrategy_->TryPush(chunk);
      
      // Clean up
      for (auto* rec : batch) {
        auto* ptr = reinterpret_cast<const databento::MboMsg*>(rec);
        delete ptr;
      }
      batch.clear();
      stats_.aggregator_chunks.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& ex) {
      stats_.aggregator_exceptions.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "[Engine::AggregatorThreadLoop leftover] exception: " << ex.what() << std::endl;
    }
  }
}

void Engine::PythonStrategyThreadLoop() {
  if (!strategy_) {
    return; // No strategy configured, nothing to do
  }
  
  while (!stopRequested_.load()) {
    try {
      StrategyChunk chunk;
      if (!ringBufferStrategy_->TryPop(chunk)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      
      // Process the chunk with the strategy
      if (chunk.recordBatch && chunk.marketView) {
        strategy_->OnDataChunk(*(chunk.recordBatch), chunk.marketView.get());
        stats_.python_chunks.fetch_add(1, std::memory_order_relaxed);
      }
    } catch (const std::exception& ex) {
      stats_.python_exceptions.fetch_add(1, std::memory_order_relaxed);
      std::cerr << "[Engine::PythonStrategyThreadLoop] exception: " << ex.what() << std::endl;
    }
  }
  
  // Call shutdown on the strategy when the thread is stopping
  if (strategy_) {
    try {
      strategy_->Shutdown();
    } catch (const std::exception& ex) {
      std::cerr << "[Engine::PythonStrategyThreadLoop shutdown] exception: " << ex.what() << std::endl;
    }
  }
}

} // end namespace engine
} // end namespace applications
} // end namespace constellation
