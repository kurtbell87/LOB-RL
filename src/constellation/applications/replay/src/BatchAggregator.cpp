#include "replay/BatchAggregator.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "databento/dbn_file_store.hpp"
#include "databento/record.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "orderbook/OrderBookFactory.hpp"

#ifdef __has_include
#  if __has_include(<Python.h>)
#    define BATCHAGGREGATOR_HAS_PYTHON
#    include <Python.h>
#  endif
#endif

namespace constellation {
namespace applications {
namespace replay {

using constellation::interfaces::logging::ILogger;
using constellation::modules::orderbook::MarketBook;

static std::uint64_t ExtractTimestampMicros(const databento::Record& rec) {
  if (!rec.Holds<databento::MboMsg>()) {
    return 0ULL;
  }
  const auto& mbo = rec.Get<databento::MboMsg>();
  return static_cast<std::uint64_t>(mbo.hd.ts_event.time_since_epoch().count());
}

BatchAggregator::BatchAggregator()
  : initialized_(false),
    lastChunkMaxTimestamp_(0)
{
}

BatchAggregator::~BatchAggregator() = default;

void BatchAggregator::Initialize(const BatchAggregatorConfig& config) {
  std::lock_guard<std::mutex> lock(init_mutex_);
  if (initialized_) {
    throw std::runtime_error("BatchAggregator is already initialized");
  }
  config_ = config;
  if (!config_.logger) {
    logger_ = std::make_shared<constellation::interfaces::logging::NullLogger>();
  } else {
    logger_ = config_.logger;
  }

  // create MarketBook
  marketBook_ = std::static_pointer_cast<MarketBook>(
      constellation::modules::orderbook::CreateMarketBook(logger_)
  );

  stats_.Reset();
  recordBatch_.Clear();
  lastChunkMaxTimestamp_ = 0;
  initialized_ = true;

  if (logger_ && config_.enable_logging) {
    logger_->Info("[BatchAggregator::Initialize] aggregator init with batch_size=%u, mem_factor=%zu",
                  config_.batch_size,
                  config_.memory_factor_limit);
    if (config_.enable_instrument_boundary) {
      logger_->Info("[BatchAggregator::Initialize] instrument-boundary: instrument_id=%u, trades=%llu",
                    config_.boundary_instrument_id,
                    (unsigned long long)config_.boundary_instrument_trades);
    }
    if (config_.enable_event_count_boundary) {
      logger_->Info("[BatchAggregator::Initialize] event-boundary: eventType=%s, count=%llu",
                    config_.boundary_event_type.c_str(),
                    (unsigned long long)config_.boundary_event_count);
    }
  }
}

void BatchAggregator::ProcessFiles(const std::vector<std::string>& dbn_files) {
  if (!initialized_) {
    throw std::runtime_error("BatchAggregator not initialized. Call Initialize() first.");
  }
  for (auto& file : dbn_files) {
    ProcessOneFileInternal(file);
  }
}

void BatchAggregator::ProcessSingleFile(const std::string& file_path) {
  if (!initialized_) {
    throw std::runtime_error("BatchAggregator not initialized. Call Initialize() first.");
  }
  ProcessOneFileInternal(file_path);
}

std::shared_ptr<MarketBook> BatchAggregator::GetMarketBook() const {
  return marketBook_;
}

BatchAggregatorStats BatchAggregator::GetStats() const {
  return stats_;
}

void BatchAggregator::ResetStats() {
  stats_.Reset();
  lastChunkMaxTimestamp_ = 0;
}

const constellation::interfaces::batch::RecordBatch&
BatchAggregator::GetLastRecordBatch() const
{
  return recordBatch_;
}

void BatchAggregator::SetStrategy(
    const std::shared_ptr<constellation::interfaces::strategy::IStrategy>& strategy)
{
  std::lock_guard<std::mutex> lock(init_mutex_);
  strategy_ = strategy;
}

void BatchAggregator::TestDistributeBatch(
    const std::vector<const databento::Record*>& batch)
{
  void* gil_state = ReleaseGILIfConfigured();
  DistributeBatch(batch);
  ReacquireGILIfNeeded(gil_state);
}

void BatchAggregator::SetOrdersEngine(
    const std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>& ordersEngine)
{
  std::lock_guard<std::mutex> lock(init_mutex_);
  ordersEngine_ = ordersEngine;
}

// --------------------------------------------------
// Private
// --------------------------------------------------
void BatchAggregator::ProcessOneFileInternal(const std::string& file_path) {
  if (logger_ && config_.enable_logging) {
    logger_->Info("[BatchAggregator] Processing file: %s", file_path.c_str());
  }
  using namespace std::chrono;
  auto start_time = steady_clock::now();

  std::vector<const databento::Record*> batch;
  batch.reserve(config_.batch_size);

  // For "instrument boundary"
  std::uint64_t instrumentTradeCount = 0;  // Day1–2 logic

  // **New in Sprint 3**: global event boundary
  std::uint64_t globalTradeCount = 0; 

  try {
    databento::DbnFileStore store(file_path);
    auto& md = store.GetMetadata();
    if (logger_ && config_.enable_logging) {
      logger_->Debug("[BatchAggregator] dataset=%s, schema=%d, file=%s",
                     md.dataset.c_str(),
                     md.schema ? static_cast<int>(*md.schema) : -1,
                     file_path.c_str());
    }

    while (true) {
      const databento::Record* rec = store.NextRecord();
      if (!rec) {
        break; // EOF
      }
      stats_.total_records.fetch_add(1, std::memory_order_relaxed);
      if (rec->Holds<databento::MboMsg>()) {
        stats_.total_mbo_messages.fetch_add(1, std::memory_order_relaxed);
      }

      batch.push_back(rec);

      // forced chunk if batch_size is reached
      if (batch.size() >= config_.batch_size) {
        EnforceMemoryConstraint();
        void* gil_state = ReleaseGILIfConfigured();
        DistributeBatch(batch);
        ReacquireGILIfNeeded(gil_state);
        batch.clear();
        // reset local trade counters
        instrumentTradeCount = 0;
        globalTradeCount = 0;
      }

      // Day1–2: If enable_instrument_boundary is set, watch trades/fills for boundary_instrument_id
      if (config_.enable_instrument_boundary && rec->Holds<databento::MboMsg>()) {
        const auto& mbo = rec->Get<databento::MboMsg>();
        if (mbo.hd.instrument_id == config_.boundary_instrument_id) {
          using databento::Action;
          if (mbo.action == Action::Trade || mbo.action == Action::Fill) {
            instrumentTradeCount++;
            if (instrumentTradeCount >= config_.boundary_instrument_trades &&
                config_.boundary_instrument_trades > 0)
            {
              EnforceMemoryConstraint();
              void* gil_state = ReleaseGILIfConfigured();
              DistributeBatch(batch);
              ReacquireGILIfNeeded(gil_state);
              batch.clear();
              instrumentTradeCount = 0;
              globalTradeCount = 0;
            }
          }
        }
      }

      // **New in Sprint 3**: If enable_event_count_boundary is set with eventType="trade", watch all trades/fills globally
      if (config_.enable_event_count_boundary && !config_.boundary_event_type.empty()) {
        // if user set boundary_event_type="trade" then increment on Action::Trade or Action::Fill
        if (config_.boundary_event_type == "trade" && rec->Holds<databento::MboMsg>()) {
          const auto& mbo = rec->Get<databento::MboMsg>();
          using databento::Action;
          if (mbo.action == Action::Trade || mbo.action == Action::Fill) {
            globalTradeCount++;
            if (globalTradeCount >= config_.boundary_event_count &&
                config_.boundary_event_count > 0)
            {
              EnforceMemoryConstraint();
              void* gil_state = ReleaseGILIfConfigured();
              DistributeBatch(batch);
              ReacquireGILIfNeeded(gil_state);
              batch.clear();
              globalTradeCount = 0;
              instrumentTradeCount = 0;
            }
          }
        }
        // in the future, user might set boundary_event_type="mbo" or "cancel", etc.
        // extension point for more event types
      }
    }
    // leftover
    if (!batch.empty()) {
      EnforceMemoryConstraint();
      void* gil_state = ReleaseGILIfConfigured();
      DistributeBatch(batch);
      ReacquireGILIfNeeded(gil_state);
      batch.clear();
    }
  } catch (const std::exception& ex) {
    if (logger_ && config_.enable_logging) {
      logger_->Error("[BatchAggregator::ProcessOneFileInternal] Exception reading '%s': %s",
                     file_path.c_str(), ex.what());
    }
  }

  auto end_time = steady_clock::now();
  auto micros = duration_cast<microseconds>(end_time - start_time).count();
  stats_.total_microseconds.fetch_add(static_cast<std::uint64_t>(micros), std::memory_order_relaxed);
}

void BatchAggregator::DistributeBatch(const std::vector<const databento::Record*>& batch) {
  if (batch.empty()) {
    return;
  }
  // copy + sort
  std::vector<const databento::Record*> working(batch.begin(), batch.end());
  std::sort(working.begin(), working.end(),
    [](const databento::Record* a, const databento::Record* b){
      return ExtractTimestampMicros(*a) < ExtractTimestampMicros(*b);
    }
  );

  // remove out-of-order
  auto new_end = std::remove_if(working.begin(), working.end(),
    [this](const databento::Record* r){
      return ExtractTimestampMicros(*r) < this->lastChunkMaxTimestamp_;
    });
  if (new_end != working.end()) {
    std::size_t removed_count = std::distance(new_end, working.end());
    if (removed_count > 0 && logger_ && config_.enable_logging) {
      logger_->Warn("[BatchAggregator::DistributeBatch] Discarded %zu out-of-order record(s).", removed_count);
    }
    working.erase(new_end, working.end());
  }
  if (working.empty()) {
    return;
  }

  recordBatch_.Clear();
  recordBatch_.Reserve(working.size());

  std::vector<databento::MboMsg> mbo_batch;
  mbo_batch.reserve(working.size());

  std::uint64_t chunkMaxTs = lastChunkMaxTimestamp_;

  for (auto* rec : working) {
    std::uint64_t ts_event = ExtractTimestampMicros(*rec);
    if (ts_event > chunkMaxTs) {
      chunkMaxTs = ts_event;
    }
    if (rec->Holds<databento::MboMsg>()) {
      const auto& mbo = rec->Get<databento::MboMsg>();
      mbo_batch.push_back(mbo);

      const std::uint64_t event_ts  = ts_event;
      const std::uint32_t instr_id  = mbo.hd.instrument_id;
      const std::int64_t  price     = static_cast<std::int64_t>(mbo.price);
      const std::uint32_t sz        = mbo.size;
      int side_val = 2;
      using databento::Side;
      if (mbo.side == Side::Bid) side_val = 0;
      else if (mbo.side == Side::Ask) side_val = 1;

      int action_val = 9;
      using databento::Action;
      switch (mbo.action) {
        case Action::Add:    action_val = 0; break;
        case Action::Modify: action_val = 1; break;
        case Action::Cancel: action_val = 2; break;
        case Action::Trade:
        case Action::Fill:   action_val = 3; break;
        case Action::Clear:  action_val = 4; break;
        default:             action_val = 9; break;
      }
      std::uint64_t oid = mbo.order_id;

      recordBatch_.Append(event_ts, instr_id, price, sz, side_val, action_val, oid);
    }
  }

  marketBook_->BatchOnMboUpdate(mbo_batch);
  lastChunkMaxTimestamp_ = chunkMaxTs;

  // FIRST OnMarketViewUpdate (fills any existing or stale orders)
  if (ordersEngine_) {
    ordersEngine_->SetCurrentTimestamp(chunkMaxTs);
    ordersEngine_->OnMarketViewUpdate(marketBook_.get());
  }

  // Now call the chunk-based strategy if set
  if (strategy_) {
    strategy_->OnDataChunk(recordBatch_, marketBook_.get());
  }

  // SECOND OnMarketViewUpdate to fill newly placed orders from strategy
  if (ordersEngine_) {
    ordersEngine_->OnMarketViewUpdate(marketBook_.get());
  }
}

void BatchAggregator::EnforceMemoryConstraint() {
  std::size_t allowed = static_cast<std::size_t>(config_.batch_size) * config_.memory_factor_limit;

  std::size_t cap_ts     = recordBatch_.TimestampsCapacity();
  std::size_t cap_instr  = recordBatch_.InstrumentIdsCapacity();
  std::size_t cap_prices = recordBatch_.PricesCapacity();
  if (cap_ts > allowed || cap_instr > allowed || cap_prices > allowed) {
    throw std::runtime_error(
      "[BatchAggregator::EnforceMemoryConstraint] Exceeded memory_factor_limit x batch_size."
    );
  }
}

void* BatchAggregator::ReleaseGILIfConfigured() {
#ifdef BATCHAGGREGATOR_HAS_PYTHON
  if (config_.release_gil_during_aggregation) {
    return (void*) PyEval_SaveThread();
  }
#endif
  return nullptr;
}

void BatchAggregator::ReacquireGILIfNeeded(void* gil_state) {
#ifdef BATCHAGGREGATOR_HAS_PYTHON
  if (gil_state && config_.release_gil_during_aggregation) {
    PyEval_RestoreThread((PyThreadState*) gil_state);
  }
#endif
}

} // namespace replay
} // namespace applications
} // namespace constellation
