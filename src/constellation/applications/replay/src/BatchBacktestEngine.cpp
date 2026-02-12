#include "replay/BatchBacktestEngine.hpp"

#include "orders/OrdersFactory.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "orders/OrdersEngine.hpp"

namespace constellation {
namespace applications {
namespace replay {

// ------------------------------
// BacktestFillLogger
// ------------------------------
BacktestFillLogger::BacktestFillLogger() = default;
BacktestFillLogger::~BacktestFillLogger() = default;

void BacktestFillLogger::OnOrderFilled(std::uint64_t order_id,
                                       std::int64_t fill_price,
                                       std::uint32_t fill_qty)
{
  // default usage if aggregator doesn't supply chunk timestamps
  std::lock_guard<std::mutex> lock(mtx_);
  auto it = order_map_.find(order_id);
  if (it == order_map_.end()) {
    // unknown -> minimal
    FillRecord rec;
    rec.order_id      = order_id;
    rec.fill_price    = fill_price;
    rec.fill_qty      = fill_qty;
    rec.is_buy        = true; // default
    rec.instrument_id = 0;
    rec.timestamp     = 0;
    fill_log_.push_back(rec);
  } else {
    const auto& ctx = it->second;
    FillRecord rec;
    rec.timestamp     = ctx.timestamp;
    rec.order_id      = order_id;
    rec.instrument_id = ctx.instrument_id;
    rec.fill_price    = fill_price;
    rec.fill_qty      = fill_qty;
    rec.is_buy        = ctx.is_buy;
    fill_log_.push_back(rec);
  }
}

void BacktestFillLogger::OnOrderDone(std::uint64_t order_id,
                                     constellation::interfaces::orders::OrderFinalState /*final_state*/)
{
  std::lock_guard<std::mutex> lock(mtx_);
  order_map_.erase(order_id);
}

void BacktestFillLogger::OnOrderFillWithTimestamp(std::uint64_t timestamp,
                                                  std::uint64_t order_id,
                                                  std::uint32_t instrument_id,
                                                  constellation::interfaces::orders::OrderSide side,
                                                  std::int64_t fill_price,
                                                  std::uint32_t fill_qty)
{
  std::lock_guard<std::mutex> lock(mtx_);
  FillRecord rec;
  rec.timestamp      = timestamp;
  rec.order_id       = order_id;
  rec.instrument_id  = instrument_id;
  rec.fill_price     = fill_price;
  rec.fill_qty       = fill_qty;
  rec.is_buy         = (side == constellation::interfaces::orders::OrderSide::Buy);
  fill_log_.push_back(rec);

  // Also store context so old OnOrderFilled can remain consistent:
  auto& ctx = order_map_[order_id];
  ctx.timestamp     = timestamp;
  ctx.instrument_id = instrument_id;
  ctx.is_buy        = rec.is_buy;
}

void BacktestFillLogger::RecordTimestampAndSide(std::uint64_t order_id,
                                                bool is_buy,
                                                std::uint32_t instr_id,
                                                std::uint64_t ts)
{
  std::lock_guard<std::mutex> lock(mtx_);
  auto& ctx = order_map_[order_id];
  ctx.is_buy        = is_buy;
  ctx.instrument_id = instr_id;
  ctx.timestamp     = ts;
}

std::vector<FillRecord> BacktestFillLogger::GetFillRecords() const {
  std::lock_guard<std::mutex> lock(mtx_);
  return fill_log_;
}

void BacktestFillLogger::Clear() {
  std::lock_guard<std::mutex> lock(mtx_);
  fill_log_.clear();
  order_map_.clear();
}

// ------------------------------
// BatchBacktestEngine
// ------------------------------
BatchBacktestEngine::BatchBacktestEngine()
{
  // Batch aggregator
  aggregator_ = std::make_shared<BatchAggregator>();

  // Orders engine
  ordersEngine_ = constellation::modules::orders::CreateIOrdersEngine();

  // Fill logger
  fillLogger_ = std::make_shared<BacktestFillLogger>();

  // connect fill logger to engine
  auto realEngine = std::dynamic_pointer_cast<constellation::orders::OrdersEngine>(ordersEngine_);
  if (realEngine) {
    realEngine->SetOrderEvents(fillLogger_);
  }

  // Phase 4: Feature manager starts as nullptr, can be set with SetFeatureManager()
  featureManager_ = nullptr;
}

BatchBacktestEngine::~BatchBacktestEngine() = default;

void BatchBacktestEngine::SetAggregatorConfig(const BatchAggregatorConfig& config) {
  // 1) Initialize aggregator with config
  aggregator_->Initialize(config);

  // 2) Attach our orders engine so aggregator can generate fills
  aggregator_->SetOrdersEngine(ordersEngine_);

  // Phase 4: If we have a feature manager, connect it to the aggregator
  if (featureManager_) {
    // For Phase 4, we'd ideally have a setFeatureManager method in BatchAggregator,
    // but for now we'll assume we can use the feature manager directly through
    // BatchBacktestEngine, updating it in the engine's data flow
    // TODO: Add SetFeatureManager to BatchAggregator in the future
  }
}

void BatchBacktestEngine::SetStrategy(const std::shared_ptr<constellation::interfaces::strategy::IStrategy>& strategy)
{
  aggregator_->SetStrategy(strategy);
}

void BatchBacktestEngine::ProcessFiles(const std::vector<std::string>& dbn_files)
{
  // Process through aggregator
  aggregator_->ProcessFiles(dbn_files);
  
  // Phase 4: Update feature manager with the latest market data if present
  if (featureManager_ && aggregator_) {
    auto market_book = aggregator_->GetMarketBook();
    featureManager_->OnDataUpdate(*market_book, market_book.get());
  }
}

void BatchBacktestEngine::ProcessSingleFile(const std::string& dbn_file)
{
  // Process single file
  aggregator_->ProcessSingleFile(dbn_file);
  
  // Phase 4: Update feature manager with the latest market data if present
  if (featureManager_ && aggregator_) {
    auto market_book = aggregator_->GetMarketBook();
    featureManager_->OnDataUpdate(*market_book, market_book.get());
  }
}

std::vector<FillRecord> BatchBacktestEngine::GetFills() const
{
  return fillLogger_->GetFillRecords();
}

std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>
BatchBacktestEngine::GetOrdersEngine() const
{
  return ordersEngine_;
}

void BatchBacktestEngine::SetFeatureManager(
    const std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager>& feature_manager)
{
  featureManager_ = feature_manager;
}

std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager>
BatchBacktestEngine::GetFeatureManager() const
{
  return featureManager_;
}

void BatchBacktestEngine::ResetStats()
{
  // aggregator has ResetStats, fill logger can be cleared
  if (aggregator_) {
    aggregator_->ResetStats();
  }
  if (fillLogger_) {
    fillLogger_->Clear();
  }
}

std::shared_ptr<constellation::interfaces::orderbook::IMarketView> BatchBacktestEngine::GetMarketView() const
{
  if (!aggregator_) {
    return nullptr;
  }
  
  // Get the MarketBook from the aggregator and cast it to IMarketView
  // MarketBook implements IMarketView interface
  auto marketBook = aggregator_->GetMarketBook();
  return std::dynamic_pointer_cast<constellation::interfaces::orderbook::IMarketView>(marketBook);
}

} // end namespace replay
} // end namespace applications
} // end namespace constellation
