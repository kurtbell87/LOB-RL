#include "strategy/SampleBatchStrategy.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include <cstdio>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace strategy {

SampleBatchStrategy::SampleBatchStrategy(
    std::shared_ptr<constellation::interfaces::orders::IOrdersCommand> ordersCmd,
    std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>   ordersQuery,
    std::shared_ptr<constellation::interfaces::logging::ILogger>       logger
)
  : ordersCmd_{ordersCmd},
    ordersQuery_{ordersQuery},
    logger_{ logger ? logger : std::make_shared<constellation::interfaces::logging::NullLogger>() }
{
}

SampleBatchStrategy::~SampleBatchStrategy() = default;

void SampleBatchStrategy::OnDataChunk(
    const constellation::interfaces::batch::IRecordBatch& recordBatch,
    const constellation::interfaces::orderbook::IMarketView* marketView)
{
  // 1) Increment counters for demonstration
  totalChunks_.fetch_add(1, std::memory_order_relaxed);
  totalRecords_.fetch_add(recordBatch.Size(), std::memory_order_relaxed);

  // 2) Optionally log
  if (logger_) {
    logger_->Info("[SampleBatchStrategy] Received chunk of size %zu. totalChunks=%llu, totalRecords=%llu",
                  recordBatch.Size(),
                  (unsigned long long)totalChunks_.load(),
                  (unsigned long long)totalRecords_.load());
  }

  // 3) (Optional) Place an order if we see some condition in the MarketView
  if (marketView) {
    // Example: if global trades > 100, place a trivial limit buy. (Purely a random condition.)
    auto tradeCount = marketView->GetGlobalTradeCount();
    if (tradeCount > 100 && ordersCmd_) {
      constellation::interfaces::orders::OrderSpec spec;
      spec.instrument_id = 12345;  // example ID
      spec.type          = constellation::interfaces::orders::OrderType::Limit;
      spec.side          = constellation::interfaces::orders::OrderSide::Buy;
      spec.quantity      = 10;
      spec.limit_price   = 12345;
      auto newId = ordersCmd_->PlaceOrder(spec);
      if (logger_) {
        logger_->Debug("[SampleBatchStrategy] Placed new limit buy order_id=%llu", (unsigned long long)newId);
      }
    }
  }

  // 4) Additional Batch strategy logic or calls to ordersCmd_ can go here.
}

void SampleBatchStrategy::Shutdown() {
  if (logger_) {
    logger_->Info("[SampleBatchStrategy] Shutdown called. Final chunk count=%llu, record count=%llu",
      (unsigned long long)totalChunks_.load(),
      (unsigned long long)totalRecords_.load()
    );
  }
}

} // end namespace strategy
} // end namespace modules
} // end namespace constellation
