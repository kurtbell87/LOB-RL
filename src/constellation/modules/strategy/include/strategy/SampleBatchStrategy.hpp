#pragma once

#include <memory>
#include <string>
#include <atomic>
#include "interfaces/strategy/IStrategy.hpp"
#include "interfaces/logging/ILogger.hpp"
#include "interfaces/orders/IOrdersCommand.hpp"
#include "interfaces/orders/IOrdersQuery.hpp"
#include "interfaces/batch/IRecordBatch.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation {
namespace modules {
namespace strategy {

/**
 * @brief SampleBatchStrategy demonstrates how to implement a chunk-based
 *        Batch strategy. Each chunk of MBO data is delivered via OnDataChunk(...).
 *
 *        This replaces the older per-message SampleBatchStrategy, guaranteeing
 *        no look-ahead. The aggregator calls us only after a chunk [t0..t1] is
 *        fully applied to the MarketBook. We can also place or cancel orders
 *        through an injected IOrdersCommand (if needed).
 */
class SampleBatchStrategy : public constellation::interfaces::strategy::IStrategy {
public:
  /**
   * @brief Constructor
   * @param ordersCmd    (optional) If provided, we can place or cancel orders from OnDataChunk
   * @param ordersQuery  (optional) For checking existing order statuses
   * @param logger       (optional) Logger
   */
  SampleBatchStrategy(
    std::shared_ptr<constellation::interfaces::orders::IOrdersCommand> ordersCmd = nullptr,
    std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>   ordersQuery = nullptr,
    std::shared_ptr<constellation::interfaces::logging::ILogger>       logger = nullptr
  );

  ~SampleBatchStrategy() override;

  /**
   * @brief The Batch aggregator calls this after a chunk of MBO data (time range [T0..T1])
   *        is fully applied. No look-ahead is possible; we see data up to T1 only.
   */
  void OnDataChunk(const constellation::interfaces::batch::IRecordBatch& recordBatch,
                   const constellation::interfaces::orderbook::IMarketView* marketView) override;

  /**
   * @brief Called before system shutdown or aggregator completes. We can do cleanup here.
   */
  void Shutdown() override;

private:
  std::shared_ptr<constellation::interfaces::orders::IOrdersCommand> ordersCmd_;
  std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>   ordersQuery_;
  std::shared_ptr<constellation::interfaces::logging::ILogger>       logger_;

  // Example usage counters:
  std::atomic<std::uint64_t> totalChunks_{0};
  std::atomic<std::uint64_t> totalRecords_{0};
};

} // end namespace strategy
} // end namespace modules
} // end namespace constellation
