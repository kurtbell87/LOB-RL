#pragma once

#include "interfaces/batch/IRecordBatch.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation {
namespace interfaces {
namespace strategy {

/**
 * @brief A chunk-based strategy interface for Batch usage. Unlike the old
 * per-message approach, this interface is called once per fully
 * ingested chunk of MBO data, ensuring no look-ahead and minimal overhead.
 */
class IStrategy {
public:
  virtual ~IStrategy() = default;

  /**
   * @brief Called after the Batch aggregator has read and applied a full chunk
   *        of MBO messages to the market book. This chunk is guaranteed to be
   *        strictly time-ordered with no look-ahead.
   *
   * @param recordBatch A reference to the SoA record batch for the newly ingested chunk.
   *                    Contains all MBO updates from time T0..T1.
   * @param marketView  A pointer to the updated market view (e.g. MarketBook) after
   *                    applying the chunk. The strategy may query best quotes, volumes,
   *                    or counters here. Must not modify it.
   */
  virtual void OnDataChunk(const constellation::interfaces::batch::IRecordBatch& recordBatch,
                           const constellation::interfaces::orderbook::IMarketView* marketView) = 0;

  /**
   * @brief Called when the strategy is being shut down. The strategy should
   *        release resources or finalize state here.
   */
  virtual void Shutdown() = 0;
};

} // end namespace strategy
} // end namespace interfaces
} // end namespace constellation
