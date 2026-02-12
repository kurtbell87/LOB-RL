#pragma once

#include <cstdint>
#include "interfaces/orders/IOrdersCommand.hpp"
#include "interfaces/orders/IOrdersQuery.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation {
namespace interfaces {
namespace orders {

/**
 * @brief IOrdersEngine extends IOrdersCommand and IOrdersQuery
 *        with a market-view update hook for real-time data integration,
 *        and an optional method to set the chunk-based timestamp.
 *
 * This unifies place/modify/cancel queries + read queries + real-time
 * fill generation.
 */
class IOrdersEngine
  : public IOrdersCommand,
    public IOrdersQuery
{
public:
  virtual ~IOrdersEngine() = default;

  /**
   * @brief OnMarketViewUpdate is called whenever the orchestrator
   *        has a fresh snapshot or pointer to the market view.
   *
   * The engine can decide if any orders are fillable based on best bid/ask
   * (which are now int64_t at nano scale if consistent with the LOB).
   */
  virtual void OnMarketViewUpdate(const constellation::interfaces::orderbook::IMarketView* market_view) = 0;

  /**
   * @brief Return interface version info for debugging or introspection.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Sets the aggregator chunk-based timestamp to be used for fill events
   *        generated on the next OnMarketViewUpdate(...) call.
   *
   * @param ts  The aggregator's chunk-based timestamp (e.g. lastChunkMaxTimestamp).
   */
  virtual void SetCurrentTimestamp(std::uint64_t ts) = 0;
};

} // end namespace orders
} // end namespace interfaces
} // end namespace constellation
