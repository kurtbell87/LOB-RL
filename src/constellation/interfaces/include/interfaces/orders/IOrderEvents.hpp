#pragma once

#include <cstdint>
#include "interfaces/orders/IOrderModels.hpp"

namespace constellation {
namespace interfaces {
namespace orders {

/**
 * @brief IOrderEvents allows external observers to receive notifications
 *        when orders fill, cancel, or reach final states.
 *
 * In this refactored version, fill_price is now an int64_t
 * so that we keep the price in nano-scale if needed.
 */
class IOrderEvents {
public:
  virtual ~IOrderEvents() = default;

  /**
   * @brief Called when a partial or full fill occurs. 
   *        (Older, minimal signature: no timestamp, instrument, or side.)
   *
   * @param order_id    The ID of the order that got filled
   * @param fill_price  The price at which the fill occurred (int64 nano price)
   * @param fill_qty    The quantity that was filled in this event
   */
  virtual void OnOrderFilled(std::uint64_t order_id,
                             std::int64_t fill_price,
                             std::uint32_t fill_qty)
  {
    // Default no-op (or minimal). 
    (void)order_id;
    (void)fill_price;
    (void)fill_qty;
  }

  /**
   * @brief Called when an order transitions to a final state (Filled, Canceled, Rejected, Expired).
   * @param order_id    The ID of the order
   * @param final_state A descriptive final state code
   */
  virtual void OnOrderDone(std::uint64_t order_id,
                           OrderFinalState final_state) = 0;

  /**
   * @brief Extended fill callback including chunk-based timestamp, instrument, side.
   *        Default implementation calls OnOrderFilled(...) so existing code isn't broken.
   *
   * @param timestamp      The approximate chunk or message timestamp
   * @param order_id       Which order got filled
   * @param instrument_id  The instrument ID of this order
   * @param side           Buy or Sell
   * @param fill_price     Fill/Trade price as int64
   * @param fill_qty       Quantity filled in this event
   */
  virtual void OnOrderFillWithTimestamp(std::uint64_t timestamp,
                                        std::uint64_t order_id,
                                        std::uint32_t instrument_id,
                                        OrderSide side,
                                        std::int64_t fill_price,
                                        std::uint32_t fill_qty)
  {
    // By default, fallback to old OnOrderFilled for partial compatibility
    OnOrderFilled(order_id, fill_price, fill_qty);
  }
};

} // end namespace orders
} // end namespace interfaces
} // end namespace constellation
