#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "interfaces/orders/IOrderModels.hpp"

namespace constellation {
namespace interfaces {
namespace orders {

/**
 * @brief IOrdersQuery defines the query-oriented (read-only) methods
 *        for inspecting order states.
 */
class IOrdersQuery {
public:
  virtual ~IOrdersQuery() = default;

  /**
   * @brief Retrieve the current status of a specific order.
   * @param order_id The ID of the order
   * @return The order status (New, PartiallyFilled, Filled, Canceled, Rejected, Expired, Unknown).
   */
  virtual OrderStatus GetOrderStatus(std::uint64_t order_id) const = 0;

  /**
   * @brief Get detailed info (filled qty, average fill price, etc.) about a specific order.
   * @param order_id The ID of the order
   * @return An OrderInfo struct if found, else std::nullopt
   */
  virtual std::optional<OrderInfo> GetOrderInfo(std::uint64_t order_id) const = 0;

  /**
   * @brief List all open orders for the specified instrument, or all if instrument_id=0.
   * @param instrument_id If 0, list across all instruments. Otherwise filter by instrument_id.
   */
  virtual std::vector<OrderInfo> ListOpenOrders(std::uint32_t instrument_id) const = 0;
};

} // end namespace orders
} // end namespace interfaces
} // end namespace constellation
