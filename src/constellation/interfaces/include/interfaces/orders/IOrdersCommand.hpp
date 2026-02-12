#pragma once

#include <cstdint>
#include "interfaces/orders/IOrderModels.hpp"

namespace constellation::interfaces::orders {

/**
 * @brief IOrdersCommand defines the command-oriented methods
 *        for placing, modifying, and canceling orders.
 *
 * Strategies or other modules use this interface to send new orders or changes.
 */
class IOrdersCommand {
public:
  virtual ~IOrdersCommand() = default;

  /**
   * @brief Place a new order using a specified OrderSpec (which now uses int64_t prices).
   * @param order_spec The specification (type, side, quantity, prices, TIF, bracket info, etc.)
   * @return A unique numeric order ID assigned by the system.
   */
  virtual std::uint64_t PlaceOrder(const OrderSpec& order_spec) = 0;

  /**
   * @brief Modify an existing order by ID. Only certain fields (like limit price, quantity) are
   *        typically modifiable. Each OrderUpdate describes what to change.
   * @param order_id The ID of the order to modify
   * @param update The fields to be updated
   * @return true if modification succeeded, false if not found or not allowed
   */
  virtual bool ModifyOrder(std::uint64_t order_id, const OrderUpdate& update) = 0;

  /**
   * @brief Cancel an order that is not yet fully filled or canceled.
   * @param order_id The ID of the order to cancel
   * @return true if the order was canceled, false if not found or already in final state
   */
  virtual bool CancelOrder(std::uint64_t order_id) = 0;
};

} // end namespace constellation::interfaces::orders
