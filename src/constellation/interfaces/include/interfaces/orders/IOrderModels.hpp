#pragma once

#include <cstdint>
#include <string>
#include <optional>

/**
 * @brief This header collects basic data structures relevant to orders.
 *        They do not depend on any specific implementation, so we place
 *        them in module_interfaces.
 */

namespace constellation::interfaces::orders {

/**
 * @brief Supported order types.
 */
enum class OrderType {
  Market,
  Limit,
  Stop,
  StopLimit,
  // Potential expansions...
};

/**
 * @brief Time In Force specification
 */
enum class TimeInForce {
  Day,
  GTC,  // Good 'Til Canceled
  IOC,  // Immediate or Cancel
  FOK   // Fill or Kill
};

/**
 * @brief The side of the order
 */
enum class OrderSide {
  Buy,
  Sell
};

/**
 * @brief High-level final states for an order
 */
enum class OrderFinalState {
  Filled,
  Canceled,
  Rejected,
  Expired
};

/**
 * @brief Overall status of an order (including partial fill)
 */
enum class OrderStatus {
  New,
  PartiallyFilled,
  Filled,
  Canceled,
  Rejected,
  Expired,
  Unknown
};

/**
 * @brief A bracket order specification:
 *        - optional stopLossPrice, takeProfitPrice
 *        - once the main order is filled, child orders can be generated
 *
 *  We store these bracket prices also as int64_t (nano scale).
 */
struct BracketOrderSpec {
  bool use_bracket{false};
  std::int64_t stopLossPrice{0};     // nano-scale stop loss
  std::int64_t takeProfitPrice{0};   // nano-scale take profit
};

/**
 * @brief The input needed to place an order. The OrdersEngine will assign
 *        a unique order_id automatically.
 *
 *  Now uses int64_t for limit_price & stop_price.
 */
struct OrderSpec {
  std::uint32_t instrument_id{0};
  OrderType     type{OrderType::Limit};
  OrderSide     side{OrderSide::Buy};
  std::uint32_t quantity{0};
  std::int64_t  limit_price{0}; // used if type=Limit or StopLimit (nano scale)
  std::int64_t  stop_price{0};  // used if type=Stop or StopLimit (nano scale)
  TimeInForce   tif{TimeInForce::GTC};
  BracketOrderSpec bracket;     // optional bracket config
  std::string   strategy_tag;   // optional free-form strategy descriptor
};

/**
 * @brief Describes which fields to update in an existing order.
 *        Typically, we might allow updating limit_price or quantity.
 */
struct OrderUpdate {
  std::optional<std::int64_t> new_limit_price;
  std::optional<std::uint32_t> new_quantity;
};

/**
 * @brief Detailed info about an order for read-only queries (IOrdersQuery).
 *
 *        Also refactored so that limit_price, stop_price, avg_fill_price
 *        are stored as int64_t in nano scale.
 */
struct OrderInfo {
  std::uint64_t order_id{0};
  std::uint32_t instrument_id{0};
  OrderType     type{OrderType::Limit};
  OrderSide     side{OrderSide::Buy};
  std::uint32_t original_quantity{0};
  std::uint32_t filled_quantity{0};
  std::int64_t  avg_fill_price{0};  // aggregated fill price in nano scale
  std::int64_t  limit_price{0};
  std::int64_t  stop_price{0};
  TimeInForce   tif{TimeInForce::GTC};
  OrderStatus   status{OrderStatus::Unknown};

  // bracket info
  bool   bracket_active{false};
  std::int64_t bracket_stop_loss{0};
  std::int64_t bracket_take_profit{0};
};

} // end namespace constellation::interfaces::orders
