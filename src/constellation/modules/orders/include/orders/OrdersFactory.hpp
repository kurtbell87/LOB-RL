#pragma once

#include <memory>
#include <utility>
#include "interfaces/logging/ILogger.hpp"
#include "interfaces/orders/IOrdersCommand.hpp"
#include "interfaces/orders/IOrdersQuery.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"

namespace constellation {
namespace modules {
namespace orders {

/**
 * @brief Create a combined OrdersEngine that implements both
 *        IOrdersCommand and IOrdersQuery. Callers receive two pointers
 *        referencing the same underlying engine object.
 *
 * Example usage for pure C++:
 *   auto [cmd, qry] = CreateOrdersEngine(logger);
 *   auto oid = cmd->PlaceOrder(...);
 *   auto info = qry->GetOrderInfo(oid);
 */
std::pair<
    std::shared_ptr<constellation::interfaces::orders::IOrdersCommand>,
    std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>
>
CreateOrdersEngine(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

/**
 * @brief Create a single pointer to IOrdersEngine, which unifies both
 *        IOrdersCommand and IOrdersQuery in one interface. Useful for
 *        Python bindings or orchestrators that want a single pointer.
 *
 * Internally, this returns the same concrete OrdersEngine but upcast
 * to the IOrdersEngine interface.
 */
std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>
CreateIOrdersEngine(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

} // end namespace orders
} // end namespace modules
} // end namespace constellation
