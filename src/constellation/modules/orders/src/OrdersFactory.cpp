#include "orders/OrdersFactory.hpp"
#include "orders/OrdersEngine.hpp"
#include "interfaces/logging/NullLogger.hpp"

namespace constellation {
namespace modules {
namespace orders {

std::pair<
    std::shared_ptr<constellation::interfaces::orders::IOrdersCommand>,
    std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>
>
CreateOrdersEngine(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
{
    if (!logger) {
        logger = std::make_shared<constellation::interfaces::logging::NullLogger>();
    }
    // Construct the internal OrdersEngine, which implements both interfaces
    auto engineImpl = std::make_shared<constellation::orders::OrdersEngine>(logger);

    // Upcast to IOrdersCommand + IOrdersQuery
    std::shared_ptr<constellation::interfaces::orders::IOrdersCommand> cmd = engineImpl;
    std::shared_ptr<constellation::interfaces::orders::IOrdersQuery>   qry = engineImpl;

    return {cmd, qry};
}

std::shared_ptr<constellation::interfaces::orders::IOrdersEngine>
CreateIOrdersEngine(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
{
    if (!logger) {
        logger = std::make_shared<constellation::interfaces::logging::NullLogger>();
    }
    // Construct the concrete OrdersEngine
    auto engineImpl = std::make_shared<constellation::orders::OrdersEngine>(logger);

    // Upcast to IOrdersEngine
    return std::static_pointer_cast<constellation::interfaces::orders::IOrdersEngine>(engineImpl);
}

} // end namespace orders
} // end namespace modules
} // end namespace constellation
