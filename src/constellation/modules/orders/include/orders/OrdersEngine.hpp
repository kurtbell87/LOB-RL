#pragma once

#include <memory>
#include <optional>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include "interfaces/logging/ILogger.hpp"
#include "interfaces/orders/IOrdersEngine.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/orders/IOrderEvents.hpp"

namespace constellation {
namespace orders {

/**
 * @class OrdersEngine
 * @brief Concrete, final class implementing IOrdersEngine (including
 *        IOrdersCommand & IOrdersQuery) with a single-writer concurrency model.
 *
 * In this refactored version, all references to price are stored as std::int64_t
 * (nano scale). The fill logic no longer uses double for price arithmetic.
 */
class OrdersEngine final : public constellation::interfaces::orders::IOrdersEngine {
public:
  explicit OrdersEngine(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);
  ~OrdersEngine() override;

  /**
   * @brief Allow hooking an external events observer (e.g., fill logger).
   */
  void SetOrderEvents(std::shared_ptr<constellation::interfaces::orders::IOrderEvents> events);

  // -----------------------------
  // IOrdersCommand interface
  // -----------------------------
  std::uint64_t PlaceOrder(const constellation::interfaces::orders::OrderSpec& order_spec) override;
  bool ModifyOrder(std::uint64_t order_id, const constellation::interfaces::orders::OrderUpdate& update) override;
  bool CancelOrder(std::uint64_t order_id) override;

  // -----------------------------
  // IOrdersQuery interface
  // -----------------------------
  constellation::interfaces::orders::OrderStatus GetOrderStatus(std::uint64_t order_id) const override;
  std::optional<constellation::interfaces::orders::OrderInfo> GetOrderInfo(std::uint64_t order_id) const override;
  std::vector<constellation::interfaces::orders::OrderInfo> ListOpenOrders(std::uint32_t instrument_id) const override;

  // -----------------------------
  // IOrdersEngine interface
  // -----------------------------
  void OnMarketViewUpdate(const constellation::interfaces::orderbook::IMarketView* market_view) override;
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override;

  /**
   * @brief Sets the aggregator chunk-based timestamp to be used for fill events
   *        on the next OnMarketViewUpdate(...) call.
   */
  void SetCurrentTimestamp(std::uint64_t ts) override;

private:
  /**
   * @struct Impl
   * @brief Hidden PIMPL data to maintain concurrency and order info
   */
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // end namespace orders
} // end namespace constellation
