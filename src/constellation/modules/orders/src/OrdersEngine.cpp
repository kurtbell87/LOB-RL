#include "orders/OrdersEngine.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include "interfaces/orders/IOrderEvents.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

#include <stdexcept>
#include <shared_mutex>
#include <unordered_map>

namespace constellation {
namespace orders {

/**
 * @struct InternalOrder
 * @brief Replaces all references to double price with int64_t (nano).
 *
 * We keep a running sum for fill price to compute an average in integer domain.
 */
struct InternalOrder {
  std::uint64_t order_id;
  std::uint32_t instrument_id;
  constellation::interfaces::orders::OrderType  type;
  constellation::interfaces::orders::OrderSide  side;
  std::uint32_t original_qty;
  std::uint32_t remaining_qty;
  std::int64_t  limit_price;  // nano
  std::int64_t  stop_price;   // nano
  constellation::interfaces::orders::TimeInForce tif;
  constellation::interfaces::orders::OrderStatus status;
  // For computing average fill price in integer domain:
  // total_fill_cost = sum of (fill_price * fill_qty)
  // then avg_fill_price = total_fill_cost / filled_qty_so_far
  std::int64_t  total_fill_cost{0};  
  std::uint64_t filled_qty_so_far{0};

  std::string   strategy_tag;

  // bracket logic
  bool   bracket_parent{false};
  std::int64_t bracket_stop_loss{0};   // nano
  std::int64_t bracket_take_profit{0}; // nano
  bool   bracket_child{false};
  std::uint64_t bracket_parent_id{0};
  std::uint64_t bracket_sl_order_id{0};
  std::uint64_t bracket_tp_order_id{0};
};

/**
 * @class OrdersEngine::Impl
 * @brief Holds concurrency, order map, and fill logic for the OrdersEngine.
 */
class OrdersEngine::Impl {
public:
  explicit Impl(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
    : logger_(std::move(logger))
  {}

  mutable std::shared_mutex mtx_;
  std::unordered_map<std::uint64_t, InternalOrder> orders_;
  std::uint64_t next_order_id_{1};

  // The external events observer (fill logger, etc.)
  std::shared_ptr<constellation::interfaces::orders::IOrderEvents> events_;
  // Logger
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  // aggregator chunk-based timestamp for fill events
  std::uint64_t current_timestamp_{0};

  // ---- fill logic / helper methods ----
  static bool IsStopTriggered(const InternalOrder& ord,
                              const constellation::interfaces::orderbook::IMarketView* view);
  static void TriggerStop(InternalOrder& ord);

  static std::pair<std::int64_t,std::uint32_t> ComputeMarketFill(
      const InternalOrder& ord,
      const constellation::interfaces::orderbook::IMarketView* market);
  static std::pair<std::int64_t,std::uint32_t> ComputeLimitFill(
      const InternalOrder& ord,
      const constellation::interfaces::orderbook::IMarketView* market);

  void ApplyFill(InternalOrder& ord,
                 std::int64_t fill_price,
                 std::uint32_t fill_qty);

  void MarkOrderDone(InternalOrder& ord,
                     constellation::interfaces::orders::OrderFinalState final_state);

  void CreateBracketChildrenIfNeeded(InternalOrder& parent);
};

// ---------------------------
// OrdersEngine constructor
// ---------------------------
OrdersEngine::OrdersEngine(
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
  : impl_(std::make_unique<Impl>(
       logger ? logger 
              : std::make_shared<constellation::interfaces::logging::NullLogger>())
    )
{}

OrdersEngine::~OrdersEngine() = default;

void OrdersEngine::SetOrderEvents(
    std::shared_ptr<constellation::interfaces::orders::IOrderEvents> events)
{
  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);
  impl_->events_ = std::move(events);
}

// ---------------------------
// IOrdersCommand
// ---------------------------
std::uint64_t OrdersEngine::PlaceOrder(
    const constellation::interfaces::orders::OrderSpec& spec)
{
  using namespace constellation::interfaces::orders;
  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);

  if (spec.quantity == 0) {
    impl_->logger_->Warn("OrdersEngine::PlaceOrder => zero quantity not allowed");
    return 0ULL;
  }

  std::uint64_t oid = impl_->next_order_id_++;
  InternalOrder ord;
  ord.order_id       = oid;
  ord.instrument_id  = spec.instrument_id;
  ord.type           = spec.type;
  ord.side           = spec.side;
  ord.original_qty   = spec.quantity;
  ord.remaining_qty  = spec.quantity;
  ord.limit_price    = spec.limit_price; // already nano
  ord.stop_price     = spec.stop_price;  // nano
  ord.tif            = spec.tif;
  ord.status         = OrderStatus::New;
  ord.total_fill_cost= 0;
  ord.filled_qty_so_far = 0;
  ord.strategy_tag   = spec.strategy_tag;

  if (spec.bracket.use_bracket) {
    ord.bracket_parent      = true;
    ord.bracket_stop_loss   = spec.bracket.stopLossPrice;   // nano
    ord.bracket_take_profit = spec.bracket.takeProfitPrice; // nano
  }

  impl_->orders_[oid] = std::move(ord);
  impl_->logger_->Debug("OrdersEngine::PlaceOrder => new order_id={}", oid);

  return oid;
}

bool OrdersEngine::ModifyOrder(
    std::uint64_t order_id,
    const constellation::interfaces::orders::OrderUpdate& update)
{
  using namespace constellation::interfaces::orders;
  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);
  auto it = impl_->orders_.find(order_id);
  if (it == impl_->orders_.end()) {
    impl_->logger_->Warn("OrdersEngine::ModifyOrder => order_id={} not found", order_id);
    return false;
  }
  auto& ord = it->second;
  if (   ord.status == OrderStatus::Filled
      || ord.status == OrderStatus::Canceled
      || ord.status == OrderStatus::Rejected
      || ord.status == OrderStatus::Expired)
  {
    impl_->logger_->Debug("OrdersEngine::ModifyOrder => order_id={} already final", order_id);
    return false;
  }

  // limit price
  if (update.new_limit_price.has_value()) {
    ord.limit_price = update.new_limit_price.value();
  }
  // quantity
  if (update.new_quantity.has_value()) {
    std::uint32_t newQty = update.new_quantity.value();
    std::uint64_t alreadyFilled = ord.filled_qty_so_far;
    if (newQty < alreadyFilled) {
      // The user is effectively setting a new quantity below the filled portion => fully filled
      ord.remaining_qty = 0;
      ord.status = OrderStatus::Filled;
      impl_->MarkOrderDone(ord, OrderFinalState::Filled);
      impl_->CreateBracketChildrenIfNeeded(ord);
    } else {
      std::uint32_t remain = static_cast<std::uint32_t>(newQty - alreadyFilled);
      ord.remaining_qty = remain;
      ord.original_qty  = newQty;
      if (remain == 0) {
        ord.status = OrderStatus::Filled;
        impl_->MarkOrderDone(ord, OrderFinalState::Filled);
        impl_->CreateBracketChildrenIfNeeded(ord);
      }
    }
  }
  return true;
}

bool OrdersEngine::CancelOrder(std::uint64_t order_id) {
  using namespace constellation::interfaces::orders;
  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);
  auto it = impl_->orders_.find(order_id);
  if (it == impl_->orders_.end()) {
    impl_->logger_->Warn("OrdersEngine::CancelOrder => order_id={} not found", order_id);
    return false;
  }
  auto& ord = it->second;
  if (   ord.status == OrderStatus::New 
      || ord.status == OrderStatus::PartiallyFilled)
  {
    ord.status = OrderStatus::Canceled;
    impl_->MarkOrderDone(ord, OrderFinalState::Canceled);
    return true;
  }
  return false;
}

// ---------------------------
// IOrdersQuery
// ---------------------------
constellation::interfaces::orders::OrderStatus
OrdersEngine::GetOrderStatus(std::uint64_t order_id) const
{
  std::shared_lock<std::shared_mutex> lock(impl_->mtx_);
  auto it = impl_->orders_.find(order_id);
  if (it == impl_->orders_.end()) {
    return constellation::interfaces::orders::OrderStatus::Unknown;
  }
  return it->second.status;
}

std::optional<constellation::interfaces::orders::OrderInfo>
OrdersEngine::GetOrderInfo(std::uint64_t order_id) const
{
  using namespace constellation::interfaces::orders;
  std::shared_lock<std::shared_mutex> lock(impl_->mtx_);
  auto it = impl_->orders_.find(order_id);
  if (it == impl_->orders_.end()) {
    return std::nullopt;
  }
  const auto& o = it->second;
  OrderInfo info;
  info.order_id          = o.order_id;
  info.instrument_id     = o.instrument_id;
  info.type              = o.type;
  info.side              = o.side;
  info.original_quantity = o.original_qty;
  info.filled_quantity   = static_cast<std::uint32_t>(o.filled_qty_so_far);
  if (o.filled_qty_so_far == 0) {
    info.avg_fill_price = 0;
  } else {
    // integer average => total_fill_cost / fill_qty
    std::int64_t avg = static_cast<std::int64_t>(
        o.total_fill_cost / o.filled_qty_so_far);
    info.avg_fill_price = avg;
  }
  info.limit_price       = o.limit_price;
  info.stop_price        = o.stop_price;
  info.tif               = o.tif;
  info.status            = o.status;

  if (o.bracket_parent) {
    info.bracket_active      = true;
    info.bracket_stop_loss   = o.bracket_stop_loss;
    info.bracket_take_profit = o.bracket_take_profit;
  }
  return info;
}

std::vector<constellation::interfaces::orders::OrderInfo>
OrdersEngine::ListOpenOrders(std::uint32_t instrument_id) const
{
  using namespace constellation::interfaces::orders;
  std::shared_lock<std::shared_mutex> lock(impl_->mtx_);
  std::vector<OrderInfo> result;
  result.reserve(impl_->orders_.size());
  for (auto& kv : impl_->orders_) {
    const auto& o = kv.second;
    if (instrument_id != 0 && o.instrument_id != instrument_id) {
      continue;
    }
    if (   o.status == OrderStatus::New
        || o.status == OrderStatus::PartiallyFilled)
    {
      OrderInfo info;
      info.order_id         = o.order_id;
      info.instrument_id    = o.instrument_id;
      info.type             = o.type;
      info.side             = o.side;
      info.original_quantity= o.original_qty;
      info.filled_quantity  = static_cast<std::uint32_t>(o.filled_qty_so_far);
      if (o.filled_qty_so_far == 0) {
        info.avg_fill_price = 0;
      } else {
        std::int64_t avg = static_cast<std::int64_t>(
            o.total_fill_cost / o.filled_qty_so_far);
        info.avg_fill_price = avg;
      }
      info.limit_price      = o.limit_price;
      info.stop_price       = o.stop_price;
      info.tif              = o.tif;
      info.status           = o.status;
      if (o.bracket_parent) {
        info.bracket_active      = true;
        info.bracket_stop_loss   = o.bracket_stop_loss;
        info.bracket_take_profit = o.bracket_take_profit;
      }
      result.push_back(info);
    }
  }
  return result;
}

// ---------------------------
// IOrdersEngine
// ---------------------------
void OrdersEngine::OnMarketViewUpdate(
    const constellation::interfaces::orderbook::IMarketView* market_view)
{
  using namespace constellation::interfaces::orders;
  if (!market_view) {
    return;
  }

  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);
  for (auto& kv : impl_->orders_) {
    auto& ord = kv.second;
    if (   ord.status != OrderStatus::New 
        && ord.status != OrderStatus::PartiallyFilled)
    {
      continue;
    }

    // If Stop or StopLimit, check if triggered:
    if ((ord.type == OrderType::Stop || ord.type == OrderType::StopLimit)) {
      if (Impl::IsStopTriggered(ord, market_view)) {
        Impl::TriggerStop(ord); 
        // Now ord.type is either Market or Limit if triggered. 
      } else {
        // Not triggered => skip fill logic
        continue;
      }
    }

    // Next we do fill logic based on the updated type:
    std::int64_t px = 0;
    std::uint32_t fillQty = 0;

    if (ord.type == OrderType::Market) {
      auto ret = Impl::ComputeMarketFill(ord, market_view);
      px = ret.first;
      fillQty = ret.second;
    } else if (ord.type == OrderType::Limit) {
      auto ret = Impl::ComputeLimitFill(ord, market_view);
      px = ret.first;
      fillQty = ret.second;
    } else {
      // if still Stop or StopLimit => means not triggered or some invalid state => skip
      continue;
    }

    if (fillQty > 0 && px != 0) {
      impl_->ApplyFill(ord, px, fillQty);
    }
    if (ord.remaining_qty == 0) {
      ord.status = OrderStatus::Filled;
      impl_->MarkOrderDone(ord, OrderFinalState::Filled);
      impl_->CreateBracketChildrenIfNeeded(ord);
    } else if (ord.filled_qty_so_far > 0 && ord.status == OrderStatus::New) {
      ord.status = OrderStatus::PartiallyFilled;
    }
  }
}

void OrdersEngine::SetCurrentTimestamp(std::uint64_t ts) {
  std::unique_lock<std::shared_mutex> lock(impl_->mtx_);
  impl_->current_timestamp_ = ts;
}

constellation::interfaces::common::InterfaceVersionInfo
OrdersEngine::GetVersionInfo() const noexcept {
  return {1, 0};
}

//---------------------------
// Implementation helpers
//---------------------------
bool OrdersEngine::Impl::IsStopTriggered(
    const InternalOrder& ord,
    const constellation::interfaces::orderbook::IMarketView* view)
{
  using namespace constellation::interfaces::orderbook;
  if (!view) return false;
  // For a STOP-BUY, typical trigger if bestAsk >= stop_price
  // For a STOP-SELL, typical trigger if bestBid <= stop_price
  auto bestBidOpt = view->GetBestBid(ord.instrument_id);
  auto bestAskOpt = view->GetBestAsk(ord.instrument_id);

  if (ord.side == constellation::interfaces::orders::OrderSide::Buy && bestAskOpt) {
    std::int64_t askPx = bestAskOpt->price; 
    // STOP-BUY triggers if ask >= stop_price
    if (askPx >= ord.stop_price && askPx > 0) {
      return true;
    }
  } else if (ord.side == constellation::interfaces::orders::OrderSide::Sell && bestBidOpt) {
    std::int64_t bidPx = bestBidOpt->price;
    // STOP-SELL triggers if bid <= stop_price
    if (bidPx <= ord.stop_price && bidPx > 0) {
      return true;
    }
  }
  return false;
}

void OrdersEngine::Impl::TriggerStop(InternalOrder& ord) {
  using namespace constellation::interfaces::orders;
  // Turn STOP -> Market, STOPLIMIT -> Limit
  if (ord.type == OrderType::Stop) {
    ord.type = OrderType::Market;
  } else if (ord.type == OrderType::StopLimit) {
    ord.type = OrderType::Limit;
  }
}

std::pair<std::int64_t,std::uint32_t>
OrdersEngine::Impl::ComputeMarketFill(
    const InternalOrder& ord,
    const constellation::interfaces::orderbook::IMarketView* view)
{
  using namespace constellation::interfaces::orderbook;
  // Market buy => fill at bestAsk if ask>0
  // Market sell => fill at bestBid if bid>0
  if (!view) {
    return {0, 0};
  }

  if (ord.side == constellation::interfaces::orders::OrderSide::Buy) {
    auto bestAskOpt = view->GetBestAsk(ord.instrument_id);
    if (!bestAskOpt) {
      return {0, 0};
    }
    std::int64_t askPx = bestAskOpt->price;
    std::uint64_t askSz = bestAskOpt->total_quantity;
    if (askPx <= 0 || askSz == 0) {
      return {0, 0};
    }
    std::uint32_t fillQty = (askSz < ord.remaining_qty)
                              ? static_cast<std::uint32_t>(askSz)
                              : ord.remaining_qty;
    if (fillQty > 0) {
      return {askPx, fillQty};
    }
  } else {
    // Sell => fill at bestBid
    auto bestBidOpt = view->GetBestBid(ord.instrument_id);
    if (!bestBidOpt) {
      return {0, 0};
    }
    std::int64_t bidPx = bestBidOpt->price;
    std::uint64_t bidSz = bestBidOpt->total_quantity;
    if (bidPx <= 0 || bidSz == 0) {
      return {0, 0};
    }
    std::uint32_t fillQty = (bidSz < ord.remaining_qty)
                              ? static_cast<std::uint32_t>(bidSz)
                              : ord.remaining_qty;
    if (fillQty > 0) {
      return {bidPx, fillQty};
    }
  }
  return {0, 0};
}

std::pair<std::int64_t,std::uint32_t>
OrdersEngine::Impl::ComputeLimitFill(
    const InternalOrder& ord,
    const constellation::interfaces::orderbook::IMarketView* view)
{
  using namespace constellation::interfaces::orderbook;
  // For buy => fill if bestAsk <= limit_price
  // For sell => fill if bestBid >= limit_price
  if (!view) {
    return {0, 0};
  }

  if (ord.side == constellation::interfaces::orders::OrderSide::Buy) {
    auto bestAskOpt = view->GetBestAsk(ord.instrument_id);
    if (!bestAskOpt) {
      return {0, 0};
    }
    std::int64_t askPx = bestAskOpt->price;
    if (askPx <= 0) {
      return {0,0};
    }
    if (askPx <= ord.limit_price) {
      std::uint64_t askSz = bestAskOpt->total_quantity;
      std::uint32_t fillQty = (askSz < ord.remaining_qty)
                                ? static_cast<std::uint32_t>(askSz)
                                : ord.remaining_qty;
      if (fillQty > 0) {
        return {askPx, fillQty};
      }
    }
  } else {
    // Sell => fill if bestBid >= limit_price
    auto bestBidOpt = view->GetBestBid(ord.instrument_id);
    if (!bestBidOpt) {
      return {0, 0};
    }
    std::int64_t bidPx = bestBidOpt->price;
    if (bidPx <= 0) {
      return {0,0};
    }
    if (bidPx >= ord.limit_price) {
      std::uint64_t bidSz = bestBidOpt->total_quantity;
      std::uint32_t fillQty = (bidSz < ord.remaining_qty)
                                ? static_cast<std::uint32_t>(bidSz)
                                : ord.remaining_qty;
      if (fillQty > 0) {
        return {bidPx, fillQty};
      }
    }
  }
  return {0,0};
}

/**
 * @brief Actually apply the fill to the order: reduce remaining qty,
 *        update total_fill_cost, call events.
 */
void OrdersEngine::Impl::ApplyFill(InternalOrder& ord,
                                   std::int64_t fill_price,
                                   std::uint32_t fill_qty)
{
  if (fill_qty == 0) return;
  if (fill_price <= 0) return;

  if (fill_qty > ord.remaining_qty) {
    fill_qty = ord.remaining_qty;
  }

  ord.remaining_qty -= fill_qty;
  // update cost for average fill
  // total_fill_cost += fill_price * fill_qty
  std::int64_t fillCost = static_cast<std::int64_t>(fill_price) 
                        * static_cast<std::int64_t>(fill_qty);
  ord.total_fill_cost += fillCost;
  ord.filled_qty_so_far += fill_qty;

  // Raise event
  if (events_) {
    events_->OnOrderFillWithTimestamp(
      current_timestamp_,
      ord.order_id,
      ord.instrument_id,
      ord.side,
      fill_price,
      fill_qty
    );
  }
}

void OrdersEngine::Impl::MarkOrderDone(
    InternalOrder& ord,
    constellation::interfaces::orders::OrderFinalState final_state)
{
  if (events_) {
    events_->OnOrderDone(ord.order_id, final_state);
  }
}

void OrdersEngine::Impl::CreateBracketChildrenIfNeeded(InternalOrder& parent)
{
  using namespace constellation::interfaces::orders;
  if (!parent.bracket_parent) return;
  std::int64_t slp = parent.bracket_stop_loss;
  std::int64_t tpp = parent.bracket_take_profit;
  if (slp <= 0 && tpp <= 0) return;

  OrderSide childSide = (parent.side == OrderSide::Buy)
                        ? OrderSide::Sell
                        : OrderSide::Buy;
  std::uint32_t qty = parent.original_qty;

  if (slp > 0) {
    // create STOP child
    InternalOrder stChild;
    stChild.order_id       = next_order_id_++;
    stChild.instrument_id  = parent.instrument_id;
    stChild.type           = OrderType::Stop;
    stChild.side           = childSide;
    stChild.original_qty   = qty;
    stChild.remaining_qty  = qty;
    stChild.limit_price    = 0;
    stChild.stop_price     = slp;
    stChild.tif            = TimeInForce::GTC;
    stChild.status         = OrderStatus::New;
    stChild.bracket_child  = true;
    stChild.bracket_parent_id = parent.order_id;
    orders_[stChild.order_id] = stChild;
    parent.bracket_sl_order_id = stChild.order_id;
  }

  if (tpp > 0) {
    // create LIMIT child
    InternalOrder limChild;
    limChild.order_id       = next_order_id_++;
    limChild.instrument_id  = parent.instrument_id;
    limChild.type           = OrderType::Limit;
    limChild.side           = childSide;
    limChild.original_qty   = qty;
    limChild.remaining_qty  = qty;
    limChild.limit_price    = tpp;
    limChild.stop_price     = 0;
    limChild.tif            = TimeInForce::GTC;
    limChild.status         = OrderStatus::New;
    limChild.bracket_child  = true;
    limChild.bracket_parent_id = parent.order_id;
    orders_[limChild.order_id] = limChild;
    parent.bracket_tp_order_id = limChild.order_id;
  }
}

} // end namespace orders
} // end namespace constellation
