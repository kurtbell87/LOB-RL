#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include <unordered_map>
#include "orders/OrdersEngine.hpp"
#include "interfaces/orders/IOrderEvents.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

using namespace constellation::orders;

using constellation::interfaces::orderbook::PriceLevel;
using constellation::interfaces::orderbook::IMarketView;
using constellation::interfaces::orders::OrderType;
using constellation::interfaces::orders::OrderSide;
using constellation::interfaces::orders::TimeInForce;
using constellation::interfaces::orders::OrderStatus;
using constellation::interfaces::orders::OrderFinalState;
using constellation::interfaces::orders::OrderInfo;
using constellation::interfaces::orders::IOrderEvents;
using constellation::interfaces::orders::OrderSpec;
using constellation::interfaces::orders::OrderUpdate;

/**
 * @brief A simple IOrderEvents to track fill notifications for test verification.
 *        Because the new interface uses std::int64_t for fill_price, we store
 *        it here internally as a double for convenience in these tests.
 */
class TestOrderEvents : public IOrderEvents {
public:
  struct FillEvent {
    std::uint64_t order_id;
    double px;  // we store as double, but the engine gives us int64 in nano
    std::uint32_t qty;
  };
  struct DoneEvent {
    std::uint64_t order_id;
    OrderFinalState final_state;
  };

  std::vector<FillEvent> fills;
  std::vector<DoneEvent> dones;

  // Updated signature to match the new IOrderEvents
  void OnOrderFilled(std::uint64_t order_id,
                     std::int64_t fill_price,
                     std::uint32_t fill_qty) override
  {
    // convert from nano-based int64 to a double for test display ( dividing by 1e9 )
    double px_double = static_cast<double>(fill_price) / 1e9;
    fills.push_back({order_id, px_double, fill_qty});
  }

  void OnOrderDone(std::uint64_t order_id,
                   OrderFinalState final_state) override
  {
    dones.push_back({order_id, final_state});
  }

  void OnOrderFillWithTimestamp(std::uint64_t timestamp,
                                std::uint64_t order_id,
                                std::uint32_t instrument_id,
                                constellation::interfaces::orders::OrderSide side,
                                std::int64_t fill_price,
                                std::uint32_t fill_qty) override
  {
    // By default, call older OnOrderFilled => or store the fill with extra info
    double px_double = static_cast<double>(fill_price) / 1e9;
    fills.push_back({order_id, px_double, fill_qty});
  }
};

/**
 * @brief A mock market view for testing.
 *        We do not do actual best-bid/ask logic, just store one instrument's data.
 *        Now uses int64 for bestBidPrice, bestAskPrice as required.
 */
class MockMarketView final : public IMarketView {
public:
  std::int64_t bid_px_nano{0};
  std::uint64_t bid_size{0};
  bool has_bid{false};

  std::int64_t ask_px_nano{0};
  std::uint64_t ask_size{0};
  bool has_ask{false};

  std::uint32_t stored_instrument_id{0};

  // IMarketStateView overrides
  std::optional<PriceLevel> GetBestBid(std::uint32_t instrument_id) const override {
    if (!has_bid || instrument_id != stored_instrument_id) {
      return std::nullopt;
    }
    PriceLevel pl;
    pl.price = bid_px_nano;
    pl.total_quantity = bid_size;
    pl.order_count = 1;
    return pl;
  }
  std::optional<PriceLevel> GetBestAsk(std::uint32_t instrument_id) const override {
    if (!has_ask || instrument_id != stored_instrument_id) {
      return std::nullopt;
    }
    PriceLevel pl;
    pl.price = ask_px_nano;
    pl.total_quantity = ask_size;
    pl.order_count = 1;
    return pl;
  }

  // IInstrumentRegistry
  std::size_t InstrumentCount() const noexcept override { return 1; }
  std::vector<std::uint32_t> GetInstrumentIds() const override {
    return { stored_instrument_id };
  }

  // IMarketStatistics
  std::uint64_t GetGlobalAddCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalCancelCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalModifyCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalTradeCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalClearCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalTotalEventCount() const noexcept override { return 0; }

  // IMarketView composite includes these:
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {1,0};
  }
};

// -----------------------------------------------------------------------------

TEST_CASE("Basic place/modify/cancel test", "[orders]") {
  OrdersEngine engine;
  // place an order
  OrderSpec spec;
  spec.instrument_id = 1234;
  spec.type = OrderType::Limit;
  spec.side = OrderSide::Buy;
  spec.quantity = 100;
  // Example: $50 => 50.0 => 50 * 1e9 => 50000000000
  spec.limit_price = 50000000000LL;  // $50.000000000

  std::uint64_t oid = engine.PlaceOrder(spec);
  REQUIRE(oid != 0ULL);

  // check status
  auto st = engine.GetOrderStatus(oid);
  CHECK(st == OrderStatus::New);

  // modify
  OrderUpdate upd;
  // Example new limit price $49.50 => 49500000000
  upd.new_limit_price = 49500000000LL;
  upd.new_quantity = 120; 
  bool ok = engine.ModifyOrder(oid, upd);
  CHECK(ok == true);

  auto infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->limit_price == 49500000000LL);
  CHECK(infoOpt->original_quantity == 120);

  // cancel
  bool canceled = engine.CancelOrder(oid);
  CHECK(canceled == true);

  st = engine.GetOrderStatus(oid);
  CHECK(st == OrderStatus::Canceled);
}

TEST_CASE("Market order fill logic test", "[orders][fills]") {
  auto events = std::make_shared<TestOrderEvents>();
  OrdersEngine engine;
  engine.SetOrderEvents(events);

  // Prepare a mock market with ask_px_nano=49000000000 for instrument=1234
  MockMarketView market;
  market.has_ask = true;
  market.ask_px_nano = 49000000000LL; // $49.000000000
  market.ask_size  = 50;
  market.stored_instrument_id = 1234;

  OrderSpec spec;
  spec.instrument_id = 1234;
  spec.type = OrderType::Market;
  spec.side = OrderSide::Buy;
  spec.quantity = 30;
  std::uint64_t oid = engine.PlaceOrder(spec);
  REQUIRE(oid != 0);

  engine.OnMarketViewUpdate(&market);

  auto infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->filled_quantity == 30);
  CHECK(infoOpt->status == OrderStatus::Filled);

  // The fill price in TestOrderEvents was stored as double from int64 fill
  REQUIRE(events->fills.size() == 1);
  CHECK(events->fills[0].qty == 30);
  // Should be 49.0 if the aggregator used ask_px_nano=49000000000 => 49.0
  CHECK(events->fills[0].px == Catch::Approx(49.0));

  REQUIRE(events->dones.size() == 1);
  CHECK(events->dones[0].final_state == OrderFinalState::Filled);
}

TEST_CASE("Limit order partial fill test", "[orders][fills]") {
  auto events = std::make_shared<TestOrderEvents>();
  OrdersEngine engine;
  engine.SetOrderEvents(events);

  // bestAsk=$100 => 100.0 => int64=100000000000
  MockMarketView market;
  market.has_ask = true;
  market.ask_px_nano = 100000000000LL; // $100.0
  market.ask_size  = 5;
  market.stored_instrument_id = 555;

  // place a limit buy at $101 => 101000000000
  OrderSpec spec;
  spec.instrument_id = 555;
  spec.type = OrderType::Limit;
  spec.side = OrderSide::Buy;
  spec.quantity = 10;
  spec.limit_price = 101000000000LL;
  auto oid = engine.PlaceOrder(spec);

  engine.OnMarketViewUpdate(&market);

  // partial fill of 5
  auto infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->filled_quantity == 5);
  CHECK(infoOpt->status == OrderStatus::PartiallyFilled);

  REQUIRE(events->fills.size() == 1);
  CHECK(events->fills[0].qty == 5);
  CHECK(events->fills[0].px == Catch::Approx(100.0));

  // next chunk => bestAsk changes => e.g. $99.50 => 99500000000
  market.ask_px_nano = 99500000000LL;
  market.ask_size  = 100; // plenty
  engine.OnMarketViewUpdate(&market);

  infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->filled_quantity == 10);
  CHECK(infoOpt->status == OrderStatus::Filled);

  // we had partial fill of 5, then partial fill of 5 => total 10
  REQUIRE(events->fills.size() == 2);
  CHECK(events->fills[1].qty == 5);
  CHECK(events->fills[1].px == Catch::Approx(99.5));

  // done event
  REQUIRE(events->dones.size() == 1);
  CHECK(events->dones[0].order_id == oid);
  CHECK(events->dones[0].final_state == OrderFinalState::Filled);
}

TEST_CASE("Stop order triggered logic", "[orders][stop]") {
  OrdersEngine engine;

  // place a stop buy at $50 => 50000000000
  OrderSpec spec;
  spec.instrument_id = 42;
  spec.type = OrderType::Stop;
  spec.side = OrderSide::Buy;
  spec.quantity = 10;
  spec.stop_price = 50000000000LL; // $50
  auto oid = engine.PlaceOrder(spec);

  MockMarketView market;
  market.stored_instrument_id = 42;

  // bestAsk=$49 => not triggered
  market.has_ask = true;
  market.ask_px_nano = 49000000000LL; 
  market.ask_size    = 100;
  engine.OnMarketViewUpdate(&market);

  auto infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->status == OrderStatus::New);
  CHECK(infoOpt->filled_quantity == 0);

  // Raise bestAsk to $50 => triggers => becomes Market => fill
  market.ask_px_nano = 50000000000LL;
  engine.OnMarketViewUpdate(&market);

  infoOpt = engine.GetOrderInfo(oid);
  REQUIRE(infoOpt.has_value());
  CHECK(infoOpt->status == OrderStatus::Filled);
  CHECK(infoOpt->filled_quantity == 10);
}

TEST_CASE("Bracket order test (simplified)", "[orders][bracket]") {
  OrdersEngine engine;

  // bracket: place a limit buy at $100 => 100000000000 for 5 shares
  // bracket_stop_loss=$95 => 95000000000
  // bracket_take_profit=$110 => 110000000000
  OrderSpec spec;
  spec.instrument_id=777;
  spec.type=OrderType::Limit;
  spec.side=OrderSide::Buy;
  spec.quantity=5;
  spec.limit_price=100000000000LL;
  spec.bracket.use_bracket=true;
  spec.bracket.stopLossPrice=95000000000LL;
  spec.bracket.takeProfitPrice=110000000000LL;

  auto mainOid = engine.PlaceOrder(spec);

  // Fill that main order => must create two child orders
  MockMarketView market;
  market.has_ask = true;
  market.ask_px_nano = 99500000000LL; // <100 => fill
  market.ask_size = 100;
  market.stored_instrument_id = 777;

  engine.OnMarketViewUpdate(&market);

  // main order fully filled => check that two children exist
  auto openOrders = engine.ListOpenOrders(777);
  // We expect 2 bracket children
  REQUIRE(openOrders.size() == 2);
}

TEST_CASE("OrdersEngine state machine transitions", "[orders][state-machine]") {
  OrdersEngine engine;
  auto events = std::make_shared<TestOrderEvents>();
  engine.SetOrderEvents(events);

  // We'll create a limit buy with quantity=10, limit=$100 => 100000000000
  OrderSpec spec;
  spec.instrument_id = 9000;
  spec.type = OrderType::Limit;
  spec.side = OrderSide::Buy;
  spec.quantity = 10;
  spec.limit_price = 100000000000LL;
  std::uint64_t oid = engine.PlaceOrder(spec);
  REQUIRE(oid != 0);

  // Step 1: "New" => partial fill of 5
  {
    MockMarketView market;
    market.has_ask       = true;
    market.ask_px_nano   = 99000000000LL; // $99
    market.ask_size      = 5;
    market.stored_instrument_id = 9000;
    engine.OnMarketViewUpdate(&market);

    auto info = engine.GetOrderInfo(oid);
    REQUIRE(info.has_value());
    CHECK(info->filled_quantity == 5);
    CHECK(info->status == OrderStatus::PartiallyFilled);
  }

  // Step 2: fill next 3 => leftover=2 => still partial
  {
    MockMarketView market2;
    market2.has_ask       = true;
    market2.ask_px_nano   = 99000000000LL;
    market2.ask_size      = 3;
    market2.stored_instrument_id = 9000;
    engine.OnMarketViewUpdate(&market2);

    auto info = engine.GetOrderInfo(oid);
    REQUIRE(info.has_value());
    CHECK(info->filled_quantity == 8);
    CHECK(info->status == OrderStatus::PartiallyFilled);
  }

  // Step 3: final fill => leftover=2
  {
    MockMarketView market3;
    market3.has_ask       = true;
    market3.ask_px_nano   = 99000000000LL;
    market3.ask_size      = 10; // enough
    market3.stored_instrument_id = 9000;
    engine.OnMarketViewUpdate(&market3);

    auto info = engine.GetOrderInfo(oid);
    REQUIRE(info.has_value());
    CHECK(info->filled_quantity == 10);
    CHECK(info->status == OrderStatus::Filled);
  }

  // Attempt a cancel => should fail, as it's already filled
  bool cxl_ok = engine.CancelOrder(oid);
  CHECK(cxl_ok == false);

  auto final_info = engine.GetOrderInfo(oid);
  REQUIRE(final_info.has_value());
  CHECK(final_info->status == OrderStatus::Filled);
}
