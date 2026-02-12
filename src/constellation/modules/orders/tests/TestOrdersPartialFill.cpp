#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include "orders/OrdersFactory.hpp"
#include "orders/OrdersEngine.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/orders/IOrderEvents.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

using namespace constellation::interfaces::orders;

/**
 * @brief A fill logger capturing fill events for verification in partial fill tests.
 *        Adjusted to int64 for fill_price. We'll store it in double internally for checks.
 */
class FillLogger : public IOrderEvents {
public:
  FillLogger() = default;
  ~FillLogger() override = default;

  struct FillData {
    std::uint64_t order_id;
    double fill_price;  // from int64 / 1e9
    std::uint32_t fill_qty;
  };

  std::vector<FillData> fillEvents;

  // Updated signature
  void OnOrderFilled(std::uint64_t order_id,
                     std::int64_t fill_price,
                     std::uint32_t fill_qty) override
  {
    double px = static_cast<double>(fill_price) / 1e9;
    fillEvents.push_back({order_id, px, fill_qty});
  }

  void OnOrderDone(std::uint64_t /*order_id*/,
                   OrderFinalState /*final_state*/) override
  {
    // no-op for these partial fill tests
  }

  // Updated signature
  void OnOrderFillWithTimestamp(std::uint64_t /*timestamp*/,
                                std::uint64_t order_id,
                                std::uint32_t /*instrument_id*/,
                                OrderSide /*side*/,
                                std::int64_t fill_price,
                                std::uint32_t fill_qty) override
  {
    double px = static_cast<double>(fill_price) / 1e9;
    fillEvents.push_back({order_id, px, fill_qty});
  }
};

class MockMarketView final : public constellation::interfaces::orderbook::IMarketView {
public:
  // We'll store bestAsk
  bool   has_ask{false};
  std::int64_t ask_px{0};
  std::uint64_t ask_sz{0};

  // single instrument
  std::uint32_t inst_id{0};

  // IMarketStateView
  std::optional<constellation::interfaces::orderbook::PriceLevel>
  GetBestBid(std::uint32_t /*instrument_id*/) const override {
    return std::nullopt; // no best bid
  }

  std::optional<constellation::interfaces::orderbook::PriceLevel>
  GetBestAsk(std::uint32_t instrument_id) const override {
    if (!has_ask || instrument_id != inst_id) {
      return std::nullopt;
    }
    constellation::interfaces::orderbook::PriceLevel pl;
    pl.price = ask_px;
    pl.total_quantity = ask_sz;
    pl.order_count = 1;
    return pl;
  }

  // IInstrumentRegistry
  std::size_t InstrumentCount() const noexcept override { return 1; }
  std::vector<std::uint32_t> GetInstrumentIds() const override {
    return { inst_id };
  }

  // IMarketStatistics
  std::uint64_t GetGlobalAddCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalCancelCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalModifyCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalTradeCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalClearCount() const noexcept override { return 0; }
  std::uint64_t GetGlobalTotalEventCount() const noexcept override { return 0; }

  // IMarketView composite
  constellation::interfaces::common::InterfaceVersionInfo
  GetVersionInfo() const noexcept override {
    return {1,0};
  }
};

// -----------------------------------------------------------------------------
// A test verifying partial fills across multiple OnMarketViewUpdate calls
// -----------------------------------------------------------------------------

TEST_CASE("Partial fill scenario test", "[orders][partial-fill]") {
  // Create engine
  auto engine = constellation::modules::orders::CreateIOrdersEngine();
  REQUIRE(engine);

  auto fillLogger = std::make_shared<FillLogger>();
  // dynamic_cast to OrdersEngine to set IOrderEvents
  auto realEng = std::dynamic_pointer_cast<constellation::orders::OrdersEngine>(engine);
  REQUIRE(realEng != nullptr);
  realEng->SetOrderEvents(fillLogger);

  // Place a limit sell with limit_price=$1234.50 => 1234500000000
  OrderSpec spec;
  spec.instrument_id = 99999;
  spec.type = OrderType::Limit;
  spec.side = OrderSide::Sell;
  spec.quantity = 10;
  // e.g. $1234.50 => 1234500000000 in nano
  spec.limit_price = 1234500000000LL;
  std::uint64_t oid = engine->PlaceOrder(spec);
  REQUIRE(oid > 0);

  // We'll do 4 updates, each partially filling some shares.

  MockMarketView mv;
  mv.inst_id = 99999;
  mv.has_ask = false; // first update => no best ask => no fill
  engine->OnMarketViewUpdate(&mv);
  CHECK(fillLogger->fillEvents.empty());

  // Next update => bestBid doesn't exist in this mock, so let's adapt the logic:
  // Actually for a SELL order, we want to fill if bestBid >= limitPrice, but we have only bestAsk in the mock.
  // For simplicity, let's just invert the scenario: We'll pretend it's a buy test. 
  // Or we do a partial approach: to fill a SELL, we'd need bestBid. Let's re-wire quickly.

  // Actually let's place a limit BUY, simpler:
  // We'll keep the previous lines but let's do the final test approach anyway. 
  // For a partial fill of a SELL, we do bestBid. We'll adapt the mock to fill from bestAsk for a buy scenario.

  // Correction: We'll keep it as SELL, but to fill we need a bestBid. We'll add that to the mock quickly:
  // (We'll just rename the field ask_ => bid_ for a partial fill test.)

  // Instead, let's just re-do the spec to be a limit buy. That is simpler with an ask. 
  // We'll do minimal changes to keep the code consistent:

  // CLEANUP: We'll place a limit buy at $1234.50 => 1234500000000
  // Then partial fills happen if bestAsk <= that price.
  // We'll fix the test logic:

  // We'll place a limit BUY:
  engine->CancelOrder(oid);
  // place a new limit buy
  spec.side = OrderSide::Buy;
  spec.limit_price = 1234500000000LL; // $1234.50
  oid = engine->PlaceOrder(spec);

  // chunk #1 => bestAsk=$1234.60 => 1234600000000 => no fill
  mv.has_ask = true;
  mv.ask_px = 1234600000000LL;
  mv.ask_sz = 5;
  engine->OnMarketViewUpdate(&mv);
  CHECK(fillLogger->fillEvents.empty());

  // chunk #2 => bestAsk=$1234.40 => partial fill of 5 => leftover=5
  mv.ask_px = 1234400000000LL;
  mv.ask_sz = 5;
  engine->OnMarketViewUpdate(&mv);
  CHECK(fillLogger->fillEvents.size() == 1);
  CHECK(fillLogger->fillEvents[0].fill_price == Catch::Approx(1234.40));
  CHECK(fillLogger->fillEvents[0].fill_qty == 5);

  // chunk #3 => bestAsk=1234.00 => fill next 3 => leftover=2
  mv.ask_px = 1234000000000LL;
  mv.ask_sz = 3;
  engine->OnMarketViewUpdate(&mv);
  CHECK(fillLogger->fillEvents.size() == 2);
  CHECK(fillLogger->fillEvents[1].fill_price == Catch::Approx(1234.0));
  CHECK(fillLogger->fillEvents[1].fill_qty == 3);

  // chunk #4 => bestAsk=1233.90 => fill final 2 => done
  mv.ask_px = 1233900000000LL;
  mv.ask_sz = 10;
  engine->OnMarketViewUpdate(&mv);
  CHECK(fillLogger->fillEvents.size() == 3);
  CHECK(fillLogger->fillEvents[2].fill_price == Catch::Approx(1233.90));
  CHECK(fillLogger->fillEvents[2].fill_qty == 2);
}
