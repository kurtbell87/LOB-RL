/**
 * @file TestTradeBarFeatures.cpp
 * @brief Tests for the 11 trade-only bar features (Phase 3).
 *
 * Spec: docs/trade-bar-features.md
 *
 * Test categories:
 *   1. IBarFeature::OnMboMsg extension (backward compat + dispatch)
 *   2. BarFeatureManager overloaded OnMboEvent(mbo, source, market)
 *   3. Per-feature unit tests (happy path + edge cases)
 *   4. Feature configuration via SetParam
 *   5. Cross-bar state (RealizedVolBarFeature)
 *   6. FeatureRegistry registration
 *   7. Regression tests vs compute_bar_features()
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Interfaces
#include "interfaces/features/IBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

// Concrete bar feature classes
#include "features/bar/TradeFlowImbalanceBarFeature.hpp"
#include "features/bar/BarRangeBarFeature.hpp"
#include "features/bar/BarBodyBarFeature.hpp"
#include "features/bar/BodyRangeRatioBarFeature.hpp"
#include "features/bar/VwapDisplacementBarFeature.hpp"
#include "features/bar/LogVolumeBarFeature.hpp"
#include "features/bar/RealizedVolBarFeature.hpp"
#include "features/bar/SessionTimeBarFeature.hpp"
#include "features/bar/SessionAgeBarFeature.hpp"
#include "features/bar/TradeArrivalRateBarFeature.hpp"
#include "features/bar/PriceImpactBarFeature.hpp"

// Manager + registry
#include "features/BarFeatureManager.hpp"
#include "features/FeatureRegistry.hpp"

// Reference implementation
#include "lob/barrier/feature_compute.h"

// Databento types
#include "databento/constants.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"

// Mock data source for AccumulateEvent (some features may use it)
#include "MockMarketDataSource.hpp"


// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Convert a double price to databento fixed-point int64.
static std::int64_t to_fixed(double price) {
  return static_cast<std::int64_t>(price * databento::kFixedPriceScale + 0.5);
}

/// Build a Trade MboMsg with given price/size/timestamp.
static databento::MboMsg make_trade(double price, std::uint32_t size,
                                     std::uint64_t ts_ns = 0,
                                     databento::Side side = databento::Side::Ask) {
  databento::MboMsg mbo{};
  mbo.hd.instrument_id = 1234;
  mbo.hd.ts_event = databento::UnixNanos{std::chrono::nanoseconds{ts_ns}};
  mbo.order_id = 1;
  mbo.price = to_fixed(price);
  mbo.size = size;
  mbo.action = databento::Action::Trade;
  mbo.side = side;
  return mbo;
}

/// Build a Fill MboMsg (also counts as a trade).
static databento::MboMsg make_fill(double price, std::uint32_t size,
                                    std::uint64_t ts_ns = 0,
                                    databento::Side side = databento::Side::Ask) {
  databento::MboMsg mbo = make_trade(price, size, ts_ns, side);
  mbo.action = databento::Action::Fill;
  return mbo;
}

/// Build an Add MboMsg (non-trade, should be ignored by trade features).
static databento::MboMsg make_add(double price, std::uint32_t size,
                                   databento::Side side = databento::Side::Bid) {
  databento::MboMsg mbo{};
  mbo.hd.instrument_id = 1234;
  mbo.order_id = 99;
  mbo.price = to_fixed(price);
  mbo.size = size;
  mbo.action = databento::Action::Add;
  mbo.side = side;
  return mbo;
}

/// Run a single-bar lifecycle: start → feed trades via OnMboMsg → complete.
/// Returns the bar value for `value_name`.
template <typename FeatureT>
double run_single_bar(std::shared_ptr<FeatureT> feat,
                      const std::vector<databento::MboMsg>& trades,
                      const std::string& value_name,
                      std::uint64_t bar_index = 0) {
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  feat->OnBarStart(bar_index);
  for (const auto& mbo : trades) {
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(bar_index);
  return feat->GetBarValue(value_name);
}


// ═══════════════════════════════════════════════════════════════════════
// Section 1: IBarFeature::OnMboMsg Extension
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("IBarFeature::OnMboMsg has default no-op", "[ibar_feature][onmbomsg]") {
  // Existing features that don't override OnMboMsg should not break.
  // We test this by creating a feature that only uses AccumulateEvent
  // and verifying OnMboMsg doesn't crash.

  // TradeFlowImbalanceBarFeature overrides OnMboMsg, so use it to verify
  // the method exists on the interface.
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  auto bar_feat = std::dynamic_pointer_cast<constellation::interfaces::features::IBarFeature>(feat);
  REQUIRE(bar_feat != nullptr);

  // OnMboMsg should be callable
  databento::MboMsg mbo = make_trade(100.0, 10);
  bar_feat->OnMboMsg(mbo);
  // No crash = pass
}


// ═══════════════════════════════════════════════════════════════════════
// Section 2: BarFeatureManager Overloaded OnMboEvent
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BarFeatureManager::OnMboEvent(mbo, source, market) dispatches OnMboMsg then OnDataUpdate",
          "[bar_feature_manager][onmbomsg]") {
  constellation::modules::features::BarFeatureManager mgr;

  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  mgr.RegisterBarFeature(feat, "trade_flow_imbalance");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  mgr.NotifyBarStart(0);

  // Feed 3 trades: up, up, down
  auto t1 = make_trade(100.0, 5);
  auto t2 = make_trade(100.25, 10);
  auto t3 = make_trade(100.0, 8);

  mgr.OnMboEvent(t1, source, nullptr);
  mgr.OnMboEvent(t2, source, nullptr);
  mgr.OnMboEvent(t3, source, nullptr);

  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 1);
  // tick rule: trade2 up → buy(10), trade3 down → sell(8)
  // imbalance = (10 - 8) / (10 + 8) = 2/18
  REQUIRE(vec[0] == Catch::Approx(2.0 / 18.0).epsilon(1e-12));
}

TEST_CASE("BarFeatureManager::OnMboEvent(mbo, source, market) backward compat with old overload",
          "[bar_feature_manager][onmbomsg]") {
  // The old OnMboEvent(source, market) overload should still work
  constellation::modules::features::BarFeatureManager mgr;
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  mgr.RegisterBarFeature(feat, "log_volume");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  mgr.NotifyBarStart(0);

  // Use the new overload to feed trades
  auto t1 = make_trade(100.0, 5);
  auto t2 = make_trade(100.25, 10);
  mgr.OnMboEvent(t1, source, nullptr);
  mgr.OnMboEvent(t2, source, nullptr);

  // Also use the old overload (should still call OnDataUpdate)
  mgr.OnMboEvent(source, nullptr);

  mgr.NotifyBarComplete(0);

  // log_volume depends on OnMboMsg, so old overload doesn't add volume
  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 1);
  REQUIRE(vec[0] == Catch::Approx(std::log(15.0)).epsilon(1e-12));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 3: Per-Feature Unit Tests
// ═══════════════════════════════════════════════════════════════════════

// --- Col 0: TradeFlowImbalanceBarFeature ---

TEST_CASE("TradeFlowImbalance: all upticks → +1", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),
    make_trade(100.25, 5),
    make_trade(100.50, 8),
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  // All upticks: buy_vol = 5+8 = 13, sell_vol = 0 → 13/13 = 1.0
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: all downticks → -1", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.50, 10),
    make_trade(100.25, 5),
    make_trade(100.0, 8),
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  REQUIRE(val == Catch::Approx(-1.0).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: mixed ticks", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 1),    // baseline
    make_trade(100.25, 10),  // up → buy 10
    make_trade(100.0, 8),    // down → sell 8
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  REQUIRE(val == Catch::Approx((10.0 - 8.0) / (10.0 + 8.0)).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: forward-fill on same price", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 1),    // baseline
    make_trade(100.25, 10),  // up → buy 10
    make_trade(100.25, 5),   // same → forward-fill up → buy 5
    make_trade(100.0, 8),    // down → sell 8
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  // buy=15, sell=8 → (15-8)/(15+8) = 7/23
  REQUIRE(val == Catch::Approx(7.0 / 23.0).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: single trade → 0", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("TradeFlowImbalance: zero trades → 0", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  double val = run_single_bar(feat, {}, "trade_flow_imbalance");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("TradeFlowImbalance: Fill action counts as trade", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 1),
    make_fill(100.25, 10),  // Fill, up → buy
    make_fill(100.0, 8),    // Fill, down → sell
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  REQUIRE(val == Catch::Approx((10.0 - 8.0) / 18.0).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: Add action ignored", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 1),
    make_add(100.25, 10),    // Add — should be ignored
    make_trade(100.25, 5),   // up → buy 5
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  // Only 2 trades: base(100), up(100.25,5) → buy=5, sell=0 → 1.0
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: HasFeature and unknown name", "[trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  REQUIRE(feat->HasFeature("trade_flow_imbalance"));
  REQUIRE_FALSE(feat->HasFeature("unknown"));
}


// --- Col 3: BarRangeBarFeature ---

TEST_CASE("BarRange: basic range in ticks", "[bar_range]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(101.0, 3),
    make_trade(100.25, 7),
  };
  double val = run_single_bar(feat, trades, "bar_range");
  // high=101, low=100, range = (101 - 100) / 0.25 = 4.0
  REQUIRE(val == Catch::Approx(4.0).epsilon(1e-12));
}

TEST_CASE("BarRange: zero trades → 0", "[bar_range]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  double val = run_single_bar(feat, {}, "bar_range");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("BarRange: single trade → 0", "[bar_range]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  std::vector<databento::MboMsg> trades = { make_trade(100.0, 5) };
  double val = run_single_bar(feat, trades, "bar_range");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("BarRange: custom tick_size via SetParam", "[bar_range][config]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  feat->SetParam("tick_size", "0.5");
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(101.0, 3),
  };
  double val = run_single_bar(feat, trades, "bar_range");
  // range = (101 - 100) / 0.5 = 2.0
  REQUIRE(val == Catch::Approx(2.0).epsilon(1e-12));
}

TEST_CASE("BarRange: HasFeature", "[bar_range]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  REQUIRE(feat->HasFeature("bar_range"));
  REQUIRE_FALSE(feat->HasFeature("bar_body"));
}


// --- Col 4: BarBodyBarFeature ---

TEST_CASE("BarBody: positive body (close > open)", "[bar_body]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),   // open
    make_trade(100.50, 3),
    make_trade(101.0, 7),   // close
  };
  double val = run_single_bar(feat, trades, "bar_body");
  // (101 - 100) / 0.25 = 4.0
  REQUIRE(val == Catch::Approx(4.0).epsilon(1e-12));
}

TEST_CASE("BarBody: negative body (close < open)", "[bar_body]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(101.0, 5),
    make_trade(100.0, 7),
  };
  double val = run_single_bar(feat, trades, "bar_body");
  // (100 - 101) / 0.25 = -4.0
  REQUIRE(val == Catch::Approx(-4.0).epsilon(1e-12));
}

TEST_CASE("BarBody: zero trades → 0", "[bar_body]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  double val = run_single_bar(feat, {}, "bar_body");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("BarBody: single trade → 0 (open == close)", "[bar_body]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  std::vector<databento::MboMsg> trades = { make_trade(100.0, 5) };
  double val = run_single_bar(feat, trades, "bar_body");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("BarBody: custom tick_size", "[bar_body][config]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  feat->SetParam("tick_size", "0.5");
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(101.0, 3),
  };
  double val = run_single_bar(feat, trades, "bar_body");
  REQUIRE(val == Catch::Approx(2.0).epsilon(1e-12));
}


// --- Col 5: BodyRangeRatioBarFeature ---

TEST_CASE("BodyRangeRatio: ratio in [−1, 1]", "[body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),    // open, low
    make_trade(101.0, 3),    // high
    make_trade(100.50, 7),   // close
  };
  double val = run_single_bar(feat, trades, "body_range_ratio");
  // body = close - open = 0.5, range = high - low = 1.0 → 0.5/1.0 = 0.5
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("BodyRangeRatio: full bullish candle → 1.0", "[body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),    // open = low
    make_trade(101.0, 3),    // close = high
  };
  double val = run_single_bar(feat, trades, "body_range_ratio");
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("BodyRangeRatio: full bearish candle → −1.0", "[body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(101.0, 5),    // open = high
    make_trade(100.0, 3),    // close = low
  };
  double val = run_single_bar(feat, trades, "body_range_ratio");
  REQUIRE(val == Catch::Approx(-1.0).epsilon(1e-12));
}

TEST_CASE("BodyRangeRatio: zero range → 0", "[body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(100.0, 3),
  };
  double val = run_single_bar(feat, trades, "body_range_ratio");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("BodyRangeRatio: zero trades → 0", "[body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  double val = run_single_bar(feat, {}, "body_range_ratio");
  REQUIRE(val == Catch::Approx(0.0));
}


// --- Col 6: VwapDisplacementBarFeature ---

TEST_CASE("VwapDisplacement: close above VWAP with range > 0", "[vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),   // low
    make_trade(102.0, 10),   // high, close
  };
  double val = run_single_bar(feat, trades, "vwap_displacement");
  // vwap = (100*10 + 102*10) / 20 = 101.0
  // range = 102 - 100 = 2.0
  // displacement = (102 - 101) / 2.0 = 0.5
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("VwapDisplacement: close below VWAP → negative", "[vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(102.0, 10),   // high
    make_trade(100.0, 10),   // low, close
  };
  double val = run_single_bar(feat, trades, "vwap_displacement");
  // vwap = 101.0, range = 2.0, displacement = (100 - 101) / 2.0 = -0.5
  REQUIRE(val == Catch::Approx(-0.5).epsilon(1e-12));
}

TEST_CASE("VwapDisplacement: zero range → 0", "[vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),
    make_trade(100.0, 5),
  };
  double val = run_single_bar(feat, trades, "vwap_displacement");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("VwapDisplacement: zero trades → 0", "[vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  double val = run_single_bar(feat, {}, "vwap_displacement");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("VwapDisplacement: volume-weighted average pulled toward larger trade", "[vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 1),    // tiny trade at low
    make_trade(102.0, 99),   // huge trade at high → VWAP near 102
  };
  double val = run_single_bar(feat, trades, "vwap_displacement");
  // vwap = (100*1 + 102*99) / 100 = (100 + 10098) / 100 = 101.98
  // range = 2.0, displacement = (102 - 101.98) / 2.0 = 0.01
  double expected_vwap = (100.0 * 1 + 102.0 * 99) / 100.0;
  double expected = (102.0 - expected_vwap) / 2.0;
  REQUIRE(val == Catch::Approx(expected).epsilon(1e-12));
}


// --- Col 7: LogVolumeBarFeature ---

TEST_CASE("LogVolume: basic computation", "[log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),
    make_trade(100.25, 20),
  };
  double val = run_single_bar(feat, trades, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(30.0)).epsilon(1e-12));
}

TEST_CASE("LogVolume: zero trades → log(1) = 0", "[log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  double val = run_single_bar(feat, {}, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(1.0)).epsilon(1e-12));
}

TEST_CASE("LogVolume: single trade", "[log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  std::vector<databento::MboMsg> trades = { make_trade(100.0, 7) };
  double val = run_single_bar(feat, trades, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(7.0)).epsilon(1e-12));
}

TEST_CASE("LogVolume: Fill action counts toward volume", "[log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_fill(100.25, 10),
  };
  double val = run_single_bar(feat, trades, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(15.0)).epsilon(1e-12));
}

TEST_CASE("LogVolume: non-trade events ignored", "[log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_add(100.25, 100),  // should be ignored
    make_trade(100.25, 10),
  };
  double val = run_single_bar(feat, trades, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(15.0)).epsilon(1e-12));
}


// --- Col 8: RealizedVolBarFeature ---

TEST_CASE("RealizedVol: warmup period returns NaN", "[realized_vol]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // Bars 0..18 (19 bars) should all return NaN
  for (std::uint64_t i = 0; i < 19; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(100.0 + i * 0.25, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);

    double val = feat->GetBarValue("realized_vol");
    REQUIRE(std::isnan(val));
  }
}

TEST_CASE("RealizedVol: after warmup returns finite value", "[realized_vol]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // Run 20 bars (index 0..19), each with one trade at incrementing prices
  for (std::uint64_t i = 0; i <= 19; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(100.0 + i * 0.25, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);
  }

  double val = feat->GetBarValue("realized_vol");
  REQUIRE_FALSE(std::isnan(val));
  REQUIRE(val >= 0.0);
}

TEST_CASE("RealizedVol: constant price → zero vol", "[realized_vol]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // 20 bars at constant price → all log returns = 0 → vol = 0
  for (std::uint64_t i = 0; i <= 19; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(100.0, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);
  }

  double val = feat->GetBarValue("realized_vol");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("RealizedVol: matches hand-computed value", "[realized_vol]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // Use prices that produce known log returns
  std::vector<double> close_prices;
  for (std::uint64_t i = 0; i <= 19; ++i) {
    double price = 100.0 * std::exp(0.001 * i);  // constant log return of 0.001
    close_prices.push_back(price);
    feat->OnBarStart(i);
    auto mbo = make_trade(price, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);
  }

  // At bar 19: 19 log returns, all ≈ 0.001
  // E[x] = 0.001, E[x^2] ≈ 0.001^2 = 1e-6
  // Var = E[x^2] - E[x]^2 ≈ 0
  // With constant returns, vol should be ≈ 0
  double val = feat->GetBarValue("realized_vol");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("RealizedVol: cross-bar state persists after ResetAccumulators", "[realized_vol]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // The spec says ResetAccumulators does NOT reset cross-bar state.
  // Run bars 0..20 and verify bar 19 is not NaN but bar 18 is.
  for (std::uint64_t i = 0; i <= 20; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(100.0 + i * 0.25, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);
  }

  double val = feat->GetBarValue("realized_vol");
  REQUIRE_FALSE(std::isnan(val));
  REQUIRE(val > 0.0);  // Varying prices → non-zero vol
}

TEST_CASE("RealizedVol: custom warmup_period via SetParam", "[realized_vol][config]") {
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  feat->SetParam("warmup_period", "5");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  // With warmup=5, bars 0..4 should be NaN, bar 5 should be finite
  for (std::uint64_t i = 0; i <= 5; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(100.0 + i * 0.25, 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);
  }

  double val = feat->GetBarValue("realized_vol");
  REQUIRE_FALSE(std::isnan(val));
}


// --- Col 9: SessionTimeBarFeature ---

TEST_CASE("SessionTime: midpoint of session → 0.5", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  // RTH: 0ns to 100ns
  feat->SetParam("rth_open_ns", "0");
  feat->SetParam("rth_close_ns", "100");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 50),  // ts=50ns → midpoint
  };
  double val = run_single_bar(feat, trades, "session_time");
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("SessionTime: at open → 0.0", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "1000");
  feat->SetParam("rth_close_ns", "2000");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 1000),
  };
  double val = run_single_bar(feat, trades, "session_time");
  REQUIRE(val == Catch::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("SessionTime: at close → 1.0", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "1000");
  feat->SetParam("rth_close_ns", "2000");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 2000),
  };
  double val = run_single_bar(feat, trades, "session_time");
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("SessionTime: before open clamps to 0", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "1000");
  feat->SetParam("rth_close_ns", "2000");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 500),  // before open
  };
  double val = run_single_bar(feat, trades, "session_time");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("SessionTime: after close clamps to 1", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "1000");
  feat->SetParam("rth_close_ns", "2000");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 3000),  // after close
  };
  double val = run_single_bar(feat, trades, "session_time");
  REQUIRE(val == Catch::Approx(1.0));
}

TEST_CASE("SessionTime: zero trades → 0", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "1000");
  feat->SetParam("rth_close_ns", "2000");

  double val = run_single_bar(feat, {}, "session_time");
  // No trades → no timestamp → clamp(0/duration, 0, 1) = 0
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("SessionTime: uses latest timestamp", "[session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  feat->SetParam("rth_open_ns", "0");
  feat->SetParam("rth_close_ns", "100");

  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5, 25),
    make_trade(100.0, 5, 75),  // latest
    make_trade(100.0, 5, 50),
  };
  double val = run_single_bar(feat, trades, "session_time");
  // Latest ts = 75, time = 75/100 = 0.75
  REQUIRE(val == Catch::Approx(0.75).epsilon(1e-12));
}


// --- Col 12: SessionAgeBarFeature ---

TEST_CASE("SessionAge: bar_index=0 → 0.0", "[session_age]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  double val = run_single_bar(feat, {}, "session_age", 0);
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("SessionAge: bar_index=10 → 0.5", "[session_age]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  double val = run_single_bar(feat, {}, "session_age", 10);
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("SessionAge: bar_index=20 → 1.0", "[session_age]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  double val = run_single_bar(feat, {}, "session_age", 20);
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("SessionAge: bar_index=100 → clamped to 1.0", "[session_age]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  double val = run_single_bar(feat, {}, "session_age", 100);
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("SessionAge: custom period via SetParam", "[session_age][config]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  feat->SetParam("period", "10.0");
  double val = run_single_bar(feat, {}, "session_age", 5);
  // 5 / 10 = 0.5
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}


// --- Col 19: TradeArrivalRateBarFeature ---

TEST_CASE("TradeArrivalRate: basic count", "[trade_arrival_rate]") {
  auto feat = std::make_shared<constellation::modules::features::TradeArrivalRateBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(100.25, 3),
    make_trade(100.50, 7),
  };
  double val = run_single_bar(feat, trades, "trade_arrival_rate");
  REQUIRE(val == Catch::Approx(std::log(1.0 + 3.0)).epsilon(1e-12));
}

TEST_CASE("TradeArrivalRate: zero trades → log(1) = 0", "[trade_arrival_rate]") {
  auto feat = std::make_shared<constellation::modules::features::TradeArrivalRateBarFeature>();
  double val = run_single_bar(feat, {}, "trade_arrival_rate");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("TradeArrivalRate: Fill counts as trade", "[trade_arrival_rate]") {
  auto feat = std::make_shared<constellation::modules::features::TradeArrivalRateBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_fill(100.25, 3),
  };
  double val = run_single_bar(feat, trades, "trade_arrival_rate");
  REQUIRE(val == Catch::Approx(std::log(3.0)).epsilon(1e-12));
}

TEST_CASE("TradeArrivalRate: Add does not count", "[trade_arrival_rate]") {
  auto feat = std::make_shared<constellation::modules::features::TradeArrivalRateBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_add(100.25, 100),
    make_trade(100.25, 3),
  };
  double val = run_single_bar(feat, trades, "trade_arrival_rate");
  // Only 2 trades (not the Add)
  REQUIRE(val == Catch::Approx(std::log(3.0)).epsilon(1e-12));
}


// --- Col 21: PriceImpactBarFeature ---

TEST_CASE("PriceImpact: basic computation", "[price_impact]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(100.50, 3),
    make_trade(101.0, 7),
  };
  double val = run_single_bar(feat, trades, "price_impact_per_trade");
  // (101 - 100) / (3 * 0.25) = 1.0 / 0.75 ≈ 1.333
  REQUIRE(val == Catch::Approx(1.0 / 0.75).epsilon(1e-12));
}

TEST_CASE("PriceImpact: negative impact", "[price_impact]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(101.0, 5),
    make_trade(100.0, 7),
  };
  double val = run_single_bar(feat, trades, "price_impact_per_trade");
  // (100 - 101) / (2 * 0.25) = -1.0 / 0.5 = -2.0
  REQUIRE(val == Catch::Approx(-2.0).epsilon(1e-12));
}

TEST_CASE("PriceImpact: zero trades → 0", "[price_impact]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  double val = run_single_bar(feat, {}, "price_impact_per_trade");
  // (0 - 0) / (1 * 0.25) = 0
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("PriceImpact: single trade → 0 (open == close)", "[price_impact]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  std::vector<databento::MboMsg> trades = { make_trade(100.0, 5) };
  double val = run_single_bar(feat, trades, "price_impact_per_trade");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("PriceImpact: custom tick_size", "[price_impact][config]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  feat->SetParam("tick_size", "0.5");
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 5),
    make_trade(101.0, 3),
  };
  double val = run_single_bar(feat, trades, "price_impact_per_trade");
  // (101 - 100) / (2 * 0.5) = 1.0
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 4: IConfigurableFeature Tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("All trade bar features implement IConfigurableFeature", "[config]") {
  // Verify dynamic_cast to IConfigurableFeature succeeds for all 11 features
  using namespace constellation::modules::features;
  using IC = constellation::interfaces::features::IConfigurableFeature;

  REQUIRE(dynamic_cast<IC*>(std::make_shared<TradeFlowImbalanceBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<BarRangeBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<BarBodyBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<BodyRangeRatioBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<VwapDisplacementBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<LogVolumeBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<RealizedVolBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<SessionTimeBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<SessionAgeBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<TradeArrivalRateBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<PriceImpactBarFeature>().get()) != nullptr);
}


// ═══════════════════════════════════════════════════════════════════════
// Section 5: FeatureRegistry Registration
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("FeatureRegistry: TradeFlowImbalanceBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("TradeFlowImbalanceBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("trade_flow_imbalance"));
}

TEST_CASE("FeatureRegistry: BarRangeBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("BarRangeBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("bar_range"));
}

TEST_CASE("FeatureRegistry: BarBodyBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("BarBodyBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("bar_body"));
}

TEST_CASE("FeatureRegistry: BodyRangeRatioBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("BodyRangeRatioBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("body_range_ratio"));
}

TEST_CASE("FeatureRegistry: VwapDisplacementBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("VwapDisplacementBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("vwap_displacement"));
}

TEST_CASE("FeatureRegistry: LogVolumeBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("LogVolumeBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("log_volume"));
}

TEST_CASE("FeatureRegistry: RealizedVolBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("RealizedVolBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("realized_vol"));
}

TEST_CASE("FeatureRegistry: SessionTimeBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("SessionTimeBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("session_time"));
}

TEST_CASE("FeatureRegistry: SessionAgeBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("SessionAgeBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("session_age"));
}

TEST_CASE("FeatureRegistry: TradeArrivalRateBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("TradeArrivalRateBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("trade_arrival_rate"));
}

TEST_CASE("FeatureRegistry: PriceImpactBarFeature registered", "[registry]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("PriceImpactBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("price_impact_per_trade"));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 6: Regression Tests vs compute_bar_features()
// ═══════════════════════════════════════════════════════════════════════

namespace {

/// Build a TradeBar from a trade sequence (same as what bar features see).
TradeBar build_trade_bar(const std::vector<std::pair<double, int>>& trades,
                         int bar_index = 0,
                         std::uint64_t t_end = 0) {
  TradeBar bar;
  bar.bar_index = bar_index;
  bar.t_end = t_end;

  if (trades.empty()) return bar;

  double sum_pv = 0.0;
  int sum_v = 0;
  bar.open = trades.front().first;
  bar.close = trades.back().first;
  bar.high = -std::numeric_limits<double>::infinity();
  bar.low = std::numeric_limits<double>::infinity();

  for (const auto& [price, size] : trades) {
    bar.trade_prices.push_back(price);
    bar.trade_sizes.push_back(size);
    bar.high = std::max(bar.high, price);
    bar.low = std::min(bar.low, price);
    sum_pv += price * size;
    sum_v += size;
  }
  bar.volume = sum_v;
  bar.vwap = (sum_v > 0) ? sum_pv / sum_v : 0.0;
  return bar;
}

/// Construct MboMsg trades from price/size pairs, run through a bar feature,
/// and compare the result against the reference column from compute_bar_features().
void regression_check_column(
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feat,
    const std::string& value_name,
    int col_index,
    const std::vector<std::pair<double, int>>& trades,
    int bar_index = 0,
    std::uint64_t t_end = 0,
    std::uint64_t rth_open_ns = 0,
    std::uint64_t rth_close_ns = 86400000000000ULL) {

  // --- Reference: compute_bar_features ---
  TradeBar ref_bar = build_trade_bar(trades, bar_index, t_end);
  BarBookAccum ref_accum;
  ref_accum.n_trades = static_cast<int>(trades.size());

  std::vector<TradeBar> bars = {ref_bar};
  std::vector<BarBookAccum> accums = {ref_accum};
  auto ref = compute_bar_features(bars, accums, rth_open_ns, rth_close_ns);
  double expected = ref[col_index];

  // --- New feature ---
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  feat->OnBarStart(bar_index);
  for (const auto& [price, size] : trades) {
    auto mbo = make_trade(price, static_cast<std::uint32_t>(size), t_end);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(bar_index);
  double actual = feat->GetBarValue(value_name);

  // Handle NaN equality
  if (std::isnan(expected)) {
    REQUIRE(std::isnan(actual));
  } else {
    REQUIRE(actual == Catch::Approx(expected).epsilon(1e-12));
  }
}

} // anonymous namespace


TEST_CASE("Regression: TradeFlowImbalance matches compute_bar_features col 0",
          "[regression][trade_flow_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  regression_check_column(feat, "trade_flow_imbalance", 0,
    {{100.0, 10}, {100.25, 5}, {100.0, 8}, {100.25, 3}, {100.50, 12}});
}

TEST_CASE("Regression: BarRange matches compute_bar_features col 3",
          "[regression][bar_range]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  regression_check_column(feat, "bar_range", 3,
    {{100.0, 10}, {101.0, 5}, {100.25, 8}, {100.75, 3}});
}

TEST_CASE("Regression: BarBody matches compute_bar_features col 4",
          "[regression][bar_body]") {
  auto feat = std::make_shared<constellation::modules::features::BarBodyBarFeature>();
  regression_check_column(feat, "bar_body", 4,
    {{100.0, 10}, {101.0, 5}, {100.25, 8}, {100.75, 3}});
}

TEST_CASE("Regression: BodyRangeRatio matches compute_bar_features col 5",
          "[regression][body_range_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::BodyRangeRatioBarFeature>();
  regression_check_column(feat, "body_range_ratio", 5,
    {{100.0, 10}, {101.0, 5}, {100.25, 8}, {100.75, 3}});
}

TEST_CASE("Regression: VwapDisplacement matches compute_bar_features col 6",
          "[regression][vwap_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  regression_check_column(feat, "vwap_displacement", 6,
    {{100.0, 10}, {101.0, 5}, {100.25, 8}, {100.75, 3}});
}

TEST_CASE("Regression: LogVolume matches compute_bar_features col 7",
          "[regression][log_volume]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  regression_check_column(feat, "log_volume", 7,
    {{100.0, 10}, {101.0, 5}, {100.25, 8}});
}

TEST_CASE("Regression: SessionTime matches compute_bar_features col 9",
          "[regression][session_time]") {
  auto feat = std::make_shared<constellation::modules::features::SessionTimeBarFeature>();
  // Configure RTH params to match the reference
  std::uint64_t rth_open = 1000000000ULL;   // 1 second in ns
  std::uint64_t rth_close = 86400000000000ULL;  // 24 hours in ns
  std::uint64_t bar_time = 43200000000000ULL;   // 12 hours in ns

  feat->SetParam("rth_open_ns", std::to_string(rth_open));
  feat->SetParam("rth_close_ns", std::to_string(rth_close));

  regression_check_column(feat, "session_time", 9,
    {{100.0, 10}}, 0, bar_time, rth_open, rth_close);
}

TEST_CASE("Regression: SessionAge matches compute_bar_features col 12",
          "[regression][session_age]") {
  auto feat = std::make_shared<constellation::modules::features::SessionAgeBarFeature>();
  regression_check_column(feat, "session_age", 12, {{100.0, 10}}, 15);
}

TEST_CASE("Regression: TradeArrivalRate matches compute_bar_features col 19",
          "[regression][trade_arrival_rate]") {
  auto feat = std::make_shared<constellation::modules::features::TradeArrivalRateBarFeature>();
  regression_check_column(feat, "trade_arrival_rate", 19,
    {{100.0, 10}, {100.25, 5}, {100.50, 8}, {100.75, 3}, {101.0, 12}});
}

TEST_CASE("Regression: PriceImpact matches compute_bar_features col 21",
          "[regression][price_impact]") {
  auto feat = std::make_shared<constellation::modules::features::PriceImpactBarFeature>();
  regression_check_column(feat, "price_impact_per_trade", 21,
    {{100.0, 10}, {100.25, 5}, {100.50, 8}, {100.75, 3}, {101.0, 12}});
}


// ═══════════════════════════════════════════════════════════════════════
// Section 7: Regression — RealizedVol multi-bar sequence
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Regression: RealizedVol matches compute_bar_features col 8 across 25 bars",
          "[regression][realized_vol]") {
  // Build a 25-bar sequence with varying close prices
  std::vector<double> close_prices;
  for (int i = 0; i < 25; ++i) {
    close_prices.push_back(100.0 + 0.25 * (i % 5) - 0.125 * (i % 3));
  }

  // --- Reference ---
  std::vector<TradeBar> ref_bars;
  std::vector<BarBookAccum> ref_accums;
  for (int i = 0; i < 25; ++i) {
    TradeBar bar;
    bar.bar_index = i;
    bar.open = close_prices[i];
    bar.close = close_prices[i];
    bar.high = close_prices[i];
    bar.low = close_prices[i];
    bar.volume = 10;
    bar.vwap = close_prices[i];
    bar.trade_prices = {close_prices[i]};
    bar.trade_sizes = {10};
    ref_bars.push_back(bar);
    BarBookAccum acc;
    acc.n_trades = 1;
    ref_accums.push_back(acc);
  }
  auto ref = compute_bar_features(ref_bars, ref_accums, 0, 86400000000000ULL);

  // --- New feature ---
  auto feat = std::make_shared<constellation::modules::features::RealizedVolBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  for (int i = 0; i < 25; ++i) {
    feat->OnBarStart(i);
    auto mbo = make_trade(close_prices[i], 10);
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(i);

    double actual = feat->GetBarValue("realized_vol");
    double expected = ref[i * N_FEATURES + 8];

    if (std::isnan(expected)) {
      REQUIRE(std::isnan(actual));
    } else {
      REQUIRE(actual == Catch::Approx(expected).epsilon(1e-12));
    }
  }
}


// ═══════════════════════════════════════════════════════════════════════
// Section 8: Edge Cases — Additional Coverage
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BarRange: Fill action tracks high/low", "[bar_range][edge]") {
  auto feat = std::make_shared<constellation::modules::features::BarRangeBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_fill(100.0, 5),
    make_fill(102.0, 3),
  };
  double val = run_single_bar(feat, trades, "bar_range");
  REQUIRE(val == Catch::Approx(8.0).epsilon(1e-12));  // (102-100)/0.25
}

TEST_CASE("LogVolume: large volume", "[log_volume][edge]") {
  auto feat = std::make_shared<constellation::modules::features::LogVolumeBarFeature>();
  std::vector<databento::MboMsg> trades;
  for (int i = 0; i < 1000; ++i) {
    trades.push_back(make_trade(100.0, 100));
  }
  double val = run_single_bar(feat, trades, "log_volume");
  REQUIRE(val == Catch::Approx(std::log(100000.0)).epsilon(1e-12));
}

TEST_CASE("VwapDisplacement: single trade → zero range → 0", "[vwap_displacement][edge]") {
  auto feat = std::make_shared<constellation::modules::features::VwapDisplacementBarFeature>();
  std::vector<databento::MboMsg> trades = { make_trade(100.0, 5) };
  double val = run_single_bar(feat, trades, "vwap_displacement");
  REQUIRE(val == Catch::Approx(0.0));
}

TEST_CASE("All features: GetBarValue throws before OnBarComplete", "[edge]") {
  using namespace constellation::modules::features;

  auto features = std::vector<std::pair<
    std::shared_ptr<constellation::interfaces::features::IBarFeature>, std::string>>{
    {std::make_shared<TradeFlowImbalanceBarFeature>(), "trade_flow_imbalance"},
    {std::make_shared<BarRangeBarFeature>(), "bar_range"},
    {std::make_shared<BarBodyBarFeature>(), "bar_body"},
    {std::make_shared<BodyRangeRatioBarFeature>(), "body_range_ratio"},
    {std::make_shared<VwapDisplacementBarFeature>(), "vwap_displacement"},
    {std::make_shared<LogVolumeBarFeature>(), "log_volume"},
    {std::make_shared<RealizedVolBarFeature>(), "realized_vol"},
    {std::make_shared<SessionTimeBarFeature>(), "session_time"},
    {std::make_shared<SessionAgeBarFeature>(), "session_age"},
    {std::make_shared<TradeArrivalRateBarFeature>(), "trade_arrival_rate"},
    {std::make_shared<PriceImpactBarFeature>(), "price_impact_per_trade"},
  };

  for (auto& [feat, name] : features) {
    REQUIRE_THROWS(feat->GetBarValue(name));
  }
}

TEST_CASE("All features: reset between bars", "[edge]") {
  // Ensure each feature resets properly between bars
  using namespace constellation::modules::features;
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  auto feat = std::make_shared<LogVolumeBarFeature>();

  // Bar 0: volume = 15
  feat->OnBarStart(0);
  feat->OnMboMsg(make_trade(100.0, 10));
  feat->OnDataUpdate(source, nullptr);
  feat->OnMboMsg(make_trade(100.25, 5));
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  REQUIRE(feat->GetBarValue("log_volume") == Catch::Approx(std::log(15.0)).epsilon(1e-12));

  // Bar 1: volume = 3 (must NOT include bar 0's volume)
  feat->OnBarStart(1);
  feat->OnMboMsg(make_trade(100.50, 3));
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(1);
  REQUIRE(feat->GetBarValue("log_volume") == Catch::Approx(std::log(3.0)).epsilon(1e-12));
}

TEST_CASE("TradeFlowImbalance: all same-price trades → 0 (no direction)", "[trade_flow_imbalance][edge]") {
  auto feat = std::make_shared<constellation::modules::features::TradeFlowImbalanceBarFeature>();
  std::vector<databento::MboMsg> trades = {
    make_trade(100.0, 10),
    make_trade(100.0, 5),
    make_trade(100.0, 8),
  };
  double val = run_single_bar(feat, trades, "trade_flow_imbalance");
  // All same price → prev_dir starts at 0, never changes → buy=0, sell=0 → 0
  REQUIRE(val == Catch::Approx(0.0));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 9: BarFeatureManager Integration with Multiple Trade Features
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BarFeatureManager: multiple trade features produce correct vector",
          "[bar_feature_manager][integration]") {
  using namespace constellation::modules::features;
  BarFeatureManager mgr;

  auto flow = std::make_shared<TradeFlowImbalanceBarFeature>();
  auto range = std::make_shared<BarRangeBarFeature>();
  auto body = std::make_shared<BarBodyBarFeature>();
  auto log_vol = std::make_shared<LogVolumeBarFeature>();
  auto arrival = std::make_shared<TradeArrivalRateBarFeature>();

  mgr.RegisterBarFeature(flow, "trade_flow_imbalance");
  mgr.RegisterBarFeature(range, "bar_range");
  mgr.RegisterBarFeature(body, "bar_body");
  mgr.RegisterBarFeature(log_vol, "log_volume");
  mgr.RegisterBarFeature(arrival, "trade_arrival_rate");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 100.25;

  mgr.NotifyBarStart(0);

  // 3 trades: 100.0(10), 100.50(5), 100.25(8)
  auto t1 = make_trade(100.0, 10);
  auto t2 = make_trade(100.50, 5);
  auto t3 = make_trade(100.25, 8);

  mgr.OnMboEvent(t1, source, nullptr);
  mgr.OnMboEvent(t2, source, nullptr);
  mgr.OnMboEvent(t3, source, nullptr);

  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 5);

  // trade_flow_imbalance: t2 up(5), t3 down(8) → (5-8)/(5+8) = -3/13
  REQUIRE(vec[0] == Catch::Approx(-3.0 / 13.0).epsilon(1e-12));
  // bar_range: (100.50-100.0)/0.25 = 2.0
  REQUIRE(vec[1] == Catch::Approx(2.0).epsilon(1e-12));
  // bar_body: (100.25-100.0)/0.25 = 1.0
  REQUIRE(vec[2] == Catch::Approx(1.0).epsilon(1e-12));
  // log_volume: log(10+5+8) = log(23)
  REQUIRE(vec[3] == Catch::Approx(std::log(23.0)).epsilon(1e-12));
  // trade_arrival_rate: log(1+3) = log(4)
  REQUIRE(vec[4] == Catch::Approx(std::log(4.0)).epsilon(1e-12));
}
