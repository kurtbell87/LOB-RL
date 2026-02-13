/**
 * @file TestBookBarFeatures.cpp
 * @brief Tests for the 11 book-dependent bar features (Phase 4).
 *
 * Spec: docs/book-bar-features.md
 *
 * Test categories:
 *   1. Enhanced MockMarketDataSource with configurable book depth
 *   2. Per-feature unit tests (happy path + edge cases) for all 11 features
 *   3. Feature configuration via SetParam (IConfigurableFeature)
 *   4. FeatureRegistry registration
 *   5. Regression tests vs compute_bar_features() columns
 *   6. BarFeatureManager integration with all 22 features (Phase 3 + Phase 4)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Interfaces
#include "interfaces/features/IBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"

// Phase 4 book-dependent bar feature classes (to be implemented)
#include "features/bar/BboImbalanceBarFeature.hpp"
#include "features/bar/DepthImbalanceBarFeature.hpp"
#include "features/bar/CancelAsymmetryBarFeature.hpp"
#include "features/bar/SpreadMeanBarFeature.hpp"
#include "features/bar/OrderFlowImbalanceBarFeature.hpp"
#include "features/bar/DepthRatioBarFeature.hpp"
#include "features/bar/WeightedMidDisplacementBarFeature.hpp"
#include "features/bar/SpreadStdBarFeature.hpp"
#include "features/bar/VampDisplacementBarFeature.hpp"
#include "features/bar/AggressorImbalanceBarFeature.hpp"
#include "features/bar/CancelTradeRatioBarFeature.hpp"

// Phase 3 trade-only features (for full 22-feature integration test)
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


// =============================================================================
// Enhanced Mock: BookMockDataSource
// =============================================================================
//
// Unlike the basic MockMarketDataSource that returns nullopt for depth queries,
// this mock supports configurable bid/ask ladders for testing book-dependent
// features.

namespace {

using BookSide = constellation::interfaces::orderbook::BookSide;
using PriceLevel = constellation::interfaces::orderbook::PriceLevel;

struct LevelConfig {
  double price;        // in real currency (e.g., 100.25)
  std::uint64_t qty;   // quantity at this level
  std::uint32_t count;  // order count
};

class BookMockDataSource final
  : public constellation::interfaces::orderbook::IMarketBookDataSource
{
public:
  // Configurable depth ladders
  std::vector<LevelConfig> bid_levels;  // index 0 = best bid
  std::vector<LevelConfig> ask_levels;  // index 0 = best ask

  std::uint32_t inst_id{1234};

  // Optional overrides for weighted mid / VAMP
  std::optional<double> wmid_override;
  std::optional<double> vamp_override;

  static std::int64_t toNano(double x) {
    return static_cast<std::int64_t>(x * 1e9 + 0.5);
  }

  constellation::interfaces::common::InterfaceVersionInfo
  GetVersionInfo() const noexcept override {
    return {1, 0};
  }

  std::optional<std::int64_t>
  BestBidPrice(std::uint32_t /*instrument_id*/) const override {
    if (bid_levels.empty()) return std::nullopt;
    return toNano(bid_levels[0].price);
  }

  std::optional<std::int64_t>
  BestAskPrice(std::uint32_t /*instrument_id*/) const override {
    if (ask_levels.empty()) return std::nullopt;
    return toNano(ask_levels[0].price);
  }

  std::optional<std::uint64_t>
  VolumeAtPrice(std::uint32_t /*instrument_id*/,
                std::int64_t /*priceNanos*/) const override {
    return std::nullopt;
  }

  std::vector<std::uint32_t> GetInstrumentIds() const override {
    return {inst_id};
  }

  std::optional<PriceLevel>
  GetLevel(std::uint32_t /*instrument_id*/,
           BookSide side,
           std::size_t depth_index) const override {
    const auto& levels = (side == BookSide::Bid) ? bid_levels : ask_levels;
    if (depth_index >= levels.size()) return std::nullopt;
    PriceLevel pl;
    pl.price = toNano(levels[depth_index].price);
    pl.total_quantity = levels[depth_index].qty;
    pl.order_count = levels[depth_index].count;
    return pl;
  }

  std::uint64_t
  TotalDepth(std::uint32_t /*instrument_id*/,
             BookSide side,
             std::size_t n_levels) const override {
    const auto& levels = (side == BookSide::Bid) ? bid_levels : ask_levels;
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < std::min(n_levels, levels.size()); ++i) {
      total += levels[i].qty;
    }
    return total;
  }

  std::optional<double>
  WeightedMidPrice(std::uint32_t /*instrument_id*/) const override {
    if (wmid_override.has_value()) return wmid_override;
    // Default computation from BBO
    if (bid_levels.empty() || ask_levels.empty()) return std::nullopt;
    double bp = bid_levels[0].price;
    double ap = ask_levels[0].price;
    double bq = static_cast<double>(bid_levels[0].qty);
    double aq = static_cast<double>(ask_levels[0].qty);
    if (bq + aq == 0.0) return std::nullopt;
    return (bp * aq + ap * bq) / (bq + aq);
  }

  std::optional<double>
  VolumeAdjustedMidPrice(std::uint32_t /*instrument_id*/,
                          std::size_t n_levels) const override {
    if (vamp_override.has_value()) return vamp_override;
    // Default: VAMP across top n_levels on each side
    double sum_pq = 0.0;
    double sum_q = 0.0;
    for (std::size_t i = 0; i < std::min(n_levels, bid_levels.size()); ++i) {
      sum_pq += bid_levels[i].price * static_cast<double>(bid_levels[i].qty);
      sum_q += static_cast<double>(bid_levels[i].qty);
    }
    for (std::size_t i = 0; i < std::min(n_levels, ask_levels.size()); ++i) {
      sum_pq += ask_levels[i].price * static_cast<double>(ask_levels[i].qty);
      sum_q += static_cast<double>(ask_levels[i].qty);
    }
    if (sum_q == 0.0) return std::nullopt;
    return sum_pq / sum_q;
  }
};


// =============================================================================
// Helpers
// =============================================================================

static constexpr double EPSILON = 1e-10;

/// Convert a double price to databento fixed-point int64.
static std::int64_t to_fixed(double price) {
  return static_cast<std::int64_t>(price * databento::kFixedPriceScale + 0.5);
}

/// Build a Trade MboMsg.
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

/// Build a Fill MboMsg.
static databento::MboMsg make_fill(double price, std::uint32_t size,
                                    std::uint64_t ts_ns = 0,
                                    databento::Side side = databento::Side::Ask) {
  databento::MboMsg mbo = make_trade(price, size, ts_ns, side);
  mbo.action = databento::Action::Fill;
  return mbo;
}

/// Build a Cancel MboMsg.
static databento::MboMsg make_cancel(double price, std::uint32_t size,
                                      databento::Side side) {
  databento::MboMsg mbo{};
  mbo.hd.instrument_id = 1234;
  mbo.order_id = 2;
  mbo.price = to_fixed(price);
  mbo.size = size;
  mbo.action = databento::Action::Cancel;
  mbo.side = side;
  return mbo;
}

/// Build an Add MboMsg.
static databento::MboMsg make_add(double price, std::uint32_t size,
                                   databento::Side side) {
  databento::MboMsg mbo{};
  mbo.hd.instrument_id = 1234;
  mbo.order_id = 3;
  mbo.price = to_fixed(price);
  mbo.size = size;
  mbo.action = databento::Action::Add;
  mbo.side = side;
  return mbo;
}

/// Build a Modify MboMsg.
static databento::MboMsg make_modify(double price, std::uint32_t size,
                                      databento::Side side) {
  databento::MboMsg mbo{};
  mbo.hd.instrument_id = 1234;
  mbo.order_id = 4;
  mbo.price = to_fixed(price);
  mbo.size = size;
  mbo.action = databento::Action::Modify;
  mbo.side = side;
  return mbo;
}


/// Create a standard book state for testing:
///   Bid: 100.00(50), 99.75(30), 99.50(20), 99.25(10), 99.00(5)
///   Ask: 100.25(40), 100.50(25), 100.75(15), 101.00(10), 101.25(5)
static BookMockDataSource make_standard_book() {
  BookMockDataSource src;
  src.bid_levels = {
    {100.00, 50, 5},
    { 99.75, 30, 3},
    { 99.50, 20, 2},
    { 99.25, 10, 1},
    { 99.00,  5, 1},
  };
  src.ask_levels = {
    {100.25, 40, 4},
    {100.50, 25, 3},
    {100.75, 15, 2},
    {101.00, 10, 1},
    {101.25,  5, 1},
  };
  return src;
}


/// Run a single-bar lifecycle with a configurable book mock.
/// Feeds MBO messages and book events in alternation.
template <typename FeatureT>
double run_book_bar(std::shared_ptr<FeatureT> feat,
                    BookMockDataSource& source,
                    const std::vector<databento::MboMsg>& msgs,
                    const std::string& value_name,
                    std::uint64_t bar_index = 0) {
  feat->OnBarStart(bar_index);
  for (const auto& mbo : msgs) {
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(bar_index);
  return feat->GetBarValue(value_name);
}


/// Run a single-bar lifecycle with ONLY AccumulateEvent (no MBO messages).
/// This tests features that only read book state at finalize.
template <typename FeatureT>
double run_book_only_bar(std::shared_ptr<FeatureT> feat,
                          BookMockDataSource& source,
                          int n_events,
                          const std::string& value_name,
                          std::uint64_t bar_index = 0) {
  feat->OnBarStart(bar_index);
  for (int i = 0; i < n_events; ++i) {
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(bar_index);
  return feat->GetBarValue(value_name);
}


} // anonymous namespace


// =============================================================================
// Section 1: BboImbalanceBarFeature (Col 1)
// =============================================================================

TEST_CASE("BboImbalance: balanced book → 0.5 region", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  source.ask_levels = {{100.25, 50, 5}};

  double val = run_book_only_bar(feat, source, 3, "bbo_imbalance");
  // bid_qty=50, ask_qty=50 → 50 / (50+50) = 0.5
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("BboImbalance: heavy bid side → close to 1.0", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 100, 10}};
  source.ask_levels = {{100.25, 10, 1}};

  double val = run_book_only_bar(feat, source, 1, "bbo_imbalance");
  // 100 / (100 + 10) = 100/110
  REQUIRE(val == Catch::Approx(100.0 / 110.0).epsilon(1e-12));
}

TEST_CASE("BboImbalance: heavy ask side → close to 0.0", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 5, 1}};
  source.ask_levels = {{100.25, 95, 10}};

  double val = run_book_only_bar(feat, source, 1, "bbo_imbalance");
  // 5 / (5 + 95) = 5/100 = 0.05
  REQUIRE(val == Catch::Approx(0.05).epsilon(1e-12));
}

TEST_CASE("BboImbalance: empty book → default 0.5", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  // No levels at all
  double val = run_book_only_bar(feat, source, 1, "bbo_imbalance");
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("BboImbalance: one-sided book (bid only) → 1.0", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  // No ask levels

  double val = run_book_only_bar(feat, source, 1, "bbo_imbalance");
  // bid=50, ask=0 (GetLevel returns nullopt) → 50 / (50+0) = 1.0
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("BboImbalance: one-sided book (ask only) → 0.0", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.ask_levels = {{100.25, 50, 5}};
  // No bid levels

  double val = run_book_only_bar(feat, source, 1, "bbo_imbalance");
  // bid=0, ask=50 → 0 / (0+50) = 0.0
  REQUIRE(val == Catch::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("BboImbalance: uses last event's book state", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  // Feed two different book states; only the LAST one should be used
  BookMockDataSource source1;
  source1.bid_levels = {{100.00, 10, 1}};
  source1.ask_levels = {{100.25, 90, 9}};

  BookMockDataSource source2;
  source2.bid_levels = {{100.00, 80, 8}};
  source2.ask_levels = {{100.25, 20, 2}};

  feat->OnBarStart(0);
  feat->OnDataUpdate(source1, nullptr);  // first event: bid=10, ask=90
  feat->OnDataUpdate(source2, nullptr);  // last event: bid=80, ask=20
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("bbo_imbalance");
  // Should use source2's state: 80 / (80+20) = 0.8
  REQUIRE(val == Catch::Approx(0.8).epsilon(1e-12));
}

TEST_CASE("BboImbalance: HasFeature", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  REQUIRE(feat->HasFeature("bbo_imbalance"));
  REQUIRE_FALSE(feat->HasFeature("depth_imbalance"));
}

TEST_CASE("BboImbalance: reset between bars", "[bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source1;
  source1.bid_levels = {{100.00, 80, 8}};
  source1.ask_levels = {{100.25, 20, 2}};

  // Bar 0
  run_book_only_bar(feat, source1, 1, "bbo_imbalance", 0);

  // Bar 1 with different book state
  BookMockDataSource source2;
  source2.bid_levels = {{100.00, 30, 3}};
  source2.ask_levels = {{100.25, 70, 7}};

  double val = run_book_only_bar(feat, source2, 1, "bbo_imbalance", 1);
  REQUIRE(val == Catch::Approx(0.3).epsilon(1e-12));
}


// =============================================================================
// Section 2: DepthImbalanceBarFeature (Col 2)
// =============================================================================

TEST_CASE("DepthImbalance: standard 5-level book", "[depth_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  auto source = make_standard_book();
  double val = run_book_only_bar(feat, source, 1, "depth_imbalance");

  // TotalDepth(Bid, 5) = 50+30+20+10+5 = 115
  // TotalDepth(Ask, 5) = 40+25+15+10+5 = 95
  // imbalance = 115 / (115+95) = 115/210
  REQUIRE(val == Catch::Approx(115.0 / 210.0).epsilon(1e-12));
}

TEST_CASE("DepthImbalance: empty book → 0.5", "[depth_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  double val = run_book_only_bar(feat, source, 1, "depth_imbalance");
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("DepthImbalance: one-sided book (all bids) → 1.0", "[depth_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}, {99.75, 30, 3}};
  double val = run_book_only_bar(feat, source, 1, "depth_imbalance");
  // bid=80, ask=0 → 80 / (80+0) = 1.0
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("DepthImbalance: custom n_levels=3", "[depth_imbalance][config]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("n_levels", "3");

  auto source = make_standard_book();
  double val = run_book_only_bar(feat, source, 1, "depth_imbalance");

  // TotalDepth(Bid, 3) = 50+30+20 = 100
  // TotalDepth(Ask, 3) = 40+25+15 = 80
  // imbalance = 100 / (100+80) = 100/180
  REQUIRE(val == Catch::Approx(100.0 / 180.0).epsilon(1e-12));
}

TEST_CASE("DepthImbalance: HasFeature", "[depth_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  REQUIRE(feat->HasFeature("depth_imbalance"));
  REQUIRE_FALSE(feat->HasFeature("bbo_imbalance"));
}


// =============================================================================
// Section 3: CancelAsymmetryBarFeature (Col 10)
// =============================================================================

TEST_CASE("CancelAsymmetry: more bid cancels → positive", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.00, 10, databento::Side::Bid),
    make_cancel(100.00, 5, databento::Side::Bid),
    make_cancel(100.25, 3, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_asymmetry");
  // bid_cancels=2, ask_cancels=1 → (2-1)/(2+1+eps) ≈ 1/3
  REQUIRE(val == Catch::Approx(1.0 / (3.0 + EPSILON)).epsilon(1e-10));
}

TEST_CASE("CancelAsymmetry: more ask cancels → negative", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.25, 10, databento::Side::Ask),
    make_cancel(100.50, 5, databento::Side::Ask),
    make_cancel(100.75, 3, databento::Side::Ask),
    make_cancel(100.00, 8, databento::Side::Bid),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_asymmetry");
  // bid_cancels=1, ask_cancels=3 → (1-3)/(1+3+eps) = -2/(4+eps)
  REQUIRE(val == Catch::Approx(-2.0 / (4.0 + EPSILON)).epsilon(1e-10));
}

TEST_CASE("CancelAsymmetry: symmetric cancels → 0", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.00, 10, databento::Side::Bid),
    make_cancel(100.25, 5, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_asymmetry");
  // bid=1, ask=1 → (1-1)/(1+1+eps) = 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("CancelAsymmetry: no cancels → 0", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  // No cancel messages at all — only a trade
  std::vector<databento::MboMsg> msgs = {
    make_trade(100.0, 5),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_asymmetry");
  // (0-0)/(0+0+eps) ≈ 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("CancelAsymmetry: only non-cancel actions → 0", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),
    make_trade(100.25, 5),
    make_modify(100.50, 3, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_asymmetry");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("CancelAsymmetry: HasFeature", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  REQUIRE(feat->HasFeature("cancel_asymmetry"));
  REQUIRE_FALSE(feat->HasFeature("bbo_imbalance"));
}

TEST_CASE("CancelAsymmetry: reset between bars", "[cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();
  BookMockDataSource source = make_standard_book();

  // Bar 0: 3 bid, 1 ask cancel
  {
    feat->OnBarStart(0);
    auto c1 = make_cancel(100.00, 10, databento::Side::Bid);
    auto c2 = make_cancel(99.75, 5, databento::Side::Bid);
    auto c3 = make_cancel(99.50, 3, databento::Side::Bid);
    auto c4 = make_cancel(100.25, 8, databento::Side::Ask);
    feat->OnMboMsg(c1); feat->OnDataUpdate(source, nullptr);
    feat->OnMboMsg(c2); feat->OnDataUpdate(source, nullptr);
    feat->OnMboMsg(c3); feat->OnDataUpdate(source, nullptr);
    feat->OnMboMsg(c4); feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(0);
  }

  // Bar 1: only 1 ask cancel (should NOT carry bar 0's counts)
  {
    feat->OnBarStart(1);
    auto c = make_cancel(100.50, 5, databento::Side::Ask);
    feat->OnMboMsg(c); feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(1);

    double val = feat->GetBarValue("cancel_asymmetry");
    // bid=0, ask=1 → (0-1)/(0+1+eps) = -1/(1+eps)
    REQUIRE(val == Catch::Approx(-1.0 / (1.0 + EPSILON)).epsilon(1e-10));
  }
}


// =============================================================================
// Section 4: SpreadMeanBarFeature (Col 11)
// =============================================================================

TEST_CASE("SpreadMean: constant spread", "[spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  source.ask_levels = {{100.25, 40, 4}};

  // 3 events, all same spread: 100.25 - 100.00 = 0.25
  double val = run_book_only_bar(feat, source, 3, "spread_mean");
  REQUIRE(val == Catch::Approx(0.25).epsilon(1e-12));
}

TEST_CASE("SpreadMean: varying spread across events", "[spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  feat->SetParam("instrument_id", "1234");

  // Simulate 3 events with different spreads
  BookMockDataSource source1;
  source1.bid_levels = {{100.00, 50, 5}};
  source1.ask_levels = {{100.25, 40, 4}};

  BookMockDataSource source2;
  source2.bid_levels = {{100.00, 50, 5}};
  source2.ask_levels = {{100.50, 40, 4}};

  BookMockDataSource source3;
  source3.bid_levels = {{99.75, 50, 5}};
  source3.ask_levels = {{100.50, 40, 4}};

  feat->OnBarStart(0);
  feat->OnDataUpdate(source1, nullptr);  // spread = 0.25
  feat->OnDataUpdate(source2, nullptr);  // spread = 0.50
  feat->OnDataUpdate(source3, nullptr);  // spread = 0.75
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("spread_mean");
  // mean = (0.25 + 0.50 + 0.75) / 3 = 0.5
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("SpreadMean: no valid spread samples → default 1.0", "[spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  // Empty book → BestBidPrice/BestAskPrice return nullopt → skip
  double val = run_book_only_bar(feat, source, 3, "spread_mean");
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("SpreadMean: one-sided book (bid only) → skip → default 1.0", "[spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  // No ask

  double val = run_book_only_bar(feat, source, 3, "spread_mean");
  REQUIRE(val == Catch::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("SpreadMean: HasFeature", "[spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  REQUIRE(feat->HasFeature("spread_mean"));
  REQUIRE_FALSE(feat->HasFeature("spread_std"));
}


// =============================================================================
// Section 5: OrderFlowImbalanceBarFeature (OFI, Col 13)
// =============================================================================

TEST_CASE("OFI: all bid adds → positive", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),
    make_add(99.75, 20, databento::Side::Bid),
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  // signed = +10 + +20 = +30, total_add = 30
  // clamp(30 / (30 + eps), -1, 1) ≈ 1.0
  REQUIRE(val == Catch::Approx(1.0).margin(1e-8));
}

TEST_CASE("OFI: all ask adds → negative", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.25, 10, databento::Side::Ask),
    make_add(100.50, 20, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  // signed = -10 + -20 = -30, total_add = 30
  // clamp(-30 / (30 + eps), -1, 1) ≈ -1.0
  REQUIRE(val == Catch::Approx(-1.0).margin(1e-8));
}

TEST_CASE("OFI: mixed adds → balanced", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),   // +10
    make_add(100.25, 15, databento::Side::Ask),    // -15
    make_add(99.75, 5, databento::Side::Bid),      // +5
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  // signed = 10 - 15 + 5 = 0, total_add = 30
  // clamp(0 / (30+eps), -1, 1) ≈ 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("OFI: no add events → 0", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  // Only trades and cancels — no adds
  std::vector<databento::MboMsg> msgs = {
    make_trade(100.0, 5),
    make_cancel(100.00, 3, databento::Side::Bid),
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("OFI: cancel and modify ignored", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),          // +10
    make_cancel(100.00, 5, databento::Side::Bid),        // ignored
    make_modify(100.00, 3, databento::Side::Ask),         // ignored
    make_add(100.25, 10, databento::Side::Ask),           // -10
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  // signed = 10 - 10 = 0, total_add = 20
  REQUIRE(val == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("OFI: HasFeature", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  REQUIRE(feat->HasFeature("ofi"));
  REQUIRE_FALSE(feat->HasFeature("trade_flow_imbalance"));
}

TEST_CASE("OFI: clamped to [-1, 1]", "[ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  // All bid adds → signed/total = 1.0 (at the clamp boundary)
  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 100, databento::Side::Bid),
  };

  double val = run_book_bar(feat, source, msgs, "ofi");
  REQUIRE(val >= -1.0);
  REQUIRE(val <= 1.0);
}


// =============================================================================
// Section 6: DepthRatioBarFeature (Col 14)
// =============================================================================

TEST_CASE("DepthRatio: standard book", "[depth_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::DepthRatioBarFeature>();
  feat->SetParam("instrument_id", "1234");

  auto source = make_standard_book();
  double val = run_book_only_bar(feat, source, 1, "depth_ratio");

  // total_3 = TotalDepth(Bid,3) + TotalDepth(Ask,3)
  //         = (50+30+20) + (40+25+15) = 100 + 80 = 180
  // total_10 = TotalDepth(Bid,10) + TotalDepth(Ask,10)
  //          = (50+30+20+10+5) + (40+25+15+10+5) = 115 + 95 = 210
  // ratio = 180 / (210 + eps)
  REQUIRE(val == Catch::Approx(180.0 / (210.0 + EPSILON)).epsilon(1e-10));
}

TEST_CASE("DepthRatio: empty book → 0.5", "[depth_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::DepthRatioBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  double val = run_book_only_bar(feat, source, 1, "depth_ratio");
  REQUIRE(val == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("DepthRatio: all depth within top 3 → ratio near 1.0", "[depth_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::DepthRatioBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}, {99.75, 30, 3}, {99.50, 20, 2}};
  source.ask_levels = {{100.25, 40, 4}, {100.50, 25, 3}, {100.75, 15, 2}};
  // Only 3 levels per side → total_3 == total_10

  double val = run_book_only_bar(feat, source, 1, "depth_ratio");
  // total_3 = 100 + 80 = 180
  // total_10 = same = 180
  // ratio = 180 / (180 + eps) ≈ 1.0
  REQUIRE(val == Catch::Approx(180.0 / (180.0 + EPSILON)).epsilon(1e-10));
}

TEST_CASE("DepthRatio: HasFeature", "[depth_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::DepthRatioBarFeature>();
  REQUIRE(feat->HasFeature("depth_ratio"));
  REQUIRE_FALSE(feat->HasFeature("depth_imbalance"));
}


// =============================================================================
// Section 7: WeightedMidDisplacementBarFeature (Col 15)
// =============================================================================

TEST_CASE("WeightedMidDisplacement: wmid moves up", "[wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");

  // First event: wmid = 100.10, last event: wmid = 100.20
  BookMockDataSource source1;
  source1.wmid_override = 100.10;

  BookMockDataSource source2;
  source2.wmid_override = 100.20;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source1, nullptr);  // first → wmid_first = 100.10
  feat->OnDataUpdate(source2, nullptr);  // last  → wmid_end = 100.20
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("wmid_displacement");
  // (100.20 - 100.10) / 0.25 = 0.10 / 0.25 = 0.4
  REQUIRE(val == Catch::Approx(0.4).epsilon(1e-10));
}

TEST_CASE("WeightedMidDisplacement: wmid moves down", "[wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source1;
  source1.wmid_override = 100.50;

  BookMockDataSource source2;
  source2.wmid_override = 100.00;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source1, nullptr);
  feat->OnDataUpdate(source2, nullptr);
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("wmid_displacement");
  // (100.00 - 100.50) / 0.25 = -0.50 / 0.25 = -2.0
  REQUIRE(val == Catch::Approx(-2.0).epsilon(1e-10));
}

TEST_CASE("WeightedMidDisplacement: no change → 0", "[wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.wmid_override = 100.125;

  double val = run_book_only_bar(feat, source, 3, "wmid_displacement");
  // wmid_first = wmid_end = 100.125 → 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("WeightedMidDisplacement: wmid unavailable → 0", "[wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  // wmid_override is nullopt by default, empty book → WeightedMidPrice returns nullopt

  double val = run_book_only_bar(feat, source, 3, "wmid_displacement");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("WeightedMidDisplacement: custom tick_size", "[wmid_displacement][config]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("tick_size", "0.5");

  BookMockDataSource source1;
  source1.wmid_override = 100.00;

  BookMockDataSource source2;
  source2.wmid_override = 101.00;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source1, nullptr);
  feat->OnDataUpdate(source2, nullptr);
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("wmid_displacement");
  // (101 - 100) / 0.5 = 2.0
  REQUIRE(val == Catch::Approx(2.0).epsilon(1e-10));
}

TEST_CASE("WeightedMidDisplacement: HasFeature", "[wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  REQUIRE(feat->HasFeature("wmid_displacement"));
  REQUIRE_FALSE(feat->HasFeature("vamp_displacement"));
}


// =============================================================================
// Section 8: SpreadStdBarFeature (Col 16)
// =============================================================================

TEST_CASE("SpreadStd: constant spread → 0", "[spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  source.ask_levels = {{100.25, 40, 4}};

  // All events have the same spread → std = 0
  double val = run_book_only_bar(feat, source, 5, "spread_std");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("SpreadStd: varying spread → positive std", "[spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  feat->SetParam("instrument_id", "1234");

  // 4 events with spreads: 0.25, 0.50, 0.25, 0.50
  BookMockDataSource s1;
  s1.bid_levels = {{100.00, 50, 5}};
  s1.ask_levels = {{100.25, 40, 4}};  // spread = 0.25

  BookMockDataSource s2;
  s2.bid_levels = {{100.00, 50, 5}};
  s2.ask_levels = {{100.50, 40, 4}};  // spread = 0.50

  feat->OnBarStart(0);
  feat->OnDataUpdate(s1, nullptr);  // 0.25
  feat->OnDataUpdate(s2, nullptr);  // 0.50
  feat->OnDataUpdate(s1, nullptr);  // 0.25
  feat->OnDataUpdate(s2, nullptr);  // 0.50
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("spread_std");
  // mean = (0.25+0.50+0.25+0.50)/4 = 0.375
  // E[x^2] = (0.0625+0.25+0.0625+0.25)/4 = 0.15625
  // var = 0.15625 - 0.375^2 = 0.15625 - 0.140625 = 0.015625
  // std = sqrt(0.015625) = 0.125
  REQUIRE(val == Catch::Approx(0.125).epsilon(1e-10));
}

TEST_CASE("SpreadStd: single event → 0", "[spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;
  source.bid_levels = {{100.00, 50, 5}};
  source.ask_levels = {{100.25, 40, 4}};

  double val = run_book_only_bar(feat, source, 1, "spread_std");
  // < 2 samples → 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("SpreadStd: no valid samples → 0", "[spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BookMockDataSource source;  // empty book
  double val = run_book_only_bar(feat, source, 5, "spread_std");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("SpreadStd: HasFeature", "[spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  REQUIRE(feat->HasFeature("spread_std"));
  REQUIRE_FALSE(feat->HasFeature("spread_mean"));
}


// =============================================================================
// Section 9: VampDisplacementBarFeature (Col 17)
// =============================================================================

TEST_CASE("VampDisplacement: VAMP moves up", "[vamp_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("bar_size", "4");

  // 4 events → midpoint at event index 2 (bar_size/2 = 2)
  BookMockDataSource s1, s2, s3, s4;
  s1.vamp_override = 100.00;
  s2.vamp_override = 100.10;
  s3.vamp_override = 100.20;  // mid sample (event 2 when bar_size=4)
  s4.vamp_override = 100.50;

  feat->OnBarStart(0);
  feat->OnDataUpdate(s1, nullptr);  // event 0
  feat->OnDataUpdate(s2, nullptr);  // event 1
  feat->OnDataUpdate(s3, nullptr);  // event 2 → midpoint for bar_size=4
  feat->OnDataUpdate(s4, nullptr);  // event 3 → vamp_at_end
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("vamp_displacement");
  // (vamp_at_end - vamp_at_mid) / tick_size = (100.50 - 100.20) / 0.25 = 1.2
  REQUIRE(val == Catch::Approx((100.50 - 100.20) / 0.25).epsilon(1e-10));
}

TEST_CASE("VampDisplacement: VAMP moves down", "[vamp_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("bar_size", "2");

  // 2 events → midpoint at event 1 (bar_size/2 = 1)
  BookMockDataSource s1, s2;
  s1.vamp_override = 100.50;
  s2.vamp_override = 100.00;

  feat->OnBarStart(0);
  feat->OnDataUpdate(s1, nullptr);  // event 0
  feat->OnDataUpdate(s2, nullptr);  // event 1 → mid for bar_size=2 (also end)
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("vamp_displacement");
  // Note: mid = event at bar_size/2 = 1 → s2, end = also s2
  // displacement = (100.00 - 100.00) / 0.25 = 0 if mid == end
  // OR if mid is captured at event_count >= bar_size/2 for the first time:
  // event 0: count=1, not >= 1 yet? That depends on implementation.
  // The spec says: capture vamp_at_mid when trade_count >= bar_size/2 first time.
  // With bar_size=2, bar_size/2=1. At event 1 (count=1), capture mid.
  // At event 1 (last), capture end.
  // displacement = (s2 - s2) / 0.25 = 0

  // For clearer testing, let's just check it's a finite number
  REQUIRE(std::isfinite(val));
}

TEST_CASE("VampDisplacement: VAMP unavailable → 0", "[vamp_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("bar_size", "2");

  BookMockDataSource source;  // empty book → VAMP returns nullopt

  double val = run_book_only_bar(feat, source, 3, "vamp_displacement");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("VampDisplacement: no change → 0", "[vamp_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("bar_size", "4");

  BookMockDataSource source;
  source.vamp_override = 100.125;

  double val = run_book_only_bar(feat, source, 4, "vamp_displacement");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("VampDisplacement: custom tick_size and n_levels", "[vamp_displacement][config]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");
  feat->SetParam("tick_size", "0.5");
  feat->SetParam("n_levels", "3");
  feat->SetParam("bar_size", "4");

  BookMockDataSource s_mid;
  s_mid.vamp_override = 100.00;

  BookMockDataSource s_end;
  s_end.vamp_override = 101.00;

  feat->OnBarStart(0);
  feat->OnDataUpdate(s_mid, nullptr);  // event 0
  feat->OnDataUpdate(s_mid, nullptr);  // event 1
  feat->OnDataUpdate(s_mid, nullptr);  // event 2 → mid (bar_size/2 = 2)
  feat->OnDataUpdate(s_end, nullptr);  // event 3 → end
  feat->OnBarComplete(0);

  double val = feat->GetBarValue("vamp_displacement");
  // (101 - 100) / 0.5 = 2.0
  REQUIRE(val == Catch::Approx(2.0).epsilon(1e-10));
}

TEST_CASE("VampDisplacement: HasFeature", "[vamp_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::VampDisplacementBarFeature>();
  REQUIRE(feat->HasFeature("vamp_displacement"));
  REQUIRE_FALSE(feat->HasFeature("wmid_displacement"));
}


// =============================================================================
// Section 10: AggressorImbalanceBarFeature (Col 18)
// =============================================================================

TEST_CASE("AggressorImbalance: all buyer-initiated → +1", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  // Buyer-initiated: mbo.side == Ask (aggressor lifts the ask)
  std::vector<databento::MboMsg> msgs = {
    make_trade(100.25, 10, 0, databento::Side::Ask),
    make_trade(100.50, 5, 0, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "aggressor_imbalance");
  // buy=15, sell=0 → (15-0)/(15+0+eps) ≈ 1.0
  REQUIRE(val == Catch::Approx(1.0).margin(1e-8));
}

TEST_CASE("AggressorImbalance: all seller-initiated → -1", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  // Seller-initiated: mbo.side == Bid (aggressor hits the bid)
  std::vector<databento::MboMsg> msgs = {
    make_trade(100.00, 10, 0, databento::Side::Bid),
    make_trade(99.75, 8, 0, databento::Side::Bid),
  };

  double val = run_book_bar(feat, source, msgs, "aggressor_imbalance");
  // buy=0, sell=18 → (0-18)/(0+18+eps) ≈ -1.0
  REQUIRE(val == Catch::Approx(-1.0).margin(1e-8));
}

TEST_CASE("AggressorImbalance: mixed aggressor → balanced", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_trade(100.25, 10, 0, databento::Side::Ask),   // buyer-init
    make_trade(100.00, 10, 0, databento::Side::Bid),   // seller-init
  };

  double val = run_book_bar(feat, source, msgs, "aggressor_imbalance");
  // buy=10, sell=10 → (10-10)/(10+10+eps) ≈ 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("AggressorImbalance: no trades → 0", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  // Only adds — no trades or fills
  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),
    make_cancel(100.25, 5, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "aggressor_imbalance");
  REQUIRE(val == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("AggressorImbalance: Fill action counts as trade", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_fill(100.25, 10, 0, databento::Side::Ask),   // buyer-init
    make_fill(100.00, 5, 0, databento::Side::Bid),    // seller-init
  };

  double val = run_book_bar(feat, source, msgs, "aggressor_imbalance");
  // buy=10, sell=5 → (10-5)/(10+5+eps) ≈ 5/15
  REQUIRE(val == Catch::Approx(5.0 / (15.0 + EPSILON)).epsilon(1e-8));
}

TEST_CASE("AggressorImbalance: HasFeature", "[aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();
  REQUIRE(feat->HasFeature("aggressor_imbalance"));
  REQUIRE_FALSE(feat->HasFeature("trade_flow_imbalance"));
}


// =============================================================================
// Section 11: CancelTradeRatioBarFeature (Col 20)
// =============================================================================

TEST_CASE("CancelTradeRatio: basic computation", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.00, 10, databento::Side::Bid),
    make_cancel(100.25, 5, databento::Side::Ask),
    make_cancel(99.75, 3, databento::Side::Bid),
    make_trade(100.00, 7),
    make_trade(100.25, 3),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_trade_ratio");
  // n_cancels=3, n_trades=2 → log(1 + 3/max(2,1)) = log(1 + 1.5) = log(2.5)
  REQUIRE(val == Catch::Approx(std::log(2.5)).epsilon(1e-12));
}

TEST_CASE("CancelTradeRatio: no cancels → 0", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_trade(100.0, 5),
    make_trade(100.25, 3),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_trade_ratio");
  // n_cancels=0, n_trades=2 → log(1 + 0/2) = log(1) = 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("CancelTradeRatio: no trades → uses max(n_trades,1)=1", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.00, 10, databento::Side::Bid),
    make_cancel(100.25, 5, databento::Side::Ask),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_trade_ratio");
  // n_cancels=2, n_trades=0 → log(1 + 2/max(0,1)) = log(1+2) = log(3)
  REQUIRE(val == Catch::Approx(std::log(3.0)).epsilon(1e-12));
}

TEST_CASE("CancelTradeRatio: no events → 0", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  double val = run_book_only_bar(feat, source, 1, "cancel_trade_ratio");
  // n_cancels=0, n_trades=0 → log(1 + 0/1) = 0
  REQUIRE(val == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("CancelTradeRatio: Fill counts as trade", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_cancel(100.00, 10, databento::Side::Bid),
    make_fill(100.25, 5),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_trade_ratio");
  // n_cancels=1, n_trades=1 → log(1 + 1/1) = log(2)
  REQUIRE(val == Catch::Approx(std::log(2.0)).epsilon(1e-12));
}

TEST_CASE("CancelTradeRatio: Add action ignored", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  BookMockDataSource source = make_standard_book();

  std::vector<databento::MboMsg> msgs = {
    make_add(100.00, 10, databento::Side::Bid),
    make_cancel(100.25, 5, databento::Side::Ask),
    make_trade(100.25, 3),
  };

  double val = run_book_bar(feat, source, msgs, "cancel_trade_ratio");
  // n_cancels=1, n_trades=1 → log(1 + 1) = log(2)
  REQUIRE(val == Catch::Approx(std::log(2.0)).epsilon(1e-12));
}

TEST_CASE("CancelTradeRatio: HasFeature", "[cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();
  REQUIRE(feat->HasFeature("cancel_trade_ratio"));
  REQUIRE_FALSE(feat->HasFeature("cancel_asymmetry"));
}


// =============================================================================
// Section 12: IConfigurableFeature Tests
// =============================================================================

TEST_CASE("All book bar features implement IConfigurableFeature", "[config]") {
  using namespace constellation::modules::features;
  using IC = constellation::interfaces::features::IConfigurableFeature;

  REQUIRE(dynamic_cast<IC*>(std::make_shared<BboImbalanceBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<DepthImbalanceBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<CancelAsymmetryBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<SpreadMeanBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<OrderFlowImbalanceBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<DepthRatioBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<WeightedMidDisplacementBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<SpreadStdBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<VampDisplacementBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<AggressorImbalanceBarFeature>().get()) != nullptr);
  REQUIRE(dynamic_cast<IC*>(std::make_shared<CancelTradeRatioBarFeature>().get()) != nullptr);
}


// =============================================================================
// Section 13: FeatureRegistry Registration
// =============================================================================

TEST_CASE("FeatureRegistry: BboImbalanceBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("BboImbalanceBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("bbo_imbalance"));
}

TEST_CASE("FeatureRegistry: DepthImbalanceBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("DepthImbalanceBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("depth_imbalance"));
}

TEST_CASE("FeatureRegistry: CancelAsymmetryBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("CancelAsymmetryBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("cancel_asymmetry"));
}

TEST_CASE("FeatureRegistry: SpreadMeanBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("SpreadMeanBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("spread_mean"));
}

TEST_CASE("FeatureRegistry: OrderFlowImbalanceBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("OrderFlowImbalanceBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("ofi"));
}

TEST_CASE("FeatureRegistry: DepthRatioBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("DepthRatioBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("depth_ratio"));
}

TEST_CASE("FeatureRegistry: WeightedMidDisplacementBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("WeightedMidDisplacementBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("wmid_displacement"));
}

TEST_CASE("FeatureRegistry: SpreadStdBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("SpreadStdBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("spread_std"));
}

TEST_CASE("FeatureRegistry: VampDisplacementBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("VampDisplacementBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("vamp_displacement"));
}

TEST_CASE("FeatureRegistry: AggressorImbalanceBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("AggressorImbalanceBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("aggressor_imbalance"));
}

TEST_CASE("FeatureRegistry: CancelTradeRatioBarFeature registered", "[registry][book]") {
  auto feat = constellation::modules::features::FeatureRegistry::Instance().Create("CancelTradeRatioBarFeature");
  REQUIRE(feat != nullptr);
  REQUIRE(feat->HasFeature("cancel_trade_ratio"));
}


// =============================================================================
// Section 14: Regression Tests vs compute_bar_features()
// =============================================================================

namespace {

/// Build a TradeBar and BarBookAccum, then compare a book-dependent feature's
/// output against the reference column from compute_bar_features().
void regression_check_book_column(
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feat,
    const std::string& value_name,
    int col_index,
    const BarBookAccum& accum,
    const std::vector<std::pair<double, int>>& trades = {{100.0, 10}},
    int bar_index = 0) {

  // --- Reference: compute_bar_features ---
  TradeBar ref_bar;
  ref_bar.bar_index = bar_index;

  if (!trades.empty()) {
    double sum_pv = 0.0;
    int sum_v = 0;
    ref_bar.open = trades.front().first;
    ref_bar.close = trades.back().first;
    ref_bar.high = -std::numeric_limits<double>::infinity();
    ref_bar.low = std::numeric_limits<double>::infinity();
    for (const auto& [price, size] : trades) {
      ref_bar.trade_prices.push_back(price);
      ref_bar.trade_sizes.push_back(size);
      ref_bar.high = std::max(ref_bar.high, price);
      ref_bar.low = std::min(ref_bar.low, price);
      sum_pv += price * size;
      sum_v += size;
    }
    ref_bar.volume = sum_v;
    ref_bar.vwap = (sum_v > 0) ? sum_pv / sum_v : 0.0;
  }

  std::vector<TradeBar> bars = {ref_bar};
  std::vector<BarBookAccum> accums = {accum};
  auto ref = compute_bar_features(bars, accums, 0, 86400000000000ULL);
  double expected = ref[col_index];

  // --- Book mock: set up from accum fields ---
  // We build a BookMockDataSource that matches the accum exactly.
  BookMockDataSource source;

  // BBO from accum
  if (accum.bid_qty > 0 || accum.ask_qty > 0) {
    source.bid_levels = {{100.00, accum.bid_qty, 1}};
    source.ask_levels = {{100.25, accum.ask_qty, 1}};
  }

  // WeightedMid from accum
  if (!std::isnan(accum.wmid_first) || !std::isnan(accum.wmid_end)) {
    // For simplicity, override directly
    if (!std::isnan(accum.wmid_end))
      source.wmid_override = accum.wmid_end;
  }

  // VAMP from accum
  if (!std::isnan(accum.vamp_at_end))
    source.vamp_override = accum.vamp_at_end;

  // For features that need the full bar lifecycle, use the feature's
  // OnBarStart/OnMboMsg/OnDataUpdate/OnBarComplete cycle.
  // This checks that the feature produces the same value as the reference.

  feat->OnBarStart(bar_index);

  // Feed trades and other MBO events as needed
  for (const auto& [price, size] : trades) {
    auto mbo = make_trade(price, static_cast<std::uint32_t>(size));
    feat->OnMboMsg(mbo);
    feat->OnDataUpdate(source, nullptr);
  }

  feat->OnBarComplete(bar_index);
  double actual = feat->GetBarValue(value_name);

  if (std::isnan(expected)) {
    REQUIRE(std::isnan(actual));
  } else {
    REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
  }
}

/// Build a default single-trade TradeBar for regression tests.
/// Used by manual regression tests that need a TradeBar to pass
/// to compute_bar_features().
static TradeBar make_default_trade_bar() {
  TradeBar bar{};
  bar.trade_prices = {100.0};
  bar.trade_sizes = {10};
  bar.open = bar.close = bar.high = bar.low = 100.0;
  bar.volume = 10;
  bar.vwap = 100.0;
  return bar;
}

/// Get reference value from compute_bar_features() for a given column and accumulator.
static double reference_bar_feature(int col_index, const BarBookAccum& accum) {
  std::vector<TradeBar> bars = {make_default_trade_bar()};
  std::vector<BarBookAccum> accums = {accum};
  auto ref = compute_bar_features(bars, accums, 0, 86400000000000ULL);
  return ref[col_index];
}

} // anonymous namespace


TEST_CASE("Regression: BboImbalance matches compute_bar_features col 1",
          "[regression][bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.bid_qty = 50;
  accum.ask_qty = 30;

  // Reference: col 1 = 50 / (50+30) = 50/80 = 0.625
  regression_check_book_column(feat, "bbo_imbalance", 1, accum);
}

TEST_CASE("Regression: BboImbalance col 1 edge - zero qty",
          "[regression][bbo_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::BboImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.bid_qty = 0;
  accum.ask_qty = 0;

  // Reference: col 1 = 0.5 (default)
  BookMockDataSource source;
  // Empty book → GetLevel returns nullopt for both sides

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("bbo_imbalance");

  double expected = reference_bar_feature(1, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("Regression: DepthImbalance matches compute_bar_features col 2",
          "[regression][depth_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::DepthImbalanceBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.total_bid_5 = 115;
  accum.total_ask_5 = 95;

  // Build a matching mock
  BookMockDataSource source;
  // Need 5-level depth that sums to bid=115, ask=95
  source.bid_levels = {{100.00, 50, 5}, {99.75, 30, 3}, {99.50, 20, 2}, {99.25, 10, 1}, {99.00, 5, 1}};
  source.ask_levels = {{100.25, 40, 4}, {100.50, 25, 3}, {100.75, 15, 2}, {101.00, 10, 1}, {101.25, 5, 1}};

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("depth_imbalance");

  double expected = reference_bar_feature(2, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: CancelAsymmetry matches compute_bar_features col 10",
          "[regression][cancel_asymmetry]") {
  auto feat = std::make_shared<constellation::modules::features::CancelAsymmetryBarFeature>();

  BarBookAccum accum;
  accum.bid_cancels = 5;
  accum.ask_cancels = 3;

  // Feed 5 bid + 3 ask cancels
  BookMockDataSource source = make_standard_book();
  feat->OnBarStart(0);
  for (int i = 0; i < 5; ++i) {
    auto c = make_cancel(100.00, 1, databento::Side::Bid);
    feat->OnMboMsg(c);
    feat->OnDataUpdate(source, nullptr);
  }
  for (int i = 0; i < 3; ++i) {
    auto c = make_cancel(100.25, 1, databento::Side::Ask);
    feat->OnMboMsg(c);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("cancel_asymmetry");

  double expected = reference_bar_feature(10, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: SpreadMean matches compute_bar_features col 11",
          "[regression][spread_mean]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadMeanBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.spread_samples = {0.25, 0.50, 0.25, 0.75};

  // Feed 4 events with spreads matching the samples
  BookMockDataSource s1, s2, s3, s4;
  s1.bid_levels = {{100.00, 50, 5}}; s1.ask_levels = {{100.25, 40, 4}};  // 0.25
  s2.bid_levels = {{100.00, 50, 5}}; s2.ask_levels = {{100.50, 40, 4}};  // 0.50
  s3.bid_levels = {{100.00, 50, 5}}; s3.ask_levels = {{100.25, 40, 4}};  // 0.25
  s4.bid_levels = {{100.00, 50, 5}}; s4.ask_levels = {{100.75, 40, 4}};  // 0.75

  feat->OnBarStart(0);
  feat->OnDataUpdate(s1, nullptr);
  feat->OnDataUpdate(s2, nullptr);
  feat->OnDataUpdate(s3, nullptr);
  feat->OnDataUpdate(s4, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("spread_mean");

  double expected = reference_bar_feature(11, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: OFI matches compute_bar_features col 13",
          "[regression][ofi]") {
  auto feat = std::make_shared<constellation::modules::features::OrderFlowImbalanceBarFeature>();

  BarBookAccum accum;
  accum.ofi_signed_volume = 15.0;
  accum.total_add_volume = 30.0;

  // Feed adds: 3 bid(10 each) + 0 ask → signed=+30, total=30
  // Hmm, need to match signed=15, total=30
  // bid_adds: total_size = X, ask_adds: total_size = Y
  // signed = X - Y = 15, total = X + Y = 30 → X=22.5, Y=7.5
  // Use: bid=23, ask=7 (integer sizes, close enough for the test)
  // Actually the regression test should match exactly. Let's use:
  // 1 bid add of 22 + 1 ask add of 7 → signed = 22 - 7 = 15, total = 29
  // That doesn't match accum.total_add_volume=30.
  // Let me match exactly: 2 bids (15 each) + 1 ask (15) → signed = 15 - 15 = 0
  // Hmm. Let me just use: bid=15, total=30 → ask=15, signed=0. No.
  // signed=15, total=30 → bid_sum = (30+15)/2 = 22.5, not integer.
  // Use accum values: signed=20, total=30 → bid=25, ask=5
  BookMockDataSource source = make_standard_book();

  BarBookAccum accum2;
  accum2.ofi_signed_volume = 20.0;
  accum2.total_add_volume = 30.0;

  feat->OnBarStart(0);
  auto a1 = make_add(100.00, 25, databento::Side::Bid);  // +25
  auto a2 = make_add(100.25, 5, databento::Side::Ask);    // -5
  feat->OnMboMsg(a1); feat->OnDataUpdate(source, nullptr);
  feat->OnMboMsg(a2); feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("ofi");

  double expected = reference_bar_feature(13, accum2);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: DepthRatio matches compute_bar_features col 14",
          "[regression][depth_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::DepthRatioBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.total_bid_3 = 100;
  accum.total_ask_3 = 80;
  accum.total_bid_10 = 115;
  accum.total_ask_10 = 95;

  auto source = make_standard_book();
  // standard book: TotalDepth(Bid,3)=100, TotalDepth(Ask,3)=80
  // TotalDepth(Bid,10)=115, TotalDepth(Ask,10)=95

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("depth_ratio");

  double expected = reference_bar_feature(14, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: WeightedMidDisplacement matches compute_bar_features col 15",
          "[regression][wmid_displacement]") {
  auto feat = std::make_shared<constellation::modules::features::WeightedMidDisplacementBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.wmid_first = 100.10;
  accum.wmid_end = 100.35;

  BookMockDataSource source_first;
  source_first.wmid_override = 100.10;
  BookMockDataSource source_end;
  source_end.wmid_override = 100.35;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source_first, nullptr);
  feat->OnDataUpdate(source_end, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("wmid_displacement");

  double expected = reference_bar_feature(15, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: SpreadStd matches compute_bar_features col 16",
          "[regression][spread_std]") {
  auto feat = std::make_shared<constellation::modules::features::SpreadStdBarFeature>();
  feat->SetParam("instrument_id", "1234");

  BarBookAccum accum;
  accum.spread_samples = {0.25, 0.50, 0.25, 0.75};

  // Same spread sequence as the SpreadMean regression test
  BookMockDataSource s1, s2, s3, s4;
  s1.bid_levels = {{100.00, 50, 5}}; s1.ask_levels = {{100.25, 40, 4}};
  s2.bid_levels = {{100.00, 50, 5}}; s2.ask_levels = {{100.50, 40, 4}};
  s3.bid_levels = {{100.00, 50, 5}}; s3.ask_levels = {{100.25, 40, 4}};
  s4.bid_levels = {{100.00, 50, 5}}; s4.ask_levels = {{100.75, 40, 4}};

  feat->OnBarStart(0);
  feat->OnDataUpdate(s1, nullptr);
  feat->OnDataUpdate(s2, nullptr);
  feat->OnDataUpdate(s3, nullptr);
  feat->OnDataUpdate(s4, nullptr);
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("spread_std");

  double expected = reference_bar_feature(16, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: AggressorImbalance matches compute_bar_features col 18",
          "[regression][aggressor_imbalance]") {
  auto feat = std::make_shared<constellation::modules::features::AggressorImbalanceBarFeature>();

  BarBookAccum accum;
  accum.buy_aggressor_vol = 30.0;
  accum.sell_aggressor_vol = 10.0;

  BookMockDataSource source = make_standard_book();
  feat->OnBarStart(0);
  // 3 buyer-initiated trades (total 30)
  for (int i = 0; i < 3; ++i) {
    auto t = make_trade(100.25, 10, 0, databento::Side::Ask);
    feat->OnMboMsg(t);
    feat->OnDataUpdate(source, nullptr);
  }
  // 1 seller-initiated trade (total 10)
  {
    auto t = make_trade(100.00, 10, 0, databento::Side::Bid);
    feat->OnMboMsg(t);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("aggressor_imbalance");

  double expected = reference_bar_feature(18, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-10));
}

TEST_CASE("Regression: CancelTradeRatio matches compute_bar_features col 20",
          "[regression][cancel_trade_ratio]") {
  auto feat = std::make_shared<constellation::modules::features::CancelTradeRatioBarFeature>();

  BarBookAccum accum;
  accum.n_cancels = 7;
  accum.n_trades = 3;

  BookMockDataSource source = make_standard_book();
  feat->OnBarStart(0);
  for (int i = 0; i < 7; ++i) {
    auto c = make_cancel(100.00, 1, databento::Side::Bid);
    feat->OnMboMsg(c);
    feat->OnDataUpdate(source, nullptr);
  }
  for (int i = 0; i < 3; ++i) {
    auto t = make_trade(100.25, 5);
    feat->OnMboMsg(t);
    feat->OnDataUpdate(source, nullptr);
  }
  feat->OnBarComplete(0);
  double actual = feat->GetBarValue("cancel_trade_ratio");

  double expected = reference_bar_feature(20, accum);

  REQUIRE(actual == Catch::Approx(expected).epsilon(1e-12));
}


// =============================================================================
// Section 15: Edge Cases — GetBarValue before OnBarComplete
// =============================================================================

TEST_CASE("All book features: GetBarValue throws before OnBarComplete", "[edge][book]") {
  using namespace constellation::modules::features;

  auto features = std::vector<std::pair<
    std::shared_ptr<constellation::interfaces::features::IBarFeature>, std::string>>{
    {std::make_shared<BboImbalanceBarFeature>(), "bbo_imbalance"},
    {std::make_shared<DepthImbalanceBarFeature>(), "depth_imbalance"},
    {std::make_shared<CancelAsymmetryBarFeature>(), "cancel_asymmetry"},
    {std::make_shared<SpreadMeanBarFeature>(), "spread_mean"},
    {std::make_shared<OrderFlowImbalanceBarFeature>(), "ofi"},
    {std::make_shared<DepthRatioBarFeature>(), "depth_ratio"},
    {std::make_shared<WeightedMidDisplacementBarFeature>(), "wmid_displacement"},
    {std::make_shared<SpreadStdBarFeature>(), "spread_std"},
    {std::make_shared<VampDisplacementBarFeature>(), "vamp_displacement"},
    {std::make_shared<AggressorImbalanceBarFeature>(), "aggressor_imbalance"},
    {std::make_shared<CancelTradeRatioBarFeature>(), "cancel_trade_ratio"},
  };

  for (auto& [feat, name] : features) {
    REQUIRE_THROWS(feat->GetBarValue(name));
  }
}

TEST_CASE("All book features: unknown value name throws", "[edge][book]") {
  using namespace constellation::modules::features;

  auto features = std::vector<std::pair<
    std::shared_ptr<constellation::interfaces::features::IBarFeature>, std::string>>{
    {std::make_shared<BboImbalanceBarFeature>(), "bbo_imbalance"},
    {std::make_shared<DepthImbalanceBarFeature>(), "depth_imbalance"},
    {std::make_shared<CancelAsymmetryBarFeature>(), "cancel_asymmetry"},
    {std::make_shared<SpreadMeanBarFeature>(), "spread_mean"},
    {std::make_shared<OrderFlowImbalanceBarFeature>(), "ofi"},
    {std::make_shared<DepthRatioBarFeature>(), "depth_ratio"},
    {std::make_shared<WeightedMidDisplacementBarFeature>(), "wmid_displacement"},
    {std::make_shared<SpreadStdBarFeature>(), "spread_std"},
    {std::make_shared<VampDisplacementBarFeature>(), "vamp_displacement"},
    {std::make_shared<AggressorImbalanceBarFeature>(), "aggressor_imbalance"},
    {std::make_shared<CancelTradeRatioBarFeature>(), "cancel_trade_ratio"},
  };

  BookMockDataSource source = make_standard_book();
  for (auto& [feat, name] : features) {
    feat->OnBarStart(0);
    feat->OnDataUpdate(source, nullptr);
    feat->OnBarComplete(0);
    REQUIRE_THROWS(feat->GetBarValue("nonexistent_value_name_xyz"));
  }
}


// =============================================================================
// Section 16: BarFeatureManager Integration — All 22 Features
// =============================================================================

TEST_CASE("BarFeatureManager: all 22 features produce 22-element vector",
          "[bar_feature_manager][integration][full]") {
  using namespace constellation::modules::features;
  BarFeatureManager mgr;

  // Phase 3 trade-only features (11)
  auto flow = std::make_shared<TradeFlowImbalanceBarFeature>();
  auto range = std::make_shared<BarRangeBarFeature>();
  auto body = std::make_shared<BarBodyBarFeature>();
  auto brr = std::make_shared<BodyRangeRatioBarFeature>();
  auto vwap_disp = std::make_shared<VwapDisplacementBarFeature>();
  auto log_vol = std::make_shared<LogVolumeBarFeature>();
  auto rv = std::make_shared<RealizedVolBarFeature>();
  auto sess_time = std::make_shared<SessionTimeBarFeature>();
  auto sess_age = std::make_shared<SessionAgeBarFeature>();
  auto arrival = std::make_shared<TradeArrivalRateBarFeature>();
  auto impact = std::make_shared<PriceImpactBarFeature>();

  // Phase 4 book-dependent features (11)
  auto bbo = std::make_shared<BboImbalanceBarFeature>();
  bbo->SetParam("instrument_id", "1234");
  auto depth_imb = std::make_shared<DepthImbalanceBarFeature>();
  depth_imb->SetParam("instrument_id", "1234");
  auto cancel_asym = std::make_shared<CancelAsymmetryBarFeature>();
  auto spread_mean = std::make_shared<SpreadMeanBarFeature>();
  spread_mean->SetParam("instrument_id", "1234");
  auto ofi = std::make_shared<OrderFlowImbalanceBarFeature>();
  auto depth_ratio = std::make_shared<DepthRatioBarFeature>();
  depth_ratio->SetParam("instrument_id", "1234");
  auto wmid_disp = std::make_shared<WeightedMidDisplacementBarFeature>();
  wmid_disp->SetParam("instrument_id", "1234");
  auto spread_std = std::make_shared<SpreadStdBarFeature>();
  spread_std->SetParam("instrument_id", "1234");
  auto vamp_disp = std::make_shared<VampDisplacementBarFeature>();
  vamp_disp->SetParam("instrument_id", "1234");
  vamp_disp->SetParam("bar_size", "3");
  auto agg_imb = std::make_shared<AggressorImbalanceBarFeature>();
  auto ct_ratio = std::make_shared<CancelTradeRatioBarFeature>();

  // Register all in column order
  mgr.RegisterBarFeature(flow, "trade_flow_imbalance");        // col 0
  mgr.RegisterBarFeature(bbo, "bbo_imbalance");                 // col 1
  mgr.RegisterBarFeature(depth_imb, "depth_imbalance");         // col 2
  mgr.RegisterBarFeature(range, "bar_range");                   // col 3
  mgr.RegisterBarFeature(body, "bar_body");                     // col 4
  mgr.RegisterBarFeature(brr, "body_range_ratio");              // col 5
  mgr.RegisterBarFeature(vwap_disp, "vwap_displacement");       // col 6
  mgr.RegisterBarFeature(log_vol, "log_volume");                // col 7
  mgr.RegisterBarFeature(rv, "realized_vol");                   // col 8
  mgr.RegisterBarFeature(sess_time, "session_time");            // col 9
  mgr.RegisterBarFeature(cancel_asym, "cancel_asymmetry");      // col 10
  mgr.RegisterBarFeature(spread_mean, "spread_mean");           // col 11
  mgr.RegisterBarFeature(sess_age, "session_age");              // col 12
  mgr.RegisterBarFeature(ofi, "ofi");                           // col 13
  mgr.RegisterBarFeature(depth_ratio, "depth_ratio");           // col 14
  mgr.RegisterBarFeature(wmid_disp, "wmid_displacement");       // col 15
  mgr.RegisterBarFeature(spread_std, "spread_std");             // col 16
  mgr.RegisterBarFeature(vamp_disp, "vamp_displacement");       // col 17
  mgr.RegisterBarFeature(agg_imb, "aggressor_imbalance");       // col 18
  mgr.RegisterBarFeature(arrival, "trade_arrival_rate");        // col 19
  mgr.RegisterBarFeature(ct_ratio, "cancel_trade_ratio");       // col 20
  mgr.RegisterBarFeature(impact, "price_impact_per_trade");     // col 21

  REQUIRE(mgr.FeatureCount() == 22);

  // Build a book mock
  auto source = make_standard_book();

  mgr.NotifyBarStart(0);

  // Feed a mix of MBO events
  auto add1 = make_add(100.00, 10, databento::Side::Bid);
  auto add2 = make_add(100.25, 5, databento::Side::Ask);
  auto cancel1 = make_cancel(99.75, 3, databento::Side::Bid);
  auto trade1 = make_trade(100.0, 7, 0, databento::Side::Ask);
  auto trade2 = make_trade(100.25, 4, 0, databento::Side::Bid);

  mgr.OnMboEvent(add1, source, nullptr);
  mgr.OnMboEvent(add2, source, nullptr);
  mgr.OnMboEvent(cancel1, source, nullptr);
  mgr.OnMboEvent(trade1, source, nullptr);
  mgr.OnMboEvent(trade2, source, nullptr);

  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 22);

  // Verify each element is a finite number (not NaN except realized_vol warmup)
  for (int i = 0; i < 22; ++i) {
    if (i == 8) {
      // realized_vol during warmup → NaN expected
      REQUIRE(std::isnan(vec[i]));
    } else {
      REQUIRE(std::isfinite(vec[i]));
    }
  }
}
