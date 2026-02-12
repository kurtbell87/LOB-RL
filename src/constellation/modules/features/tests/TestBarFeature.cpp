/**
 * @file TestBarFeature.cpp
 * @brief Tests for IBarFeature interface, AbstractBarFeature base class,
 *        and BarFeatureManager.
 *
 * Spec: docs/ibar-feature.md
 *
 * Test categories:
 *   1. AbstractBarFeature lifecycle (~8 tests)
 *   2. BarFeatureManager dispatch (~8 tests)
 *   3. Integration — full bar lifecycle (~4 tests)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "features/AbstractBarFeature.hpp"
#include "features/BarFeatureManager.hpp"
#include "interfaces/features/IBarFeature.hpp"
#include "MockMarketDataSource.hpp"

// ═══════════════════════════════════════════════════════════════════════
// Test helper: EventCountBarFeature
//
// A minimal concrete bar feature that counts the number of MBO events
// accumulated within a bar. This is NOT implementation code — it's a
// test fixture that exercises the AbstractBarFeature contract.
// ═══════════════════════════════════════════════════════════════════════

class EventCountBarFeature final : public constellation::modules::features::AbstractBarFeature {
public:
  int count() const { return count_; }

  double GetBarValue(const std::string& name) const override {
    if (!IsBarComplete()) {
      throw std::runtime_error("GetBarValue called before bar completion");
    }
    if (name == "event_count") {
      return static_cast<double>(finalized_count_);
    }
    throw std::runtime_error("Unknown bar value name: " + name);
  }

  bool HasFeature(const std::string& name) const override {
    return name == "event_count";
  }

protected:
  void AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) override {
    ++count_;
  }

  void ResetAccumulators() override {
    count_ = 0;
  }

  void FinalizeBar() override {
    finalized_count_ = count_;
  }

private:
  int count_{0};
  int finalized_count_{0};
};

// A second test feature — accumulates sum of best bid prices.
class BidSumBarFeature final : public constellation::modules::features::AbstractBarFeature {
public:
  double GetBarValue(const std::string& name) const override {
    if (!IsBarComplete()) {
      throw std::runtime_error("GetBarValue called before bar completion");
    }
    if (name == "bid_sum") {
      return finalized_sum_;
    }
    throw std::runtime_error("Unknown bar value name: " + name);
  }

  bool HasFeature(const std::string& name) const override {
    return name == "bid_sum";
  }

protected:
  void AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) override {
    auto bid = source.BestBidPrice(0);
    if (bid.has_value()) {
      sum_ += static_cast<double>(*bid) / 1e9;
    }
  }

  void ResetAccumulators() override {
    sum_ = 0.0;
  }

  void FinalizeBar() override {
    finalized_sum_ = sum_;
  }

private:
  double sum_{0.0};
  double finalized_sum_{0.0};
};


// ═══════════════════════════════════════════════════════════════════════
// Section 1: AbstractBarFeature Lifecycle Tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("AbstractBarFeature: OnBarStart resets accumulators", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Start a bar, accumulate some events
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(2.0));

  // Start a new bar — accumulators must be reset
  feat->OnBarStart(1);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(1);
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(1.0));
}

TEST_CASE("AbstractBarFeature: AccumulateEvent called during bar", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 50.0;
  source.best_ask = 51.0;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);

  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(3.0));
}

TEST_CASE("AbstractBarFeature: OnBarComplete finalizes values", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);

  // After OnBarComplete, IsBarComplete() should be true
  REQUIRE(feat->IsBarComplete());
  // And the finalized value should be accessible
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(2.0));
}

TEST_CASE("AbstractBarFeature: IsBarComplete transitions correctly", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Before any bar: not complete
  REQUIRE_FALSE(feat->IsBarComplete());

  // After OnBarStart: not complete (we're inside a bar)
  feat->OnBarStart(0);
  REQUIRE_FALSE(feat->IsBarComplete());

  // After OnBarComplete: complete
  feat->OnBarComplete(0);
  REQUIRE(feat->IsBarComplete());

  // After OnBarStart for next bar: not complete again
  feat->OnBarStart(1);
  REQUIRE_FALSE(feat->IsBarComplete());
}

TEST_CASE("AbstractBarFeature: GetBarValue throws before bar completion", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // No bar started at all — should throw
  REQUIRE_THROWS_AS(feat->GetBarValue("event_count"), std::runtime_error);

  // Inside a bar — should throw
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  REQUIRE_THROWS_AS(feat->GetBarValue("event_count"), std::runtime_error);
}

TEST_CASE("AbstractBarFeature: events outside bar are silently ignored", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Events before any OnBarStart — should be no-op
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);

  // Now start a bar and accumulate one event
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);

  // Only the event inside the bar should be counted
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(1.0));
}

TEST_CASE("AbstractBarFeature: GetValue delegates to GetBarValue", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);

  // GetValue (from IFeature) should delegate to GetBarValue
  REQUIRE(feat->GetValue("event_count") == Catch::Approx(2.0));
}

TEST_CASE("AbstractBarFeature: OnBarStart twice without OnBarComplete resets", "[bar_feature][lifecycle]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Start bar 0, accumulate events
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);

  // Start bar 1 without completing bar 0 — should reset
  feat->OnBarStart(1);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(1);

  // Only the one event from bar 1 should be counted
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(1.0));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 2: BarFeatureManager Dispatch Tests
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("BarFeatureManager: register and extract single feature", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;

  auto feat = std::make_shared<EventCountBarFeature>();
  mgr.RegisterBarFeature(feat, "event_count");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  mgr.NotifyBarStart(0);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 1);
  REQUIRE(vec[0] == Catch::Approx(2.0));
}

TEST_CASE("BarFeatureManager: multiple features in registration order", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;

  auto count_feat = std::make_shared<EventCountBarFeature>();
  auto bid_feat = std::make_shared<BidSumBarFeature>();

  // Register in specific order — vector should follow this order
  mgr.RegisterBarFeature(count_feat, "event_count");
  mgr.RegisterBarFeature(bid_feat, "bid_sum");

  MockMarketDataSource source;
  source.best_bid = 50.0;
  source.best_ask = 51.0;

  mgr.NotifyBarStart(0);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == Catch::Approx(2.0));          // event_count
  REQUIRE(vec[1] == Catch::Approx(100.0));         // bid_sum = 50.0 * 2
}

TEST_CASE("BarFeatureManager: empty manager returns empty vector", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  REQUIRE(mgr.FeatureCount() == 0);

  // GetBarFeatureVector on empty manager should return empty
  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.empty());
}

TEST_CASE("BarFeatureManager: FeatureCount reports correct count", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  REQUIRE(mgr.FeatureCount() == 0);

  mgr.RegisterBarFeature(std::make_shared<EventCountBarFeature>(), "event_count");
  REQUIRE(mgr.FeatureCount() == 1);

  mgr.RegisterBarFeature(std::make_shared<BidSumBarFeature>(), "bid_sum");
  REQUIRE(mgr.FeatureCount() == 2);
}

TEST_CASE("BarFeatureManager: AllBarsComplete reports correctly", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  auto feat1 = std::make_shared<EventCountBarFeature>();
  auto feat2 = std::make_shared<BidSumBarFeature>();
  mgr.RegisterBarFeature(feat1, "event_count");
  mgr.RegisterBarFeature(feat2, "bid_sum");

  // Before any bar: not complete
  REQUIRE_FALSE(mgr.AllBarsComplete());

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  mgr.NotifyBarStart(0);
  REQUIRE_FALSE(mgr.AllBarsComplete());

  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);
  REQUIRE(mgr.AllBarsComplete());
}

TEST_CASE("BarFeatureManager: GetBarFeatureVector throws before any bar completes", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  auto feat = std::make_shared<EventCountBarFeature>();
  mgr.RegisterBarFeature(feat, "event_count");

  // No bar completed yet — should throw
  REQUIRE_THROWS(mgr.GetBarFeatureVector());
}

TEST_CASE("BarFeatureManager: NotifyBarStart and NotifyBarComplete dispatch to all features", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  auto feat1 = std::make_shared<EventCountBarFeature>();
  auto feat2 = std::make_shared<EventCountBarFeature>();
  mgr.RegisterBarFeature(feat1, "event_count");
  mgr.RegisterBarFeature(feat2, "event_count");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  mgr.NotifyBarStart(0);

  // Both features should be in-bar (not complete)
  REQUIRE_FALSE(feat1->IsBarComplete());
  REQUIRE_FALSE(feat2->IsBarComplete());

  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);

  // Both should now be complete
  REQUIRE(feat1->IsBarComplete());
  REQUIRE(feat2->IsBarComplete());
}

TEST_CASE("BarFeatureManager: OnMboEvent forwards to all registered features", "[bar_feature_manager]") {
  constellation::modules::features::BarFeatureManager mgr;
  auto feat1 = std::make_shared<EventCountBarFeature>();
  auto feat2 = std::make_shared<EventCountBarFeature>();
  mgr.RegisterBarFeature(feat1, "event_count");
  mgr.RegisterBarFeature(feat2, "event_count");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  mgr.NotifyBarStart(0);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == Catch::Approx(3.0));
  REQUIRE(vec[1] == Catch::Approx(3.0));
}


// ═══════════════════════════════════════════════════════════════════════
// Section 3: Integration — Full Bar Lifecycle
// ═══════════════════════════════════════════════════════════════════════

TEST_CASE("Integration: full bar lifecycle with test feature", "[bar_feature][integration]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Warmup events (before bar) — should be ignored
  feat->OnDataUpdate(source, nullptr);

  // Bar 0
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(0);

  REQUIRE(feat->IsBarComplete());
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(2.0));
  REQUIRE(feat->GetValue("event_count") == Catch::Approx(2.0));
  REQUIRE(feat->HasFeature("event_count"));
  REQUIRE_FALSE(feat->HasFeature("nonexistent"));
}

TEST_CASE("Integration: multiple bars in sequence", "[bar_feature][integration]") {
  constellation::modules::features::BarFeatureManager mgr;
  auto feat = std::make_shared<EventCountBarFeature>();
  mgr.RegisterBarFeature(feat, "event_count");

  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Bar 0: 3 events
  mgr.NotifyBarStart(0);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(0);
  {
    auto vec = mgr.GetBarFeatureVector();
    REQUIRE(vec[0] == Catch::Approx(3.0));
  }

  // Bar 1: 1 event
  mgr.NotifyBarStart(1);
  mgr.OnMboEvent(source, nullptr);
  mgr.NotifyBarComplete(1);
  {
    auto vec = mgr.GetBarFeatureVector();
    REQUIRE(vec[0] == Catch::Approx(1.0));
  }

  // Bar 2: 5 events
  mgr.NotifyBarStart(2);
  for (int i = 0; i < 5; ++i) {
    mgr.OnMboEvent(source, nullptr);
  }
  mgr.NotifyBarComplete(2);
  {
    auto vec = mgr.GetBarFeatureVector();
    REQUIRE(vec[0] == Catch::Approx(5.0));
  }
}

TEST_CASE("Integration: GetBarFeatureVector returns correct values from multiple features", "[bar_feature][integration]") {
  constellation::modules::features::BarFeatureManager mgr;

  auto count_feat = std::make_shared<EventCountBarFeature>();
  auto bid_feat = std::make_shared<BidSumBarFeature>();

  mgr.RegisterBarFeature(count_feat, "event_count");
  mgr.RegisterBarFeature(bid_feat, "bid_sum");

  MockMarketDataSource source;
  source.best_bid = 25.0;
  source.best_ask = 26.0;

  mgr.NotifyBarStart(0);
  mgr.OnMboEvent(source, nullptr);  // count=1, bid_sum=25
  source.best_bid = 75.0;
  mgr.OnMboEvent(source, nullptr);  // count=2, bid_sum=100
  mgr.NotifyBarComplete(0);

  auto vec = mgr.GetBarFeatureVector();
  REQUIRE(vec.size() == 2);
  REQUIRE(vec[0] == Catch::Approx(2.0));          // event_count
  REQUIRE(vec[1] == Catch::Approx(100.0));         // bid_sum = 25 + 75
}

TEST_CASE("Integration: bar_index is informational, not enforced", "[bar_feature][integration]") {
  auto feat = std::make_shared<EventCountBarFeature>();
  MockMarketDataSource source;
  source.best_bid = 100.0;
  source.best_ask = 101.0;

  // Start with bar_index 0, complete with bar_index 99 — should not error
  feat->OnBarStart(0);
  feat->OnDataUpdate(source, nullptr);
  feat->OnBarComplete(99);  // Mismatched index — implementation ignores this

  REQUIRE(feat->IsBarComplete());
  REQUIRE(feat->GetBarValue("event_count") == Catch::Approx(1.0));
}
