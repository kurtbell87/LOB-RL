#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <memory>
#include "features/FeatureManager.hpp"
#include "features/primitives/SpreadFeature.hpp"
#include "features/primitives/MicroPriceFeature.hpp"
#include "features/primitives/OrderImbalanceFeature.hpp"
#include "features/primitives/LogReturnFeature.hpp"
#include "features/derived/RollingVolatilityFeature.hpp"
#include "features/derived/MidPriceFeature.hpp"
#include "MockMarketDataSource.hpp"

using namespace constellation::modules::features;

TEST_CASE("ExtendedFeatures: SpreadFeature basic usage") {
  // SpreadFeature that references instrument_id=777
  primitives::SpreadFeature::Config cfg{777};
  auto spread = std::make_shared<primitives::SpreadFeature>(cfg);

  FeatureManager manager;
  manager.Register(spread);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 100.0;
  src.best_ask = 103.0;

  manager.OnDataUpdate(src, nullptr);
  CHECK(manager.GetValue("bid_ask_spread") == Catch::Approx(3.0));
}

TEST_CASE("ExtendedFeatures: MicroPriceFeature basic usage") {
  FeatureManager manager;
  auto mp_feat = std::make_shared<primitives::MicroPriceFeature>(
                    primitives::MicroPriceFeature::Config{777});
  manager.Register(mp_feat);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 98.0;
  src.best_ask = 102.0;
  src.volumes[98.0]  = 10;  // best bid vol
  src.volumes[102.0] = 20;  // best ask vol

  manager.OnDataUpdate(src, nullptr);
  double mp = manager.GetValue("micro_price");
  // Weighted: askVol=20 => bidPx=98, bidVol=10 => askPx=102
  // Weighted formula => (20*98 + 10*102)/(20+10) = (1960 + 1020)/30=2980/30=99.3333
  CHECK(mp == Catch::Approx(99.3333).margin(1e-4));
}

TEST_CASE("ExtendedFeatures: OrderImbalanceFeature basic usage") {
  FeatureManager manager;
  auto imb = std::make_shared<primitives::OrderImbalanceFeature>(
                primitives::OrderImbalanceFeature::Config{777});
  manager.Register(imb);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 100.0;
  src.best_ask = 101.0;
  src.volumes[100.0] = 30;
  src.volumes[101.0] = 10;

  manager.OnDataUpdate(src, nullptr);
  // imbalance=(30-10)/(30+10)=0.5
  CHECK(manager.GetValue("order_imbalance") == Catch::Approx(0.5));
}

TEST_CASE("ExtendedFeatures: LogReturnFeature basic usage") {
  FeatureManager manager;
  auto logret = std::make_shared<primitives::LogReturnFeature>(
                   primitives::LogReturnFeature::Config{777});
  manager.Register(logret);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 100.0;

  // first update => no previous => 0.0
  manager.OnDataUpdate(src, nullptr);
  CHECK(manager.GetValue("log_return") == Catch::Approx(0.0));

  // next => from 100->102 => log(102/100)
  src.best_bid = 102.0;
  manager.OnDataUpdate(src, nullptr);
  double lr = manager.GetValue("log_return");
  CHECK(lr == Catch::Approx(std::log(102.0/100.0)));
}

TEST_CASE("ExtendedFeatures: RollingVolatilityFeature basic usage") {
  derived::RollingVolatilityFeature::Config cfg{777, 3};
  auto rollvol = std::make_shared<derived::RollingVolatilityFeature>(cfg);

  FeatureManager manager;
  manager.Register(rollvol);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 100.0;

  // 1) first update => no prior => vol=0
  manager.OnDataUpdate(src, nullptr);
  CHECK(manager.GetValue("rolling_volatility") == Catch::Approx(0.0));

  // 2) from 100->101
  src.best_bid = 101.0;
  manager.OnDataUpdate(src, nullptr);
  CHECK(manager.GetValue("rolling_volatility") == Catch::Approx(0.0));

  // 3) from 101->102 => 2 data points => stdev>0
  src.best_bid = 102.0;
  manager.OnDataUpdate(src, nullptr);
  double vol1 = manager.GetValue("rolling_volatility");
  CHECK(vol1 > 0.0);

  // 4) 102->104 => 3 data points => stdev changes
  src.best_bid = 104.0;
  manager.OnDataUpdate(src, nullptr);
  double vol2 = manager.GetValue("rolling_volatility");
  CHECK(vol2 != vol1);
}

TEST_CASE("ExtendedFeatures: MidPriceFeature basic usage") {
  FeatureManager manager;
  auto mid_feat = std::make_shared<derived::MidPriceFeature>(
                    derived::MidPriceFeature::Config{777});
  manager.Register(mid_feat);

  MockMarketDataSource src;
  src.instrument_ids = {777};
  src.best_bid = 100.0;
  src.best_ask = 102.0;
  manager.OnDataUpdate(src, nullptr);

  CHECK(manager.GetValue("mid_price") == Catch::Approx(101.0));

  // if ask < bid => zero
  src.best_bid = 105.0;
  src.best_ask = 104.0;
  manager.OnDataUpdate(src, nullptr);
  CHECK(manager.GetValue("mid_price") == Catch::Approx(0.0));
}
