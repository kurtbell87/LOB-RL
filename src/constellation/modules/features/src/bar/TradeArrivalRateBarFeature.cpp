#include "features/bar/TradeArrivalRateBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <cmath>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void TradeArrivalRateBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  ++n_trades_;
}

void TradeArrivalRateBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void TradeArrivalRateBarFeature::ResetAccumulators() {
  n_trades_ = 0;
  finalized_value_ = 0.0;
}

void TradeArrivalRateBarFeature::FinalizeBar() {
  finalized_value_ = std::log(1.0 + n_trades_);
}

double TradeArrivalRateBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("TradeArrivalRateBarFeature: bar not complete");
  if (name != "trade_arrival_rate")
    throw std::runtime_error("TradeArrivalRateBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool TradeArrivalRateBarFeature::HasFeature(const std::string& name) const {
  return name == "trade_arrival_rate";
}

void TradeArrivalRateBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("TradeArrivalRateBarFeature", TradeArrivalRateBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
