#include "features/bar/AggressorImbalanceBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void AggressorImbalanceBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  auto size = static_cast<double>(mbo.size);

  // Ask side = buyer-initiated (aggressor lifts the ask)
  // Bid side = seller-initiated (aggressor hits the bid)
  if (mbo.side == databento::Side::Ask)
    buy_aggressor_vol_ += size;
  else if (mbo.side == databento::Side::Bid)
    sell_aggressor_vol_ += size;
}

void AggressorImbalanceBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // Trade data handled in OnMboMsg
}

void AggressorImbalanceBarFeature::ResetAccumulators() {
  buy_aggressor_vol_ = 0.0;
  sell_aggressor_vol_ = 0.0;
  finalized_value_ = 0.0;
}

void AggressorImbalanceBarFeature::FinalizeBar() {
  double total = buy_aggressor_vol_ + sell_aggressor_vol_;
  if (total == 0.0) {
    finalized_value_ = 0.0;
  } else {
    finalized_value_ = (buy_aggressor_vol_ - sell_aggressor_vol_) / (total + kBarFeatureEpsilon);
  }
}

double AggressorImbalanceBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("AggressorImbalanceBarFeature: bar not complete");
  if (name != "aggressor_imbalance")
    throw std::runtime_error("AggressorImbalanceBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool AggressorImbalanceBarFeature::HasFeature(const std::string& name) const {
  return name == "aggressor_imbalance";
}

void AggressorImbalanceBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("AggressorImbalanceBarFeature", AggressorImbalanceBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
