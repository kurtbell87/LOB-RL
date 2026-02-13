#include "features/bar/VwapDisplacementBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void VwapDisplacementBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  double size = static_cast<double>(mbo.size);
  sum_pv_ += price * size;
  sum_v_ += size;
  close_ = price;
  high_ = std::max(high_, price);
  low_ = std::min(low_, price);
  has_trades_ = true;
}

void VwapDisplacementBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void VwapDisplacementBarFeature::ResetAccumulators() {
  sum_pv_ = 0.0;
  sum_v_ = 0.0;
  close_ = 0.0;
  high_ = -std::numeric_limits<double>::infinity();
  low_ = std::numeric_limits<double>::infinity();
  has_trades_ = false;
  finalized_value_ = 0.0;
}

void VwapDisplacementBarFeature::FinalizeBar() {
  double range = high_ - low_;
  if (!has_trades_ || range == 0.0 || sum_v_ == 0.0) {
    finalized_value_ = 0.0;
  } else {
    double vwap = sum_pv_ / sum_v_;
    finalized_value_ = (close_ - vwap) / range;
  }
}

double VwapDisplacementBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("VwapDisplacementBarFeature: bar not complete");
  if (name != "vwap_displacement")
    throw std::runtime_error("VwapDisplacementBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool VwapDisplacementBarFeature::HasFeature(const std::string& name) const {
  return name == "vwap_displacement";
}

void VwapDisplacementBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("VwapDisplacementBarFeature", VwapDisplacementBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
