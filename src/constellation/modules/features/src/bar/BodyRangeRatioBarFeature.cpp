#include "features/bar/BodyRangeRatioBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void BodyRangeRatioBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  if (!has_trades_) {
    open_ = price;
  }
  close_ = price;
  high_ = std::max(high_, price);
  low_ = std::min(low_, price);
  has_trades_ = true;
}

void BodyRangeRatioBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void BodyRangeRatioBarFeature::ResetAccumulators() {
  open_ = 0.0;
  close_ = 0.0;
  high_ = -std::numeric_limits<double>::infinity();
  low_ = std::numeric_limits<double>::infinity();
  has_trades_ = false;
  finalized_value_ = 0.0;
}

void BodyRangeRatioBarFeature::FinalizeBar() {
  double range = high_ - low_;
  if (!has_trades_ || range == 0.0) {
    finalized_value_ = 0.0;
  } else {
    finalized_value_ = (close_ - open_) / range;
  }
}

double BodyRangeRatioBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("BodyRangeRatioBarFeature: bar not complete");
  if (name != "body_range_ratio")
    throw std::runtime_error("BodyRangeRatioBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool BodyRangeRatioBarFeature::HasFeature(const std::string& name) const {
  return name == "body_range_ratio";
}

void BodyRangeRatioBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("BodyRangeRatioBarFeature", BodyRangeRatioBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
