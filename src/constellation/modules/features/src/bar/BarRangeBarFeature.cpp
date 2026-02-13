#include "features/bar/BarRangeBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void BarRangeBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  high_ = std::max(high_, price);
  low_ = std::min(low_, price);
  has_trades_ = true;
}

void BarRangeBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void BarRangeBarFeature::ResetAccumulators() {
  high_ = -std::numeric_limits<double>::infinity();
  low_ = std::numeric_limits<double>::infinity();
  has_trades_ = false;
  finalized_value_ = 0.0;
}

void BarRangeBarFeature::FinalizeBar() {
  if (!has_trades_) {
    finalized_value_ = 0.0;
  } else {
    finalized_value_ = (high_ - low_) / tick_size_;
  }
}

double BarRangeBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("BarRangeBarFeature: bar not complete");
  if (name != "bar_range")
    throw std::runtime_error("BarRangeBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool BarRangeBarFeature::HasFeature(const std::string& name) const {
  return name == "bar_range";
}

void BarRangeBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "tick_size") {
    tick_size_ = std::stod(value);
  }
}

REGISTER_FEATURE("BarRangeBarFeature", BarRangeBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
