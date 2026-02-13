#include "features/bar/PriceImpactBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void PriceImpactBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  if (!has_trades_) {
    open_ = price;
  }
  close_ = price;
  ++n_trades_;
  has_trades_ = true;
}

void PriceImpactBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void PriceImpactBarFeature::ResetAccumulators() {
  open_ = 0.0;
  close_ = 0.0;
  n_trades_ = 0;
  has_trades_ = false;
  finalized_value_ = 0.0;
}

void PriceImpactBarFeature::FinalizeBar() {
  int denom = std::max(n_trades_, 1);
  finalized_value_ = (close_ - open_) / (denom * tick_size_);
}

double PriceImpactBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("PriceImpactBarFeature: bar not complete");
  if (name != "price_impact_per_trade")
    throw std::runtime_error("PriceImpactBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool PriceImpactBarFeature::HasFeature(const std::string& name) const {
  return name == "price_impact_per_trade";
}

void PriceImpactBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "tick_size") {
    tick_size_ = std::stod(value);
  }
}

REGISTER_FEATURE("PriceImpactBarFeature", PriceImpactBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
