#include "features/bar/BarBodyBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void BarBodyBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  if (!has_trades_) {
    open_ = price;
  }
  close_ = price;
  has_trades_ = true;
}

void BarBodyBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void BarBodyBarFeature::ResetAccumulators() {
  open_ = 0.0;
  close_ = 0.0;
  has_trades_ = false;
  finalized_value_ = 0.0;
}

void BarBodyBarFeature::FinalizeBar() {
  if (!has_trades_) {
    finalized_value_ = 0.0;
  } else {
    finalized_value_ = (close_ - open_) / tick_size_;
  }
}

double BarBodyBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("BarBodyBarFeature: bar not complete");
  if (name != "bar_body")
    throw std::runtime_error("BarBodyBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool BarBodyBarFeature::HasFeature(const std::string& name) const {
  return name == "bar_body";
}

void BarBodyBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "tick_size") {
    tick_size_ = std::stod(value);
  }
}

REGISTER_FEATURE("BarBodyBarFeature", BarBodyBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
