#include "features/bar/OrderFlowImbalanceBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void OrderFlowImbalanceBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (mbo.action != databento::Action::Add) return;

  auto size = static_cast<double>(mbo.size);
  total_add_volume_ += size;

  if (mbo.side == databento::Side::Bid)
    signed_volume_ += size;
  else if (mbo.side == databento::Side::Ask)
    signed_volume_ -= size;
}

void OrderFlowImbalanceBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // Add data handled in OnMboMsg
}

void OrderFlowImbalanceBarFeature::ResetAccumulators() {
  signed_volume_ = 0.0;
  total_add_volume_ = 0.0;
  finalized_value_ = 0.0;
}

void OrderFlowImbalanceBarFeature::FinalizeBar() {
  if (signed_volume_ == 0.0 && total_add_volume_ == 0.0) {
    finalized_value_ = 0.0;
  } else {
    finalized_value_ = std::clamp(signed_volume_ / (total_add_volume_ + kBarFeatureEpsilon), -1.0, 1.0);
  }
}

double OrderFlowImbalanceBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("OrderFlowImbalanceBarFeature: bar not complete");
  if (name != "ofi")
    throw std::runtime_error("OrderFlowImbalanceBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool OrderFlowImbalanceBarFeature::HasFeature(const std::string& name) const {
  return name == "ofi";
}

void OrderFlowImbalanceBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("OrderFlowImbalanceBarFeature", OrderFlowImbalanceBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
