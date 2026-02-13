#include "features/bar/CancelAsymmetryBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void CancelAsymmetryBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (mbo.action != databento::Action::Cancel) return;

  if (mbo.side == databento::Side::Bid)
    ++bid_cancels_;
  else if (mbo.side == databento::Side::Ask)
    ++ask_cancels_;
}

void CancelAsymmetryBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // Cancel data handled in OnMboMsg
}

void CancelAsymmetryBarFeature::ResetAccumulators() {
  bid_cancels_ = 0;
  ask_cancels_ = 0;
  finalized_value_ = 0.0;
}

void CancelAsymmetryBarFeature::FinalizeBar() {
  double b = static_cast<double>(bid_cancels_);
  double a = static_cast<double>(ask_cancels_);
  finalized_value_ = (b - a) / (b + a + kBarFeatureEpsilon);
}

double CancelAsymmetryBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("CancelAsymmetryBarFeature: bar not complete");
  if (name != "cancel_asymmetry")
    throw std::runtime_error("CancelAsymmetryBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool CancelAsymmetryBarFeature::HasFeature(const std::string& name) const {
  return name == "cancel_asymmetry";
}

void CancelAsymmetryBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("CancelAsymmetryBarFeature", CancelAsymmetryBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
