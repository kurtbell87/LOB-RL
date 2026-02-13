#include "features/bar/CancelTradeRatioBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void CancelTradeRatioBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;

  if (mbo.action == databento::Action::Cancel)
    ++n_cancels_;
  else if (is_trade(mbo))
    ++n_trades_;
}

void CancelTradeRatioBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // Cancel/trade data handled in OnMboMsg
}

void CancelTradeRatioBarFeature::ResetAccumulators() {
  n_cancels_ = 0;
  n_trades_ = 0;
  finalized_value_ = 0.0;
}

void CancelTradeRatioBarFeature::FinalizeBar() {
  double cancels = static_cast<double>(n_cancels_);
  double trades = static_cast<double>(std::max(n_trades_, std::uint64_t{1}));
  finalized_value_ = std::log(1.0 + cancels / trades);
}

double CancelTradeRatioBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("CancelTradeRatioBarFeature: bar not complete");
  if (name != "cancel_trade_ratio")
    throw std::runtime_error("CancelTradeRatioBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool CancelTradeRatioBarFeature::HasFeature(const std::string& name) const {
  return name == "cancel_trade_ratio";
}

void CancelTradeRatioBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("CancelTradeRatioBarFeature", CancelTradeRatioBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
