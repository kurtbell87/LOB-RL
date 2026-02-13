#include "features/bar/TradeFlowImbalanceBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void TradeFlowImbalanceBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  double price = trade_price(mbo);
  auto size = static_cast<double>(mbo.size);

  if (has_prev_price_) {
    double diff = price - prev_price_;
    int dir;
    if (diff > 0.0)
      dir = 1;
    else if (diff < 0.0)
      dir = -1;
    else
      dir = prev_dir_;  // forward-fill

    if (dir > 0)
      buy_vol_ += size;
    else if (dir < 0)
      sell_vol_ += size;

    prev_dir_ = dir;
  }

  prev_price_ = price;
  has_prev_price_ = true;
}

void TradeFlowImbalanceBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // Trade data handled in OnMboMsg
}

void TradeFlowImbalanceBarFeature::ResetAccumulators() {
  buy_vol_ = 0.0;
  sell_vol_ = 0.0;
  prev_price_ = 0.0;
  prev_dir_ = 0;
  has_prev_price_ = false;
  finalized_value_ = 0.0;
}

void TradeFlowImbalanceBarFeature::FinalizeBar() {
  double total = buy_vol_ + sell_vol_;
  finalized_value_ = (total == 0.0) ? 0.0 : (buy_vol_ - sell_vol_) / total;
}

double TradeFlowImbalanceBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("TradeFlowImbalanceBarFeature: bar not complete");
  if (name != "trade_flow_imbalance")
    throw std::runtime_error("TradeFlowImbalanceBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool TradeFlowImbalanceBarFeature::HasFeature(const std::string& name) const {
  return name == "trade_flow_imbalance";
}

void TradeFlowImbalanceBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("TradeFlowImbalanceBarFeature", TradeFlowImbalanceBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
