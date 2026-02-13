#include "features/bar/RealizedVolBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void RealizedVolBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  bar_close_ = trade_price(mbo);
  has_bar_close_ = true;
}

void RealizedVolBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void RealizedVolBarFeature::ResetAccumulators() {
  // Only reset intra-bar state. Cross-bar state persists.
  bar_close_ = 0.0;
  has_bar_close_ = false;
  finalized_value_ = std::numeric_limits<double>::quiet_NaN();
}

void RealizedVolBarFeature::FinalizeBar() {
  auto bar_idx = current_bar_index();

  // Update cross-bar state with log return
  if (has_bar_close_ && has_prev_close_) {
    double lr = std::log(bar_close_ / prev_close_);
    rv_sum_ += lr;
    rv_sum_sq_ += lr * lr;
  }

  // Compute realized vol
  if (static_cast<int>(bar_idx) < warmup_period_) {
    finalized_value_ = std::numeric_limits<double>::quiet_NaN();
  } else {
    // bar_idx log returns available (from bars 0..bar_idx, giving bar_idx returns)
    double count = static_cast<double>(bar_idx);
    if (count > 0.0) {
      double mean = rv_sum_ / count;
      double var = rv_sum_sq_ / count - mean * mean;
      finalized_value_ = std::sqrt(std::max(var, 0.0));
    } else {
      finalized_value_ = 0.0;
    }
  }

  // Update prev_close for next bar
  if (has_bar_close_) {
    prev_close_ = bar_close_;
    has_prev_close_ = true;
  }
}

double RealizedVolBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("RealizedVolBarFeature: bar not complete");
  if (name != "realized_vol")
    throw std::runtime_error("RealizedVolBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool RealizedVolBarFeature::HasFeature(const std::string& name) const {
  return name == "realized_vol";
}

void RealizedVolBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "warmup_period") {
    warmup_period_ = std::stoi(value);
  }
}

REGISTER_FEATURE("RealizedVolBarFeature", RealizedVolBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
