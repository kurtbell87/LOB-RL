#include "features/bar/LogVolumeBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void LogVolumeBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;
  if (!is_trade(mbo)) return;

  volume_ += static_cast<std::int64_t>(mbo.size);
}

void LogVolumeBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void LogVolumeBarFeature::ResetAccumulators() {
  volume_ = 0;
  finalized_value_ = 0.0;
}

void LogVolumeBarFeature::FinalizeBar() {
  finalized_value_ = std::log(std::max(volume_, std::int64_t{1}));
}

double LogVolumeBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("LogVolumeBarFeature: bar not complete");
  if (name != "log_volume")
    throw std::runtime_error("LogVolumeBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool LogVolumeBarFeature::HasFeature(const std::string& name) const {
  return name == "log_volume";
}

void LogVolumeBarFeature::SetParam(
    const std::string& /*key*/, const std::string& /*value*/) {
  // No configurable parameters
}

REGISTER_FEATURE("LogVolumeBarFeature", LogVolumeBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
