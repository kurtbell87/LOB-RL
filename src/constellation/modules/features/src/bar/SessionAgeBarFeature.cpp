#include "features/bar/SessionAgeBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void SessionAgeBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  // No accumulation needed — uses bar_index
}

void SessionAgeBarFeature::ResetAccumulators() {
  finalized_value_ = 0.0;
}

void SessionAgeBarFeature::FinalizeBar() {
  finalized_value_ = std::min(
    static_cast<double>(current_bar_index()) / period_, 1.0);
}

double SessionAgeBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("SessionAgeBarFeature: bar not complete");
  if (name != "session_age")
    throw std::runtime_error("SessionAgeBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool SessionAgeBarFeature::HasFeature(const std::string& name) const {
  return name == "session_age";
}

void SessionAgeBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "period") {
    period_ = std::stod(value);
  }
}

REGISTER_FEATURE("SessionAgeBarFeature", SessionAgeBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
