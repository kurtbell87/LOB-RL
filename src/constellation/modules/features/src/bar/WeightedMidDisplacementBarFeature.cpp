#include "features/bar/WeightedMidDisplacementBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void WeightedMidDisplacementBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  auto wmid = source.WeightedMidPrice(instrument_id_);
  if (!wmid.has_value()) return;

  if (!has_first_) {
    wmid_first_ = *wmid;
    has_first_ = true;
  }
  wmid_end_ = *wmid;
  has_end_ = true;
}

void WeightedMidDisplacementBarFeature::ResetAccumulators() {
  wmid_first_ = 0.0;
  wmid_end_ = 0.0;
  has_first_ = false;
  has_end_ = false;
  finalized_value_ = 0.0;
}

void WeightedMidDisplacementBarFeature::FinalizeBar() {
  if (!has_first_ || !has_end_) {
    finalized_value_ = 0.0;
    return;
  }
  finalized_value_ = (wmid_end_ - wmid_first_) / tick_size_;
}

double WeightedMidDisplacementBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("WeightedMidDisplacementBarFeature: bar not complete");
  if (name != "wmid_displacement")
    throw std::runtime_error("WeightedMidDisplacementBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool WeightedMidDisplacementBarFeature::HasFeature(const std::string& name) const {
  return name == "wmid_displacement";
}

void WeightedMidDisplacementBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "tick_size") {
    tick_size_ = std::stod(value);
  }
}

REGISTER_FEATURE("WeightedMidDisplacementBarFeature", WeightedMidDisplacementBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
