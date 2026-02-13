#include "features/bar/VampDisplacementBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void VampDisplacementBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  auto vamp = source.VolumeAdjustedMidPrice(instrument_id_, n_levels_);

  if (!vamp.has_value()) {
    ++event_count_;
    return;
  }

  // Capture mid sample at 0-based event index == bar_size/2
  if (!has_mid_ && bar_size_ > 0 && event_count_ >= bar_size_ / 2) {
    vamp_at_mid_ = *vamp;
    has_mid_ = true;
  }

  ++event_count_;

  // Always update end
  vamp_at_end_ = *vamp;
  has_end_ = true;
}

void VampDisplacementBarFeature::ResetAccumulators() {
  event_count_ = 0;
  vamp_at_mid_ = 0.0;
  vamp_at_end_ = 0.0;
  has_mid_ = false;
  has_end_ = false;
  finalized_value_ = 0.0;
}

void VampDisplacementBarFeature::FinalizeBar() {
  if (!has_mid_ || !has_end_) {
    finalized_value_ = 0.0;
    return;
  }
  finalized_value_ = (vamp_at_end_ - vamp_at_mid_) / tick_size_;
}

double VampDisplacementBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("VampDisplacementBarFeature: bar not complete");
  if (name != "vamp_displacement")
    throw std::runtime_error("VampDisplacementBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool VampDisplacementBarFeature::HasFeature(const std::string& name) const {
  return name == "vamp_displacement";
}

void VampDisplacementBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "n_levels") {
    n_levels_ = static_cast<std::size_t>(std::stoul(value));
  } else if (key == "tick_size") {
    tick_size_ = std::stod(value);
  } else if (key == "bar_size") {
    bar_size_ = static_cast<std::uint64_t>(std::stoull(value));
  }
}

REGISTER_FEATURE("VampDisplacementBarFeature", VampDisplacementBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
