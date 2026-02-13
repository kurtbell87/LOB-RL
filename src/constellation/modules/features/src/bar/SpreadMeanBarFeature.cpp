#include "features/bar/SpreadMeanBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void SpreadMeanBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  auto bid = source.BestBidPrice(instrument_id_);
  auto ask = source.BestAskPrice(instrument_id_);

  if (!bid.has_value() || !ask.has_value()) return;

  double spread = static_cast<double>(*ask - *bid) / 1e9;
  spread_sum_ += spread;
  ++spread_count_;
}

void SpreadMeanBarFeature::ResetAccumulators() {
  spread_sum_ = 0.0;
  spread_count_ = 0;
  finalized_value_ = 0.0;
}

void SpreadMeanBarFeature::FinalizeBar() {
  if (spread_count_ == 0) {
    finalized_value_ = 1.0;  // default when no samples
  } else {
    finalized_value_ = spread_sum_ / static_cast<double>(spread_count_);
  }
}

double SpreadMeanBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("SpreadMeanBarFeature: bar not complete");
  if (name != "spread_mean")
    throw std::runtime_error("SpreadMeanBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool SpreadMeanBarFeature::HasFeature(const std::string& name) const {
  return name == "spread_mean";
}

void SpreadMeanBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  }
}

REGISTER_FEATURE("SpreadMeanBarFeature", SpreadMeanBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
