#include "features/bar/SpreadStdBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include <cmath>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void SpreadStdBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  auto bid = source.BestBidPrice(instrument_id_);
  auto ask = source.BestAskPrice(instrument_id_);

  if (!bid.has_value() || !ask.has_value()) return;

  double spread = static_cast<double>(*ask - *bid) / 1e9;
  spread_sum_ += spread;
  spread_sum_sq_ += spread * spread;
  ++spread_count_;
}

void SpreadStdBarFeature::ResetAccumulators() {
  spread_sum_ = 0.0;
  spread_sum_sq_ = 0.0;
  spread_count_ = 0;
  finalized_value_ = 0.0;
}

void SpreadStdBarFeature::FinalizeBar() {
  if (spread_count_ < 2) {
    finalized_value_ = 0.0;
    return;
  }
  double n = static_cast<double>(spread_count_);
  double mean = spread_sum_ / n;
  double mean_sq = spread_sum_sq_ / n;
  double variance = mean_sq - mean * mean;
  // Guard against floating-point underflow producing tiny negative values
  if (variance < 0.0) variance = 0.0;
  finalized_value_ = std::sqrt(variance);
}

double SpreadStdBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("SpreadStdBarFeature: bar not complete");
  if (name != "spread_std")
    throw std::runtime_error("SpreadStdBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool SpreadStdBarFeature::HasFeature(const std::string& name) const {
  return name == "spread_std";
}

void SpreadStdBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  }
}

REGISTER_FEATURE("SpreadStdBarFeature", SpreadStdBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
