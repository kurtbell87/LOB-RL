#include "features/bar/DepthImbalanceBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

using BookSide = constellation::interfaces::orderbook::BookSide;

void DepthImbalanceBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  last_source_ = &source;
}

void DepthImbalanceBarFeature::ResetAccumulators() {
  last_source_ = nullptr;
  finalized_value_ = 0.0;
}

void DepthImbalanceBarFeature::FinalizeBar() {
  if (!last_source_) {
    finalized_value_ = 0.5;
    return;
  }

  double bid_depth = static_cast<double>(
      last_source_->TotalDepth(instrument_id_, BookSide::Bid, n_levels_));
  double ask_depth = static_cast<double>(
      last_source_->TotalDepth(instrument_id_, BookSide::Ask, n_levels_));

  double total = bid_depth + ask_depth;
  finalized_value_ = (total == 0.0) ? 0.5 : bid_depth / total;
}

double DepthImbalanceBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("DepthImbalanceBarFeature: bar not complete");
  if (name != "depth_imbalance")
    throw std::runtime_error("DepthImbalanceBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool DepthImbalanceBarFeature::HasFeature(const std::string& name) const {
  return name == "depth_imbalance";
}

void DepthImbalanceBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "n_levels") {
    n_levels_ = static_cast<std::size_t>(std::stoul(value));
  }
}

REGISTER_FEATURE("DepthImbalanceBarFeature", DepthImbalanceBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
