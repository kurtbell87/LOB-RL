#include "features/bar/DepthRatioBarFeature.hpp"
#include "features/bar/MboTradeUtils.hpp"
#include "features/FeatureRegistry.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

using BookSide = constellation::interfaces::orderbook::BookSide;

void DepthRatioBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  last_source_ = &source;
}

void DepthRatioBarFeature::ResetAccumulators() {
  last_source_ = nullptr;
  finalized_value_ = 0.0;
}

void DepthRatioBarFeature::FinalizeBar() {
  if (!last_source_) {
    finalized_value_ = 0.5;
    return;
  }

  double total_3 = static_cast<double>(
      last_source_->TotalDepth(instrument_id_, BookSide::Bid, 3) +
      last_source_->TotalDepth(instrument_id_, BookSide::Ask, 3));
  double total_10 = static_cast<double>(
      last_source_->TotalDepth(instrument_id_, BookSide::Bid, 10) +
      last_source_->TotalDepth(instrument_id_, BookSide::Ask, 10));

  if (total_10 == 0.0) {
    finalized_value_ = 0.5;
  } else {
    finalized_value_ = total_3 / (total_10 + kBarFeatureEpsilon);
  }
}

double DepthRatioBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("DepthRatioBarFeature: bar not complete");
  if (name != "depth_ratio")
    throw std::runtime_error("DepthRatioBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool DepthRatioBarFeature::HasFeature(const std::string& name) const {
  return name == "depth_ratio";
}

void DepthRatioBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  }
}

REGISTER_FEATURE("DepthRatioBarFeature", DepthRatioBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
