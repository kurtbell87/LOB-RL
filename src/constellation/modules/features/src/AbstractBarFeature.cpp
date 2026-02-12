#include "features/AbstractBarFeature.hpp"

namespace constellation {
namespace modules {
namespace features {

void AbstractBarFeature::OnBarStart(std::uint64_t bar_index) {
  current_bar_index_ = bar_index;
  in_bar_ = true;
  bar_complete_ = false;
  ResetAccumulators();
}

void AbstractBarFeature::OnBarComplete(std::uint64_t /*bar_index*/) {
  FinalizeBar();
  in_bar_ = false;
  bar_complete_ = true;
}

bool AbstractBarFeature::IsBarComplete() const {
  return bar_complete_;
}

void AbstractBarFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) {
  if (in_bar_) {
    AccumulateEvent(source, market);
  }
}

double AbstractBarFeature::GetValue(const std::string& name) const {
  return GetBarValue(name);
}

} // namespace features
} // namespace modules
} // namespace constellation
