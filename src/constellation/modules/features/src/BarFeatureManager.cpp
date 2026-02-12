#include "features/BarFeatureManager.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void BarFeatureManager::RegisterBarFeature(
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feat,
    const std::string& value_name) {
  features_.push_back({std::move(feat), value_name});
}

void BarFeatureManager::OnMboEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) {
  for (auto& entry : features_) {
    entry.feature->OnDataUpdate(source, market);
  }
}

void BarFeatureManager::NotifyBarStart(std::uint64_t bar_index) {
  for (auto& entry : features_) {
    entry.feature->OnBarStart(bar_index);
  }
}

void BarFeatureManager::NotifyBarComplete(std::uint64_t bar_index) {
  for (auto& entry : features_) {
    entry.feature->OnBarComplete(bar_index);
  }
}

std::vector<double> BarFeatureManager::GetBarFeatureVector() const {
  if (features_.empty()) {
    return {};
  }
  std::vector<double> result;
  result.reserve(features_.size());
  for (const auto& entry : features_) {
    result.push_back(entry.feature->GetBarValue(entry.value_name));
  }
  return result;
}

std::size_t BarFeatureManager::FeatureCount() const {
  return features_.size();
}

bool BarFeatureManager::AllBarsComplete() const {
  if (features_.empty()) {
    return false;
  }
  for (const auto& entry : features_) {
    if (!entry.feature->IsBarComplete()) {
      return false;
    }
  }
  return true;
}

} // namespace features
} // namespace modules
} // namespace constellation
