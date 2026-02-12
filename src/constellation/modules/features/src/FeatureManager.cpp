#include "features/FeatureManager.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include <stdexcept>

namespace constellation::modules::features {

FeatureManager::FeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
  : logger_{ logger ? logger : std::make_shared<constellation::interfaces::logging::NullLogger>() }
{
}

FeatureManager::~FeatureManager() = default;

void FeatureManager::Register(const std::shared_ptr<constellation::interfaces::features::IFeature>& feature) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  features_.push_back(feature);
  logger_->Debug("FeatureManager::Register - feature registered");
}

void FeatureManager::OnDataUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& aggregator,
                                  const constellation::interfaces::orderbook::IMarketView* market_view)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  for (auto& feat : features_) {
    feat->OnDataUpdate(aggregator, market_view);
  }
}

double FeatureManager::GetValue(const std::string& feature_name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  for (auto& feat : features_) {
    if (feat->HasFeature(feature_name)) {
      return feat->GetValue(feature_name);
    }
  }
  throw std::runtime_error("FeatureManager::GetValue: No feature provides '" + feature_name + "'");
}

} // end namespace
