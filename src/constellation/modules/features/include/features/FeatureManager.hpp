#pragma once

#include <shared_mutex>
#include <memory>
#include <vector>
#include <string>
#include "interfaces/features/IFeature.hpp"
#include "interfaces/features/IFeatureManager.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation::modules::features {

/**
 * @class FeatureManager
 * @brief Concrete class implementing IFeatureManager, managing multiple IFeature objects,
 *        updating them on new data, and retrieving named metric values.
 */
class FeatureManager final : public constellation::interfaces::features::IFeatureManager {
public:
  explicit FeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);
  ~FeatureManager() override;

  void Register(const std::shared_ptr<constellation::interfaces::features::IFeature>& feature) override;
  void OnDataUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& aggregator,
                    const constellation::interfaces::orderbook::IMarketView* market_view) override;
  double GetValue(const std::string& feature_name) const override;

private:
  mutable std::shared_mutex mutex_;
  std::vector<std::shared_ptr<constellation::interfaces::features::IFeature>> features_;
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;
};

} // end namespace constellation::modules::features
