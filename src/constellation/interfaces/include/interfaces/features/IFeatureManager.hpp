#pragma once

#include <string>
#include <memory>
#include "interfaces/features/IFeature.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation::interfaces::features {

/**
 * @brief IFeatureManager is an interface for managing multiple IFeature
 *        objects, updating them on new market data, and retrieving their values by name.
 */
class IFeatureManager {
public:
  virtual ~IFeatureManager() = default;

  /**
   * @brief Register a new IFeature instance into this manager.
   */
  virtual void Register(const std::shared_ptr<IFeature>& feature) = 0;

  /**
   * @brief Provide a global or local data update (IMarketBookDataSource + IMarketView).
   *        The manager will forward to all registered features.
   */
  virtual void OnDataUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& aggregator,
                            const constellation::interfaces::orderbook::IMarketView* market_view) = 0;

  /**
   * @brief Obtain the numeric value of a named sub-feature.
   * @throws std::runtime_error if no feature provides this name.
   */
  virtual double GetValue(const std::string& feature_name) const = 0;
};

} // end namespace constellation::interfaces::features
