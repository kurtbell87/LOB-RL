#pragma once

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "interfaces/features/IFeature.hpp"
#include "interfaces/features/IFeatureManager.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation::interfaces::features {

/**
 * @brief Extension of IFeatureManager that adds multi-instrument support
 */
class MultiInstrumentFeatureManager : public IFeatureManager {
public:
  virtual ~MultiInstrumentFeatureManager() = default;

  /**
   * @brief Register a new feature for a specific instrument
   * 
   * @param feature The feature to register
   * @param instrument_id The instrument ID this feature applies to
   */
  virtual void RegisterForInstrument(const std::shared_ptr<IFeature>& feature, 
                                     std::uint32_t instrument_id) = 0;

  /**
   * @brief Get all instrument IDs that have registered features
   * 
   * @return Vector of instrument IDs
   */
  virtual std::vector<std::uint32_t> GetInstrumentIds() const = 0;

  /**
   * @brief Check if any features are registered for a specific instrument
   * 
   * @param instrument_id The instrument ID to check
   * @return true if instrument has features, false otherwise
   */
  virtual bool HasFeaturesForInstrument(std::uint32_t instrument_id) const = 0;

  /**
   * @brief Get all feature values for a specific instrument
   * 
   * @param instrument_id The instrument ID to get features for
   * @return Map of feature name to value
   */
  virtual std::unordered_map<std::string, double> GetInstrumentFeatureValues(
      std::uint32_t instrument_id) const = 0;

  /**
   * @brief Get a specific feature value for a specific instrument
   * 
   * @param feature_name The name of the feature
   * @param instrument_id The instrument ID
   * @return The feature value
   * @throws std::runtime_error if no such feature exists
   */
  virtual double GetInstrumentValue(const std::string& feature_name,
                                    std::uint32_t instrument_id) const = 0;
};

} // namespace constellation::interfaces::features