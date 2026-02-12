#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include "interfaces/features/MultiInstrumentFeatureManager.hpp"
#include "interfaces/features/IFeature.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation::modules::features {

/**
 * @class MultiInstrumentFeatureManager
 * @brief Implementation of MultiInstrumentFeatureManager interface that manages features
 *        organized by instrument ID. Supports Phase 4 multi-instrument capabilities.
 */
class MultiInstrumentFeatureManager final : public constellation::interfaces::features::MultiInstrumentFeatureManager {
public:
  explicit MultiInstrumentFeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);
  ~MultiInstrumentFeatureManager() override;

  // --- IFeatureManager interface ---
  void Register(const std::shared_ptr<constellation::interfaces::features::IFeature>& feature) override;
  void OnDataUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& aggregator,
                    const constellation::interfaces::orderbook::IMarketView* market_view) override;
  double GetValue(const std::string& feature_name) const override;

  // --- MultiInstrumentFeatureManager extensions ---
  void RegisterForInstrument(const std::shared_ptr<constellation::interfaces::features::IFeature>& feature,
                            std::uint32_t instrument_id) override;
  std::vector<std::uint32_t> GetInstrumentIds() const override;
  bool HasFeaturesForInstrument(std::uint32_t instrument_id) const override;
  std::unordered_map<std::string, double> GetInstrumentFeatureValues(std::uint32_t instrument_id) const override;
  double GetInstrumentValue(const std::string& feature_name, std::uint32_t instrument_id) const override;

private:
  mutable std::shared_mutex mutex_;
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  // Global features (not tied to a specific instrument)
  std::vector<std::shared_ptr<constellation::interfaces::features::IFeature>> global_features_;
  
  // Features organized by instrument ID
  std::unordered_map<std::uint32_t, 
                    std::vector<std::shared_ptr<constellation::interfaces::features::IFeature>>> 
                    instrument_features_;
                    
  // Test mode for returning mock values
  bool test_mode_{false};
  double test_bid_price_{100.0};
  double test_ask_price_{101.0};
  
public:
  // For testing only
  void EnableTestMode(bool enable) { test_mode_ = enable; }
  void SetTestValues(double bid, double ask) { 
    test_bid_price_ = bid; 
    test_ask_price_ = ask; 
  }
};

} // namespace constellation::modules::features