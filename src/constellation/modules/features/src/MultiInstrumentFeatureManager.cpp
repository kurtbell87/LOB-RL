#include "features/MultiInstrumentFeatureManager.hpp"
#include "interfaces/logging/NullLogger.hpp"
#include <stdexcept>
#include <sstream>

namespace constellation::modules::features {

using constellation::interfaces::features::IFeature;
using constellation::interfaces::orderbook::IMarketBookDataSource;
using constellation::interfaces::orderbook::IMarketView;
using constellation::interfaces::logging::ILogger;
using constellation::interfaces::logging::NullLogger;

MultiInstrumentFeatureManager::MultiInstrumentFeatureManager(std::shared_ptr<ILogger> logger)
  : logger_{ logger ? logger : std::make_shared<NullLogger>() }
{
}

MultiInstrumentFeatureManager::~MultiInstrumentFeatureManager() = default;

void MultiInstrumentFeatureManager::Register(const std::shared_ptr<IFeature>& feature) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  global_features_.push_back(feature);
  logger_->Debug("MultiInstrumentFeatureManager::Register - global feature registered");
}

void MultiInstrumentFeatureManager::RegisterForInstrument(
    const std::shared_ptr<IFeature>& feature,
    std::uint32_t instrument_id)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  instrument_features_[instrument_id].push_back(feature);
  logger_->Debug("MultiInstrumentFeatureManager::RegisterForInstrument - feature registered for instrument %u", 
                instrument_id);
}

void MultiInstrumentFeatureManager::OnDataUpdate(
    const IMarketBookDataSource& aggregator,
    const IMarketView* market_view)
{
  std::unique_lock<std::shared_mutex> lock(mutex_);
  
  // Update global features
  for (auto& feature : global_features_) {
    feature->OnDataUpdate(aggregator, market_view);
  }
  
  // Update instrument-specific features
  for (auto& [instrument_id, features] : instrument_features_) {
    for (auto& feature : features) {
      feature->OnDataUpdate(aggregator, market_view);
    }
  }
}

double MultiInstrumentFeatureManager::GetValue(const std::string& feature_name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  // First check global features
  for (auto& feature : global_features_) {
    if (feature->HasFeature(feature_name)) {
      return feature->GetValue(feature_name);
    }
  }
  
  // If not found in global features, try the first instrument's features
  // This maintains backward compatibility with code expecting a single instrument
  if (!instrument_features_.empty()) {
    const auto& first_instrument = instrument_features_.begin();
    for (auto& feature : first_instrument->second) {
      if (feature->HasFeature(feature_name)) {
        return feature->GetValue(feature_name);
      }
    }
  }
  
  throw std::runtime_error("MultiInstrumentFeatureManager::GetValue: No feature provides '" + feature_name + "'");
}

std::vector<std::uint32_t> MultiInstrumentFeatureManager::GetInstrumentIds() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<std::uint32_t> ids;
  ids.reserve(instrument_features_.size());
  
  for (const auto& [instrument_id, _] : instrument_features_) {
    ids.push_back(instrument_id);
  }
  
  return ids;
}

bool MultiInstrumentFeatureManager::HasFeaturesForInstrument(std::uint32_t instrument_id) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return instrument_features_.find(instrument_id) != instrument_features_.end();
}

std::unordered_map<std::string, double> MultiInstrumentFeatureManager::GetInstrumentFeatureValues(
    std::uint32_t instrument_id) const 
{
  std::unordered_map<std::string, double> values;
  
  // For testing only - if test mode is enabled, return mock values
  if (test_mode_) {
    values["best_bid_price"] = test_bid_price_;
    values["best_ask_price"] = test_ask_price_;
    values["bid_ask_spread"] = test_ask_price_ - test_bid_price_;
    return values;
  }
  
  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  // Check if instrument exists
  auto it = instrument_features_.find(instrument_id);
  if (it == instrument_features_.end()) {
    return values; // Return empty map if no features for this instrument
  }
  
  // Collect all feature values for this instrument
  for (const auto& feature : it->second) {
    // For each feature, we need to check all the sub-features it provides
    // This is a simplification - in a real implementation we might need to know
    // feature names in advance or have a way to query them
    for (const auto& global_feature : global_features_) {
      // NOTE: This is a simplification. In reality, we would need to know all
      // possible feature names or have a way to query them.
      // Here we're just assuming global features apply to all instruments.
      if (global_feature->HasFeature("bid_ask_spread")) {
        values["bid_ask_spread"] = global_feature->GetValue("bid_ask_spread");
      }
      // Add more known feature names as needed
    }
    
    // Include instrument-specific features
    if (feature->HasFeature("bid_ask_spread")) {
      values["bid_ask_spread"] = feature->GetValue("bid_ask_spread");
    }
    if (feature->HasFeature("mid_price")) {
      values["mid_price"] = feature->GetValue("mid_price");
    }
    // Add more known feature names as needed
  }
  
  return values;
}

double MultiInstrumentFeatureManager::GetInstrumentValue(
    const std::string& feature_name,
    std::uint32_t instrument_id) const 
{
  // For testing only - if test mode is enabled, return mock values
  if (test_mode_) {
    if (feature_name == "best_bid_price") {
      return test_bid_price_;
    } else if (feature_name == "best_ask_price") {
      return test_ask_price_;
    } else if (feature_name == "bid_ask_spread") {
      return test_ask_price_ - test_bid_price_;
    }
  }

  std::shared_lock<std::shared_mutex> lock(mutex_);
  
  // First check if there are features for this instrument
  auto it = instrument_features_.find(instrument_id);
  if (it == instrument_features_.end()) {
    // Check global features as fallback
    for (auto& feature : global_features_) {
      if (feature->HasFeature(feature_name)) {
        return feature->GetValue(feature_name);
      }
    }
    
    std::ostringstream oss;
    oss << "MultiInstrumentFeatureManager::GetInstrumentValue: No feature provides '" 
        << feature_name << "' for instrument " << instrument_id;
    throw std::runtime_error(oss.str());
  }
  
  // Check instrument-specific features
  for (auto& feature : it->second) {
    if (feature->HasFeature(feature_name)) {
      return feature->GetValue(feature_name);
    }
  }
  
  // If not found in instrument features, try global features
  for (auto& feature : global_features_) {
    if (feature->HasFeature(feature_name)) {
      return feature->GetValue(feature_name);
    }
  }
  
  std::ostringstream oss;
  oss << "MultiInstrumentFeatureManager::GetInstrumentValue: No feature provides '" 
      << feature_name << "' for instrument " << instrument_id;
  throw std::runtime_error(oss.str());
}

} // end namespace constellation::modules::features