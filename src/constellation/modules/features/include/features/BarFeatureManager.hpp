#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "interfaces/features/IBarFeature.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "databento/record.hpp"

namespace constellation {
namespace modules {
namespace features {

/// Orchestrates bar feature lifecycle and event dispatch.
///
/// Owns a collection of IBarFeature instances. On each MBO event, updates all features.
/// Manages bar boundary notifications (start/complete).
///
/// Features are stored in registration order. GetBarFeatureVector() returns values
/// in that order, which enables deterministic feature column mapping.
class BarFeatureManager {
public:
  BarFeatureManager() = default;
  ~BarFeatureManager() = default;

  /// Register a bar feature with an explicit value name.
  /// Features are stored in registration order.
  void RegisterBarFeature(
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feat,
    const std::string& value_name);

  /// Forward an MBO event to all registered bar features.
  /// Calls OnDataUpdate(source, market) on each feature.
  void OnMboEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market);

  /// Forward an MBO event to all registered bar features.
  /// First calls OnMboMsg(mbo) on each feature, then OnDataUpdate(source, market).
  void OnMboEvent(
    const databento::MboMsg& mbo,
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market);

  /// Notify all features that a new bar has started.
  void NotifyBarStart(std::uint64_t bar_index);

  /// Notify all features that the current bar is complete.
  void NotifyBarComplete(std::uint64_t bar_index);

  /// Extract the feature vector for the completed bar.
  /// Each registered feature contributes one value via GetBarValue(value_name).
  /// Returns values in registration order.
  /// Throws if any feature has not completed its bar (non-empty manager only).
  std::vector<double> GetBarFeatureVector() const;

  /// Number of registered bar features.
  std::size_t FeatureCount() const;

  /// Check if all features report bar complete.
  bool AllBarsComplete() const;

private:
  struct FeatureEntry {
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feature;
    std::string value_name;
  };
  std::vector<FeatureEntry> features_;
};

} // namespace features
} // namespace modules
} // namespace constellation
