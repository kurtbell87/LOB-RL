#pragma once

#include <string>
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::features {

/**
 * @brief IFeature: a composable metric that can be updated from a multi-instrument
 *        aggregator (IMarketBookDataSource) and an IMarketView pointer for advanced counters.
 */
class IFeature {
public:
  virtual ~IFeature() = default;

  /**
   * @brief Return interface version info for this feature implementation.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Recalculate internal state given new data from 'aggregator' plus an
   *        optional 'market' pointer for advanced queries.
   *
   * aggregator => multi-instrument aggregator for best quotes, volumes, etc.
   * market     => aggregates counters, multi-instrument stats
   */
  virtual void OnDataUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& aggregator,
                            const constellation::interfaces::orderbook::IMarketView* market) = 0;

  /**
   * @brief Retrieve a sub-feature value by name (e.g., "spread", "rolling_volatility", etc.).
   */
  virtual double GetValue(const std::string& name) const = 0;

  /**
   * @brief Check if this feature offers a sub-feature with the given name.
   */
  virtual bool HasFeature(const std::string& name) const = 0;
};

} // end namespace constellation::interfaces::features
