#pragma once

#include "interfaces/features/IFeature.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation {
namespace modules {
namespace features {

/**
 * @brief AbstractFeature uses a Template Method pattern: derived classes override
 *        `ComputeUpdate(...)` for the main logic. We store no "price scale = /100" anywhere.
 */
class AbstractFeature : public virtual constellation::interfaces::features::IFeature {
public:
  virtual ~AbstractFeature() = default;

  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {2, 0}; // default version info
  }

  void OnDataUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market
  ) final {
    PreDataUpdate(source, market);
    ComputeUpdate(source, market);
    PostDataUpdate(source, market);
  }

protected:
  virtual void PreDataUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*src*/,
    const constellation::interfaces::orderbook::IMarketView* /*mv*/
  ) {}

  /**
   * @brief The derived class must implement the main feature calculation here.
   *        NOTE: Any int64 price from aggregator is raw "nano" scale, so do
   *              `(double)raw / 1e9` if you want real currency.
   */
  virtual void ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market
  ) = 0;

  virtual void PostDataUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*src*/,
    const constellation::interfaces::orderbook::IMarketView* /*mv*/
  ) {}
};

} // end namespace features
} // end namespace modules
} // end namespace constellation
