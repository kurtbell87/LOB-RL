#pragma once

#include "interfaces/orderbook/IMarketStateView.hpp"
#include "interfaces/orderbook/IMarketStatistics.hpp"
#include "interfaces/orderbook/IInstrumentRegistry.hpp"

namespace constellation::interfaces::orderbook {

/**
 * @brief Composite interface that inherits from IMarketStateView,
 *        IMarketStatistics, and IInstrumentRegistry. This replaces
 *        the old monolithic IMarketView while preserving the same
 *        name to avoid breaking existing references or code.
 *
 * All existing references to IMarketView will still work,
 * but implementers should now also implement the three
 * more granular interfaces.
 */
class IMarketView
  : public IMarketStateView,
    public IMarketStatistics,
    public IInstrumentRegistry
{
public:
  ~IMarketView() override = default;
};

} // end namespace constellation::interfaces::orderbook
