#pragma once

#include <vector>
#include "databento/record.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation {
namespace interfaces {
namespace orderbook {

/**
 * @brief IMarketBook provides read + write access to a multi-instrument
 *        order book aggregator. It combines IMarketBookDataSource (read),
 *        IMarketView (read), and adds methods to update it with new MBO messages.
 *
 * This interface is introduced to allow the Orchestrator to call OnMboUpdate
 * and BatchOnMboUpdate without referencing concrete classes like MarketBook.
 */
class IMarketBook
  : public IMarketBookDataSource,
    public IMarketView
{
public:
  virtual ~IMarketBook() = default;

  /**
   * @brief Insert or update a single MBO message.
   */
  virtual void OnMboUpdate(const databento::MboMsg& msg) = 0;

  /**
   * @brief Insert or update a batch of MBO messages in one pass.
   */
  virtual void BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages) = 0;

  /**
   * @brief Return interface version info.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;
};

} // end namespace orderbook
} // end namespace interfaces
} // end namespace constellation
