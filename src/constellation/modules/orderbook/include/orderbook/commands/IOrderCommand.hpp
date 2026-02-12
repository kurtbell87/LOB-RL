#pragma once

#include <memory>
#include "databento/record.hpp"

namespace constellation::modules::orderbook {

class LimitOrderBook;  // forward
class MarketBook;      // forward

/**
 * @brief The Command pattern interface. Each command encapsulates
 *        a single "order action" to be executed on a LimitOrderBook
 *        or MarketBook.
 */
class IOrderCommand {
public:
  virtual ~IOrderCommand() = default;

  /**
   * @brief Execute this command on a single-instrument LOB.
   */
  virtual void Execute(LimitOrderBook& lob) = 0;

  /**
   * @brief Execute this command on the multi-instrument MarketBook.
   *        Defaults to no-op unless the command explicitly addresses
   *        multiple instruments or route logic.
   */
  virtual void Execute(MarketBook& book) {}
};

} // end namespace constellation::modules::orderbook {
