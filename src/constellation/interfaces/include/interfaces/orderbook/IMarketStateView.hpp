#pragma once

#include <cstdint>
#include <optional>
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::orderbook {

struct PriceLevel {
  std::int64_t price;
  std::uint64_t total_quantity;
  std::uint32_t order_count;
};

/**
 * @brief Provides read-only access to top-of-book or price-level queries
 *        for each instrument. Focuses on "live state" (best bid/ask).
 */
class IMarketStateView {
public:
  virtual ~IMarketStateView() = default;

  /**
   * @brief Return interface version info.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Return the best bid PriceLevel for a given instrument, if any.
   */
  virtual std::optional<PriceLevel> GetBestBid(std::uint32_t instrument_id) const = 0;

  /**
   * @brief Return the best ask PriceLevel for a given instrument, if any.
   */
  virtual std::optional<PriceLevel> GetBestAsk(std::uint32_t instrument_id) const = 0;
};
} // end namespace constellation::interfaces::orderbook
