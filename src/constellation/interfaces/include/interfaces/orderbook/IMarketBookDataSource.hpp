#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation {
namespace interfaces {
namespace orderbook {

/**
 * @brief IMarketBookDataSource provides read-only queries across
 *        multiple instruments in an aggregated order book or
 *        multi-LOB aggregator. Each method returns **int64** for prices,
 *        preserving nano precision. The user is free to do (price / 1e9)
 *        if they need a floating representation externally.
 */
class IMarketBookDataSource {
public:
  virtual ~IMarketBookDataSource() = default;

  /**
   * @brief Return interface version info.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Retrieve the current best bid price (nano-based) for the given instrument, if any.
   */
  virtual std::optional<std::int64_t> BestBidPrice(std::uint32_t instrument_id) const = 0;

  /**
   * @brief Retrieve the current best ask price (nano-based) for the given instrument, if any.
   */
  virtual std::optional<std::int64_t> BestAskPrice(std::uint32_t instrument_id) const = 0;

  /**
   * @brief Return aggregated volume at the specified instrument & exact nano-based price.
   *        The caller must pass `priceNanos` as an int64. 
   */
  virtual std::optional<std::uint64_t> VolumeAtPrice(std::uint32_t instrument_id, std::int64_t priceNanos) const = 0;

  /**
   * @brief List all instrument IDs known in this aggregator.
   */
  virtual std::vector<std::uint32_t> GetInstrumentIds() const = 0;
};

} // end namespace orderbook
} // end namespace interfaces
} // end namespace constellation
