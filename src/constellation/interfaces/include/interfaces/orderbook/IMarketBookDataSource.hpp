#pragma once

#include <cstdint>
#include <optional>
#include <vector>
#include "interfaces/common/InterfaceVersionInfo.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"  // BookSide, PriceLevel

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

  /**
   * @brief Return the price level at depth_index (0 = best) for the given instrument and side.
   * @return PriceLevel if the level exists, nullopt if unknown instrument or insufficient depth.
   */
  virtual std::optional<PriceLevel> GetLevel(std::uint32_t instrument_id,
                                              BookSide side,
                                              std::size_t depth_index) const = 0;

  /**
   * @brief Sum total quantity across the top n_levels on the given side.
   * @return 0 if the instrument is unknown, the book is empty, or n_levels is 0.
   */
  virtual std::uint64_t TotalDepth(std::uint32_t instrument_id,
                                    BookSide side,
                                    std::size_t n_levels) const = 0;

  /**
   * @brief Volume-weighted mid-price using best bid/ask quantities.
   *        wmid = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
   * @return Price as double in real currency (not nanos), or nullopt if either side is empty.
   */
  virtual std::optional<double> WeightedMidPrice(std::uint32_t instrument_id) const = 0;

  /**
   * @brief Volume-Adjusted Mid-Price (VAMP) using top n_levels on each side.
   *        VAMP = sum(price_i * qty_i) / sum(qty_i) across both sides.
   * @return Price as double in real currency, or nullopt if either side is empty or n_levels is 0.
   */
  virtual std::optional<double> VolumeAdjustedMidPrice(std::uint32_t instrument_id,
                                                        std::size_t n_levels) const = 0;
};

} // end namespace orderbook
} // end namespace interfaces
} // end namespace constellation
