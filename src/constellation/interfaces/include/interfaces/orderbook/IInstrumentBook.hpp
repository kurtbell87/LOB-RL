#pragma once

#include <cstdint>
#include <cstddef>
#include <optional>
#include "interfaces/orderbook/IMarketView.hpp" // PriceLevel
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::orderbook {

/**
 * @brief Side of the limit order book: Bid or Ask.
 */
enum class BookSide {
  Bid,
  Ask
};

/**
 * @brief A read-only interface for a single-instrument limit order book view.
 *
 * Provides aggregated counters and partial depth queries.
 */
class IInstrumentBook {
public:
  virtual ~IInstrumentBook() = default;

  /**
   * @brief Returns version info (major, minor) for debugging or introspection.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  // -- MBO counters for this single instrument --
  virtual std::uint64_t GetAddCount() const noexcept = 0;
  virtual std::uint64_t GetCancelCount() const noexcept = 0;
  virtual std::uint64_t GetModifyCount() const noexcept = 0;
  virtual std::uint64_t GetTradeCount() const noexcept = 0;
  virtual std::uint64_t GetClearCount() const noexcept = 0;
  virtual std::uint64_t GetTotalEventCount() const noexcept = 0;

  /**
   * @brief Return aggregated volume at a given side & exact int64 nano price.
   */
  virtual std::uint64_t VolumeAtPrice(BookSide side, std::int64_t priceNanos) const = 0;

  /**
   * @brief Retrieve the 0-based depth level for the specified side.
   * @return A PriceLevel if the level_index is within range, else nullopt.
   */
  virtual std::optional<PriceLevel> GetLevel(BookSide side, std::size_t level_index) const = 0;
};

} // end namespace constellation::interfaces::orderbook
