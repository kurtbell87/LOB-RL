#pragma once

#include <cstdint>
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::orderbook {
/**
 * @brief Provides aggregated counters across all instruments (Add, Cancel, etc.).
 */
class IMarketStatistics {
public:
  virtual ~IMarketStatistics() = default;

  /**
   * @brief Return interface version info.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  // -- Global action counters across *all* instruments --
  virtual std::uint64_t GetGlobalAddCount() const noexcept = 0;
  virtual std::uint64_t GetGlobalCancelCount() const noexcept = 0;
  virtual std::uint64_t GetGlobalModifyCount() const noexcept = 0;
  virtual std::uint64_t GetGlobalTradeCount() const noexcept = 0;
  virtual std::uint64_t GetGlobalClearCount() const noexcept = 0;
  virtual std::uint64_t GetGlobalTotalEventCount() const noexcept = 0;
};
} // end namespace constellation::interfaces::orderbook
