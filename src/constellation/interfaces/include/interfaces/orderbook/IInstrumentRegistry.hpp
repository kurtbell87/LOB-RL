#pragma once

#include <cstdint>
#include <vector>
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::orderbook {

/**
 * @brief Provides a registry of which instruments are being tracked,
 *        along with a count of how many there are.
 */
class IInstrumentRegistry {
public:
  virtual ~IInstrumentRegistry() = default;

  /**
   * @brief Return interface version info.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Return the number of instruments tracked.
   */
  virtual std::size_t InstrumentCount() const noexcept = 0;

  /**
   * @brief Return the list of instrument IDs currently managed.
   */
  virtual std::vector<std::uint32_t> GetInstrumentIds() const = 0;
};

} // end namespace constellation::interfaces::orderbook
