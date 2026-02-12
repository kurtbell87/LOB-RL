#pragma once

#include <cstdint>

/**
 * @brief A lightweight struct for interface versioning.
 *
 * Constellation has replaced the old 'InterfaceVersion()' approach
 * with a more flexible major/minor scheme, returned by 
 * 'GetVersionInfo()'.
 */
namespace constellation::interfaces::common {

/**
 * @brief InterfaceVersionInfo holds a major and minor version number
 *        for identifying interface revisions.
 */
struct InterfaceVersionInfo {
  std::uint16_t major;
  std::uint16_t minor;
};

} // end namespace constellation::interfaces::common
