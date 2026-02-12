#pragma once

#include <stdexcept>
#include <string>

namespace constellation::interfaces::errors {

/**
 * @brief ConstellationError is the universal base class for all
 *        module-specific exceptions throughout the Constellation codebase.
 *
 * By deriving from std::runtime_error, we preserve compatibility
 * with standard C++ exception handling. Modules that wish to throw
 * domain-specific exceptions should derive from ConstellationError.
 */
class ConstellationError : public std::runtime_error {
public:
  explicit ConstellationError(const std::string& msg)
    : std::runtime_error(msg)
  {}
};

} // end namespace constellation::interfaces::errors
