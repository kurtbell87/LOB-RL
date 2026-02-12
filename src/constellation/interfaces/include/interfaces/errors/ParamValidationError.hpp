#pragma once

#include "interfaces/errors/ConstellationError.hpp"
#include <string>

namespace constellation::interfaces::errors {

/**
 * @brief ParamValidationError is thrown by parameter validation routines
 *        whenever a parameter is invalid or out of range.
 */
class ParamValidationError : public ConstellationError {
public:
  explicit ParamValidationError(const std::string& msg)
    : ConstellationError(msg)
  {}
};

} // end namespace constellation::interfaces::errors
