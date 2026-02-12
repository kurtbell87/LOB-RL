#pragma once

#include <stdexcept>
#include <string>
#include "interfaces/errors/ParamValidationError.hpp"

/**
 * @file ParameterValidator.hpp
 * @brief Central, minimal param-validation utilities that throw standard exceptions
 *        to avoid cross-module dependencies.
 *
 * Any higher-level modules (like 'features' or 'market_data') should catch
 * these standard exceptions and rethrow their custom exceptions if needed.
 */

namespace constellation::interfaces::validation {

/**
 * @brief Validate a rolling-window size. Throws ParamValidationError if zero.
 *        Leaves it to the caller to wrap or rethrow as a module-specific exception.
 */
inline void ValidateRollingWindowSize(std::size_t window_size, const char* context = "RollingWindow") {
  if (window_size == 0) {
    throw constellation::interfaces::errors::ParamValidationError(std::string(context) + ": window_size cannot be zero");
  }
}

/**
 * @brief Validate DBN file path. Throws ParamValidationError if empty.
 */
inline void ValidateDbnFilePath(const std::string& file_path, const char* context = "DbnFileFeed") {
  if (file_path.empty()) {
    throw constellation::interfaces::errors::ParamValidationError(std::string(context) + ": file_path cannot be empty");
  }
}

} // end namespace constellation::interfaces::validation
