#pragma once

#include <string>
#include "interfaces/errors/ConstellationError.hpp"

namespace constellation::modules::features {

/**
 * @brief FeatureException is thrown when a feature encounters an unrecoverable error
 *        or invalid state (e.g. missing data, config, etc.).
 */
class FeatureException : public constellation::interfaces::errors::ConstellationError {
public:
  explicit FeatureException(const std::string& msg)
    : constellation::interfaces::errors::ConstellationError(msg)
  {}
};

} // end namespace constellation::modules::features
