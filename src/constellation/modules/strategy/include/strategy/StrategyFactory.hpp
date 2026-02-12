#pragma once

#include <memory>
#include "interfaces/strategy/IStrategy.hpp"

namespace constellation {
namespace modules {
namespace strategy {

/**
 * @brief Optional placeholder config struct if needed later.
 *        Currently empty, provided as an example extension point.
 */
struct SampleBatchStrategyConfig {
  // Future fields or constructor parameters can go here.
  // e.g., an int debug_level{0};
};

/**
 * @brief Factory function that creates a new IStrategy implementation.
 *        Currently returns the "SampleBatchStrategy" behind the interface.
 *
 * @param cfg Optional configuration struct for advanced usage
 * @return A shared pointer to an IStrategy interface
 */
std::shared_ptr<constellation::interfaces::strategy::IStrategy>
CreateSampleBatchStrategy(const SampleBatchStrategyConfig& cfg = {});

} // end namespace strategy
} // end namespace modules
} // end namespace constellation
