#include "strategy/StrategyFactory.hpp"
#include "strategy/SampleBatchStrategy.hpp"

namespace constellation {
namespace modules {
namespace strategy {

std::shared_ptr<constellation::interfaces::strategy::IStrategy>
CreateSampleBatchStrategy(const SampleBatchStrategyConfig& /*cfg*/)
{
  // Old name left for compatibility, but we now return the Batch sample strategy
  // with default null orders/loggers. 
  return std::make_shared<SampleBatchStrategy>();
}

} // end namespace strategy
} // end namespace modules
} // end namespace constellation
