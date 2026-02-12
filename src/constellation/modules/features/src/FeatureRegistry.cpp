#include "features/FeatureRegistry.hpp"

namespace constellation {
namespace modules {
namespace features {

FeatureRegistry& FeatureRegistry::Instance() {
  static FeatureRegistry instance;
  return instance;
}

} // end namespace
} // end namespace
} // end namespace
