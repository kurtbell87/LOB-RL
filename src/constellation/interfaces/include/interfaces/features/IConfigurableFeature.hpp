#pragma once

#include <string>
#include "interfaces/features/IFeature.hpp"

namespace constellation {
namespace interfaces {
namespace features {

/**
 * @brief Extends the IFeature concept with a method for receiving arbitrary "key/value" parameters.
 */
class IConfigurableFeature : public virtual IFeature {
public:
  virtual ~IConfigurableFeature() = default;

  /**
   * @brief Called for each (key, value) from user config, except "type".
   *        Typically parse the string as needed.
   */
  virtual void SetParam(const std::string& key, const std::string& value) = 0;
};

} // end namespace features
} // end namespace interfaces
} // end namespace constellation
