#pragma once

#include "interfaces/features/IFeature.hpp"
#include <string>
#include <cstdint>

namespace constellation::interfaces::features {

/// Extension of IFeature for bar-granularity features.
/// Bar features accumulate state across multiple MBO events within a bar,
/// then produce final values when the bar completes.
class IBarFeature : public virtual IFeature {
public:
  virtual ~IBarFeature() = default;

  /// Called when a new bar begins. Implementations should reset intra-bar accumulators.
  virtual void OnBarStart(std::uint64_t bar_index) = 0;

  /// Called when the current bar is complete. Implementations should finalize values.
  virtual void OnBarComplete(std::uint64_t bar_index) = 0;

  /// Returns true if the current bar has been completed (OnBarComplete was called
  /// after the most recent OnBarStart).
  virtual bool IsBarComplete() const = 0;

  /// Get a named bar-level value. Only valid after OnBarComplete().
  /// Throws if called before bar completion or if name is unknown.
  virtual double GetBarValue(const std::string& name) const = 0;
};

} // namespace constellation::interfaces::features
