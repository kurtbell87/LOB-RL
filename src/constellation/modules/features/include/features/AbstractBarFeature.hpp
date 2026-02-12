#pragma once

#include "features/AbstractFeature.hpp"
#include "interfaces/features/IBarFeature.hpp"

namespace constellation {
namespace modules {
namespace features {

/// Base class for bar-granularity features.
///
/// Extends AbstractFeature (for ComputeUpdate routing) and IBarFeature (for bar lifecycle).
/// Derived classes implement AccumulateEvent() instead of ComputeUpdate().
///
/// GetValue() delegates to GetBarValue() — bar features only expose values through
/// the bar interface, ensuring callers can't read stale intra-bar state.
class AbstractBarFeature
  : public virtual AbstractFeature,
    public virtual constellation::interfaces::features::IBarFeature
{
public:
  virtual ~AbstractBarFeature() = default;

  // IBarFeature
  void OnBarStart(std::uint64_t bar_index) override;
  void OnBarComplete(std::uint64_t bar_index) override;
  bool IsBarComplete() const override;

  // IFeature — delegate to GetBarValue
  double GetValue(const std::string& name) const override;

protected:
  /// Called on each MBO event within the current bar.
  /// Implementations accumulate intra-bar state here.
  virtual void AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) = 0;

  /// Called by OnBarStart — derived classes override to reset accumulators.
  virtual void ResetAccumulators() = 0;

  /// Called by OnBarComplete — derived classes override to finalize bar values.
  virtual void FinalizeBar() = 0;

  /// Access current bar index (set by OnBarStart)
  std::uint64_t current_bar_index() const { return current_bar_index_; }

  /// Whether we're currently inside a bar (between OnBarStart and OnBarComplete)
  bool is_in_bar() const { return in_bar_; }

private:
  // AbstractFeature::ComputeUpdate — routes to AccumulateEvent
  void ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) final;

  std::uint64_t current_bar_index_{0};
  bool in_bar_{false};
  bool bar_complete_{false};
};

} // namespace features
} // namespace modules
} // namespace constellation
