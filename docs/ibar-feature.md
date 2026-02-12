# Phase 2: IBarFeature Interface and AbstractBarFeature Base

## Problem

Constellation features are tick-granularity — each `OnDataUpdate()` computes a point-in-time metric. LOB-RL's 22 features are bar-granularity with complex intra-bar accumulation (e.g., summing trade volumes, sampling spreads on every event, tracking VWAP). There's no infrastructure to express bar-level features in Constellation's composable feature system.

## What to Build

### 1. IBarFeature Interface

**File:** `src/constellation/interfaces/include/interfaces/features/IBarFeature.hpp` (NEW)

```cpp
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
```

### 2. AbstractBarFeature Base Class

**File:** `src/constellation/modules/features/include/features/AbstractBarFeature.hpp` (NEW)

```cpp
#pragma once

#include "features/AbstractFeature.hpp"
#include "interfaces/features/IBarFeature.hpp"

namespace constellation::modules::features {

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
  bool HasFeature(const std::string& name) const override;

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

} // namespace constellation::modules::features
```

**Implementation file:** `src/constellation/modules/features/src/AbstractBarFeature.cpp` (NEW)

Behavior:
- `OnBarStart(idx)`: sets `current_bar_index_ = idx`, `in_bar_ = true`, `bar_complete_ = false`, calls `ResetAccumulators()`.
- `OnBarComplete(idx)`: calls `FinalizeBar()`, sets `in_bar_ = false`, `bar_complete_ = true`.
- `IsBarComplete()`: returns `bar_complete_`.
- `ComputeUpdate(source, market)`: if `in_bar_`, delegates to `AccumulateEvent(source, market)`. If not in a bar, no-op (silently ignored — events between bars are expected during warmup).
- `GetValue(name)`: delegates to `GetBarValue(name)`. This means `GetBarValue` must handle the pre-completion case itself (throw or return a sentinel).
- `HasFeature(name)`: returns true for any name the derived class's `GetBarValue` supports. Derived classes override this.

### 3. BarFeatureManager

**File:** `src/constellation/modules/features/include/features/BarFeatureManager.hpp` (NEW)
**File:** `src/constellation/modules/features/src/BarFeatureManager.cpp` (NEW)

```cpp
#pragma once

#include <memory>
#include <vector>
#include <string>
#include "interfaces/features/IBarFeature.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketView.hpp"

namespace constellation::modules::features {

/// Orchestrates bar feature lifecycle and event dispatch.
///
/// Owns a collection of IBarFeature instances. On each MBO event, updates all features.
/// Manages bar boundary notifications (start/complete).
///
/// Features are stored in registration order. GetBarFeatureVector() returns values
/// in that order, which enables deterministic feature column mapping.
class BarFeatureManager {
public:
  BarFeatureManager() = default;
  ~BarFeatureManager() = default;

  /// Register a bar feature. Features are stored in registration order.
  void RegisterBarFeature(std::shared_ptr<constellation::interfaces::features::IBarFeature> feat);

  /// Forward an MBO event to all registered bar features.
  /// Calls OnDataUpdate(source, market) on each feature.
  void OnMboEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market);

  /// Notify all features that a new bar has started.
  void NotifyBarStart(std::uint64_t bar_index);

  /// Notify all features that the current bar is complete.
  void NotifyBarComplete(std::uint64_t bar_index);

  /// Extract the feature vector for the completed bar.
  /// Iterates all registered features and collects their GetBarValue() for each
  /// feature name. Each IBarFeature contributes one or more named values.
  /// Returns values in registration order (feature 0's values first, then feature 1, etc.).
  ///
  /// The caller must provide the feature names in order (established at registration time).
  /// For simplicity, each bar feature contributes exactly one value via GetBarValue(feature_name),
  /// where feature_name is provided at registration.
  std::vector<double> GetBarFeatureVector() const;

  /// Number of registered bar features.
  std::size_t FeatureCount() const;

  /// Check if all features report bar complete.
  bool AllBarsComplete() const;

private:
  struct FeatureEntry {
    std::shared_ptr<constellation::interfaces::features::IBarFeature> feature;
    std::string value_name;  // the named value to extract from this feature
  };
  std::vector<FeatureEntry> features_;
};
```

Registration API:
```cpp
void RegisterBarFeature(shared_ptr<IBarFeature> feat);
```
Each `IBarFeature` is registered with a `value_name` that identifies which named value to extract. We need a way to specify this. Options:
- Overload: `RegisterBarFeature(feat, "trade_flow_imbalance")`
- Or: features self-declare their primary value name

**Decision:** Use overloaded registration with explicit value name:
```cpp
void RegisterBarFeature(std::shared_ptr<IBarFeature> feat, const std::string& value_name);
```

This keeps the manager simple and composable — different registrations of the same feature type can extract different named values.

### 4. Python Bindings

**File:** `src/bindings/bindings.cpp` — add bindings for:

```python
# IBarFeature (abstract, not directly constructible)
# Exposed for isinstance checks and type hints

# BarFeatureManager
mgr = core.BarFeatureManager()
mgr.register_bar_feature(feat, "trade_flow_imbalance")
mgr.on_mbo_event(source, market_view)
mgr.notify_bar_start(0)
mgr.notify_bar_complete(0)
vec = mgr.get_bar_feature_vector()  # returns list[float]
mgr.feature_count()                 # returns int
mgr.all_bars_complete()             # returns bool
```

## Edge Cases

- **OnMboEvent before OnBarStart:** Events are silently ignored (warmup behavior).
- **GetBarValue before OnBarComplete:** Should throw `std::runtime_error` with clear message.
- **GetBarFeatureVector before any bar completes:** Throws.
- **OnBarStart called twice without OnBarComplete:** Resets accumulators (previous partial bar lost). No error.
- **OnBarComplete with wrong bar_index:** Implementation ignores index mismatch — index is informational.
- **Empty BarFeatureManager:** `GetBarFeatureVector()` returns empty vector. `FeatureCount()` returns 0.
- **Registration after events started:** Allowed but the newly registered feature won't have previous bar data.

## Acceptance Criteria

1. `IBarFeature.hpp` exists with OnBarStart, OnBarComplete, IsBarComplete, GetBarValue pure virtuals.
2. `AbstractBarFeature` routes ComputeUpdate to AccumulateEvent, manages bar lifecycle state.
3. `BarFeatureManager` dispatches events and extracts ordered feature vectors.
4. A simple test feature (e.g., `EventCountBarFeature` that counts events in a bar) demonstrates the full lifecycle.
5. Python bindings expose BarFeatureManager with register/notify/extract.
6. All existing tests pass (IBarFeature is new, doesn't affect existing IFeature implementations).
7. ~20 new tests.

## Files to Create

| File | Role |
|------|------|
| `src/constellation/interfaces/include/interfaces/features/IBarFeature.hpp` | Interface |
| `src/constellation/modules/features/include/features/AbstractBarFeature.hpp` | Base class header |
| `src/constellation/modules/features/src/AbstractBarFeature.cpp` | Base class implementation |
| `src/constellation/modules/features/include/features/BarFeatureManager.hpp` | Manager header |
| `src/constellation/modules/features/src/BarFeatureManager.cpp` | Manager implementation |

## Files to Modify

| File | Change |
|------|--------|
| `src/bindings/bindings.cpp` | Add IBarFeature, BarFeatureManager bindings |
| `CMakeLists.txt` | Add new .cpp files to constellation_features target |

## Test Categories

1. **AbstractBarFeature lifecycle (~6):** OnBarStart resets state, AccumulateEvent called during bar, OnBarComplete finalizes, IsBarComplete transitions, GetBarValue throws before completion, events ignored outside bar
2. **BarFeatureManager dispatch (~6):** register + notify + extract vector, multiple features in order, empty manager, AllBarsComplete
3. **Integration (~4):** Full bar lifecycle with test feature, multiple bars in sequence, GetBarFeatureVector returns correct values
4. **Python bindings (~4):** BarFeatureManager creation, register + notify + extract from Python, feature_count, all_bars_complete
