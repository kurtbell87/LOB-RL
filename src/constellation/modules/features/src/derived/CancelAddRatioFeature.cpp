#include "features/derived/CancelAddRatioFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"
#include "features/FeatureRegistry.hpp" 

namespace constellation::modules::features::derived {

CancelAddRatioFeature::CancelAddRatioFeature()
  : ratio_{0.0}
{
}

void CancelAddRatioFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* market)
{
  if (!market) {
    ratio_.store(0.0);
    return;
  }
  double adds = static_cast<double>(market->GetGlobalAddCount());
  double cancels = static_cast<double>(market->GetGlobalCancelCount());
  if (adds > 1e-12) {
    ratio_.store(cancels / adds);
  } else {
    ratio_.store(0.0);
  }
}

double CancelAddRatioFeature::GetValue(const std::string& name) const {
  if (name == "cancel_add_ratio") {
    return ratio_.load();
  }
  throw FeatureException("CancelAddRatioFeature: unknown name " + name);
}

bool CancelAddRatioFeature::HasFeature(const std::string& name) const {
  return (name == "cancel_add_ratio");
}

} // end namespace

// -- Now define a short alias at global scope (outside the namespace):
using CancelAddRatioFeatureAlias =
    constellation::modules::features::derived::CancelAddRatioFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("CancelAddRatioFeature", CancelAddRatioFeatureAlias);
