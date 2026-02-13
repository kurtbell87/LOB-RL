#pragma once

#include "features/AbstractBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace constellation {
namespace modules {
namespace features {

class DepthImbalanceBarFeature final
  : public AbstractBarFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  double GetBarValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) override;
  void ResetAccumulators() override;
  void FinalizeBar() override;

private:
  std::uint32_t instrument_id_{0};
  std::size_t n_levels_{5};
  const constellation::interfaces::orderbook::IMarketBookDataSource* last_source_{nullptr};
  double finalized_value_{0.0};
};

} // namespace features
} // namespace modules
} // namespace constellation
