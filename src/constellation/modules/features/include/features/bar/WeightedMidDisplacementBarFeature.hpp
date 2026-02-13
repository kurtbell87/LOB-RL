#pragma once

#include "features/AbstractBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include <cstdint>
#include <string>

namespace constellation {
namespace modules {
namespace features {

class WeightedMidDisplacementBarFeature final
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
  double tick_size_{0.25};
  double wmid_first_{0.0};
  double wmid_end_{0.0};
  bool has_first_{false};
  bool has_end_{false};
  double finalized_value_{0.0};
};

} // namespace features
} // namespace modules
} // namespace constellation
