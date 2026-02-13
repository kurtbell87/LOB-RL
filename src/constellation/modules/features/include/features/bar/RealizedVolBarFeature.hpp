#pragma once

#include "features/AbstractBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include <limits>
#include <string>

namespace constellation {
namespace modules {
namespace features {

class RealizedVolBarFeature final
  : public AbstractBarFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  double GetBarValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void OnMboMsg(const databento::MboMsg& mbo) override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market) override;
  void ResetAccumulators() override;
  void FinalizeBar() override;

private:
  // Intra-bar state (reset each bar)
  double bar_close_{0.0};
  bool has_bar_close_{false};

  // Cross-bar state (persists across bars)
  double prev_close_{0.0};
  bool has_prev_close_{false};
  double rv_sum_{0.0};
  double rv_sum_sq_{0.0};
  int warmup_period_{19};

  double finalized_value_{std::numeric_limits<double>::quiet_NaN()};
};

} // namespace features
} // namespace modules
} // namespace constellation
