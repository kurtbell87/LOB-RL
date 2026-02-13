#pragma once

#include "features/AbstractBarFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include <limits>
#include <string>

namespace constellation {
namespace modules {
namespace features {

class BodyRangeRatioBarFeature final
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
  double open_{0.0};
  double close_{0.0};
  double high_{-std::numeric_limits<double>::infinity()};
  double low_{std::numeric_limits<double>::infinity()};
  bool has_trades_{false};
  double finalized_value_{0.0};
};

} // namespace features
} // namespace modules
} // namespace constellation
