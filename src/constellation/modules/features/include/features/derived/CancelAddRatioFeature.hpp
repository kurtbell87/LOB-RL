#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::derived {

/**
 * @brief ratio = global_cancel_count / global_add_count
 */
class CancelAddRatioFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  CancelAddRatioFeature();
  ~CancelAddRatioFeature() override = default;

  double GetValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;

  void SetParam(const std::string& /*key*/, const std::string& /*value*/) override {/* no-op */}

protected:
  void ComputeUpdate(const  constellation::interfaces::orderbook::IMarketBookDataSource& source,
                     const constellation::interfaces::orderbook::IMarketView* market) override;

private:
  std::atomic<double> ratio_;
};

} // end namespace constellation::modules::features::derived
