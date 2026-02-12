#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::derived {

class MidPriceFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
  };

  MidPriceFeature();
  explicit MidPriceFeature(const Config& config);
  ~MidPriceFeature() override = default;

  double GetValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* market
  ) override;

private:
  Config config_;
  std::atomic<double> mid_price_;  // store final mid price in "real" currency
};

} // end namespace constellation::modules::features::derived
