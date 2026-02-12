#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::primitives {

/**
 * @brief OrderImbalanceFeature compares volume at best bid vs best ask:
 *        imbalance = (bidVol - askVol) / (bidVol + askVol).
 */
class OrderImbalanceFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
  };

  OrderImbalanceFeature();
  explicit OrderImbalanceFeature(const Config& cfg);
  ~OrderImbalanceFeature() override = default;

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
  std::atomic<double> imbalance_;
};

} // end namespace
