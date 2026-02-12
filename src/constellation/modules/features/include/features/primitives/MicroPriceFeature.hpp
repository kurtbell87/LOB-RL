#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::primitives {

/**
 * @brief MicroPriceFeature calculates volume-weighted midpoint from best bid/ask & volumes.
 *        We store the final micro price as double in real currency (bid/ask each / 1e9).
 */
class MicroPriceFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
  };

  MicroPriceFeature();
  explicit MicroPriceFeature(const Config& cfg);
  ~MicroPriceFeature() override = default;

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
  std::atomic<double> micro_price_;  // final micro price in double
};

} // end namespace
