#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::primitives {

/**
 * @brief BestAskPriceFeature for an instrument. Stored as double in real currency (int64 / 1e9).
 */
class BestAskPriceFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
  };

  BestAskPriceFeature();
  explicit BestAskPriceFeature(const Config& config);
  ~BestAskPriceFeature() override = default;

  double GetValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void ComputeUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& source,
                     const constellation::interfaces::orderbook::IMarketView* market) override;

private:
  Config config_;
  std::atomic<double> best_ask_; // final price in float
};

} // end namespace
