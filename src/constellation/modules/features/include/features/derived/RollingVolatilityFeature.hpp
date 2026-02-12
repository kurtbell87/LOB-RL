#pragma once

#include <deque>
#include <atomic>
#include <string>
#include <cstddef>
#include "features/AbstractFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::derived {

/**
 * @brief RollingVolatilityFeature calculates a rolling standard deviation
 *        of log returns using best bid, for the specified instrument.
 *        We convert int64 price to double (by /1e9) for the log returns.
 */
class RollingVolatilityFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
    std::size_t   window_size{1};
  };

  RollingVolatilityFeature();
  explicit RollingVolatilityFeature(const Config& config);
  ~RollingVolatilityFeature() override = default;

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
  mutable std::atomic<double> volatility_;

  bool have_prev_{false};
  std::int64_t prev_bid_raw_{0};  // store previous best bid as raw int64
  std::deque<double> returns_;    // store log returns in double
  double sum_{0.0};
  double sum_sq_{0.0};
};

} // end namespace constellation::modules::features::derived
