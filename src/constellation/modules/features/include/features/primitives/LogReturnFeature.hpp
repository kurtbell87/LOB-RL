#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"

namespace constellation::modules::features::primitives {

/**
 * @brief LogReturnFeature: ln( current_bid / prev_bid ) in real currency.
 *        We store the most recent log return in an atomic<double>.
 */
class LogReturnFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
  };

  LogReturnFeature();
  explicit LogReturnFeature(const Config& config);
  ~LogReturnFeature() override = default;

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
  std::atomic<double> log_return_;   // ln(cur/prev)
  bool have_prev_;
  std::int64_t prev_bid_raw_;
};

} // end namespace
