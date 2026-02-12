#include "features/derived/MidPriceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::derived {

MidPriceFeature::MidPriceFeature()
  : config_{0},
    mid_price_{0.0}
{
}

MidPriceFeature::MidPriceFeature(const Config& config)
  : config_(config),
    mid_price_{0.0}
{
}

void MidPriceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void MidPriceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  auto ask_opt = source.BestAskPrice(config_.instrument_id);
  if (!bid_opt || !ask_opt) {
    mid_price_.store(0.0);
    return;
  }
  std::int64_t bid_raw = bid_opt.value();
  std::int64_t ask_raw = ask_opt.value();
  if (bid_raw <= 0 || ask_raw <= 0 || ask_raw < bid_raw) {
    mid_price_.store(0.0);
    return;
  }
  double b = static_cast<double>(bid_raw) / 1e9;
  double a = static_cast<double>(ask_raw) / 1e9;
  double m = 0.5 * (b + a);
  mid_price_.store(m);
}

double MidPriceFeature::GetValue(const std::string& name) const {
  if (name == "mid_price") {
    return mid_price_.load();
  }
  throw FeatureException("MidPriceFeature: unknown feature " + name);
}

bool MidPriceFeature::HasFeature(const std::string& name) const {
  return (name == "mid_price");
}

} // end namespace

// -- Now define a short alias at global scope (outside the namespace):
using MidPriceFeatureAlias =
    ::constellation::modules::features::derived::MidPriceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("MidPriceFeature", MidPriceFeatureAlias);
