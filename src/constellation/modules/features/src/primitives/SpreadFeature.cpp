#include "features/primitives/SpreadFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

SpreadFeature::SpreadFeature()
  : config_{0},
    spread_{0.0}
{
}

SpreadFeature::SpreadFeature(const Config& config)
  : config_{config},
    spread_{0.0}
{
}

void SpreadFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void SpreadFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  auto ask_opt = source.BestAskPrice(config_.instrument_id);
  if (!bid_opt || !ask_opt) {
    spread_.store(0.0);
    return;
  }
  std::int64_t braw = bid_opt.value();
  std::int64_t araw = ask_opt.value();
  if (braw <= 0 || araw <= 0 || araw < braw) {
    spread_.store(0.0);
    return;
  }
  double bid = static_cast<double>(braw) / 1e9;
  double ask = static_cast<double>(araw) / 1e9;
  double val = ask - bid;
  spread_.store(val);
}

double SpreadFeature::GetValue(const std::string& name) const {
  if (name == "bid_ask_spread") {
    return spread_.load();
  }
  throw FeatureException("SpreadFeature: unknown name " + name);
}

bool SpreadFeature::HasFeature(const std::string& name) const {
  return (name == "bid_ask_spread");
}

} // end namespace

using SpreadFeatureAlias =
    ::constellation::modules::features::primitives::SpreadFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("SpreadFeature", SpreadFeatureAlias);
