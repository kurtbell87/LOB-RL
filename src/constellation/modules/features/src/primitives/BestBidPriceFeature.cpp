#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

BestBidPriceFeature::BestBidPriceFeature()
  : config_{0},
    best_bid_{0.0}
{
}

BestBidPriceFeature::BestBidPriceFeature(const Config& config)
  : config_{config},
    best_bid_{0.0}
{
}

void BestBidPriceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void BestBidPriceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  if (!bid_opt) {
    best_bid_.store(0.0);
    return;
  }
  std::int64_t bid_raw = bid_opt.value();
  if (bid_raw <= 0) {
    best_bid_.store(0.0);
    return;
  }
  double price = static_cast<double>(bid_raw) / 1e9;
  best_bid_.store(price);
}

double BestBidPriceFeature::GetValue(const std::string& name) const {
  if (name == "best_bid_price") {
    return best_bid_.load();
  }
  throw FeatureException("BestBidPriceFeature: unknown name " + name);
}

bool BestBidPriceFeature::HasFeature(const std::string& name) const {
  return (name == "best_bid_price");
}

} // end namespace

using BestBidPriceFeatureAlias =
    ::constellation::modules::features::primitives::BestBidPriceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("BestBidPriceFeature", BestBidPriceFeatureAlias);
