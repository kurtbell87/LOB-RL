#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

BestAskPriceFeature::BestAskPriceFeature()
  : config_{0},
    best_ask_{0.0}
{
}

BestAskPriceFeature::BestAskPriceFeature(const Config& config)
  : config_{config},
    best_ask_{0.0}
{
}

void BestAskPriceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void BestAskPriceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto ask_opt = source.BestAskPrice(config_.instrument_id);
  if (!ask_opt) {
    best_ask_.store(0.0);
    return;
  }
  std::int64_t ask_raw = ask_opt.value();
  if (ask_raw <= 0) {
    best_ask_.store(0.0);
    return;
  }
  double price = static_cast<double>(ask_raw) / 1e9;
  best_ask_.store(price);
}

double BestAskPriceFeature::GetValue(const std::string& name) const {
  if (name == "best_ask_price") {
    return best_ask_.load();
  }
  throw FeatureException("BestAskPriceFeature: unknown name " + name);
}

bool BestAskPriceFeature::HasFeature(const std::string& name) const {
  return (name == "best_ask_price");
}

} // end namespace

using BestAskPriceFeatureAlias =
    ::constellation::modules::features::primitives::BestAskPriceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("BestAskPriceFeature", BestAskPriceFeatureAlias);
