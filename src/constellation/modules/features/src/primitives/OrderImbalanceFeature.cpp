#include "features/primitives/OrderImbalanceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

OrderImbalanceFeature::OrderImbalanceFeature()
  : config_{0},
    imbalance_{0.0}
{
}

OrderImbalanceFeature::OrderImbalanceFeature(const Config& cfg)
  : config_(cfg),
    imbalance_{0.0}
{
}

void OrderImbalanceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void OrderImbalanceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  auto ask_opt = source.BestAskPrice(config_.instrument_id);
  if (!bid_opt || !ask_opt) {
    imbalance_.store(0.0);
    return;
  }
  std::int64_t bid_raw = bid_opt.value();
  std::int64_t ask_raw = ask_opt.value();
  if (bid_raw <= 0 || ask_raw <= 0) {
    imbalance_.store(0.0);
    return;
  }

  auto bid_vol_opt = source.VolumeAtPrice(config_.instrument_id, bid_raw);
  auto ask_vol_opt = source.VolumeAtPrice(config_.instrument_id, ask_raw);
  std::uint64_t bid_vol = (bid_vol_opt ? bid_vol_opt.value() : 0ULL);
  std::uint64_t ask_vol = (ask_vol_opt ? ask_vol_opt.value() : 0ULL);

  std::uint64_t total = bid_vol + ask_vol;
  if (total == 0ULL) {
    imbalance_.store(0.0);
    return;
  }
  double numerator = static_cast<double>(bid_vol) - static_cast<double>(ask_vol);
  double denom = static_cast<double>(total);
  imbalance_.store(numerator / denom);
}

double OrderImbalanceFeature::GetValue(const std::string& name) const {
  if (name == "order_imbalance") {
    return imbalance_.load();
  }
  throw FeatureException("OrderImbalanceFeature: unknown name: " + name);
}

bool OrderImbalanceFeature::HasFeature(const std::string& name) const {
  return (name == "order_imbalance");
}

} // end namespace

using OrderImbalanceFeatureAlias =
    ::constellation::modules::features::primitives::OrderImbalanceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("OrderImbalanceFeature", OrderImbalanceFeatureAlias);
