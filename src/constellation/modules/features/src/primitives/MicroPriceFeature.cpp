#include "features/primitives/MicroPriceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

MicroPriceFeature::MicroPriceFeature()
  : config_{0},
    micro_price_{0.0}
{
}

MicroPriceFeature::MicroPriceFeature(const Config& cfg)
  : config_{cfg},
    micro_price_{0.0}
{
}

void MicroPriceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void MicroPriceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  auto ask_opt = source.BestAskPrice(config_.instrument_id);
  if (!bid_opt || !ask_opt) {
    micro_price_.store(0.0);
    return;
  }
  std::int64_t bid_raw = bid_opt.value();
  std::int64_t ask_raw = ask_opt.value();
  if (bid_raw <= 0 || ask_raw <= 0 || ask_raw < bid_raw) {
    micro_price_.store(0.0);
    return;
  }

  // Retrieve volumes at those exact raw prices:
  auto bid_vol_opt = source.VolumeAtPrice(config_.instrument_id, bid_raw);
  auto ask_vol_opt = source.VolumeAtPrice(config_.instrument_id, ask_raw);
  std::uint64_t bid_vol = (bid_vol_opt ? bid_vol_opt.value() : 0ULL);
  std::uint64_t ask_vol = (ask_vol_opt ? ask_vol_opt.value() : 0ULL);

  if ((bid_vol + ask_vol) == 0ULL) {
    micro_price_.store(0.0);
    return;
  }

  double b = static_cast<double>(bid_raw) / 1e9;
  double a = static_cast<double>(ask_raw) / 1e9;
  double weighted = (static_cast<double>(ask_vol) * b + static_cast<double>(bid_vol) * a)
                    / static_cast<double>(ask_vol + bid_vol);
  micro_price_.store(weighted);
}

double MicroPriceFeature::GetValue(const std::string& name) const {
  if (name == "micro_price") {
    return micro_price_.load();
  }
  throw FeatureException("MicroPriceFeature: unknown name " + name);
}

bool MicroPriceFeature::HasFeature(const std::string& name) const {
  return (name == "micro_price");
}

} // end namespace

using MicroPriceFeatureAlias =
    ::constellation::modules::features::primitives::MicroPriceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("MicroPriceFeature", MicroPriceFeatureAlias);
