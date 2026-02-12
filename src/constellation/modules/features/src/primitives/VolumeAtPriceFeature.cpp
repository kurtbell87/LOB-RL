#include "features/primitives/VolumeAtPriceFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

VolumeAtPriceFeature::VolumeAtPriceFeature()
  : config_(),
    volume_(0ULL)
{
}

VolumeAtPriceFeature::VolumeAtPriceFeature(const Config& cfg)
  : config_(cfg),
    volume_(0ULL)
{
}

void VolumeAtPriceFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "side") {
    if (value == "Bid" || value == "bid") {
      config_.side = constellation::interfaces::orderbook::BookSide::Bid;
    } else if (value == "Ask" || value == "ask") {
      config_.side = constellation::interfaces::orderbook::BookSide::Ask;
    } else {
      throw FeatureException("VolumeAtPriceFeature::SetParam side must be 'Bid' or 'Ask'");
    }
  } else if (key == "price") {
    // store as int64
    config_.price = static_cast<std::int64_t>(std::stoll(value));
  }
}

void VolumeAtPriceFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/)
{
  // We ignore 'side' for aggregator calls, because aggregator->VolumeAtPrice is side-agnostic or returns total.
  if (config_.instrument_id == 0) {
    volume_.store(0ULL);
    return;
  }
  auto vol_opt = source.VolumeAtPrice(config_.instrument_id, config_.price);
  if (!vol_opt) {
    volume_.store(0ULL);
  } else {
    volume_.store(vol_opt.value());
  }
}

double VolumeAtPriceFeature::GetValue(const std::string& name) const {
  if (name == "volume_at_price") {
    // Return as double but it's just a count
    return static_cast<double>(volume_.load());
  }
  throw FeatureException("VolumeAtPriceFeature: unknown feature " + name);
}

bool VolumeAtPriceFeature::HasFeature(const std::string& name) const {
  return (name == "volume_at_price");
}

} // namespace

using VolumeAtPriceFeatureAlias =
    ::constellation::modules::features::primitives::VolumeAtPriceFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("VolumeAtPriceFeature", VolumeAtPriceFeatureAlias);
