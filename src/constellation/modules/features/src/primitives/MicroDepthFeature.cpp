#include "features/primitives/MicroDepthFeature.hpp"
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

MicroDepthFeature::MicroDepthFeature()
  : config_(),
    price_(0.0),
    size_(0ULL)
{
}

MicroDepthFeature::MicroDepthFeature(const Config& cfg)
  : config_(cfg),
    price_(0.0),
    size_(0ULL)
{
}

void MicroDepthFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "side") {
    if (value == "Bid" || value == "bid") {
      config_.side = constellation::interfaces::orderbook::BookSide::Bid;
    } else if (value == "Ask" || value == "ask") {
      config_.side = constellation::interfaces::orderbook::BookSide::Ask;
    } else {
      throw FeatureException("MicroDepthFeature::SetParam side must be 'Bid' or 'Ask'");
    }
  } else if (key == "depth_index") {
    config_.depth_index = static_cast<std::size_t>(std::stoul(value));
  }
}

void MicroDepthFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/)
{
  auto lvl = source.GetLevel(config_.instrument_id, config_.side, config_.depth_index);
  if (!lvl.has_value()) {
    price_.store(0.0);
    size_.store(0ULL);
    return;
  }
  price_.store(static_cast<double>(lvl->price) / 1e9);
  size_.store(lvl->total_quantity);
}

double MicroDepthFeature::GetValue(const std::string& name) const {
  if (name == "micro_depth_price") {
    return price_.load();
  } else if (name == "micro_depth_size") {
    return static_cast<double>(size_.load());
  }
  throw FeatureException("MicroDepthFeature: unknown name: " + name);
}

bool MicroDepthFeature::HasFeature(const std::string& name) const {
  return (name == "micro_depth_price" || name == "micro_depth_size");
}

} // end namespace

using MicroDepthFeatureAlias =
    ::constellation::modules::features::primitives::MicroDepthFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("MicroDepthFeature", MicroDepthFeatureAlias);
