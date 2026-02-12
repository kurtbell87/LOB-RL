#include "features/FeatureFactory.hpp"
#include "features/FeatureManager.hpp"
#include "features/MultiInstrumentFeatureManager.hpp"

namespace constellation::modules::features {

using namespace constellation::interfaces::features;

std::shared_ptr<IFeatureManager>
CreateFeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger) {
  auto mgr = std::make_shared<FeatureManager>(logger);
  return std::static_pointer_cast<IFeatureManager>(mgr);
}

std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager>
CreateMultiInstrumentFeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger) {
  auto mgr = std::make_shared<constellation::modules::features::MultiInstrumentFeatureManager>(logger);
  return std::static_pointer_cast<constellation::interfaces::features::MultiInstrumentFeatureManager>(mgr);
}

// Primitives
std::shared_ptr<IFeature> CreateBestBidPriceFeature(const primitives::BestBidPriceFeature::Config& cfg) {
  return std::make_shared<primitives::BestBidPriceFeature>(cfg);
}
std::shared_ptr<IFeature> CreateBestAskPriceFeature(const primitives::BestAskPriceFeature::Config& cfg) {
  return std::make_shared<primitives::BestAskPriceFeature>(cfg);
}
std::shared_ptr<IFeature> CreateSpreadFeature(const primitives::SpreadFeature::Config& cfg) {
  return std::make_shared<primitives::SpreadFeature>(cfg);
}
std::shared_ptr<IFeature> CreateMicroPriceFeature(const primitives::MicroPriceFeature::Config& cfg) {
  return std::make_shared<primitives::MicroPriceFeature>(cfg);
}
std::shared_ptr<IFeature> CreateOrderImbalanceFeature(const primitives::OrderImbalanceFeature::Config& cfg) {
  return std::make_shared<primitives::OrderImbalanceFeature>(cfg);
}
std::shared_ptr<IFeature> CreateLogReturnFeature(const primitives::LogReturnFeature::Config& cfg) {
  return std::make_shared<primitives::LogReturnFeature>(cfg);
}
std::shared_ptr<IFeature> CreateMicroDepthFeature(const primitives::MicroDepthFeature::Config& cfg) {
  return std::make_shared<primitives::MicroDepthFeature>(cfg);
}
std::shared_ptr<IFeature> CreateVolumeAtPriceFeature(const primitives::VolumeAtPriceFeature::Config& cfg) {
  return std::make_shared<primitives::VolumeAtPriceFeature>(cfg);
}

// Derived
std::shared_ptr<IFeature> CreateMidPriceFeature(const derived::MidPriceFeature::Config& cfg) {
  return std::make_shared<derived::MidPriceFeature>(cfg);
}
std::shared_ptr<IFeature> CreateCancelAddRatioFeature() {
  return std::make_shared<derived::CancelAddRatioFeature>();
}
std::shared_ptr<IFeature> CreateRollingVolatilityFeature(const derived::RollingVolatilityFeature::Config& cfg) {
  return std::make_shared<derived::RollingVolatilityFeature>(cfg);
}

} // end namespace
