#pragma once

#include <memory>
#include "interfaces/features/IFeatureManager.hpp"
#include "interfaces/features/MultiInstrumentFeatureManager.hpp"
#include "interfaces/features/IFeature.hpp"
#include "interfaces/logging/ILogger.hpp"
#include "features/derived/MidPriceFeature.hpp"
#include "features/derived/CancelAddRatioFeature.hpp"
#include "features/derived/RollingVolatilityFeature.hpp"
#include "features/primitives/BestBidPriceFeature.hpp"
#include "features/primitives/BestAskPriceFeature.hpp"
#include "features/primitives/SpreadFeature.hpp"
#include "features/primitives/MicroPriceFeature.hpp"
#include "features/primitives/OrderImbalanceFeature.hpp"
#include "features/primitives/LogReturnFeature.hpp"
#include "features/primitives/MicroDepthFeature.hpp"
#include "features/primitives/VolumeAtPriceFeature.hpp"

namespace constellation::modules::features {

/**
 * @brief Factory functions for building a FeatureManager and each IFeature.
 */
std::shared_ptr<constellation::interfaces::features::IFeatureManager>
CreateFeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

/**
 * @brief Create a multi-instrument feature manager for Phase 4 support
 */
std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager>
CreateMultiInstrumentFeatureManager(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

std::shared_ptr<constellation::interfaces::features::IFeature>
CreateBestBidPriceFeature(const primitives::BestBidPriceFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateBestAskPriceFeature(const primitives::BestAskPriceFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateSpreadFeature(const primitives::SpreadFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateMicroPriceFeature(const primitives::MicroPriceFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateOrderImbalanceFeature(const primitives::OrderImbalanceFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateLogReturnFeature(const primitives::LogReturnFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateMicroDepthFeature(const primitives::MicroDepthFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateVolumeAtPriceFeature(const primitives::VolumeAtPriceFeature::Config& cfg);

// Derived
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateMidPriceFeature(const derived::MidPriceFeature::Config& cfg);
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateCancelAddRatioFeature();
std::shared_ptr<constellation::interfaces::features::IFeature>
CreateRollingVolatilityFeature(const derived::RollingVolatilityFeature::Config& cfg);

} // end namespace constellation::modules::features
