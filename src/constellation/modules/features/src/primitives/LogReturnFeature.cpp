#include "features/primitives/LogReturnFeature.hpp"
#include <cmath>
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::primitives {

LogReturnFeature::LogReturnFeature()
  : config_{0},
    log_return_{0.0},
    have_prev_(false),
    prev_bid_raw_(0)  // corrected here
{
}

LogReturnFeature::LogReturnFeature(const Config& config)
  : config_(config),
    log_return_{0.0},
    have_prev_(false),
    prev_bid_raw_(0)  // and here
{
}

void LogReturnFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  }
}

void LogReturnFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  if (!bid_opt) {
    log_return_.store(0.0);
    return;
  }

  std::int64_t bid_raw = bid_opt.value();
  if (bid_raw <= 0) {
    log_return_.store(0.0);
    return;
  }

  if (!have_prev_) {
    have_prev_ = true;
    prev_bid_raw_ = bid_raw;
    log_return_.store(0.0);
    return;
  }

  // Convert raw int64 to double in real currency (1 => 1e-9).
  double curr = static_cast<double>(bid_raw) / 1e9;
  double prev = static_cast<double>(prev_bid_raw_) / 1e9;
  if (prev <= 0.0 || curr <= 0.0) {
    log_return_.store(0.0);
    prev_bid_raw_ = bid_raw;
    return;
  }

  double lr = std::log(curr / prev);
  log_return_.store(lr);

  prev_bid_raw_ = bid_raw;
}

double LogReturnFeature::GetValue(const std::string& name) const {
  if (name == "log_return") {
    return log_return_.load();
  }
  throw FeatureException("LogReturnFeature: unknown name: " + name);
}

bool LogReturnFeature::HasFeature(const std::string& name) const {
  return (name == "log_return");
}

} // end namespace

using LogReturnFeatureAlias =
    ::constellation::modules::features::primitives::LogReturnFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("LogReturnFeature", LogReturnFeatureAlias);
