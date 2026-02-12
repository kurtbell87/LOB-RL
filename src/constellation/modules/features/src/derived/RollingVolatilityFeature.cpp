#include "features/derived/RollingVolatilityFeature.hpp"
#include <cmath>
#include "features/FeatureException.hpp"
#include "features/FeatureRegistry.hpp"
#include "features/FeatureRegistry.hpp"

namespace constellation::modules::features::derived {

RollingVolatilityFeature::RollingVolatilityFeature()
  : config_(),
    volatility_(0.0),
    have_prev_(false),
    prev_bid_raw_(0),
    sum_(0.0),
    sum_sq_(0.0)
{
}

RollingVolatilityFeature::RollingVolatilityFeature(const Config& config)
  : config_(config),
    volatility_(0.0),
    have_prev_(false),
    prev_bid_raw_(0),
    sum_(0.0),
    sum_sq_(0.0)
{
  if (config_.window_size == 0) {
    throw FeatureException("RollingVolatilityFeature: window_size cannot be zero");
  }
}

void RollingVolatilityFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    config_.instrument_id = static_cast<std::uint32_t>(std::stoul(value));
  } else if (key == "window_size") {
    config_.window_size = static_cast<std::size_t>(std::stoul(value));
  }
}

void RollingVolatilityFeature::ComputeUpdate(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*unused*/)
{
  if (config_.window_size == 0) {
    return;
  }
  auto bid_opt = source.BestBidPrice(config_.instrument_id);
  if (!bid_opt) {
    return;
  }
  std::int64_t current_bid_raw = bid_opt.value();
  if (current_bid_raw <= 0) {
    return;
  }
  if (!have_prev_) {
    prev_bid_raw_ = current_bid_raw;
    have_prev_ = true;
    return;
  }
  // convert to double in real currency
  double curr_bid = static_cast<double>(current_bid_raw) / 1e9;
  double prev_bid = static_cast<double>(prev_bid_raw_) / 1e9;
  if (prev_bid <= 0.0 || curr_bid <= 0.0) {
    prev_bid_raw_ = current_bid_raw;
    return;
  }

  double r = std::log(curr_bid / prev_bid);
  returns_.push_back(r);
  sum_    += r;
  sum_sq_ += (r * r);
  if (returns_.size() > config_.window_size) {
    double oldest = returns_.front();
    returns_.pop_front();
    sum_    -= oldest;
    sum_sq_ -= (oldest * oldest);
  }
  std::size_t n = returns_.size();
  double vol_val = 0.0;
  if (n >= 2) {
    double mean = sum_ / static_cast<double>(n);
    double mean_sq = sum_sq_ / static_cast<double>(n);
    double var = mean_sq - (mean * mean);
    if (var < 0.0) var = 0.0;
    vol_val = std::sqrt(var);
  }
  volatility_.store(vol_val);

  prev_bid_raw_ = current_bid_raw;
}

double RollingVolatilityFeature::GetValue(const std::string& name) const {
  if (name == "rolling_volatility") {
    return volatility_.load();
  }
  throw FeatureException("RollingVolatilityFeature: unknown name: " + name);
}

bool RollingVolatilityFeature::HasFeature(const std::string& name) const {
  return (name == "rolling_volatility");
}

} // end namespace


// -- Now define a short alias at global scope (outside the namespace):
using RollingVolatilityFeatureAlias =
    ::constellation::modules::features::derived::RollingVolatilityFeature;

// -- Then call the macro with that alias:
REGISTER_FEATURE("RollingVolatilityFeature", RollingVolatilityFeatureAlias);
