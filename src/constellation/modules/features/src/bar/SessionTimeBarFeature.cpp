#include "features/bar/SessionTimeBarFeature.hpp"
#include "databento/record.hpp"
#include "features/FeatureRegistry.hpp"
#include <algorithm>
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

void SessionTimeBarFeature::OnMboMsg(const databento::MboMsg& mbo) {
  if (!is_in_bar()) return;

  // Track latest timestamp from any event (trade or not)
  auto ts = static_cast<std::uint64_t>(
    mbo.hd.ts_event.time_since_epoch().count());
  if (!has_ts_ || ts > latest_ts_) {
    latest_ts_ = ts;
    has_ts_ = true;
  }
}

void SessionTimeBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& /*source*/,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
}

void SessionTimeBarFeature::ResetAccumulators() {
  latest_ts_ = 0;
  has_ts_ = false;
  finalized_value_ = 0.0;
}

void SessionTimeBarFeature::FinalizeBar() {
  double rth_duration = static_cast<double>(rth_close_ns_ - rth_open_ns_);
  if (rth_duration <= 0.0 || !has_ts_) {
    finalized_value_ = 0.0;
  } else {
    double elapsed = static_cast<double>(
      static_cast<std::int64_t>(latest_ts_) - static_cast<std::int64_t>(rth_open_ns_));
    double t = elapsed / rth_duration;
    finalized_value_ = std::clamp(t, 0.0, 1.0);
  }
}

double SessionTimeBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("SessionTimeBarFeature: bar not complete");
  if (name != "session_time")
    throw std::runtime_error("SessionTimeBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool SessionTimeBarFeature::HasFeature(const std::string& name) const {
  return name == "session_time";
}

void SessionTimeBarFeature::SetParam(const std::string& key, const std::string& value) {
  if (key == "rth_open_ns") {
    rth_open_ns_ = std::stoull(value);
  } else if (key == "rth_close_ns") {
    rth_close_ns_ = std::stoull(value);
  }
}

REGISTER_FEATURE("SessionTimeBarFeature", SessionTimeBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
