#include "features/bar/BboImbalanceBarFeature.hpp"
#include "features/FeatureRegistry.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include <stdexcept>

namespace constellation {
namespace modules {
namespace features {

using BookSide = constellation::interfaces::orderbook::BookSide;

void BboImbalanceBarFeature::AccumulateEvent(
    const constellation::interfaces::orderbook::IMarketBookDataSource& source,
    const constellation::interfaces::orderbook::IMarketView* /*market*/) {
  last_source_ = &source;
}

void BboImbalanceBarFeature::ResetAccumulators() {
  last_source_ = nullptr;
  finalized_value_ = 0.0;
}

void BboImbalanceBarFeature::FinalizeBar() {
  if (!last_source_) {
    finalized_value_ = 0.5;
    return;
  }

  auto bid_level = last_source_->GetLevel(instrument_id_, BookSide::Bid, 0);
  auto ask_level = last_source_->GetLevel(instrument_id_, BookSide::Ask, 0);

  double bid_qty = bid_level.has_value() ? static_cast<double>(bid_level->total_quantity) : 0.0;
  double ask_qty = ask_level.has_value() ? static_cast<double>(ask_level->total_quantity) : 0.0;

  double total = bid_qty + ask_qty;
  finalized_value_ = (total == 0.0) ? 0.5 : bid_qty / total;
}

double BboImbalanceBarFeature::GetBarValue(const std::string& name) const {
  if (!IsBarComplete())
    throw std::runtime_error("BboImbalanceBarFeature: bar not complete");
  if (name != "bbo_imbalance")
    throw std::runtime_error("BboImbalanceBarFeature: unknown value '" + name + "'");
  return finalized_value_;
}

bool BboImbalanceBarFeature::HasFeature(const std::string& name) const {
  return name == "bbo_imbalance";
}

void BboImbalanceBarFeature::SetParam(
    const std::string& key, const std::string& value) {
  if (key == "instrument_id") {
    instrument_id_ = static_cast<std::uint32_t>(std::stoul(value));
  }
}

REGISTER_FEATURE("BboImbalanceBarFeature", BboImbalanceBarFeature)

} // namespace features
} // namespace modules
} // namespace constellation
