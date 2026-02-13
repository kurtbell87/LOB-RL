#pragma once

#include "databento/record.hpp"
#include "databento/enums.hpp"
#include "databento/constants.hpp"

namespace constellation {
namespace modules {
namespace features {

/// Check if an MBO message is a trade (Trade or Fill action).
inline bool is_trade(const databento::MboMsg& mbo) {
  return mbo.action == databento::Action::Trade ||
         mbo.action == databento::Action::Fill;
}

/// Extract the trade price as a double from fixed-point MBO message.
inline double trade_price(const databento::MboMsg& mbo) {
  return static_cast<double>(mbo.price) / databento::kFixedPriceScale;
}

} // namespace features
} // namespace modules
} // namespace constellation
