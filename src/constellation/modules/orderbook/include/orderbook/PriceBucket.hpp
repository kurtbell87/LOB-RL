#pragma once

#include <cstdint>
#include <unordered_map>
#include "databento/record.hpp"

namespace constellation::modules::orderbook {

/**
 * @brief Aggregated order data at a single price, storing total qty,
 *        count, plus a per‐order map for partial fills.
 */
struct PriceBucket {
  std::uint64_t agg_qty{0};
  std::uint32_t count{0};
  std::unordered_map<std::uint64_t, databento::MboMsg> orders;
};

} // end namespace constellation::modules::orderbook
