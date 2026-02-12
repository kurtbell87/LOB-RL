#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include "databento/record.hpp"
#include "orderbook/PriceBucket.hpp"

namespace constellation::modules::orderbook {

class LimitOrderBook;

/**
 * @brief Snapshot capturing the entire internal state of a LimitOrderBook:
 *        order map, bids, asks, and counters.
 */
class LimitOrderBookSnapshot {
public:
  LimitOrderBookSnapshot() = default;
  ~LimitOrderBookSnapshot() = default;

  // copyable
  LimitOrderBookSnapshot(const LimitOrderBookSnapshot&) = default;
  LimitOrderBookSnapshot& operator=(const LimitOrderBookSnapshot&) = default;

private:
  friend class LimitOrderBook;

  std::uint32_t instrument_id_{0};
  // copy of all orders
  std::unordered_map<std::uint64_t, databento::MboMsg> orders_;

  // Bids, Asks stored as vectors of (price, PriceBucket)
  std::vector<std::pair<std::int64_t, PriceBucket>> bids_;
  std::vector<std::pair<std::int64_t, PriceBucket>> asks_;

  // counters
  std::uint64_t add_count_{0};
  std::uint64_t cancel_count_{0};
  std::uint64_t modify_count_{0};
  std::uint64_t trade_count_{0};
  std::uint64_t clear_count_{0};
};

} // end namespace constellation::modules::orderbook
