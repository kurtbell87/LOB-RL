#pragma once

#include <cstdint>
#include <unordered_map>
#include <memory>
#include "orderbook/LimitOrderBookSnapshot.hpp"

namespace constellation::modules::orderbook {

/**
 * @brief A snapshot capturing the entire state of MarketBook,
 *        i.e. each instrument's LOB snapshot plus global counters.
 */
class MarketBookSnapshot {
public:
  MarketBookSnapshot() = default;
  ~MarketBookSnapshot() = default;

  MarketBookSnapshot(const MarketBookSnapshot&) = default;
  MarketBookSnapshot& operator=(const MarketBookSnapshot&) = default;

private:
  friend class MarketBook;

  std::unordered_map<std::uint32_t, LimitOrderBookSnapshot> lob_snapshots_;

  // global counters
  std::uint64_t global_add_count_{0};
  std::uint64_t global_cancel_count_{0};
  std::uint64_t global_modify_count_{0};
  std::uint64_t global_trade_count_{0};
  std::uint64_t global_clear_count_{0};
};

} // end namespace constellation::modules::orderbook
