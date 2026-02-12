#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "interfaces/orderbook/IMarketBookDataSource.hpp"

/**
 * @brief Shared mock data source for feature tests.
 *
 * Stores bid/ask as doubles internally and converts to int64 nanos
 * for the IMarketBookDataSource interface. Supports optional
 * validity flags and a volume lookup map.
 */
class MockMarketDataSource final
  : public constellation::interfaces::orderbook::IMarketBookDataSource
{
public:
  std::atomic<double> best_bid{0.0};
  std::atomic<double> best_ask{0.0};
  bool bid_valid{true};
  bool ask_valid{true};
  std::map<double, std::uint64_t> volumes;
  std::vector<std::uint32_t> instrument_ids{1234};

  static std::int64_t toNano(double x) {
    return static_cast<std::int64_t>(x * 1e9 + 0.5);
  }

  constellation::interfaces::common::InterfaceVersionInfo
  GetVersionInfo() const noexcept override {
    return {1, 0};
  }

  std::optional<std::int64_t>
  BestBidPrice(std::uint32_t /*instrument_id*/) const override {
    if (!bid_valid) return std::nullopt;
    double b = best_bid.load();
    if (b <= 0.0) return std::nullopt;
    return toNano(b);
  }

  std::optional<std::int64_t>
  BestAskPrice(std::uint32_t /*instrument_id*/) const override {
    if (!ask_valid) return std::nullopt;
    double a = best_ask.load();
    if (a <= 0.0) return std::nullopt;
    return toNano(a);
  }

  std::optional<std::uint64_t>
  VolumeAtPrice(std::uint32_t /*instrument_id*/,
                std::int64_t priceNanos) const override {
    double keyD = static_cast<double>(priceNanos) / 1e9;
    auto it = volumes.find(keyD);
    if (it != volumes.end()) return it->second;
    return std::nullopt;
  }

  std::vector<std::uint32_t> GetInstrumentIds() const override {
    return instrument_ids;
  }

  std::optional<constellation::interfaces::orderbook::PriceLevel>
  GetLevel(std::uint32_t, constellation::interfaces::orderbook::BookSide,
           std::size_t) const override {
    return std::nullopt;
  }

  std::uint64_t
  TotalDepth(std::uint32_t, constellation::interfaces::orderbook::BookSide,
             std::size_t) const override {
    return 0;
  }

  std::optional<double>
  WeightedMidPrice(std::uint32_t) const override { return std::nullopt; }

  std::optional<double>
  VolumeAdjustedMidPrice(std::uint32_t, std::size_t) const override {
    return std::nullopt;
  }
};
