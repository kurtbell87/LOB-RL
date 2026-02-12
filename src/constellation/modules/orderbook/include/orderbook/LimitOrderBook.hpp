#pragma once

#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <optional>
#include <vector>

#include "databento/record.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include "interfaces/orderbook/IMarketStateView.hpp" // PriceLevel
#include "interfaces/logging/ILogger.hpp"
#include "orderbook/AugmentedPriceMap.hpp"
#include "orderbook/PriceBucket.hpp"
#include "orderbook/LimitOrderBookSnapshot.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::modules::orderbook {

/**
 * @brief LimitOrderBook is a single-instrument order book that implements
 *        IInstrumentBook.  It uses an AugmentedPriceMap for each side
 *        (bids and asks).
 */
class LimitOrderBook : public constellation::interfaces::orderbook::IInstrumentBook {
public:
  /**
   * @brief Construct with an instrument ID and optional logger injection.
   */
  explicit LimitOrderBook(std::uint32_t instrument_id,
      std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

  ~LimitOrderBook() override = default;

  /**
   * @brief Process a single MBO message for this instrument.
   *        Thread-safe single-writer usage.
   */
  void OnMboUpdate(const databento::MboMsg& mbo);

  /**
   * @brief Batch process many MBO messages under a single lock.
   */
  void BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages);

  // IInstrumentBook interface
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {1, 0};
  }

  std::uint64_t GetAddCount() const noexcept override;
  std::uint64_t GetCancelCount() const noexcept override;
  std::uint64_t GetModifyCount() const noexcept override;
  std::uint64_t GetTradeCount() const noexcept override;
  std::uint64_t GetClearCount() const noexcept override;
  std::uint64_t GetTotalEventCount() const noexcept override;

  std::uint64_t VolumeAtPrice(constellation::interfaces::orderbook::BookSide side, std::int64_t priceNanos) const override;
  std::optional<constellation::interfaces::orderbook::PriceLevel> GetLevel(constellation::interfaces::orderbook::BookSide side, std::size_t level_index) const override;

  /**
   * @brief Additional queries
   */
  std::uint32_t NumOrdersAtPrice(constellation::interfaces::orderbook::BookSide side, std::int64_t priceNanos) const;
  std::optional<constellation::interfaces::orderbook::PriceLevel> BestBid() const;
  std::optional<constellation::interfaces::orderbook::PriceLevel> BestAsk() const;
  std::vector<constellation::interfaces::orderbook::PriceLevel> GetBids() const;
  std::vector<constellation::interfaces::orderbook::PriceLevel> GetAsks() const;

  /**
   * @brief Create a snapshot (memento) of the current LOB state.
   */
  std::unique_ptr<LimitOrderBookSnapshot> CreateSnapshot() const;

  /**
   * @brief Restore from the snapshot, overwriting everything.
   */
  void RestoreSnapshot(const LimitOrderBookSnapshot& snapshot);

private:
  void HandleAdd(const databento::MboMsg& mbo);
  void HandleModify(const databento::MboMsg& mbo);
  void HandleFillOrCancel(const databento::MboMsg& mbo);
  void ClearAll();
  void ClearSide(databento::Side side);

private:
  mutable std::shared_mutex mtx_;
  const std::uint32_t instrument_id_;
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  // order_id => MboMsg for partial fill tracking
  std::unordered_map<std::uint64_t, databento::MboMsg> orders_;

  // Bids stored with negative price, Asks stored with positive
  AugmentedPriceMap<std::int64_t, PriceBucket> bids_;
  AugmentedPriceMap<std::int64_t, PriceBucket> asks_;

  // counters
  std::uint64_t add_count_{0};
  std::uint64_t cancel_count_{0};
  std::uint64_t modify_count_{0};
  std::uint64_t trade_count_{0};
  std::uint64_t clear_count_{0};
};

} // end namespace constellation::modules::orderbook
