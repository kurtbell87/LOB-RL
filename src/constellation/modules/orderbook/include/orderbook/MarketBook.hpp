#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <optional>
#include <vector>
#include "databento/enums.hpp"
#include "databento/record.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"
#include "interfaces/orderbook/IMarketBook.hpp"
#include "interfaces/logging/ILogger.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBookSnapshot.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::modules::orderbook {
using interfaces::orderbook::PriceLevel;

/**
 * @brief MarketBook aggregates multiple LimitOrderBooks by instrument_id,
 *        implementing IMarketBook which combines IMarketView (for global best quotes, counters) plus
 *        IMarketBookDataSource for aggregator queries.
 *        
 *        This class implements per-instrument locking, allowing concurrent access to 
 *        different instruments and parallel read operations on the same instrument.
 */
class MarketBook
  : public constellation::interfaces::orderbook::IMarketBook
{
public:
  /**
   * @brief Constructor for MarketBook
   * @param logger Optional logger for logging events
   */
  explicit MarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

  ~MarketBook() override;
  
  // Delete copy and move constructors/assignments
  MarketBook(const MarketBook&) = delete;
  MarketBook& operator=(const MarketBook&) = delete;
  MarketBook(MarketBook&&) = delete;
  MarketBook& operator=(MarketBook&&) = delete;

  /**
   * @brief Add a newly created LOB for the given instrument_id.
   */
  void AddInstrument(std::uint32_t instrument_id,
                     std::unique_ptr<LimitOrderBook> lob);

  /**
   * @brief Get the LimitOrderBook for a specific instrument
   * @param instrument_id The ID of the instrument
   * @return Pointer to the LimitOrderBook or nullptr if not found
   */
  LimitOrderBook* GetBook(std::uint32_t instrument_id) const;

  /**
   * @brief Process one MBO message => route to correct LOB or auto-create
   */
  void OnMboUpdate(const databento::MboMsg& mbo) override;

  /**
   * @brief Batch process multiple MBO messages
   */
  void BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages) override;

  // IMarketStateView
  std::optional<PriceLevel> GetBestBid(std::uint32_t instrument_id) const override;
  std::optional<PriceLevel> GetBestAsk(std::uint32_t instrument_id) const override;

  // IInstrumentRegistry
  std::size_t InstrumentCount() const noexcept override;
  std::vector<std::uint32_t> GetInstrumentIds() const override;

  // IMarketStatistics
  std::uint64_t GetGlobalAddCount() const noexcept override;
  std::uint64_t GetGlobalCancelCount() const noexcept override;
  std::uint64_t GetGlobalModifyCount() const noexcept override;
  std::uint64_t GetGlobalTradeCount() const noexcept override;
  std::uint64_t GetGlobalClearCount() const noexcept override;
  std::uint64_t GetGlobalTotalEventCount() const noexcept override;

  // IMarketBook
  constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept override {
    return {1, 2}; // Incremented minor version for per-instrument locking
  }

  std::optional<std::int64_t> BestBidPrice(std::uint32_t instrument_id) const override;
  std::optional<std::int64_t> BestAskPrice(std::uint32_t instrument_id) const override;
  std::optional<std::uint64_t> VolumeAtPrice(std::uint32_t instrument_id, std::int64_t priceNanos) const override;

  // Depth query methods (IMarketBookDataSource)
  std::optional<PriceLevel> GetLevel(std::uint32_t instrument_id,
                                      interfaces::orderbook::BookSide side,
                                      std::size_t depth_index) const override;
  std::uint64_t TotalDepth(std::uint32_t instrument_id,
                            interfaces::orderbook::BookSide side,
                            std::size_t n_levels) const override;
  std::optional<double> WeightedMidPrice(std::uint32_t instrument_id) const override;
  std::optional<double> VolumeAdjustedMidPrice(std::uint32_t instrument_id,
                                                std::size_t n_levels) const override;

  /**
   * @brief Return {instrument_id, bestBid, bestAsk} for each known LOB
   */
  std::vector<std::tuple<std::uint32_t,
                         std::optional<PriceLevel>,
                         std::optional<PriceLevel>>>
    AllBestQuotes() const;

  // Snapshots
  std::unique_ptr<MarketBookSnapshot> CreateSnapshot() const;
  void RestoreSnapshot(const MarketBookSnapshot& snapshot);
  
  /**
   * @brief Reset global counters and clear all books
   */
  void ResetGlobalCounters();

private:
  /**
   * @brief Update global counters based on MBO action
   * @param action The action from the MBO message
   */
  void UpdateGlobalCounters(databento::Action action);

  /**
   * @brief Find or create a LimitOrderBook for an instrument
   * @param instrument_id The ID of the instrument
   * @return Pointer to the LimitOrderBook
   */
  LimitOrderBook* FindOrCreateBook(std::uint32_t instrument_id);

  void InvalidateCache();

private:
  mutable std::mutex mapMutex_; // Only protects map operations (add/remove/find)
  std::unordered_map<std::uint32_t, std::unique_ptr<LimitOrderBook>> books_;
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger_;

  // Mutex for protecting global counters
  mutable std::mutex counterMutex_;
  std::uint64_t global_add_count_{0};
  std::uint64_t global_cancel_count_{0};
  std::uint64_t global_modify_count_{0};
  std::uint64_t global_trade_count_{0};
  std::uint64_t global_clear_count_{0};

  // Mutex for protecting cache state
  mutable std::mutex cacheMutex_;
  mutable std::size_t cached_instrument_count_{0};
  mutable std::vector<std::uint32_t> cached_ids_;
};

} // end namespace constellation::modules::orderbook
