#include "orderbook/MarketBook.hpp"
#include <stdexcept>
#include <mutex>

namespace constellation::modules::orderbook {

using constellation::interfaces::orderbook::PriceLevel;

MarketBook::MarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
  : logger_(logger)
{
}

MarketBook::~MarketBook() = default;

void MarketBook::UpdateGlobalCounters(databento::Action action) {
  std::lock_guard<std::mutex> lock(counterMutex_);
  switch (action) {
    case databento::Action::Add:    ++global_add_count_;    break;
    case databento::Action::Cancel: ++global_cancel_count_; break;
    case databento::Action::Modify: ++global_modify_count_; break;
    case databento::Action::Trade:
    case databento::Action::Fill:   ++global_trade_count_;  break;
    case databento::Action::Clear:  ++global_clear_count_;  break;
    default: break;
  }
}

void MarketBook::InvalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMutex_);
  cached_instrument_count_ = 0;
  cached_ids_.clear();
}

LimitOrderBook* MarketBook::FindOrCreateBook(std::uint32_t instrument_id) {
  // Lock map for find/create operations
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) {
    auto new_lob = std::make_unique<LimitOrderBook>(instrument_id, logger_);
    auto [ins_it, _] = books_.emplace(instrument_id, std::move(new_lob));
    it = ins_it;
    InvalidateCache();
  }
  return it->second.get();
}

void MarketBook::AddInstrument(std::uint32_t instrument_id,
                               std::unique_ptr<LimitOrderBook> lob) {
  std::lock_guard<std::mutex> lock(mapMutex_);
  books_[instrument_id] = std::move(lob);
  InvalidateCache();
}

LimitOrderBook* MarketBook::GetBook(std::uint32_t instrument_id) const {
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return nullptr;
  return it->second.get();
}

void MarketBook::OnMboUpdate(const databento::MboMsg& mbo) {
  // Update global counters
  UpdateGlobalCounters(mbo.action);
  
  // Find or create the LimitOrderBook for this instrument
  LimitOrderBook* lob = FindOrCreateBook(mbo.hd.instrument_id);
  
  // The LOB's internal locking will handle thread safety
  lob->OnMboUpdate(mbo);
}

void MarketBook::BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages) {
  // Group messages by instrument ID
  std::unordered_map<std::uint32_t, std::vector<databento::MboMsg>> msgsByInstrument;
  
  // First pass - update counters and group by instrument
  for (const auto& mbo : messages) {
    // Update global counters
    UpdateGlobalCounters(mbo.action);
    
    // Group by instrument ID
    msgsByInstrument[mbo.hd.instrument_id].push_back(mbo);
  }
  
  // Second pass - process each instrument's messages
  for (auto& [inst_id, inst_msgs] : msgsByInstrument) {
    // Find or create the LimitOrderBook
    LimitOrderBook* lob = FindOrCreateBook(inst_id);
    
    // Process batch for this instrument - LOB has its own locking
    if (inst_msgs.size() == 1) {
      lob->OnMboUpdate(inst_msgs[0]);
    } else {
      lob->BatchOnMboUpdate(inst_msgs);
    }
  }
}

std::optional<PriceLevel> MarketBook::GetBestBid(std::uint32_t instrument_id) const {
  // Find the LOB without creating a new one
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return std::nullopt;
  
  // LOB's BestBid method handles its own locking
  return it->second->BestBid();
}

std::optional<PriceLevel> MarketBook::GetBestAsk(std::uint32_t instrument_id) const {
  // Find the LOB without creating a new one
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return std::nullopt;
  
  // LOB's BestAsk method handles its own locking
  return it->second->BestAsk();
}

std::size_t MarketBook::InstrumentCount() const noexcept {
  // First check if we have a cached value
  {
    std::lock_guard<std::mutex> cacheLock(cacheMutex_);
    if (cached_instrument_count_ > 0) {
      return cached_instrument_count_;
    }
  }
  
  // Lock the map to count instruments
  std::lock_guard<std::mutex> lock(mapMutex_);
  std::size_t count = books_.size();
  
  // Update cache
  if (count > 0) {
    std::lock_guard<std::mutex> cacheLock(cacheMutex_);
    cached_instrument_count_ = count;
  }
  
  return count;
}

std::vector<std::uint32_t> MarketBook::GetInstrumentIds() const {
  // First check if we have cached IDs
  {
    std::lock_guard<std::mutex> cacheLock(cacheMutex_);
    if (!cached_ids_.empty()) {
      return cached_ids_;
    }
  }
  
  // Get IDs from the map
  std::vector<std::uint32_t> ids;
  {
    std::lock_guard<std::mutex> lock(mapMutex_);
    ids.reserve(books_.size());
    for (auto& kv : books_) {
      ids.push_back(kv.first);
    }
  }
  
  // Update cache if we found IDs
  if (!ids.empty()) {
    std::lock_guard<std::mutex> cacheLock(cacheMutex_);
    cached_ids_ = ids;
  }
  
  return ids;
}

std::uint64_t MarketBook::GetGlobalAddCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return global_add_count_;
}

std::uint64_t MarketBook::GetGlobalCancelCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return global_cancel_count_;
}

std::uint64_t MarketBook::GetGlobalModifyCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return global_modify_count_;
}

std::uint64_t MarketBook::GetGlobalTradeCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return global_trade_count_;
}

std::uint64_t MarketBook::GetGlobalClearCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return global_clear_count_;
}

std::uint64_t MarketBook::GetGlobalTotalEventCount() const noexcept {
  std::lock_guard<std::mutex> lock(counterMutex_);
  return (global_add_count_ + global_cancel_count_ + global_modify_count_ +
          global_trade_count_ + global_clear_count_);
}

std::optional<std::int64_t> MarketBook::BestBidPrice(std::uint32_t instrument_id) const {
  // Get best bid price level
  auto best = GetBestBid(instrument_id);
  if (!best.has_value()) {
    return std::nullopt;
  }
  // Return raw int64 price
  return best->price;
}

std::optional<std::int64_t> MarketBook::BestAskPrice(std::uint32_t instrument_id) const {
  // Get best ask price level
  auto best = GetBestAsk(instrument_id);
  if (!best.has_value()) {
    return std::nullopt;
  }
  // Return raw int64
  return best->price;
}

std::optional<std::uint64_t>
MarketBook::VolumeAtPrice(std::uint32_t instrument_id, std::int64_t priceNanos) const {
  // Find the LOB without creating a new one
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) {
    return std::nullopt;
  }
  
  // If user passes negative price, treat as no volume
  if (priceNanos < 0) {
    return 0ULL;
  }

  // Sum volume from both sides - LOB methods have their own locking
  auto bidVol = it->second->VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Bid, priceNanos);
  auto askVol = it->second->VolumeAtPrice(constellation::interfaces::orderbook::BookSide::Ask, priceNanos);
  return bidVol + askVol;
}

std::vector<std::tuple<std::uint32_t,
                       std::optional<PriceLevel>,
                       std::optional<PriceLevel>>>
MarketBook::AllBestQuotes() const {
  std::vector<std::tuple<std::uint32_t,
                         std::optional<PriceLevel>,
                         std::optional<PriceLevel>>> result;
  
  // Get list of instrument IDs
  auto instrumentIds = GetInstrumentIds();
  result.reserve(instrumentIds.size());
  
  // For each instrument ID, get best bid and ask
  for (const auto& inst_id : instrumentIds) {
    auto bid = GetBestBid(inst_id);
    auto ask = GetBestAsk(inst_id);
    result.emplace_back(inst_id, bid, ask);
  }
  
  return result;
}

std::unique_ptr<MarketBookSnapshot> MarketBook::CreateSnapshot() const {
  auto snap = std::make_unique<MarketBookSnapshot>();
  
  // Get list of instrument IDs
  auto instrumentIds = GetInstrumentIds();
  
  // Copy global counters
  {
    std::lock_guard<std::mutex> counterLock(counterMutex_);
    snap->global_add_count_    = global_add_count_;
    snap->global_cancel_count_ = global_cancel_count_;
    snap->global_modify_count_ = global_modify_count_;
    snap->global_trade_count_  = global_trade_count_;
    snap->global_clear_count_  = global_clear_count_;
  }
  
  // Get snapshot of each LOB
  {
    std::lock_guard<std::mutex> mapLock(mapMutex_);
    for (auto& kv : books_) {
      auto inst_id = kv.first;
      auto lob_snap = kv.second->CreateSnapshot();
      if (lob_snap) {
        snap->lob_snapshots_[inst_id] = *lob_snap;
      }
    }
  }
  
  return snap;
}

void MarketBook::RestoreSnapshot(const MarketBookSnapshot& snapshot) {
  // Clear all existing data
  {
    std::lock_guard<std::mutex> mapLock(mapMutex_);
    books_.clear();
  }
  
  // Restore global counters
  {
    std::lock_guard<std::mutex> counterLock(counterMutex_);
    global_add_count_    = snapshot.global_add_count_;
    global_cancel_count_ = snapshot.global_cancel_count_;
    global_modify_count_ = snapshot.global_modify_count_;
    global_trade_count_  = snapshot.global_trade_count_;
    global_clear_count_  = snapshot.global_clear_count_;
  }
  
  InvalidateCache();

  // Restore each LOB from the snapshot
  {
    std::lock_guard<std::mutex> mapLock(mapMutex_);
    for (auto& kv : snapshot.lob_snapshots_) {
      auto inst_id = kv.first;
      auto new_lob = std::make_unique<LimitOrderBook>(inst_id, logger_);
      new_lob->RestoreSnapshot(kv.second);
      books_[inst_id] = std::move(new_lob);
    }
  }
}

void MarketBook::ResetGlobalCounters() {
  // Clear all books
  {
    std::lock_guard<std::mutex> mapLock(mapMutex_);
    books_.clear();
  }
  
  // Reset counters
  {
    std::lock_guard<std::mutex> counterLock(counterMutex_);
    global_add_count_ = 0;
    global_cancel_count_ = 0;
    global_modify_count_ = 0;
    global_trade_count_ = 0;
    global_clear_count_ = 0;
  }
  
  InvalidateCache();
}

// ── Depth query methods ──────────────────────────────────────────────

std::optional<PriceLevel>
MarketBook::GetLevel(std::uint32_t instrument_id,
                     constellation::interfaces::orderbook::BookSide side,
                     std::size_t depth_index) const {
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return std::nullopt;
  return it->second->GetLevel(side, depth_index);
}

std::uint64_t
MarketBook::TotalDepth(std::uint32_t instrument_id,
                       constellation::interfaces::orderbook::BookSide side,
                       std::size_t n_levels) const {
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return 0;

  std::uint64_t total = 0;
  for (std::size_t i = 0; i < n_levels; ++i) {
    auto lvl = it->second->GetLevel(side, i);
    if (!lvl.has_value()) break;
    total += lvl->total_quantity;
  }
  return total;
}

std::optional<double>
MarketBook::WeightedMidPrice(std::uint32_t instrument_id) const {
  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return std::nullopt;

  auto bid = it->second->BestBid();
  auto ask = it->second->BestAsk();
  if (!bid.has_value() || !ask.has_value()) return std::nullopt;

  double bid_price = static_cast<double>(bid->price) / 1e9;
  double ask_price = static_cast<double>(ask->price) / 1e9;
  double bid_qty = static_cast<double>(bid->total_quantity);
  double ask_qty = static_cast<double>(ask->total_quantity);

  return (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty);
}

std::optional<double>
MarketBook::VolumeAdjustedMidPrice(std::uint32_t instrument_id,
                                    std::size_t n_levels) const {
  if (n_levels == 0) return std::nullopt;

  std::lock_guard<std::mutex> lock(mapMutex_);
  auto it = books_.find(instrument_id);
  if (it == books_.end()) return std::nullopt;

  using constellation::interfaces::orderbook::BookSide;

  double sum_pq = 0.0;
  double sum_q = 0.0;
  bool has_bid = false;
  bool has_ask = false;

  auto accumulate_side = [&](BookSide side, bool& has_side) {
    for (std::size_t i = 0; i < n_levels; ++i) {
      auto lvl = it->second->GetLevel(side, i);
      if (!lvl.has_value()) break;
      has_side = true;
      double price = static_cast<double>(lvl->price) / 1e9;
      double qty = static_cast<double>(lvl->total_quantity);
      sum_pq += price * qty;
      sum_q += qty;
    }
  };

  accumulate_side(BookSide::Bid, has_bid);
  accumulate_side(BookSide::Ask, has_ask);

  if (!has_bid || !has_ask || sum_q == 0.0) return std::nullopt;
  return sum_pq / sum_q;
}

} // end namespace constellation::modules::orderbook
