#include "orderbook/LimitOrderBook.hpp"
#include <stdexcept>
#include <algorithm>
#include <shared_mutex>
#include "databento/constants.hpp"
#include "databento/record.hpp"

namespace constellation::modules::orderbook {

LimitOrderBook::LimitOrderBook(std::uint32_t instrument_id,
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
  : instrument_id_(instrument_id),
    logger_(logger)
{
}

void LimitOrderBook::OnMboUpdate(const databento::MboMsg& mbo) {
  std::unique_lock<std::shared_mutex> lock(mtx_);

  if (mbo.hd.instrument_id != instrument_id_) {
    if (logger_) {
      logger_->Trace("LimitOrderBook::OnMboUpdate - instrument mismatch, ignoring");
    }
    return;
  }
  switch (mbo.action) {
    case databento::Action::Add:    ++add_count_;    break;
    case databento::Action::Cancel: ++cancel_count_; break;
    case databento::Action::Modify: ++modify_count_; break;
    case databento::Action::Trade:
    case databento::Action::Fill:   ++trade_count_;  break;
    case databento::Action::Clear:  ++clear_count_;  break;
    default:
      break;
  }

  switch (mbo.action) {
    case databento::Action::Add:
      HandleAdd(mbo);
      break;
    case databento::Action::Modify:
      HandleModify(mbo);
      break;
    case databento::Action::Cancel:
      HandleFillOrCancel(mbo);
      break;
    case databento::Action::Trade:
    case databento::Action::Fill:
      break;  // no-op: orders removed only by Cancel
    case databento::Action::Clear:
      ClearAll();
      break;
    default:
      break;
  }
}

void LimitOrderBook::BatchOnMboUpdate(const std::vector<databento::MboMsg>& messages) {
  std::unique_lock<std::shared_mutex> lock(mtx_);
  for (auto& mbo : messages) {
    if (mbo.hd.instrument_id != instrument_id_) {
      if (logger_) {
        logger_->Debug("LimitOrderBook::BatchOnMboUpdate - skipping msg for other instrument");
      }
      continue;
    }
    switch (mbo.action) {
      case databento::Action::Add:    ++add_count_;    break;
      case databento::Action::Cancel: ++cancel_count_; break;
      case databento::Action::Modify: ++modify_count_; break;
      case databento::Action::Trade:
      case databento::Action::Fill:   ++trade_count_;  break;
      case databento::Action::Clear:  ++clear_count_;  break;
      default:
        break;
    }
    switch (mbo.action) {
      case databento::Action::Add:
        HandleAdd(mbo);
        break;
      case databento::Action::Modify:
        HandleModify(mbo);
        break;
      case databento::Action::Cancel:
        HandleFillOrCancel(mbo);
        break;
      case databento::Action::Trade:
      case databento::Action::Fill:
        break;  // no-op: orders removed only by Cancel
      case databento::Action::Clear:
        ClearAll();
        break;
      default:
        break;
    }
  }
}

std::uint64_t LimitOrderBook::GetAddCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return add_count_;
}
std::uint64_t LimitOrderBook::GetCancelCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return cancel_count_;
}
std::uint64_t LimitOrderBook::GetModifyCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return modify_count_;
}
std::uint64_t LimitOrderBook::GetTradeCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return trade_count_;
}
std::uint64_t LimitOrderBook::GetClearCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return clear_count_;
}
std::uint64_t LimitOrderBook::GetTotalEventCount() const noexcept {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  return (add_count_ + cancel_count_ + modify_count_ + trade_count_ + clear_count_);
}

std::uint64_t LimitOrderBook::VolumeAtPrice(constellation::interfaces::orderbook::BookSide side, std::int64_t price) const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  if (side == constellation::interfaces::orderbook::BookSide::Bid) {
    auto key = -price;
    auto f = bids_.find(key);
    return f ? f->agg_qty : 0ULL;
  } else {
    auto f = asks_.find(price);
    return f ? f->agg_qty : 0ULL;
  }
}

std::optional<constellation::interfaces::orderbook::PriceLevel>
LimitOrderBook::GetLevel(constellation::interfaces::orderbook::BookSide side, std::size_t level_index) const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  if (side == constellation::interfaces::orderbook::BookSide::Bid) {
    if (level_index >= bids_.size()) return std::nullopt;
    auto p = bids_.nth(level_index);
    if (!p) return std::nullopt;
    auto& [neg_price, pb] = *p;
    return constellation::interfaces::orderbook::PriceLevel{
      -neg_price, pb.agg_qty, pb.count
    };
  } else {
    if (level_index >= asks_.size()) return std::nullopt;
    auto p = asks_.nth(level_index);
    if (!p) return std::nullopt;
    auto& [price, pb] = *p;
    return constellation::interfaces::orderbook::PriceLevel{
      price, pb.agg_qty, pb.count
    };
  }
}

std::uint32_t LimitOrderBook::NumOrdersAtPrice(constellation::interfaces::orderbook::BookSide side, std::int64_t price) const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  if (side == constellation::interfaces::orderbook::BookSide::Bid) {
    auto f = bids_.find(-price);
    return f ? f->count : 0U;
  } else {
    auto f = asks_.find(price);
    return f ? f->count : 0U;
  }
}

std::optional<constellation::interfaces::orderbook::PriceLevel> LimitOrderBook::BestBid() const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  if (bids_.size() == 0) return std::nullopt;
  auto p = bids_.nth(0);
  if (!p) return std::nullopt;
  auto& [neg_price, pb] = *p;
  return constellation::interfaces::orderbook::PriceLevel{
    -neg_price, pb.agg_qty, pb.count
  };
}

std::optional<constellation::interfaces::orderbook::PriceLevel> LimitOrderBook::BestAsk() const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  if (asks_.size() == 0) return std::nullopt;
  auto p = asks_.nth(0);
  if (!p) return std::nullopt;
  auto& [price, pb] = *p;
  return constellation::interfaces::orderbook::PriceLevel{
    price, pb.agg_qty, pb.count
  };
}

std::vector<constellation::interfaces::orderbook::PriceLevel> LimitOrderBook::GetBids() const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  std::vector<constellation::interfaces::orderbook::PriceLevel> out;
  auto n = bids_.size();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto p = bids_.nth(i);
    if (!p) break;
    auto& [neg_pr, pb] = *p;
    out.push_back({ -neg_pr, pb.agg_qty, pb.count });
  }
  return out;
}

std::vector<constellation::interfaces::orderbook::PriceLevel> LimitOrderBook::GetAsks() const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  std::vector<constellation::interfaces::orderbook::PriceLevel> out;
  auto n = asks_.size();
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto p = asks_.nth(i);
    if (!p) break;
    auto& [pr, pb] = *p;
    out.push_back({ pr, pb.agg_qty, pb.count });
  }
  return out;
}

void LimitOrderBook::HandleAdd(const databento::MboMsg& mbo) {
  if (mbo.side != databento::Side::Bid && mbo.side != databento::Side::Ask) return;

  // IsTob: clear entire side, then add one synthetic level (count=0, not tracked)
  if (mbo.flags.IsTob()) {
    ClearSide(mbo.side);
    if (mbo.price == databento::kUndefPrice) return;
    if (mbo.side == databento::Side::Bid) {
      auto key = -static_cast<std::int64_t>(mbo.price);
      PriceBucket pb;
      pb.agg_qty = mbo.size;
      pb.count   = 0;
      bids_.insert(key, pb);
    } else {
      PriceBucket pb;
      pb.agg_qty = mbo.size;
      pb.count   = 0;
      asks_.insert(mbo.price, pb);
    }
    return;
  }

  if (mbo.size == 0) return;
  orders_[mbo.order_id] = mbo;

  if (mbo.side == databento::Side::Bid) {
    auto key = -static_cast<std::int64_t>(mbo.price);
    auto f = bids_.find(key);
    if (!f) {
      PriceBucket pb;
      pb.agg_qty = mbo.size;
      pb.count   = 1;
      pb.orders[mbo.order_id] = mbo;
      bids_.insert(key, pb);
    } else {
      auto pb = *f;
      pb.agg_qty += mbo.size;
      pb.count   += 1;
      pb.orders[mbo.order_id] = mbo;
      bids_.erase(key);
      bids_.insert(key, pb);
    }
  } else {
    auto key = mbo.price;
    auto f = asks_.find(key);
    if (!f) {
      PriceBucket pb;
      pb.agg_qty = mbo.size;
      pb.count   = 1;
      pb.orders[mbo.order_id] = mbo;
      asks_.insert(key, pb);
    } else {
      auto pb = *f;
      pb.agg_qty += mbo.size;
      pb.count   += 1;
      pb.orders[mbo.order_id] = mbo;
      asks_.erase(key);
      asks_.insert(key, pb);
    }
  }
}

void LimitOrderBook::HandleModify(const databento::MboMsg& mbo) {
  auto it = orders_.find(mbo.order_id);
  if (it == orders_.end()) {
    HandleAdd(mbo);
    return;
  }

  auto old = it->second;
  if (old.side == databento::Side::Bid) {
    auto old_key = -static_cast<std::int64_t>(old.price);
    auto f = bids_.find(old_key);
    if (f) {
      auto pb = *f;
      pb.agg_qty -= old.size;
      pb.count -= 1;
      pb.orders.erase(old.order_id);
      bids_.erase(old_key);
      if (pb.count > 0) {
        bids_.insert(old_key, pb);
      }
    }
  } else if (old.side == databento::Side::Ask) {
    auto old_key = old.price;
    auto f = asks_.find(old_key);
    if (f) {
      auto pb = *f;
      pb.agg_qty -= old.size;
      pb.count -= 1;
      pb.orders.erase(old.order_id);
      asks_.erase(old_key);
      if (pb.count > 0) {
        asks_.insert(old_key, pb);
      }
    }
  }

  databento::MboMsg updated = old;
  updated.price = mbo.price;
  updated.size  = mbo.size;
  updated.side  = mbo.side;

  if ((updated.side == databento::Side::Bid || updated.side == databento::Side::Ask)
      && updated.size > 0)
  {
    if (updated.side == databento::Side::Bid) {
      auto new_key = -static_cast<std::int64_t>(updated.price);
      auto f2 = bids_.find(new_key);
      if (!f2) {
        PriceBucket pb;
        pb.agg_qty = updated.size;
        pb.count   = 1;
        pb.orders[updated.order_id] = updated;
        bids_.insert(new_key, pb);
      } else {
        auto pb = *f2;
        pb.agg_qty += updated.size;
        pb.count   += 1;
        pb.orders[updated.order_id] = updated;
        bids_.erase(new_key);
        bids_.insert(new_key, pb);
      }
    } else {
      auto new_key = updated.price;
      auto f2 = asks_.find(new_key);
      if (!f2) {
        PriceBucket pb;
        pb.agg_qty = updated.size;
        pb.count   = 1;
        pb.orders[updated.order_id] = updated;
        asks_.insert(new_key, pb);
      } else {
        auto pb = *f2;
        pb.agg_qty += updated.size;
        pb.count   += 1;
        pb.orders[updated.order_id] = updated;
        asks_.erase(new_key);
        asks_.insert(new_key, pb);
      }
    }
    orders_[mbo.order_id] = updated;
  } else {
    orders_.erase(mbo.order_id);
  }
}

void LimitOrderBook::HandleFillOrCancel(const databento::MboMsg& mbo) {
  auto it = orders_.find(mbo.order_id);
  if (it == orders_.end()) return;
  auto& old = it->second;

  if (old.side == databento::Side::Bid) {
    auto key = -static_cast<std::int64_t>(old.price);
    auto f = bids_.find(key);
    if (!f) return;
    auto pb = *f;
    bids_.erase(key);

    // Partial cancel: subtract mbo.size (not old.size)
    std::uint32_t cancel_sz = mbo.size;
    if (cancel_sz > old.size) cancel_sz = old.size;
    if (pb.agg_qty >= cancel_sz) {
      pb.agg_qty -= cancel_sz;
    }
    std::uint32_t new_qty = old.size - cancel_sz;
    if (new_qty == 0) {
      pb.count -= 1;
      pb.orders.erase(old.order_id);
      orders_.erase(it);
    } else {
      old.size = new_qty;
      pb.orders[old.order_id] = old;
    }

    if (pb.count > 0) {
      bids_.insert(key, pb);
    }
  }
  else if (old.side == databento::Side::Ask) {
    auto key = old.price;
    auto f = asks_.find(key);
    if (!f) return;
    auto pb = *f;
    asks_.erase(key);

    // Partial cancel: subtract mbo.size (not old.size)
    std::uint32_t cancel_sz = mbo.size;
    if (cancel_sz > old.size) cancel_sz = old.size;
    if (pb.agg_qty >= cancel_sz) {
      pb.agg_qty -= cancel_sz;
    }
    std::uint32_t new_qty = old.size - cancel_sz;
    if (new_qty == 0) {
      pb.count -= 1;
      pb.orders.erase(old.order_id);
      orders_.erase(it);
    } else {
      old.size = new_qty;
      pb.orders[old.order_id] = old;
    }

    if (pb.count > 0) {
      asks_.insert(key, pb);
    }
  }
}

void LimitOrderBook::ClearSide(databento::Side side) {
  if (side == databento::Side::Bid) {
    auto n = bids_.size();
    for (size_t i = 0; i < n; ++i) {
      auto p = bids_.nth(i);
      if (p) {
        for (auto& [oid, msg] : p->second.orders) {
          orders_.erase(oid);
        }
      }
    }
    bids_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
  } else if (side == databento::Side::Ask) {
    auto n = asks_.size();
    for (size_t i = 0; i < n; ++i) {
      auto p = asks_.nth(i);
      if (p) {
        for (auto& [oid, msg] : p->second.orders) {
          orders_.erase(oid);
        }
      }
    }
    asks_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
  }
}

void LimitOrderBook::ClearAll() {
  orders_.clear();
  bids_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
  asks_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
}

std::unique_ptr<LimitOrderBookSnapshot> LimitOrderBook::CreateSnapshot() const {
  std::shared_lock<std::shared_mutex> lock(mtx_);
  auto snap = std::make_unique<LimitOrderBookSnapshot>();
  snap->instrument_id_ = instrument_id_;
  snap->orders_ = orders_;

  {
    auto sz = bids_.size();
    snap->bids_.clear();
    snap->bids_.reserve(sz);
    for (size_t i = 0; i < sz; ++i) {
      auto item = bids_.nth(i);
      if (item) {
        snap->bids_.push_back(*item);
      }
    }
  }
  {
    auto sz = asks_.size();
    snap->asks_.clear();
    snap->asks_.reserve(sz);
    for (size_t i = 0; i < sz; ++i) {
      auto item = asks_.nth(i);
      if (item) {
        snap->asks_.push_back(*item);
      }
    }
  }

  snap->add_count_    = add_count_;
  snap->cancel_count_ = cancel_count_;
  snap->modify_count_ = modify_count_;
  snap->trade_count_  = trade_count_;
  snap->clear_count_  = clear_count_;
  return snap;
}

void LimitOrderBook::RestoreSnapshot(const LimitOrderBookSnapshot& snapshot) {
  std::unique_lock<std::shared_mutex> lock(mtx_);
  if (snapshot.instrument_id_ != instrument_id_) {
    if (logger_) {
      logger_->Warn("LimitOrderBook::RestoreSnapshot - instrument_id mismatch, ignoring snapshot");
    }
    return;
  }
  orders_ = snapshot.orders_;

  bids_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
  for (auto& kv : snapshot.bids_) {
    bids_.insert(kv.first, kv.second);
  }
  asks_ = AugmentedPriceMap<std::int64_t, PriceBucket>();
  for (auto& kv : snapshot.asks_) {
    asks_.insert(kv.first, kv.second);
  }

  add_count_    = snapshot.add_count_;
  cancel_count_ = snapshot.cancel_count_;
  modify_count_ = snapshot.modify_count_;
  trade_count_  = snapshot.trade_count_;
  clear_count_  = snapshot.clear_count_;
}

} // end namespace constellation::modules::orderbook
