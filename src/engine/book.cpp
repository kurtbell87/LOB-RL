#include "lob/book.h"
#include <limits>

// Saturating addition for uint32_t
static uint32_t saturating_add(uint32_t a, uint32_t b) {
    uint32_t result = a + b;
    if (result < a) {  // overflow occurred
        return std::numeric_limits<uint32_t>::max();
    }
    return result;
}

// Subtract qty from a price level; erase the level if it reaches zero.
static void reduce_level(std::map<double, uint32_t>& levels,
                         std::map<double, uint32_t>::iterator lvl_it,
                         uint32_t qty) {
    if (lvl_it->second <= qty) {
        levels.erase(lvl_it);
    } else {
        lvl_it->second -= qty;
    }
}

void Book::apply(const Message& msg) {
    auto& levels = (msg.side == Message::Side::Bid) ? bids_ : asks_;

    switch (msg.action) {
        case Message::Action::Add:    apply_add(msg, levels);    break;
        case Message::Action::Cancel: apply_cancel(msg, levels); break;
        case Message::Action::Modify: apply_modify(msg, levels); break;
        case Message::Action::Trade:  apply_trade(msg, levels);  break;
    }
}

void Book::apply_add(const Message& msg, std::map<double, uint32_t>& levels) {
    levels[msg.price] = saturating_add(levels[msg.price], msg.qty);
    orders_[msg.order_id] = {msg.side, msg.price, msg.qty};
}

void Book::apply_cancel(const Message& msg, std::map<double, uint32_t>& levels) {
    auto it = orders_.find(msg.order_id);
    if (it == orders_.end()) return;

    // Capture entry fields before erasing
    double cancel_price = it->second.price;
    Message::Side cancel_side = it->second.side;
    uint32_t cancel_qty = it->second.qty;

    auto lvl_it = levels.find(cancel_price);
    if (lvl_it != levels.end()) {
        reduce_level(levels, lvl_it, cancel_qty);
    }
    orders_.erase(it);

    // If level was erased, recalculate from remaining orders at that price
    // (needed when saturation caused loss of accounting)
    if (levels.find(cancel_price) == levels.end()) {
        uint32_t recalc = 0;
        for (const auto& [oid, oentry] : orders_) {
            if (oentry.side == cancel_side && oentry.price == cancel_price) {
                recalc = saturating_add(recalc, oentry.qty);
            }
        }
        if (recalc > 0) {
            levels[cancel_price] = recalc;
        }
    }
}

void Book::apply_modify(const Message& msg, std::map<double, uint32_t>& levels) {
    auto it = orders_.find(msg.order_id);
    if (it == orders_.end()) {
        // Treat as add if unknown
        apply_add(msg, levels);
        return;
    }
    auto& entry = it->second;
    if (msg.price == entry.price) {
        // Same price: subtract old order qty, add new order qty
        auto& level_qty = levels[entry.price];
        uint32_t remaining = (level_qty >= entry.qty) ? (level_qty - entry.qty) : 0;
        level_qty = saturating_add(remaining, msg.qty);
        if (level_qty == 0) {
            levels.erase(entry.price);
        }
    } else {
        // Price changed: remove old qty from old level, add new qty at new level
        auto lvl_it = levels.find(entry.price);
        if (lvl_it != levels.end()) {
            reduce_level(levels, lvl_it, entry.qty);
        }
        levels[msg.price] = saturating_add(levels[msg.price], msg.qty);
        entry.price = msg.price;
    }
    entry.qty = msg.qty;
}

void Book::apply_trade(const Message& /*msg*/, std::map<double, uint32_t>& /*levels*/) {
    // No-op: Databento spec says Trade/Fill messages do not affect the book.
    // Book changes are communicated entirely through Add, Cancel, and Modify.
}

void Book::reset() {
    bids_.clear();
    asks_.clear();
    orders_.clear();
}

double Book::best_bid() const {
    if (bids_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return bids_.rbegin()->first;
}

double Book::best_ask() const {
    if (asks_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return asks_.begin()->first;
}

double Book::mid_price() const {
    if (bids_.empty() || asks_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return (bids_.rbegin()->first + asks_.begin()->first) / 2.0;
}

double Book::spread() const {
    if (bids_.empty() || asks_.empty()) return std::numeric_limits<double>::quiet_NaN();
    return asks_.begin()->first - bids_.rbegin()->first;
}

size_t Book::bid_depth() const {
    return bids_.size();
}

size_t Book::ask_depth() const {
    return asks_.size();
}

// Collect up to k levels from an iterator range, padding with NaN/0.
template <typename Iter>
static std::vector<Book::PriceLevel> collect_levels(Iter begin, Iter end, int k) {
    std::vector<Book::PriceLevel> result;
    if (k <= 0) return result;
    result.reserve(static_cast<size_t>(k));

    int count = 0;
    for (auto it = begin; it != end && count < k; ++it, ++count) {
        result.push_back({it->first, it->second});
    }

    while (static_cast<int>(result.size()) < k) {
        result.push_back({std::numeric_limits<double>::quiet_NaN(), 0});
    }
    return result;
}

std::vector<Book::PriceLevel> Book::top_bids(int k) const {
    return collect_levels(bids_.rbegin(), bids_.rend(), k);
}

std::vector<Book::PriceLevel> Book::top_asks(int k) const {
    return collect_levels(asks_.begin(), asks_.end(), k);
}

uint32_t Book::best_bid_qty() const {
    if (bids_.empty()) return 0;
    return bids_.rbegin()->second;
}

uint32_t Book::best_ask_qty() const {
    if (asks_.empty()) return 0;
    return asks_.begin()->second;
}

// Sum quantities from the first n levels of an iterator range.
template <typename Iter>
static uint32_t sum_first_n(Iter begin, Iter end, int n) {
    uint32_t total = 0;
    int count = 0;
    for (auto it = begin; it != end && count < n; ++it, ++count) {
        total += it->second;
    }
    return total;
}

uint32_t Book::total_bid_depth(int n) const {
    if (n <= 0) return 0;
    return sum_first_n(bids_.rbegin(), bids_.rend(), n);
}

uint32_t Book::total_ask_depth(int n) const {
    if (n <= 0) return 0;
    return sum_first_n(asks_.begin(), asks_.end(), n);
}

double Book::weighted_mid() const {
    if (bids_.empty() || asks_.empty()) return std::numeric_limits<double>::quiet_NaN();
    double bq = static_cast<double>(bids_.rbegin()->second);
    double aq = static_cast<double>(asks_.begin()->second);
    if (bq + aq == 0.0) return std::numeric_limits<double>::quiet_NaN();
    double bp = bids_.rbegin()->first;
    double ap = asks_.begin()->first;
    return (bq * ap + aq * bp) / (bq + aq);
}

double Book::vamp(int n) const {
    if (n <= 0 || bids_.empty() || asks_.empty())
        return std::numeric_limits<double>::quiet_NaN();

    double sum_pq = 0.0;
    double sum_q = 0.0;

    int count = 0;
    for (auto it = bids_.rbegin(); it != bids_.rend() && count < n; ++it, ++count) {
        sum_pq += it->first * it->second;
        sum_q += it->second;
    }

    count = 0;
    for (auto it = asks_.begin(); it != asks_.end() && count < n; ++it, ++count) {
        sum_pq += it->first * it->second;
        sum_q += it->second;
    }

    if (sum_q == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return sum_pq / sum_q;
}
