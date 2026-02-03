#include "lob/book.h"
#include <algorithm>

namespace lob {

void Book::apply(const MBOMessage& msg) {
    switch (msg.action) {
        case Action::Add:
            add_order(msg);
            break;
        case Action::Cancel:
            cancel_order(msg);
            break;
        case Action::Modify:
            modify_order(msg);
            break;
        case Action::Trade:
            process_trade(msg);
            break;
        case Action::Clear:
            clear();
            break;
    }
}

void Book::clear() {
    orders_.clear();
    bids_.clear();
    asks_.clear();
}

void Book::add_order(const MBOMessage& msg) {
    // Add to order map
    Order order{msg.order_id, msg.price, msg.quantity, msg.side};
    orders_[msg.order_id] = order;

    // Add to price levels
    if (msg.side == Side::Bid) {
        bids_[msg.price] += msg.quantity;
    } else {
        asks_[msg.price] += msg.quantity;
    }
}

void Book::cancel_order(const MBOMessage& msg) {
    auto it = orders_.find(msg.order_id);
    if (it == orders_.end()) {
        return;  // Order not found, ignore
    }

    const Order& order = it->second;

    // Remove from price levels
    if (order.side == Side::Bid) {
        auto level_it = bids_.find(order.price);
        if (level_it != bids_.end()) {
            if (level_it->second <= order.quantity) {
                bids_.erase(level_it);
            } else {
                level_it->second -= order.quantity;
            }
        }
    } else {
        auto level_it = asks_.find(order.price);
        if (level_it != asks_.end()) {
            if (level_it->second <= order.quantity) {
                asks_.erase(level_it);
            } else {
                level_it->second -= order.quantity;
            }
        }
    }

    orders_.erase(it);
}

void Book::modify_order(const MBOMessage& msg) {
    auto it = orders_.find(msg.order_id);
    if (it == orders_.end()) {
        // Order not found, treat as add
        add_order(msg);
        return;
    }

    Order& order = it->second;

    // Remove old quantity from old price level
    if (order.side == Side::Bid) {
        auto level_it = bids_.find(order.price);
        if (level_it != bids_.end()) {
            if (level_it->second <= order.quantity) {
                bids_.erase(level_it);
            } else {
                level_it->second -= order.quantity;
            }
        }
    } else {
        auto level_it = asks_.find(order.price);
        if (level_it != asks_.end()) {
            if (level_it->second <= order.quantity) {
                asks_.erase(level_it);
            } else {
                level_it->second -= order.quantity;
            }
        }
    }

    // Update order
    order.price = msg.price;
    order.quantity = msg.quantity;

    // Add new quantity to new price level
    if (order.side == Side::Bid) {
        bids_[order.price] += order.quantity;
    } else {
        asks_[order.price] += order.quantity;
    }
}

void Book::process_trade(const MBOMessage& msg) {
    auto it = orders_.find(msg.order_id);
    if (it == orders_.end()) {
        return;  // Order not found, ignore
    }

    Order& order = it->second;
    uint32_t traded_qty = std::min(msg.quantity, order.quantity);

    // Reduce quantity at price level
    if (order.side == Side::Bid) {
        auto level_it = bids_.find(order.price);
        if (level_it != bids_.end()) {
            if (level_it->second <= traded_qty) {
                bids_.erase(level_it);
            } else {
                level_it->second -= traded_qty;
            }
        }
    } else {
        auto level_it = asks_.find(order.price);
        if (level_it != asks_.end()) {
            if (level_it->second <= traded_qty) {
                asks_.erase(level_it);
            } else {
                level_it->second -= traded_qty;
            }
        }
    }

    // Update or remove order
    if (order.quantity <= traded_qty) {
        orders_.erase(it);
    } else {
        order.quantity -= traded_qty;
    }
}

Level Book::bid(int depth) const {
    if (depth < 0 || static_cast<size_t>(depth) >= bids_.size()) {
        return Level{0, 0};
    }

    auto it = bids_.begin();
    std::advance(it, depth);
    return Level{it->first, it->second};
}

Level Book::ask(int depth) const {
    if (depth < 0 || static_cast<size_t>(depth) >= asks_.size()) {
        return Level{0, 0};
    }

    auto it = asks_.begin();
    std::advance(it, depth);
    return Level{it->first, it->second};
}

int Book::bid_depth() const {
    return static_cast<int>(bids_.size());
}

int Book::ask_depth() const {
    return static_cast<int>(asks_.size());
}

int64_t Book::mid_price() const {
    if (bids_.empty() || asks_.empty()) {
        return 0;
    }

    int64_t best_bid = bids_.begin()->first;
    int64_t best_ask = asks_.begin()->first;
    return (best_bid + best_ask) / 2;
}

int64_t Book::spread() const {
    if (bids_.empty() || asks_.empty()) {
        return 0;
    }

    int64_t best_bid = bids_.begin()->first;
    int64_t best_ask = asks_.begin()->first;
    return best_ask - best_bid;
}

double Book::imbalance(int levels) const {
    if (levels <= 0 || bids_.empty() || asks_.empty()) {
        return 0.0;
    }

    uint64_t bid_qty = 0;
    uint64_t ask_qty = 0;

    int count = 0;
    for (auto it = bids_.begin(); it != bids_.end() && count < levels; ++it, ++count) {
        bid_qty += it->second;
    }

    count = 0;
    for (auto it = asks_.begin(); it != asks_.end() && count < levels; ++it, ++count) {
        ask_qty += it->second;
    }

    uint64_t total = bid_qty + ask_qty;
    if (total == 0) {
        return 0.0;
    }

    // Imbalance: (bid - ask) / (bid + ask), ranges from -1 to 1
    return static_cast<double>(static_cast<int64_t>(bid_qty) - static_cast<int64_t>(ask_qty))
           / static_cast<double>(total);
}

}  // namespace lob
