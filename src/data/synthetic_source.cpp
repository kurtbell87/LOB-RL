#include "synthetic_source.h"
#include <algorithm>

namespace lob {

SyntheticSource::SyntheticSource(uint32_t seed, uint64_t num_messages)
    : seed_(seed), num_messages_(num_messages), rng_(seed) {
    generate_messages();
}

bool SyntheticSource::has_next() const {
    return current_idx_ < messages_.size();
}

MBOMessage SyntheticSource::next() {
    return messages_[current_idx_++];
}

void SyntheticSource::reset() {
    current_idx_ = 0;
}

uint64_t SyntheticSource::message_count() const {
    return messages_.size();
}

void SyntheticSource::generate_messages() {
    messages_.clear();
    messages_.reserve(num_messages_);

    rng_.seed(seed_);
    next_order_id_ = 1;
    timestamp_ns_ = 1'000'000'000'000;  // Start at 1000 seconds
    active_bid_orders_.clear();
    active_ask_orders_.clear();

    // Bootstrap: add initial orders to create a book
    int64_t tick_size = 250'000'000;  // 0.25 * 1e9 (ES/MES tick)

    // Add 10 levels of bids
    for (int i = 0; i < 10; ++i) {
        MBOMessage msg;
        msg.timestamp_ns = timestamp_ns_;
        msg.order_id = next_order_id_++;
        msg.price = base_price_ - (i + 1) * tick_size;
        msg.quantity = 10 + (rng_() % 50);
        msg.side = Side::Bid;
        msg.action = Action::Add;
        messages_.push_back(msg);
        active_bid_orders_.push_back(msg.order_id);
        timestamp_ns_ += 1000;  // 1 microsecond
    }

    // Add 10 levels of asks
    for (int i = 0; i < 10; ++i) {
        MBOMessage msg;
        msg.timestamp_ns = timestamp_ns_;
        msg.order_id = next_order_id_++;
        msg.price = base_price_ + (i + 1) * tick_size;
        msg.quantity = 10 + (rng_() % 50);
        msg.side = Side::Ask;
        msg.action = Action::Add;
        messages_.push_back(msg);
        active_ask_orders_.push_back(msg.order_id);
        timestamp_ns_ += 1000;
    }

    // Generate remaining messages with realistic distribution
    std::uniform_int_distribution<int> action_dist(0, 99);

    while (messages_.size() < num_messages_) {
        int r = action_dist(rng_);

        if (r < 40) {
            // 40% adds
            Side side = (rng_() % 2 == 0) ? Side::Bid : Side::Ask;
            messages_.push_back(generate_add(side));
        } else if (r < 70) {
            // 30% cancels
            if (!active_bid_orders_.empty() || !active_ask_orders_.empty()) {
                messages_.push_back(generate_cancel());
            } else {
                Side side = (rng_() % 2 == 0) ? Side::Bid : Side::Ask;
                messages_.push_back(generate_add(side));
            }
        } else if (r < 85) {
            // 15% modifies
            if (!active_bid_orders_.empty() || !active_ask_orders_.empty()) {
                messages_.push_back(generate_modify());
            } else {
                Side side = (rng_() % 2 == 0) ? Side::Bid : Side::Ask;
                messages_.push_back(generate_add(side));
            }
        } else {
            // 15% trades
            if (!active_bid_orders_.empty() || !active_ask_orders_.empty()) {
                messages_.push_back(generate_trade());
            } else {
                Side side = (rng_() % 2 == 0) ? Side::Bid : Side::Ask;
                messages_.push_back(generate_add(side));
            }
        }

        timestamp_ns_ += 1000 + (rng_() % 100000);  // 1-100 microseconds
    }
}

MBOMessage SyntheticSource::generate_add(Side side) {
    MBOMessage msg;
    msg.timestamp_ns = timestamp_ns_;
    msg.order_id = next_order_id_++;

    // Price within 10 ticks of base
    int64_t tick_size = 250'000'000;
    int offset = static_cast<int>(rng_() % 10) + 1;

    if (side == Side::Bid) {
        msg.price = base_price_ - offset * tick_size;
        active_bid_orders_.push_back(msg.order_id);
    } else {
        msg.price = base_price_ + offset * tick_size;
        active_ask_orders_.push_back(msg.order_id);
    }

    msg.quantity = 1 + (rng_() % 100);
    msg.side = side;
    msg.action = Action::Add;

    return msg;
}

MBOMessage SyntheticSource::generate_cancel() {
    MBOMessage msg;
    msg.timestamp_ns = timestamp_ns_;
    msg.action = Action::Cancel;
    msg.quantity = 0;

    // Pick a random active order to cancel
    bool use_bid = (rng_() % 2 == 0 && !active_bid_orders_.empty()) || active_ask_orders_.empty();

    if (use_bid && !active_bid_orders_.empty()) {
        size_t idx = rng_() % active_bid_orders_.size();
        msg.order_id = active_bid_orders_[idx];
        msg.side = Side::Bid;
        msg.price = 0;  // Price not needed for cancel
        active_bid_orders_.erase(active_bid_orders_.begin() + static_cast<long>(idx));
    } else if (!active_ask_orders_.empty()) {
        size_t idx = rng_() % active_ask_orders_.size();
        msg.order_id = active_ask_orders_[idx];
        msg.side = Side::Ask;
        msg.price = 0;
        active_ask_orders_.erase(active_ask_orders_.begin() + static_cast<long>(idx));
    }

    return msg;
}

MBOMessage SyntheticSource::generate_modify() {
    MBOMessage msg;
    msg.timestamp_ns = timestamp_ns_;
    msg.action = Action::Modify;

    // Pick a random active order to modify
    bool use_bid = (rng_() % 2 == 0 && !active_bid_orders_.empty()) || active_ask_orders_.empty();

    if (use_bid && !active_bid_orders_.empty()) {
        size_t idx = rng_() % active_bid_orders_.size();
        msg.order_id = active_bid_orders_[idx];
        msg.side = Side::Bid;
        int64_t tick_size = 250'000'000;
        int offset = static_cast<int>(rng_() % 10) + 1;
        msg.price = base_price_ - offset * tick_size;
    } else if (!active_ask_orders_.empty()) {
        size_t idx = rng_() % active_ask_orders_.size();
        msg.order_id = active_ask_orders_[idx];
        msg.side = Side::Ask;
        int64_t tick_size = 250'000'000;
        int offset = static_cast<int>(rng_() % 10) + 1;
        msg.price = base_price_ + offset * tick_size;
    }

    msg.quantity = 1 + (rng_() % 100);

    return msg;
}

MBOMessage SyntheticSource::generate_trade() {
    MBOMessage msg;
    msg.timestamp_ns = timestamp_ns_;
    msg.action = Action::Trade;

    // Trade happens at the best bid or ask
    bool trade_bid = (rng_() % 2 == 0 && !active_bid_orders_.empty()) || active_ask_orders_.empty();

    if (trade_bid && !active_bid_orders_.empty()) {
        // Trade against a bid (someone sold into the bid)
        size_t idx = rng_() % active_bid_orders_.size();
        msg.order_id = active_bid_orders_[idx];
        msg.side = Side::Bid;
        int64_t tick_size = 250'000'000;
        msg.price = base_price_ - tick_size;  // Best bid
        msg.quantity = 1 + (rng_() % 10);
        // Remove order if fully filled (simplified - always remove)
        active_bid_orders_.erase(active_bid_orders_.begin() + static_cast<long>(idx));
    } else if (!active_ask_orders_.empty()) {
        // Trade against an ask (someone bought into the ask)
        size_t idx = rng_() % active_ask_orders_.size();
        msg.order_id = active_ask_orders_[idx];
        msg.side = Side::Ask;
        int64_t tick_size = 250'000'000;
        msg.price = base_price_ + tick_size;  // Best ask
        msg.quantity = 1 + (rng_() % 10);
        active_ask_orders_.erase(active_ask_orders_.begin() + static_cast<long>(idx));
    }

    return msg;
}

}  // namespace lob
