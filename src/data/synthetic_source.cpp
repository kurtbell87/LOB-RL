#include "synthetic_source.h"
#include <random>
#include <algorithm>

SyntheticSource::SyntheticSource(uint64_t seed) : seed_(seed) {
    generate();
}

bool SyntheticSource::next(Message& msg) {
    if (index_ >= messages_.size()) return false;
    msg = messages_[index_++];
    return true;
}

void SyntheticSource::reset() {
    index_ = 0;
}

void SyntheticSource::generate() {
    messages_.clear();
    std::mt19937_64 rng(seed_);

    const double mid = 1000.0;
    const double tick = 0.25;
    uint64_t ts = 1000000;
    uint64_t next_id = 1;

    // Phase 1: Build initial book — 5 bid levels + 5 ask levels (10 messages)
    // Bids: mid - 1*tick, mid - 2*tick, ..., mid - 5*tick
    // Asks: mid + 1*tick, mid + 2*tick, ..., mid + 5*tick
    for (int i = 1; i <= 5; ++i) {
        Message bid;
        bid.order_id = next_id++;
        bid.side = Message::Side::Bid;
        bid.action = Message::Action::Add;
        bid.price = mid - i * tick;
        bid.qty = 100;
        bid.ts_ns = ts;
        messages_.push_back(bid);
        ts += 1000;

        Message ask;
        ask.order_id = next_id++;
        ask.side = Message::Side::Ask;
        ask.action = Message::Action::Add;
        ask.price = mid + i * tick;
        ask.qty = 100;
        ask.ts_ns = ts;
        messages_.push_back(ask);
        ts += 1000;
    }

    // Phase 2: Random activity (~90 more messages)
    // Track which orders exist at which prices for cancel/modify/trade
    struct OrderInfo {
        uint64_t id;
        Message::Side side;
        double price;
        uint32_t qty;
    };
    std::vector<OrderInfo> live_orders;

    // Populate live_orders from phase 1
    for (const auto& m : messages_) {
        live_orders.push_back({m.order_id, m.side, m.price, m.qty});
    }

    std::uniform_int_distribution<int> action_dist(0, 3);
    std::uniform_int_distribution<int> side_dist(0, 1);
    std::uniform_int_distribution<int> level_dist(1, 8);
    std::uniform_int_distribution<uint32_t> qty_dist(10, 200);

    // Pick a random live order, returning its index and a reference.
    auto pick_order = [&](std::mt19937_64& g) -> std::pair<size_t, OrderInfo&> {
        std::uniform_int_distribution<size_t> idx_dist(0, live_orders.size() - 1);
        size_t idx = idx_dist(g);
        return {idx, live_orders[idx]};
    };

    for (int i = 0; i < 90; ++i) {
        int act = action_dist(rng);
        Message msg;
        msg.ts_ns = ts;

        if (act == 0 || live_orders.empty()) {
            // Add
            auto side = side_dist(rng) == 0 ? Message::Side::Bid : Message::Side::Ask;
            int lvl = level_dist(rng);
            double price = (side == Message::Side::Bid)
                ? mid - lvl * tick
                : mid + lvl * tick;
            uint32_t qty = qty_dist(rng);

            msg.order_id = next_id++;
            msg.side = side;
            msg.action = Message::Action::Add;
            msg.price = price;
            msg.qty = qty;
            live_orders.push_back({msg.order_id, msg.side, msg.price, msg.qty});
        } else if (act == 1) {
            // Cancel
            auto [idx, o] = pick_order(rng);
            msg.order_id = o.id;
            msg.side = o.side;
            msg.action = Message::Action::Cancel;
            msg.price = o.price;
            msg.qty = o.qty;
            live_orders.erase(live_orders.begin() + static_cast<long>(idx));
        } else if (act == 2) {
            // Modify
            auto [idx, o] = pick_order(rng);
            uint32_t new_qty = qty_dist(rng);
            msg.order_id = o.id;
            msg.side = o.side;
            msg.action = Message::Action::Modify;
            msg.price = o.price;
            msg.qty = new_qty;
            o.qty = new_qty;
        } else {
            // Trade
            auto [idx, o] = pick_order(rng);
            uint32_t trade_qty = std::min(o.qty, qty_dist(rng));
            if (trade_qty == 0) trade_qty = 1;
            msg.order_id = o.id;
            msg.side = o.side;
            msg.action = Message::Action::Trade;
            msg.price = o.price;
            msg.qty = trade_qty;
            o.qty -= trade_qty;
            if (o.qty == 0) {
                live_orders.erase(live_orders.begin() + static_cast<long>(idx));
            }
        }

        messages_.push_back(msg);
        ts += 1000;
    }
}
