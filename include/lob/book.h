#pragma once
#include "lob/message.h"
#include <map>
#include <unordered_map>
#include <vector>
#include <cstdint>

class Book {
public:
    struct PriceLevel {
        double price;
        uint32_t qty;
    };

    void apply(const Message& msg);
    void reset();

    double best_bid() const;
    double best_ask() const;
    double mid_price() const;
    double spread() const;
    size_t bid_depth() const;
    size_t ask_depth() const;

    std::vector<PriceLevel> top_bids(int k = 10) const;
    std::vector<PriceLevel> top_asks(int k = 10) const;
    uint32_t best_bid_qty() const;
    uint32_t best_ask_qty() const;
    uint32_t total_bid_depth(int n) const;
    uint32_t total_ask_depth(int n) const;
    double weighted_mid() const;
    double vamp(int n) const;

private:
    void apply_add(const Message& msg, std::map<double, uint32_t>& levels);
    void apply_cancel(const Message& msg, std::map<double, uint32_t>& levels);
    void apply_modify(const Message& msg, std::map<double, uint32_t>& levels);
    void apply_trade(const Message& msg, std::map<double, uint32_t>& levels);

    // price -> total qty at that level
    std::map<double, uint32_t> bids_;  // sorted ascending, best = rbegin
    std::map<double, uint32_t> asks_;  // sorted ascending, best = begin

    // order_id -> (side, price, qty) for cancel/modify/trade tracking
    struct OrderEntry {
        Message::Side side;
        double price;
        uint32_t qty;
    };
    std::unordered_map<uint64_t, OrderEntry> orders_;
};
