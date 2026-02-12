#pragma once
#include "lob/message.h"
#include "orderbook/LimitOrderBook.hpp"
#include <vector>
#include <cstdint>
#include <memory>
#include <unordered_set>

class Book {
public:
    struct PriceLevel {
        double price;
        uint32_t qty;
    };

    Book();
    ~Book();
    Book(Book&&) noexcept;
    Book& operator=(Book&&) noexcept;

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

    /// Access the underlying Constellation LimitOrderBook.
    constellation::modules::orderbook::LimitOrderBook& constellation_lob();
    const constellation::modules::orderbook::LimitOrderBook& constellation_lob() const;

private:
    static constexpr uint32_t kInstrumentId = 1;
    std::unique_ptr<constellation::modules::orderbook::LimitOrderBook> lob_;
    std::unordered_set<uint64_t> known_orders_;  // track order IDs for modify-as-add
};
