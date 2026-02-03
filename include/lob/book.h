#pragma once
#include "lob/message.h"
#include <map>
#include <unordered_map>

namespace lob {

struct Level {
    int64_t  price    = 0;
    uint32_t quantity = 0;
};

struct Order {
    uint64_t order_id;
    int64_t  price;
    uint32_t quantity;
    Side     side;
};

class Book {
public:
    void apply(const MBOMessage& msg);
    void clear();

    Level bid(int depth = 0) const;
    Level ask(int depth = 0) const;
    int bid_depth() const;
    int ask_depth() const;
    int64_t mid_price() const;
    int64_t spread() const;
    double imbalance(int levels = 1) const;

private:
    void add_order(const MBOMessage& msg);
    void cancel_order(const MBOMessage& msg);
    void modify_order(const MBOMessage& msg);
    void process_trade(const MBOMessage& msg);

    // Order storage: order_id -> Order
    std::unordered_map<uint64_t, Order> orders_;

    // Price levels: price -> total quantity
    // Bids: descending order (highest price first)
    // Asks: ascending order (lowest price first)
    std::map<int64_t, uint32_t, std::greater<int64_t>> bids_;
    std::map<int64_t, uint32_t> asks_;
};

}  // namespace lob
