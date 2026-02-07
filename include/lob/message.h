#pragma once
#include <cstdint>
#include <cmath>

struct Message {
    enum class Side : uint8_t { Bid, Ask };
    enum class Action : uint8_t { Add, Cancel, Modify, Trade };

    uint64_t order_id = 0;
    Side side = Side::Bid;
    Action action = Action::Add;
    double price = 0.0;
    uint32_t qty = 0;
    uint64_t ts_ns = 0;
    uint8_t flags = 0;

    bool is_valid() const {
        // price must be positive and finite
        if (price <= 0.0 || !std::isfinite(price)) return false;
        // qty must be positive
        if (qty == 0) return false;
        // order_id must be non-zero
        if (order_id == 0) return false;
        return true;
    }
};
