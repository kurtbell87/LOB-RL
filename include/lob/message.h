#pragma once
#include <cstdint>

namespace lob {

enum class Side : uint8_t { Bid, Ask };

enum class Action : uint8_t {
    Add,
    Cancel,
    Modify,
    Trade,
    Clear
};

struct MBOMessage {
    uint64_t timestamp_ns;
    uint64_t order_id;
    int64_t  price;          // fixed-point (price * 1e9)
    uint32_t quantity;
    Side     side;
    Action   action;
};

}  // namespace lob
