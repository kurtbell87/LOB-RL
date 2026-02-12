#include "dbn_message_map.h"
#include "databento/constants.hpp"

bool map_action(char action_char, Message::Action& out) {
    switch (action_char) {
        case 'A': out = Message::Action::Add; return true;
        case 'C': out = Message::Action::Cancel; return true;
        case 'M': out = Message::Action::Modify; return true;
        case 'T': [[fallthrough]];
        case 'F': out = Message::Action::Trade; return true;
        case 'R': out = Message::Action::Cancel; return true;  // Clear -> Cancel
        default:  return false;
    }
}

Message::Side map_side(char side_char) {
    switch (side_char) {
        case 'A': return Message::Side::Ask;
        case 'B': [[fallthrough]];
        default:  return Message::Side::Bid;
    }
}

bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id) {
    // Filter by instrument_id (0 = accept all)
    if (instrument_id != 0 && mbo.hd.instrument_id != instrument_id) {
        return false;
    }

    if (!map_action(static_cast<char>(mbo.action), msg.action)) {
        return false;
    }
    msg.side = map_side(static_cast<char>(mbo.side));

    // Price: i64 fixed-point (1e-9) -> double
    msg.price = static_cast<double>(mbo.price) / databento::kFixedPriceScale;

    // Pass through other fields
    msg.order_id = mbo.order_id;
    msg.qty = mbo.size;
    msg.ts_ns = static_cast<uint64_t>(
        mbo.hd.ts_event.time_since_epoch().count());
    msg.flags = static_cast<uint8_t>(mbo.flags);

    return true;
}
