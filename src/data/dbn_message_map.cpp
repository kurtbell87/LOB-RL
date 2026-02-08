#include "dbn_message_map.h"

bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id) {
    // Filter by instrument_id (0 = accept all)
    if (instrument_id != 0 && mbo.hd.instrument_id != instrument_id) {
        return false;
    }

    // Map action — skip Clear ('R') and None ('N')
    switch (mbo.action) {
        case 'A': msg.action = Message::Action::Add; break;
        case 'C': msg.action = Message::Action::Cancel; break;
        case 'M': msg.action = Message::Action::Modify; break;
        case 'T': [[fallthrough]];
        case 'F': msg.action = Message::Action::Trade; break;  // Fill → Trade
        case 'R': return false;  // Clear — skip
        case 'N': return false;  // None — skip
        default:  return false;  // Unknown — skip
    }

    // Map side — None → Bid
    switch (mbo.side) {
        case 'B': msg.side = Message::Side::Bid; break;
        case 'A': msg.side = Message::Side::Ask; break;
        case 'N': [[fallthrough]];  // None → Bid
        default:  msg.side = Message::Side::Bid; break;
    }

    // Price: i64 fixed-point (1e-9) → double
    msg.price = static_cast<double>(mbo.price) / 1e9;

    // Pass through other fields
    msg.order_id = mbo.order_id;
    msg.qty = mbo.size;
    msg.ts_ns = mbo.hd.ts_event;
    msg.flags = mbo.flags;

    return true;
}
