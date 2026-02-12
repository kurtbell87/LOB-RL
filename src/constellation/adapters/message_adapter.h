#pragma once

#include "lob/message.h"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include "databento/constants.hpp"
#include <cmath>
#include <cstdint>
#include <chrono>

namespace constellation::adapters {

/// Convert LOB-RL Message -> databento::MboMsg.
/// instrument_id is embedded in the RecordHeader.
inline databento::MboMsg to_mbo_msg(const Message& msg,
                                     uint32_t instrument_id = 1) {
    databento::MboMsg mbo{};

    // RecordHeader
    mbo.hd.length = sizeof(databento::MboMsg) / 4;
    mbo.hd.rtype = databento::RType::Mbo;
    mbo.hd.publisher_id = 0;
    mbo.hd.instrument_id = instrument_id;
    mbo.hd.ts_event = databento::UnixNanos{
        std::chrono::nanoseconds{msg.ts_ns}};

    // Order fields
    mbo.order_id = msg.order_id;
    mbo.price = std::llround(
        msg.price * static_cast<double>(databento::kFixedPriceScale));
    mbo.size = msg.qty;
    mbo.flags = databento::FlagSet{msg.flags};
    mbo.channel_id = 0;

    // Action mapping
    switch (msg.action) {
        case Message::Action::Add:    mbo.action = databento::Action::Add;    break;
        case Message::Action::Cancel: mbo.action = databento::Action::Cancel; break;
        case Message::Action::Modify: mbo.action = databento::Action::Modify; break;
        case Message::Action::Trade:  mbo.action = databento::Action::Trade;  break;
    }

    // Side mapping
    mbo.side = (msg.side == Message::Side::Ask)
                   ? databento::Side::Ask
                   : databento::Side::Bid;

    mbo.ts_recv = {};
    mbo.ts_in_delta = {};
    mbo.sequence = 0;

    return mbo;
}

/// Convert databento::MboMsg -> LOB-RL Message.
/// Returns false if the action is unrecognized.
inline bool to_message(const databento::MboMsg& mbo, Message& msg) {
    switch (static_cast<char>(mbo.action)) {
        case 'A': msg.action = Message::Action::Add;    break;
        case 'C': msg.action = Message::Action::Cancel; break;
        case 'M': msg.action = Message::Action::Modify; break;
        case 'T':
        case 'F': msg.action = Message::Action::Trade;  break;
        default: return false;
    }

    msg.side = (mbo.side == databento::Side::Ask)
                   ? Message::Side::Ask
                   : Message::Side::Bid;

    msg.price = static_cast<double>(mbo.price) / databento::kFixedPriceScale;
    msg.order_id = mbo.order_id;
    msg.qty = mbo.size;
    msg.ts_ns = static_cast<uint64_t>(
        mbo.hd.ts_event.time_since_epoch().count());
    msg.flags = static_cast<uint8_t>(mbo.flags);

    return true;
}

} // namespace constellation::adapters
