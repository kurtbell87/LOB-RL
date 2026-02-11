#pragma once

#include "lob/message.h"
#include <cstdint>

// Minimal databento-compatible types for map_mbo_to_message().
// These mirror the databento-cpp field names with simple C types
// so test code can construct and assign fields directly.
// Guarded to avoid conflict with the real databento-cpp types.
// Define DBN_MESSAGE_MAP_SKIP_SHIM before including to use real databento types.
#ifndef DBN_MESSAGE_MAP_SKIP_SHIM
namespace databento {

struct RecordHeader {
    uint8_t length = 0;
    uint8_t rtype = 0;
    uint16_t publisher_id = 0;
    uint32_t instrument_id = 0;
    uint64_t ts_event = 0;
};

struct MboMsg {
    RecordHeader hd;
    uint64_t order_id = 0;
    int64_t price = 0;
    uint32_t size = 0;
    uint8_t flags = 0;
    uint8_t channel_id = 0;
    char action = 'N';
    char side = 'N';
};

}  // namespace databento
#endif  // DBN_MESSAGE_MAP_SKIP_SHIM

// Map a Databento action character to Message::Action.
// Returns false for actions we skip (None 'N', unknown).
bool map_action(char action_char, Message::Action& out);

// Map a Databento side character to Message::Side.
// Defaults to Bid for unknown/None sides.
Message::Side map_side(char side_char);

// Map a databento MboMsg to our internal Message format.
// Returns true if the record produced a valid Message, false if it should be skipped.
// Skips: Action::None ('N'), non-matching instrument_id.
// Remaps: Action::Fill ('F') → Message::Action::Trade, Action::Clear ('R') → Cancel.
// Side::None → Side::Bid (default for trades with unspecified aggressor).
// Price: i64 fixed-point (1e-9) → double.
// instrument_id=0 means accept all records (no filtering).
bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id = 0);
