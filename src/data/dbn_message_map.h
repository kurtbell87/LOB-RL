#pragma once

#include "lob/message.h"
#include <cstdint>

// Minimal databento-compatible types for map_mbo_to_message().
// These mirror the databento-cpp field names with simple C types
// so test code can construct and assign fields directly.
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

// Map a databento MboMsg to our internal Message format.
// Returns true if the record produced a valid Message, false if it should be skipped.
// Skips: Action::Clear ('R'), Action::None ('N'), non-matching instrument_id.
// Remaps: Action::Fill ('F') → Message::Action::Trade.
// Side::None → Side::Bid (default for trades with unspecified aggressor).
// Price: i64 fixed-point (1e-9) → double.
// instrument_id=0 means accept all records (no filtering).
bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id = 0);
