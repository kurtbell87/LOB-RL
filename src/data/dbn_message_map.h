#pragma once

#include "lob/message.h"
#include "databento/record.hpp"
#include "databento/enums.hpp"
#include <cstdint>

// Map a Databento action character to Message::Action.
// Returns false for actions we skip (None 'N', unknown).
bool map_action(char action_char, Message::Action& out);

// Map a Databento side character to Message::Side.
// Defaults to Bid for unknown/None sides.
Message::Side map_side(char side_char);

// Map a databento MboMsg to our internal Message format.
// Returns true if the record produced a valid Message, false if it should be skipped.
// Skips: Action::None ('N'), non-matching instrument_id.
// Remaps: Action::Fill ('F') -> Message::Action::Trade, Action::Clear ('R') -> Cancel.
// Side::None -> Side::Bid (default for trades with unspecified aggressor).
// Price: i64 fixed-point (1e-9) -> double.
// instrument_id=0 means accept all records (no filtering).
bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id = 0);
