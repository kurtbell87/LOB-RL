#pragma once
#include "lob/book.h"
#include "lob/message.h"
#include "lob/source.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Assert that all elements of a float container are finite.
#define EXPECT_ALL_FINITE(container)                                 \
    do {                                                             \
        for (size_t _i = 0; _i < (container).size(); ++_i) {        \
            EXPECT_TRUE(std::isfinite((container)[_i]))              \
                << "Element [" << _i << "] is not finite";           \
        }                                                            \
    } while (0)

// Helper to create messages easily in tests.
inline Message make_msg(uint64_t id, Message::Side side, Message::Action action,
                        double price, uint32_t qty, uint64_t ts = 0,
                        uint8_t flags = 0) {
    Message m;
    m.order_id = id;
    m.side = side;
    m.action = action;
    m.price = price;
    m.qty = qty;
    m.ts_ns = ts;
    m.flags = flags;
    return m;
}

// Common time constants for session-related tests.
static constexpr uint64_t NS_PER_SEC  = 1'000'000'000ULL;
static constexpr uint64_t NS_PER_MIN  = 60ULL * NS_PER_SEC;
static constexpr uint64_t NS_PER_HOUR = 60ULL * NS_PER_MIN;

// RTH boundaries in nanoseconds-since-midnight (UTC)
static constexpr uint64_t RTH_OPEN_NS  = 13ULL * NS_PER_HOUR + 30ULL * NS_PER_MIN;  // 13:30 UTC
static constexpr uint64_t RTH_CLOSE_NS = 20ULL * NS_PER_HOUR;                         // 20:00 UTC
static constexpr uint64_t RTH_DURATION_NS = RTH_CLOSE_NS - RTH_OPEN_NS;               // 6.5 hours

// A Unix nanosecond timestamp for 2024-01-15 00:00:00 UTC (arbitrary reference day)
static constexpr uint64_t DAY_BASE_NS = 19737ULL * 24ULL * NS_PER_HOUR;

// A message source that replays an explicit list of messages.
// Useful for precise control over message sequences and timestamps in tests.
class ScriptedSource : public IMessageSource {
public:
    explicit ScriptedSource(std::vector<Message> messages)
        : messages_(std::move(messages)) {}

    bool next(Message& msg) override {
        if (index_ >= messages_.size()) return false;
        msg = messages_[index_++];
        return true;
    }

    void reset() override { index_ = 0; }

private:
    std::vector<Message> messages_;
    size_t index_ = 0;
};

// ===========================================================================
// Helper: Build a message sequence spanning pre-market -> RTH -> post-market
// with a stable BBO throughout (no price changes at best bid/ask).
//
// Builds:
//   - `warmup_count` pre-market Add messages establishing the book
//   - `rth_count` RTH Add messages at deeper levels (don't change BBO)
//   - `post_count` post-market messages to trigger session close
//
// The best bid is (mid - tick) and best ask is (mid + tick).
// Spread = 2 * tick.  Half-spread = tick.
// ===========================================================================
inline std::vector<Message> make_stable_bbo_messages(
    int warmup_count,
    int rth_count,
    int post_count,
    double mid = 1000.0,
    double tick = 0.25)
{
    std::vector<Message> msgs;
    uint64_t next_id = 1;

    double best_bid = mid - tick;
    double best_ask = mid + tick;

    // Pre-market: establish the top-of-book bid and ask
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;

    // First two messages set the best bid and best ask
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            best_bid, 100, pre_start));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            best_ask, 100, pre_start + NS_PER_MIN));

    // Additional warmup messages at deeper levels (don't affect BBO)
    for (int i = 2; i < warmup_count; ++i) {
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        double price = (side == Message::Side::Bid)
            ? best_bid - (i / 2) * tick
            : best_ask + (i / 2) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 100, pre_start + i * NS_PER_MIN));
    }

    // RTH messages: add orders at deeper levels so mid does NOT change
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;
    uint64_t rth_duration = RTH_CLOSE_NS - RTH_OPEN_NS;
    for (int i = 0; i < rth_count; ++i) {
        uint64_t ts = rth_start + ((uint64_t)i * rth_duration) / (rth_count + 1);
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        // Place deep away from BBO so mid stays unchanged
        double price = (side == Message::Side::Bid)
            ? best_bid - (2 + i) * tick
            : best_ask + (2 + i) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 50 + i, ts));
    }

    // Post-market messages to trigger session close
    uint64_t post_start = DAY_BASE_NS + RTH_CLOSE_NS;
    for (int i = 0; i < post_count; ++i) {
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        double price = (side == Message::Side::Bid)
            ? best_bid - (1 + i / 2) * tick
            : best_ask + (1 + i / 2) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 100, post_start + i * NS_PER_MIN));
    }

    return msgs;
}

// Build a single trade message (convenience wrapper around make_msg).
inline Message make_trade_msg(uint64_t id, double price, uint32_t qty,
                               uint64_t ts, Message::Side side = Message::Side::Ask) {
    return make_msg(id, side, Message::Action::Trade, price, qty, ts);
}

// Append pre-market book-building messages centered on `mid`.
// Creates 2 tight levels + 8 deeper levels (10 total per side).
// Updates next_id in-place.
inline void append_book_warmup(std::vector<Message>& msgs, uint64_t& next_id,
                                double mid, double tick = 0.25) {
    uint64_t pre_ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - tick, 100, pre_ts));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + tick, 100, pre_ts + NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                            mid - 2 * tick, 200, pre_ts + 2 * NS_PER_MIN));
    msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                            mid + 2 * tick, 200, pre_ts + 3 * NS_PER_MIN));
    for (int i = 3; i <= 10; ++i) {
        msgs.push_back(make_msg(next_id++, Message::Side::Bid, Message::Action::Add,
                                mid - i * tick, 100, pre_ts + (2 * i) * NS_PER_MIN));
        msgs.push_back(make_msg(next_id++, Message::Side::Ask, Message::Action::Add,
                                mid + i * tick, 100, pre_ts + (2 * i + 1) * NS_PER_MIN));
    }
}

// Databento flag constants (from the spec).
// Used by test_dbn_message_map.cpp and test_fix_precompute_events.cpp.
static constexpr uint8_t F_LAST     = 0x80;
static constexpr uint8_t F_SNAPSHOT  = 0x20;
static constexpr uint8_t F_PUB_SPEC = 0x02;  // PUBLISHER_SPECIFIC

// Common real-world flag combos
static constexpr uint8_t FLAGS_EVENT_TERMINAL = F_LAST | F_PUB_SPEC;  // 0x82
static constexpr uint8_t FLAGS_MID_EVENT     = 0x00;
static constexpr uint8_t FLAGS_SNAPSHOT_REC  = F_SNAPSHOT | 0x08;     // 0x28

// Path to test fixtures directory (relative to the source file that includes this).
// Works from any test .cpp file in tests/.
inline std::string fixture_path(const std::string& filename) {
    std::filesystem::path p = std::filesystem::path(__FILE__).parent_path() / "fixtures" / filename;
    return p.string();
}

// Build a symmetric book with `n` bid levels and `n` ask levels.
// Bids: 100.0, 99.75, 99.50, ... (descending by 0.25)
// Asks: 100.25, 100.50, 100.75, ... (ascending by 0.25)
// Each level gets qty = base_qty * (level_index + 1)
inline Book make_symmetric_book(int n = 10, uint32_t base_qty = 10) {
    Book book;
    uint64_t oid = 1;
    for (int i = 0; i < n; ++i) {
        book.apply(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            100.0 - i * 0.25, base_qty * (i + 1)));
    }
    for (int i = 0; i < n; ++i) {
        book.apply(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                            100.25 + i * 0.25, base_qty * (i + 1)));
    }
    return book;
}
