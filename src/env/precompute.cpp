#include "lob/precompute.h"
#include "dbn_file_source.h"
#include "warmup.h"
#include <cmath>

// MBO record flag bits (Databento spec)
static constexpr uint8_t FLAG_LAST     = 0x80;  // Last message in event
static constexpr uint8_t FLAG_SNAPSHOT = 0x20;  // Snapshot record (not a real event)

static bool has_valid_bbo(double bid, double ask) {
    return std::isfinite(bid) && std::isfinite(ask);
}

static bool detect_bbo_change(double cur_bid, double cur_ask,
                              double prev_bid, double prev_ask) {
    if (!has_valid_bbo(cur_bid, cur_ask)) return false;
    return !has_valid_bbo(prev_bid, prev_ask)
        || cur_bid != prev_bid
        || cur_ask != prev_ask;
}

static void record_snapshot(PrecomputedDay& result, const Book& book,
                            FeatureBuilder& fb, const SessionFilter& filter,
                            uint64_t ts_ns) {
    double mid = book.mid_price();
    double spread = book.spread();
    float progress = filter.session_progress(ts_ns);
    float time_remaining = 1.0f - progress;

    auto full_obs = fb.build(book, 0.0f, time_remaining);
    result.obs.insert(result.obs.end(),
                      full_obs.begin(), full_obs.begin() + FeatureBuilder::POSITION);
    result.mid.push_back(mid);
    result.spread.push_back(spread);
    result.num_steps++;
}

// Legacy mode: snapshot on any BBO change (no flag filtering).
static void process_rth_legacy(PrecomputedDay& result, Book& book,
                               FeatureBuilder& fb, const SessionFilter& filter,
                               const std::vector<Message>& rth,
                               double prev_best_bid, double prev_best_ask) {
    for (auto& msg : rth) {
        book.apply(msg);

        double cur_bid = book.best_bid();
        double cur_ask = book.best_ask();

        if (detect_bbo_change(cur_bid, cur_ask, prev_best_bid, prev_best_ask)) {
            record_snapshot(result, book, fb, filter, msg.ts_ns);
        }

        prev_best_bid = cur_bid;
        prev_best_ask = cur_ask;
    }
}

// Flag-aware mode: buffer mid-event messages, apply complete events
// (when F_LAST arrives at the same timestamp). Orphaned mid-event
// messages and snapshot records are discarded.
static void process_rth_flag_aware(PrecomputedDay& result, Book& book,
                                   FeatureBuilder& fb, const SessionFilter& filter,
                                   const std::vector<Message>& rth,
                                   double prev_best_bid, double prev_best_ask) {
    std::vector<Message> event_buf;
    uint64_t event_ts = 0;

    for (size_t idx = 0; idx < rth.size(); ++idx) {
        auto& msg = rth[idx];
        bool is_f_last     = (msg.flags & FLAG_LAST) != 0;
        bool is_snapshot   = (msg.flags & FLAG_SNAPSHOT) != 0;

        if (is_snapshot) continue;

        // Timestamp changed: discard any orphaned mid-event messages
        if (msg.ts_ns != event_ts) {
            event_buf.clear();
            event_ts = msg.ts_ns;
        }

        if (!is_f_last) {
            event_buf.push_back(msg);
            continue;
        }

        // F_LAST: apply buffered messages + this one to the book
        for (auto& buffered : event_buf) {
            book.apply(buffered);
        }
        event_buf.clear();
        book.apply(msg);

        double cur_bid = book.best_bid();
        double cur_ask = book.best_ask();

        if (detect_bbo_change(cur_bid, cur_ask, prev_best_bid, prev_best_ask)) {
            double spread = book.spread();
            if (spread > 0.0) {  // reject crossed/locked books
                record_snapshot(result, book, fb, filter, msg.ts_ns);
            }
        }

        prev_best_bid = cur_bid;
        prev_best_ask = cur_ask;
    }
}

PrecomputedDay precompute(IMessageSource& source, const SessionConfig& cfg) {
    PrecomputedDay result;
    Book book;
    FeatureBuilder fb;
    SessionFilter filter(cfg);

    Message m;

    // Separate messages into pre-market and RTH phase buffers
    std::vector<Message> pre_market;
    std::vector<Message> rth;

    while (source.next(m)) {
        auto phase = filter.classify(m.ts_ns);
        if (phase == SessionFilter::Phase::PreMarket) {
            pre_market.push_back(m);
        } else if (phase == SessionFilter::Phase::RTH) {
            rth.push_back(m);
        }
    }

    apply_warmup(book, pre_market, cfg.warmup_messages);

    double prev_best_bid = book.best_bid();
    double prev_best_ask = book.best_ask();

    // Auto-detect: if any RTH message has non-zero flags, use flag-aware mode
    bool flag_aware = false;
    for (const auto& msg : rth) {
        if (msg.flags != 0) { flag_aware = true; break; }
    }

    if (!flag_aware) {
        process_rth_legacy(result, book, fb, filter, rth,
                           prev_best_bid, prev_best_ask);
    } else {
        process_rth_flag_aware(result, book, fb, filter, rth,
                               prev_best_bid, prev_best_ask);
    }

    return result;
}

PrecomputedDay precompute(const std::string& path, const SessionConfig& cfg,
                          uint32_t instrument_id) {
    DbnFileSource source(path, instrument_id);
    return precompute(source, cfg);
}
