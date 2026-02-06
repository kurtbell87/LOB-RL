#include "lob/precompute.h"
#include "binary_file_source.h"
#include <cmath>
#include <algorithm>

static bool has_valid_bbo(double bid, double ask) {
    return std::isfinite(bid) && std::isfinite(ask);
}

PrecomputedDay precompute(IMessageSource& source, const SessionConfig& cfg) {
    PrecomputedDay result;
    Book book;
    FeatureBuilder fb;
    SessionFilter filter(cfg);

    Message m;

    // Track previous BBO to detect changes
    double prev_best_bid = std::numeric_limits<double>::quiet_NaN();
    double prev_best_ask = std::numeric_limits<double>::quiet_NaN();

    // Separate messages into phases in a single pass (no intermediate buffer)
    std::vector<Message> pre_market;
    std::vector<Message> rth;

    while (source.next(m)) {
        auto phase = filter.classify(m.ts_ns);
        if (phase == SessionFilter::Phase::PreMarket) {
            pre_market.push_back(m);
        } else if (phase == SessionFilter::Phase::RTH) {
            rth.push_back(m);
        }
        // Post-market: skip
    }

    // Apply warmup
    if (cfg.warmup_messages < 0) {
        // All pre-market messages
        for (auto& pm : pre_market) {
            book.apply(pm);
        }
    } else if (cfg.warmup_messages > 0) {
        int start = std::max(0, static_cast<int>(pre_market.size()) - cfg.warmup_messages);
        for (int i = start; i < static_cast<int>(pre_market.size()); ++i) {
            book.apply(pre_market[i]);
        }
    }
    // warmup_messages == 0: skip all pre-market

    // Track initial BBO after warmup
    prev_best_bid = book.best_bid();
    prev_best_ask = book.best_ask();

    // Phase 2: Process RTH messages, snapshot on BBO change
    for (auto& msg : rth) {
        book.apply(msg);

        double cur_bid = book.best_bid();
        double cur_ask = book.best_ask();

        // Snapshot when BBO changes and both sides are valid (finite)
        bool bbo_changed = false;
        if (has_valid_bbo(cur_bid, cur_ask)) {
            bbo_changed = !has_valid_bbo(prev_best_bid, prev_best_ask)
                       || cur_bid != prev_best_bid
                       || cur_ask != prev_best_ask;
        }

        prev_best_bid = cur_bid;
        prev_best_ask = cur_ask;

        if (bbo_changed) {
            double mid = book.mid_price();
            double spread = book.spread();

            // Compute time_remaining: 1.0 - session_progress
            float progress = filter.session_progress(msg.ts_ns);
            float time_remaining = 1.0f - progress;

            // Build observation with position=0, take first 43 floats (no position)
            auto full_obs = fb.build(book, 0.0f, time_remaining);

            // Append first OBS_SIZE-1 floats (exclude position) to flat obs vector
            for (int i = 0; i < FeatureBuilder::POSITION; ++i) {
                result.obs.push_back(full_obs[i]);
            }
            result.mid.push_back(mid);
            result.spread.push_back(spread);
            result.num_steps++;
        }
    }

    return result;
}

PrecomputedDay precompute(const std::string& path, const SessionConfig& cfg) {
    BinaryFileSource source(path);
    return precompute(source, cfg);
}
