#pragma once
#include "lob/barrier/barrier_precompute.h"
#include "lob/barrier/bar_builder.h"
#include "lob/session.h"
#include "lob/source.h"
#include "test_helpers.h"
#include <cstdint>
#include <vector>

// Build an MBO stream that produces a known number of complete bars.
// Includes pre-market book warmup + RTH trades with a rotating price pattern.
inline std::vector<Message> make_barrier_stream(int bar_size, int num_bars,
                                                 int partial_trades = 0) {
    std::vector<Message> msgs;
    uint64_t next_id = 1;
    double mid = 4000.0;

    append_book_warmup(msgs, next_id, mid);

    uint64_t rth_ts = DAY_BASE_NS + RTH_OPEN_NS + NS_PER_MIN;
    int total_trades = num_bars * bar_size + partial_trades;

    double prices[] = {4000.25, 4000.50, 4000.00, 4000.75, 4000.25};
    int n_prices = 5;

    for (int i = 0; i < total_trades; ++i) {
        double p = prices[i % n_prices];
        Message::Side side = (i % 3 == 0) ? Message::Side::Bid : Message::Side::Ask;
        msgs.push_back(make_trade_msg(next_id++, p, 1, rth_ts + i * 1'000'000ULL, side));
    }

    return msgs;
}

// Convenience: create a barrier stream and run barrier_precompute in one call.
inline BarrierPrecomputedDay run_barrier_precompute(int bar_size, int num_bars,
                                                     int lookback = 3,
                                                     int a = 20, int b = 10, int t_max = 40,
                                                     int partial_trades = 0) {
    auto msgs = make_barrier_stream(bar_size, num_bars, partial_trades);
    ScriptedSource source(msgs);
    SessionConfig cfg = SessionConfig::default_rth();
    return barrier_precompute(source, cfg, bar_size, lookback, a, b, t_max);
}
