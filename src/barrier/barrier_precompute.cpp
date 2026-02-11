#include "lob/barrier/barrier_precompute.h"
#include "lob/barrier/bar_builder.h"
#include "lob/barrier/barrier_label.h"
#include "lob/barrier/feature_compute.h"
#include "dbn_file_source.h"

// Shared implementation for both overloads.
static BarrierPrecomputedDay barrier_precompute_impl(
    BarBuilder& builder,
    int bar_size,
    int lookback,
    int a, int b, int t_max)
{
    const auto& bars = builder.bars();
    const auto& accums = builder.accums();
    int n_bars = static_cast<int>(bars.size());

    BarrierPrecomputedDay day{};
    day.n_bars = n_bars;
    day.bar_size = bar_size;
    day.lookback = lookback;
    day.n_features = N_FEATURES;

    if (n_bars == 0) {
        return day;
    }

    // Populate bar OHLCV arrays
    day.bar_open.resize(n_bars);
    day.bar_high.resize(n_bars);
    day.bar_low.resize(n_bars);
    day.bar_close.resize(n_bars);
    day.bar_vwap.resize(n_bars);
    day.bar_volume.resize(n_bars);
    day.bar_t_start.resize(n_bars);
    day.bar_t_end.resize(n_bars);

    // Flatten trade data + build offsets
    day.bar_trade_offsets.resize(n_bars + 1);
    day.bar_trade_offsets[0] = 0;

    for (int i = 0; i < n_bars; ++i) {
        const auto& bar = bars[i];
        day.bar_open[i] = bar.open;
        day.bar_high[i] = bar.high;
        day.bar_low[i] = bar.low;
        day.bar_close[i] = bar.close;
        day.bar_vwap[i] = bar.vwap;
        day.bar_volume[i] = bar.volume;
        day.bar_t_start[i] = bar.t_start;
        day.bar_t_end[i] = bar.t_end;

        // Append trade prices and sizes
        day.trade_prices.insert(day.trade_prices.end(),
                                bar.trade_prices.begin(), bar.trade_prices.end());
        day.trade_sizes.insert(day.trade_sizes.end(),
                               bar.trade_sizes.begin(), bar.trade_sizes.end());
        day.bar_trade_offsets[i + 1] = static_cast<int64_t>(day.trade_prices.size());
    }

    // Compute labels
    auto labels = compute_labels(bars, a, b, t_max);
    day.label_values.resize(n_bars);
    day.label_tau.resize(n_bars);
    day.label_resolution_bar.resize(n_bars);
    for (int i = 0; i < n_bars; ++i) {
        day.label_values[i] = labels[i].label;
        day.label_tau[i] = labels[i].tau;
        day.label_resolution_bar[i] = labels[i].resolution_bar;
    }

    // Compute features
    // Step 1: raw features (n_bars x N_FEATURES)
    auto raw = compute_bar_features(bars, accums,
                                     builder.rth_open_ns(), builder.rth_close_ns());

    // Step 2: Drop realized-vol warmup rows (first REALIZED_VOL_WARMUP bars)
    int warmup = REALIZED_VOL_WARMUP;
    if (n_bars <= warmup) {
        // Not enough bars for feature computation
        day.n_usable = 0;
        return day;
    }

    int n_trimmed = n_bars - warmup;

    // Extract the trimmed portion of raw features (skip warmup rows)
    auto trim_begin = raw.begin() + warmup * N_FEATURES;
    std::vector<double> raw_trimmed(trim_begin, trim_begin + n_trimmed * N_FEATURES);

    // Step 3: Normalize
    auto normed = normalize_features(raw_trimmed, n_trimmed, N_FEATURES, 2000);

    // Step 4: Assemble lookback
    auto feat = assemble_lookback(normed, n_trimmed, N_FEATURES, lookback);

    if (feat.empty()) {
        day.n_usable = 0;
        return day;
    }

    int n_usable = n_trimmed - lookback + 1;
    day.n_usable = n_usable;
    day.features = std::move(feat);

    return day;
}

BarrierPrecomputedDay barrier_precompute(
    IMessageSource& source,
    const SessionConfig& cfg,
    int bar_size,
    int lookback,
    int a, int b, int t_max)
{
    BarBuilder builder(bar_size, cfg);

    Message msg;
    while (source.next(msg)) {
        builder.process(msg);
    }
    builder.flush();

    return barrier_precompute_impl(builder, bar_size, lookback, a, b, t_max);
}

BarrierPrecomputedDay barrier_precompute(
    const std::string& path,
    uint32_t instrument_id,
    int bar_size,
    int lookback,
    int a, int b, int t_max)
{
    DbnFileSource source(path, instrument_id);
    SessionConfig cfg = SessionConfig::default_rth();
    return barrier_precompute(source, cfg, bar_size, lookback, a, b, t_max);
}
