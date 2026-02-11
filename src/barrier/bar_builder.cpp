#include "lob/barrier/bar_builder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

BarBuilder::BarBuilder(int bar_size, const SessionConfig& cfg)
    : bar_size_(bar_size), cfg_(cfg), filter_(cfg) {}

void BarBuilder::process(const Message& msg) {
    // Establish day base from first message timestamp
    if (!day_base_set_ && msg.ts_ns > 0) {
        static constexpr uint64_t NS_PER_DAY = 24ULL * 3600'000'000'000ULL;
        day_base_ = (msg.ts_ns / NS_PER_DAY) * NS_PER_DAY;
        day_base_set_ = true;
    }

    // Always apply to book for state tracking
    book_.apply(msg);

    // Classify the message phase
    auto phase = filter_.classify(msg.ts_ns);

    if (phase == SessionFilter::Phase::RTH) {
        // Ensure pending bar exists for any RTH event
        if (!has_pending_) {
            pending_bar_ = TradeBar{};
            pending_accum_ = BarBookAccum{};
            trade_count_ = 0;
            has_pending_ = true;
            wmid_first_set_ = false;
            vamp_mid_sampled_ = false;
        }

        // Track non-trade events in current accumulator
        if (msg.action != Message::Action::Trade) {
            // Sample spread on every RTH event
            sample_spread(pending_accum_);

            // Track OFI and cancel counts by action type
            if (msg.action == Message::Action::Add) {
                pending_accum_.total_add_volume += msg.qty;
                if (is_at_bbo(msg)) {
                    double signed_vol = (msg.side == Message::Side::Bid)
                        ? static_cast<double>(msg.qty)
                        : -static_cast<double>(msg.qty);
                    pending_accum_.ofi_signed_volume += signed_vol;
                }
            } else if (msg.action == Message::Action::Cancel) {
                if (msg.side == Message::Side::Bid) {
                    pending_accum_.bid_cancels++;
                } else {
                    pending_accum_.ask_cancels++;
                }
                pending_accum_.n_cancels++;
                if (is_at_bbo(msg)) {
                    double signed_vol = (msg.side == Message::Side::Bid)
                        ? -static_cast<double>(msg.qty)
                        : static_cast<double>(msg.qty);
                    pending_accum_.ofi_signed_volume += signed_vol;
                }
            }
            return;
        }

        // It's a trade during RTH

        // Capture wmid_first on first event in bar
        if (!wmid_first_set_) {
            sample_wmid_first(pending_accum_);
            wmid_first_set_ = true;
        }

        // Record trade data
        double price = msg.price;
        int qty = static_cast<int>(msg.qty);

        pending_bar_.trade_prices.push_back(price);
        pending_bar_.trade_sizes.push_back(qty);

        // Update OHLCV
        if (trade_count_ == 0) {
            pending_bar_.open = price;
            pending_bar_.high = price;
            pending_bar_.low = price;
            pending_bar_.t_start = msg.ts_ns;
        } else {
            if (price > pending_bar_.high) pending_bar_.high = price;
            if (price < pending_bar_.low) pending_bar_.low = price;
        }
        pending_bar_.close = price;
        pending_bar_.t_end = msg.ts_ns;
        pending_bar_.volume += qty;

        // Aggressor tracking: side indicates passive side
        // If passive side is Ask → buyer aggressor
        // If passive side is Bid → seller aggressor
        if (msg.side == Message::Side::Ask) {
            pending_accum_.buy_aggressor_vol += qty;
        } else {
            pending_accum_.sell_aggressor_vol += qty;
        }
        pending_accum_.n_trades++;

        // Sample spread on trade events too
        sample_spread(pending_accum_);

        trade_count_++;

        // Sample VAMP at midpoint of bar
        if (!vamp_mid_sampled_ && trade_count_ >= (bar_size_ + 1) / 2) {
            sample_vamp(pending_accum_.vamp_at_mid);
            vamp_mid_sampled_ = true;
        }

        // Check if bar is complete
        if (trade_count_ >= bar_size_) {
            emit_bar();
        }
    }
    // Pre-market and post-market: book is updated above, no bar creation
}

bool BarBuilder::flush() {
    if (!has_pending_ || trade_count_ == 0) {
        return false;
    }
    emit_bar();
    return true;
}

void BarBuilder::emit_bar() {
    // Compute VWAP
    double total_pv = 0.0;
    int total_v = 0;
    for (size_t i = 0; i < pending_bar_.trade_prices.size(); ++i) {
        total_pv += pending_bar_.trade_prices[i] * pending_bar_.trade_sizes[i];
        total_v += pending_bar_.trade_sizes[i];
    }
    pending_bar_.vwap = (total_v > 0) ? total_pv / total_v : 0.0;
    pending_bar_.bar_index = next_bar_index_++;

    // Snapshot book state into accum
    snapshot_accum(pending_accum_);

    // Sample wmid_end
    sample_wmid_end(pending_accum_);

    // Sample VAMP at end
    sample_vamp(pending_accum_.vamp_at_end);

    bars_.push_back(std::move(pending_bar_));
    accums_.push_back(std::move(pending_accum_));

    // Reset pending state
    pending_bar_ = TradeBar{};
    pending_accum_ = BarBookAccum{};
    trade_count_ = 0;
    has_pending_ = false;
    wmid_first_set_ = false;
    vamp_mid_sampled_ = false;
}

void BarBuilder::snapshot_accum(BarBookAccum& acc) {
    acc.bid_qty = book_.best_bid_qty();
    acc.ask_qty = book_.best_ask_qty();
    acc.total_bid_3 = book_.total_bid_depth(3);
    acc.total_ask_3 = book_.total_ask_depth(3);
    acc.total_bid_5 = book_.total_bid_depth(5);
    acc.total_ask_5 = book_.total_ask_depth(5);
    acc.total_bid_10 = book_.total_bid_depth(10);
    acc.total_ask_10 = book_.total_ask_depth(10);
}

void BarBuilder::sample_spread(BarBookAccum& acc) {
    double sp = book_.spread();
    if (std::isfinite(sp) && sp >= 0.0) {
        acc.spread_samples.push_back(sp);
    }
}

void BarBuilder::sample_wmid_first(BarBookAccum& acc) {
    double wmid = book_.weighted_mid();
    if (std::isfinite(wmid)) {
        acc.wmid_first = wmid;
    }
}

void BarBuilder::sample_wmid_end(BarBookAccum& acc) {
    double wmid = book_.weighted_mid();
    if (std::isfinite(wmid)) {
        acc.wmid_end = wmid;
    }
}

void BarBuilder::sample_vamp(double& target) {
    double v = book_.vamp(5);
    if (std::isfinite(v)) {
        target = v;
    }
}

bool BarBuilder::is_at_bbo(const Message& msg) const {
    if (msg.side == Message::Side::Bid) {
        double bb = book_.best_bid();
        return std::isfinite(bb) && std::abs(msg.price - bb) < 1e-9;
    } else {
        double ba = book_.best_ask();
        return std::isfinite(ba) && std::abs(msg.price - ba) < 1e-9;
    }
}
