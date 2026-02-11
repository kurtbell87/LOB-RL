#pragma once
#include "lob/barrier/trade_bar.h"
#include "lob/book.h"
#include "lob/message.h"
#include "lob/session.h"
#include <cstdint>
#include <vector>

class BarBuilder {
public:
    BarBuilder(int bar_size, const SessionConfig& cfg);

    void process(const Message& msg);
    bool flush();

    const std::vector<TradeBar>& bars() const { return bars_; }
    const std::vector<BarBookAccum>& accums() const { return accums_; }

    uint64_t rth_open_ns() const { return day_base_ + cfg_.rth_open_ns; }
    uint64_t rth_close_ns() const { return day_base_ + cfg_.rth_close_ns; }

private:
    void emit_bar();
    void snapshot_accum(BarBookAccum& acc);
    void sample_spread(BarBookAccum& acc);
    void sample_wmid_first(BarBookAccum& acc);
    void sample_wmid_end(BarBookAccum& acc);
    void sample_vamp(double& target);
    bool is_at_bbo(const Message& msg) const;

    int bar_size_;
    SessionConfig cfg_;
    SessionFilter filter_;
    Book book_;

    // Current pending bar state
    TradeBar pending_bar_;
    BarBookAccum pending_accum_;
    int trade_count_ = 0;       // trades in current pending bar
    bool has_pending_ = false;
    bool wmid_first_set_ = false;
    bool vamp_mid_sampled_ = false;

    // Day base (epoch ns for midnight of the day)
    uint64_t day_base_ = 0;
    bool day_base_set_ = false;

    // Completed bars
    std::vector<TradeBar> bars_;
    std::vector<BarBookAccum> accums_;
    int next_bar_index_ = 0;
};
