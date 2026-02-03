#pragma once
#include "lob/source.h"
#include <random>
#include <vector>

namespace lob {

class SyntheticSource : public IMessageSource {
public:
    explicit SyntheticSource(uint32_t seed = 42, uint64_t num_messages = 1000);

    bool has_next() const override;
    MBOMessage next() override;
    void reset() override;
    uint64_t message_count() const override;

private:
    void generate_messages();
    MBOMessage generate_add(Side side);
    MBOMessage generate_cancel();
    MBOMessage generate_modify();
    MBOMessage generate_trade();

    uint32_t seed_;
    uint64_t num_messages_;
    std::mt19937_64 rng_;
    std::vector<MBOMessage> messages_;
    size_t current_idx_ = 0;

    // State for generation
    uint64_t next_order_id_ = 1;
    uint64_t timestamp_ns_ = 0;
    int64_t base_price_ = 5000'000'000'000;  // $5000 * 1e9 (like /MES)
    std::vector<uint64_t> active_bid_orders_;
    std::vector<uint64_t> active_ask_orders_;
};

}  // namespace lob
