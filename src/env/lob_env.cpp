#include "lob/env.h"
#include "lob/book.h"
#include "feature_builder.h"
#include "lob/reward.h"

namespace lob {

struct LOBEnv::Impl {
    EnvConfig config;
    std::unique_ptr<IMessageSource> source;
    Book book;
    FeatureBuilder feature_builder;
    RewardCalculator reward_calc;

    int position = 0;
    double pnl = 0.0;
    double entry_price = 0.0;
    uint64_t current_timestamp = 0;
    uint64_t session_start_ns = 0;
    uint64_t session_end_ns = 0;
    int messages_processed = 0;

    Impl(EnvConfig cfg)
        : config(std::move(cfg))
        , feature_builder(config.book_depth)
        , reward_calc(config.reward_type, config.inventory_penalty) {}

    double mark_to_market() const {
        if (position == 0) {
            return pnl;
        }

        // Value position at mid price
        double mid = static_cast<double>(book.mid_price()) * 1e-9;
        if (mid == 0) {
            return pnl;
        }

        double unrealized = position * (mid - entry_price);
        return pnl + unrealized;
    }

    double time_remaining() const {
        if (session_end_ns <= session_start_ns) {
            return 1.0;
        }
        if (current_timestamp <= session_start_ns) {
            return 1.0;
        }
        if (current_timestamp >= session_end_ns) {
            return 0.0;
        }

        uint64_t elapsed = current_timestamp - session_start_ns;
        uint64_t duration = session_end_ns - session_start_ns;
        return 1.0 - static_cast<double>(elapsed) / static_cast<double>(duration);
    }

    void execute_action(int action) {
        int target_position = action - 1;  // 0->-1 (short), 1->0 (flat), 2->1 (long)

        if (target_position == position) {
            return;  // No change needed
        }

        double mid = static_cast<double>(book.mid_price()) * 1e-9;
        if (mid == 0) {
            return;  // Can't trade without a price
        }

        // Close existing position if any
        if (position != 0) {
            double close_price;
            if (position > 0) {
                // Selling: get bid price
                close_price = static_cast<double>(book.bid(0).price) * 1e-9;
            } else {
                // Buying to cover: get ask price
                close_price = static_cast<double>(book.ask(0).price) * 1e-9;
            }

            if (close_price == 0) {
                close_price = mid;  // Fallback to mid
            }

            // Realize PnL
            pnl += position * (close_price - entry_price);
            position = 0;
        }

        // Open new position if target is not flat
        if (target_position != 0) {
            if (target_position > 0) {
                // Buying: pay ask price
                entry_price = static_cast<double>(book.ask(0).price) * 1e-9;
            } else {
                // Selling short: get bid price
                entry_price = static_cast<double>(book.bid(0).price) * 1e-9;
            }

            if (entry_price == 0) {
                entry_price = mid;  // Fallback to mid
            }

            position = target_position;
        }
    }
};

LOBEnv::LOBEnv(EnvConfig config, std::unique_ptr<IMessageSource> source)
    : impl_(std::make_unique<Impl>(std::move(config))) {
    impl_->source = std::move(source);
}

LOBEnv::~LOBEnv() = default;

StepResult LOBEnv::reset() {
    impl_->book.clear();
    impl_->source->reset();
    impl_->position = 0;
    impl_->pnl = 0.0;
    impl_->entry_price = 0.0;
    impl_->current_timestamp = 0;
    impl_->messages_processed = 0;

    // Process initial messages to build book state
    int warmup_messages = std::min(100, impl_->config.trades_per_step);
    while (impl_->source->has_next() && impl_->messages_processed < warmup_messages) {
        MBOMessage msg = impl_->source->next();
        impl_->book.apply(msg);
        impl_->current_timestamp = msg.timestamp_ns;
        impl_->messages_processed++;
    }

    // Set session boundaries based on first timestamp
    impl_->session_start_ns = impl_->current_timestamp;
    // Assume 6.5 hour session (US RTH)
    impl_->session_end_ns = impl_->session_start_ns + 6ULL * 3600 * 1'000'000'000ULL + 1800 * 1'000'000'000ULL;

    Observation obs = impl_->feature_builder.build(
        impl_->book, impl_->position, impl_->time_remaining()
    );

    return StepResult{
        std::move(obs),
        0.0,
        false,
        impl_->position,
        impl_->pnl,
        impl_->current_timestamp
    };
}

StepResult LOBEnv::step(int action) {
    double prev_pnl = impl_->mark_to_market();
    int prev_position = impl_->position;

    // Execute the action
    impl_->execute_action(action);

    // Process messages
    int messages_this_step = 0;
    while (impl_->source->has_next() && messages_this_step < impl_->config.trades_per_step) {
        MBOMessage msg = impl_->source->next();
        impl_->book.apply(msg);
        impl_->current_timestamp = msg.timestamp_ns;
        messages_this_step++;
        impl_->messages_processed++;
    }

    // Check if episode is done
    bool done = !impl_->source->has_next() || impl_->time_remaining() <= 0.0;

    // Force flat at end of episode
    if (done && impl_->position != 0) {
        impl_->execute_action(1);  // Go flat
    }

    // Calculate reward
    double curr_pnl = impl_->mark_to_market();
    double reward = impl_->reward_calc.calculate(
        prev_position, prev_pnl,
        impl_->position, curr_pnl,
        impl_->book
    );

    Observation obs = impl_->feature_builder.build(
        impl_->book, impl_->position, impl_->time_remaining()
    );

    return StepResult{
        std::move(obs),
        reward,
        done,
        impl_->position,
        curr_pnl,
        impl_->current_timestamp
    };
}

int LOBEnv::observation_size() const {
    return impl_->feature_builder.observation_size();
}

const EnvConfig& LOBEnv::config() const {
    return impl_->config;
}

}  // namespace lob
