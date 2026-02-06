#include "lob/env.h"
#include <algorithm>
#include <cmath>

static constexpr int INITIAL_BOOK_MESSAGES = 10;

LOBEnv::LOBEnv(std::unique_ptr<IMessageSource> src, int steps_per_episode,
               RewardMode mode, float lambda, bool execution_cost,
               float participation_bonus)
    : src_(std::move(src)), steps_per_episode_(steps_per_episode),
      reward_calc_(mode, lambda), execution_cost_enabled_(execution_cost),
      participation_bonus_(participation_bonus) {}

LOBEnv::LOBEnv(std::unique_ptr<IMessageSource> src, SessionConfig cfg, int steps_per_episode,
               RewardMode mode, float lambda, bool execution_cost,
               float participation_bonus)
    : src_(std::move(src)), steps_per_episode_(steps_per_episode),
      session_filter_(SessionFilter(cfg)), warmup_messages_(cfg.warmup_messages),
      reward_calc_(mode, lambda), execution_cost_enabled_(execution_cost),
      participation_bonus_(participation_bonus) {}

StepResult LOBEnv::reset() {
    book_.reset();
    src_->reset();
    current_step_ = 0;
    position_ = 0.0f;
    prev_position_ = 0.0f;
    source_exhausted_ = false;
    session_done_ = false;
    has_pending_msg_ = false;
    last_ts_ns_ = 0;

    return session_filter_ ? reset_with_session() : reset_basic();
}

StepResult LOBEnv::reset_with_session() {
    Message m;
    std::vector<Message> pre_market_msgs;

    // Collect pre-market messages and find first RTH message
    while (src_->next(m)) {
        auto phase = session_filter_->classify(m.ts_ns);
        if (phase == SessionFilter::Phase::PreMarket) {
            pre_market_msgs.push_back(m);
        } else if (phase == SessionFilter::Phase::RTH) {
            pending_msg_ = m;
            has_pending_msg_ = true;
            break;
        } else {
            source_exhausted_ = true;
            break;
        }
    }

    if (!has_pending_msg_ && !source_exhausted_) {
        source_exhausted_ = true;
    }

    // Apply warmup messages to book
    if (warmup_messages_ < 0) {
        for (auto& pm : pre_market_msgs) {
            book_.apply(pm);
        }
    } else if (warmup_messages_ > 0) {
        int start = std::max(0, static_cast<int>(pre_market_msgs.size()) - warmup_messages_);
        for (int i = start; i < static_cast<int>(pre_market_msgs.size()); ++i) {
            book_.apply(pre_market_msgs[i]);
        }
    }

    if (source_exhausted_ || !has_pending_msg_) {
        prev_mid_ = book_.mid_price();
        return make_initial_result(/*done=*/true);
    }

    last_ts_ns_ = pending_msg_.ts_ns;
    book_.apply(pending_msg_);
    has_pending_msg_ = false;
    prev_mid_ = book_.mid_price();
    return make_initial_result(/*done=*/false);
}

StepResult LOBEnv::reset_basic() {
    Message m;
    for (int i = 0; i < INITIAL_BOOK_MESSAGES; ++i) {
        if (!src_->next(m)) {
            source_exhausted_ = true;
            break;
        }
        last_ts_ns_ = m.ts_ns;
        book_.apply(m);
    }

    prev_mid_ = book_.mid_price();
    return make_initial_result(/*done=*/false);
}

StepResult LOBEnv::make_initial_result(bool done) {
    StepResult result;
    result.obs = make_obs();
    result.reward = 0.0f;
    result.done = done;
    return result;
}

StepResult LOBEnv::step(int action) {
    // Clamp action to [0, 2]
    action = std::clamp(action, 0, 2);

    // Map action: 0 = short (-1), 1 = flat (0), 2 = long (+1)
    position_ = static_cast<float>(action - 1);

    // Advance one message
    advance_one_message();

    double current_mid = book_.mid_price();

    // Calculate reward using RewardCalculator
    float reward = reward_calc_.compute(position_, current_mid, prev_mid_);

    // Apply execution cost if enabled and position changed
    if (execution_cost_enabled_) {
        double spread = book_.spread();
        if (std::isfinite(spread)) {
            reward += reward_calc_.execution_cost(prev_position_, position_, spread);
        }
    }
    prev_position_ = position_;

    // Apply participation bonus
    if (participation_bonus_ != 0.0f) {
        reward += static_cast<float>(
            RewardCalculator::participation_bonus(position_, participation_bonus_));
    }

    prev_mid_ = current_mid;

    ++current_step_;

    bool done = false;
    if (session_filter_) {
        // Session-aware termination
        done = session_done_ || source_exhausted_;
        if (steps_per_episode_ > 0 && current_step_ >= steps_per_episode_) {
            done = true;
        }
    } else {
        done = (current_step_ >= steps_per_episode_) || source_exhausted_;
    }

    if (done && session_filter_) {
        // Flattening penalty: cost of crossing the spread to close the position
        double spread = book_.spread();
        if (std::isfinite(spread) && position_ != 0.0f) {
            reward += reward_calc_.flattening_penalty(position_, spread);
        }
        position_ = 0.0f;
    }

    StepResult result;
    result.obs = make_obs();
    result.reward = reward;
    result.done = done;
    return result;
}

std::vector<float> LOBEnv::make_obs() {
    float time_remaining = 0.5f;  // default for non-session mode
    if (session_filter_ && last_ts_ns_ > 0) {
        float progress = session_filter_->session_progress(last_ts_ns_);
        time_remaining = 1.0f - progress;
    }
    return feature_builder_.build(book_, position_, time_remaining);
}

void LOBEnv::advance_one_message() {
    if (has_pending_msg_) {
        book_.apply(pending_msg_);
        has_pending_msg_ = false;
        return;
    }

    Message m;
    if (src_->next(m)) {
        if (session_filter_) {
            auto phase = session_filter_->classify(m.ts_ns);
            if (phase == SessionFilter::Phase::PostMarket) {
                session_done_ = true;
                return;
            }
        }
        last_ts_ns_ = m.ts_ns;
        book_.apply(m);
    } else {
        source_exhausted_ = true;
    }
}
