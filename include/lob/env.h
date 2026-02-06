#pragma once
#include "lob/book.h"
#include "lob/source.h"
#include "lob/session.h"
#include "lob/feature_builder.h"
#include "lob/reward.h"
#include <vector>
#include <memory>
#include <cstdint>
#include <optional>

struct StepResult {
    std::vector<float> obs;
    float reward = 0.0f;
    bool done = false;
};

class LOBEnv {
public:
    explicit LOBEnv(std::unique_ptr<IMessageSource> src, int steps_per_episode = 50,
                    RewardMode mode = RewardMode::PnLDelta, float lambda = 0.0f,
                    bool execution_cost = false);
    LOBEnv(std::unique_ptr<IMessageSource> src, SessionConfig cfg, int steps_per_episode,
           RewardMode mode = RewardMode::PnLDelta, float lambda = 0.0f,
           bool execution_cost = false);

    StepResult reset();
    StepResult step(int action);
    int steps_per_episode() const { return steps_per_episode_; }

private:
    std::vector<float> make_obs();
    void advance_one_message();
    StepResult reset_with_session();
    StepResult reset_basic();
    StepResult make_initial_result(bool done);

    std::unique_ptr<IMessageSource> src_;
    Book book_;
    int steps_per_episode_;
    int current_step_ = 0;
    float position_ = 0.0f;
    double prev_mid_ = 0.0;
    bool source_exhausted_ = false;

    // Session support
    std::optional<SessionFilter> session_filter_;
    int warmup_messages_ = -1;
    bool session_done_ = false;
    Message pending_msg_;
    bool has_pending_msg_ = false;

    // 44-float observation support
    FeatureBuilder feature_builder_;
    uint64_t last_ts_ns_ = 0;

    // Reward calculation
    RewardCalculator reward_calc_;
    bool execution_cost_enabled_ = false;
    float prev_position_ = 0.0f;
};
