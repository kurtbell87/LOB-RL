#pragma once
#include "lob/source.h"
#include "lob/reward.h"
#include <vector>
#include <memory>
#include <string>

namespace lob {

struct Session {
    int start_hour_utc;
    int start_minute_utc;
    int end_hour_utc;
    int end_minute_utc;
};

namespace sessions {
    constexpr Session US_RTH_EST = {14, 30, 21, 0};  // EST (no DST)
    constexpr Session US_RTH_EDT = {13, 30, 20, 0};  // EDT (DST)
}

struct EnvConfig {
    std::string data_path;
    int book_depth = 10;
    int trades_per_step = 100;
    RewardType reward_type = RewardType::PnLDelta;
    double inventory_penalty = 0.0;
    Session session = sessions::US_RTH_EST;
};

struct Observation {
    std::vector<float> data;
    static int size(int book_depth) { return 4 * book_depth + 4; }
};

struct StepResult {
    Observation obs;
    double reward;
    bool done;
    int position;
    double pnl;
    uint64_t timestamp_ns;
};

class LOBEnv {
public:
    LOBEnv(EnvConfig config, std::unique_ptr<IMessageSource> source);
    ~LOBEnv();

    StepResult reset();
    StepResult step(int action);  // 0=short, 1=flat, 2=long

    int observation_size() const;
    static constexpr int action_size() { return 3; }
    const EnvConfig& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lob
