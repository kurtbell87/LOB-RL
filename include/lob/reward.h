#pragma once
#include <cmath>

enum class RewardMode {
    PnLDelta,
    PnLDeltaPenalized
};

class RewardCalculator {
public:
    RewardCalculator() : mode_(RewardMode::PnLDelta), lambda_(0.0f) {}
    explicit RewardCalculator(RewardMode mode, float lambda = 0.0f)
        : mode_(mode), lambda_(lambda) {}

    RewardMode mode() const { return mode_; }
    float lambda() const { return lambda_; }

    float compute(float position, double current_mid, double prev_mid) const {
        if (!std::isfinite(current_mid) || !std::isfinite(prev_mid)) {
            return 0.0f;
        }
        float pnl = position * static_cast<float>(current_mid - prev_mid);
        if (mode_ == RewardMode::PnLDeltaPenalized) {
            pnl -= lambda_ * std::abs(position);
        }
        return pnl;
    }

    float flattening_penalty(float position, double spread) const {
        return -std::abs(position) * half_spread(spread);
    }

    float execution_cost(float old_pos, float new_pos, double spread) const {
        float delta = std::abs(new_pos - old_pos);
        return -delta * half_spread(spread);
    }

    static double participation_bonus(double position, double bonus_rate) {
        return bonus_rate * std::abs(position);
    }

private:
    static float half_spread(double spread) {
        return static_cast<float>(spread * 0.5);
    }

    RewardMode mode_;
    float lambda_;
};
