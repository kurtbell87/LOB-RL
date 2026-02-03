#pragma once

namespace lob {

enum class RewardType {
    PnLDelta,
    PnLDeltaPenalized
};

class Book;  // forward decl

class RewardCalculator {
public:
    RewardCalculator(RewardType type, double inventory_penalty = 0.0);

    double calculate(
        int prev_position, double prev_pnl,
        int curr_position, double curr_pnl,
        const Book& book
    );

private:
    RewardType type_;
    double inventory_penalty_;
};

}  // namespace lob
