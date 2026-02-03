#include "lob/reward.h"
#include "lob/book.h"
#include <cmath>

namespace lob {

RewardCalculator::RewardCalculator(RewardType type, double inventory_penalty)
    : type_(type), inventory_penalty_(inventory_penalty) {}

double RewardCalculator::calculate(
    int /*prev_position*/, double prev_pnl,
    int curr_position, double curr_pnl,
    const Book& /*book*/
) {
    double pnl_delta = curr_pnl - prev_pnl;

    switch (type_) {
        case RewardType::PnLDelta:
            return pnl_delta;

        case RewardType::PnLDeltaPenalized:
            // Penalize holding inventory
            return pnl_delta - inventory_penalty_ * std::abs(curr_position);

        default:
            return pnl_delta;
    }
}

}  // namespace lob
