#pragma once

#include "lob/book.h"
#include "lob/message.h"
#include <algorithm>
#include <vector>

// Apply warmup messages to the book before RTH processing.
//   warmup_count < 0 : apply ALL pre-market messages
//   warmup_count > 0 : apply last N pre-market messages
//   warmup_count == 0: skip all pre-market messages
inline void apply_warmup(Book& book, const std::vector<Message>& pre_market,
                         int warmup_count) {
    if (warmup_count < 0) {
        for (auto& m : pre_market) {
            book.apply(m);
        }
    } else if (warmup_count > 0) {
        int start = std::max(0, static_cast<int>(pre_market.size()) - warmup_count);
        for (int i = start; i < static_cast<int>(pre_market.size()); ++i) {
            book.apply(pre_market[i]);
        }
    }
}
