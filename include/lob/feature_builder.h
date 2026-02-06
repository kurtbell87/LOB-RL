#pragma once
#include "lob/book.h"
#include <vector>
#include <cstdint>

class FeatureBuilder {
public:
    static constexpr int DEPTH = 10;

    // Observation layout indices — derived from DEPTH so changing it
    // auto-adjusts all offsets and OBS_SIZE.
    static constexpr int BID_PRICE  = 0;                 // [0, DEPTH)
    static constexpr int BID_SIZE   = DEPTH;             // [DEPTH, 2*DEPTH)
    static constexpr int ASK_PRICE  = 2 * DEPTH;         // [2*DEPTH, 3*DEPTH)
    static constexpr int ASK_SIZE   = 3 * DEPTH;         // [3*DEPTH, 4*DEPTH)
    static constexpr int SPREAD     = 4 * DEPTH;         // spread / mid
    static constexpr int IMBALANCE  = 4 * DEPTH + 1;     // (bid_top - ask_top) / (bid_top + ask_top)
    static constexpr int TIME_LEFT  = 4 * DEPTH + 2;     // time remaining in session
    static constexpr int POSITION   = 4 * DEPTH + 3;     // agent position
    static constexpr int OBS_SIZE   = 4 * DEPTH + 4;

    std::vector<float> build(const Book& book, float position, float time_remaining) const;
};
