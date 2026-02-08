#pragma once
#include "lob/book.h"
#include "lob/feature_builder.h"
#include "lob/session.h"
#include "lob/source.h"
#include <vector>
#include <string>
#include <cstdint>

struct PrecomputedDay {
    int num_steps = 0;
    std::vector<float> obs;     // flat: num_steps * POSITION floats (no position)
    std::vector<double> mid;    // num_steps mid prices
    std::vector<double> spread; // num_steps spreads
};

PrecomputedDay precompute(IMessageSource& source, const SessionConfig& cfg);
PrecomputedDay precompute(const std::string& path, const SessionConfig& cfg,
                          uint32_t instrument_id = 0);
