#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "lob/env.h"
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// LOBEnv: Observation size is 44 (replaces old 4-float obs)
// ===========================================================================

TEST(LOBEnvObs44, ResetReturns44FloatObservation) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();
    EXPECT_EQ(result.obs.size(), 44u)
        << "make_obs() should now return 44 floats";
}

TEST(LOBEnvObs44, StepReturns44FloatObservation) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();
    StepResult result = env.step(1);
    EXPECT_EQ(result.obs.size(), 44u);
}

TEST(LOBEnvObs44, AllObsValuesFiniteAfterReset) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();
    EXPECT_ALL_FINITE(result.obs);
}

TEST(LOBEnvObs44, AllObsValuesFiniteThroughEpisode) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 20);
    env.reset();

    for (int s = 0; s < 20; ++s) {
        StepResult result = env.step(s % 3);
        if (result.done) break;
        EXPECT_ALL_FINITE(result.obs);
    }
}

// ===========================================================================
// LOBEnv: Position is at index 43
// ===========================================================================

TEST(LOBEnvObs44, PositionAtIndex43AfterReset) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();
    EXPECT_FLOAT_EQ(result.obs[43], 0.0f)
        << "Initial position should be 0 at obs[43]";
}

TEST(LOBEnvObs44, ActionZeroSetsPositionAtIndex43) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();
    StepResult result = env.step(0);  // short
    EXPECT_FLOAT_EQ(result.obs[43], -1.0f);
}

TEST(LOBEnvObs44, ActionOneSetsPositionAtIndex43) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();
    StepResult result = env.step(1);  // flat
    EXPECT_FLOAT_EQ(result.obs[43], 0.0f);
}

TEST(LOBEnvObs44, ActionTwoSetsPositionAtIndex43) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();
    StepResult result = env.step(2);  // long
    EXPECT_FLOAT_EQ(result.obs[43], 1.0f);
}

// ===========================================================================
// LOBEnv: Time remaining at index 42
// ===========================================================================

TEST(LOBEnvObs44, TimeRemainingIsHalfWithoutSession) {
    // Without session config, time_remaining should be 0.5
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();
    EXPECT_FLOAT_EQ(result.obs[42], 0.5f)
        << "Without session, time_remaining should be 0.5";
}

TEST(LOBEnvObs44, TimeRemainingWithSessionReflectsProgress) {
    // Create a session-aware env with scripted messages during RTH
    uint64_t rth_open = RTH_OPEN_NS;
    uint64_t rth_close = RTH_CLOSE_NS;

    // Create messages: pre-market warmup + RTH
    std::vector<Message> msgs;
    uint64_t oid = 1;

    // Pre-market messages to build the book
    for (int i = 0; i < 5; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                100.0 - i * 0.25, 10, DAY_BASE_NS + rth_open - NS_PER_HOUR + i * NS_PER_MIN));
    }
    for (int i = 0; i < 5; ++i) {
        msgs.push_back(make_msg(oid++, Message::Side::Ask, Message::Action::Add,
                                100.25 + i * 0.25, 10, DAY_BASE_NS + rth_open - NS_PER_HOUR + (5 + i) * NS_PER_MIN));
    }

    // RTH messages at various times through the session
    // Message at RTH open (progress = 0)
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            100.0, 5, DAY_BASE_NS + rth_open));
    // Message at mid-session
    uint64_t mid_time = rth_open + (rth_close - rth_open) / 2;
    msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                            100.0, 5, DAY_BASE_NS + mid_time));
    // More messages to keep the episode going
    for (int i = 0; i < 10; ++i) {
        uint64_t t = rth_open + (rth_close - rth_open) * (i + 1) / 12;
        msgs.push_back(make_msg(oid++, Message::Side::Bid, Message::Action::Add,
                                100.0 + 0.01 * i, 5, DAY_BASE_NS + t));
    }

    SessionConfig cfg;
    cfg.rth_open_ns = rth_open;
    cfg.rth_close_ns = rth_close;
    cfg.warmup_messages = -1;

    auto env = LOBEnv(std::make_unique<ScriptedSource>(msgs), cfg, 50);
    StepResult result = env.reset();

    // time_remaining = 1.0 - session_progress
    // At or near RTH open, time_remaining should be close to 1.0
    EXPECT_GT(result.obs[42], 0.5f)
        << "Near session open, time_remaining should be > 0.5";
    EXPECT_LE(result.obs[42], 1.0f)
        << "time_remaining should be <= 1.0";
}

// ===========================================================================
// LOBEnv: Spread at index 40 is normalized
// ===========================================================================

TEST(LOBEnvObs44, SpreadAtIndex40IsNormalized) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();

    // Spread should be > 0 and small (normalized by mid, not raw)
    // With SyntheticSource, raw spread is 0.50. Mid is ~100.
    // Normalized spread = 0.50 / ~100 ≈ 0.005
    EXPECT_GT(result.obs[40], 0.0f) << "Spread should be positive";
    EXPECT_LT(result.obs[40], 0.1f) << "Normalized spread should be small";
}

// ===========================================================================
// LOBEnv: Imbalance at index 41 is in [-1, 1]
// ===========================================================================

TEST(LOBEnvObs44, ImbalanceAtIndex41InRange) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();

    for (int s = 0; s < 20; ++s) {
        StepResult result = env.step(s % 3);
        if (result.done) break;
        EXPECT_GE(result.obs[41], -1.0f)
            << "Imbalance at step " << s << " should be >= -1";
        EXPECT_LE(result.obs[41], 1.0f)
            << "Imbalance at step " << s << " should be <= 1";
    }
}

// ===========================================================================
// LOBEnv: Bid prices at indices 0-9 are <= 0
// ===========================================================================

TEST(LOBEnvObs44, BidPricesNonPositiveAfterReset) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();

    for (int i = 0; i < 10; ++i) {
        EXPECT_LE(result.obs[i], 0.0f)
            << "Bid price obs[" << i << "] should be <= 0 (relative to mid)";
    }
}

// ===========================================================================
// LOBEnv: Ask prices at indices 20-29 are >= 0
// ===========================================================================

TEST(LOBEnvObs44, AskPricesNonNegativeAfterReset) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();

    for (int i = 20; i < 30; ++i) {
        EXPECT_GE(result.obs[i], 0.0f)
            << "Ask price obs[" << i << "] should be >= 0 (relative to mid)";
    }
}

// ===========================================================================
// LOBEnv: Sizes at indices 10-19 and 30-39 in [0, 1]
// ===========================================================================

TEST(LOBEnvObs44, SizesInRange01) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    StepResult result = env.reset();

    for (int i = 10; i < 20; ++i) {
        EXPECT_GE(result.obs[i], 0.0f);
        EXPECT_LE(result.obs[i], 1.0f);
    }
    for (int i = 30; i < 40; ++i) {
        EXPECT_GE(result.obs[i], 0.0f);
        EXPECT_LE(result.obs[i], 1.0f);
    }
}

// ===========================================================================
// LOBEnv: Deterministic resets produce same 44-float obs
// ===========================================================================

TEST(LOBEnvObs44, DeterministicResetsProduceSameObs) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>());

    StepResult r1 = env.reset();
    StepResult r2 = env.reset();

    ASSERT_EQ(r1.obs.size(), 44u);
    ASSERT_EQ(r2.obs.size(), 44u);
    for (int i = 0; i < 44; ++i) {
        EXPECT_FLOAT_EQ(r1.obs[i], r2.obs[i])
            << "Obs mismatch at index " << i << " after re-reset";
    }
}
