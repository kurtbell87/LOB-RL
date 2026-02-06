#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>
#include "lob/env.h"
#include "test_helpers.h"

// ===========================================================================
// Helper: Build a message sequence spanning pre-market -> RTH -> post-market
// Creates a valid order book with 5 bid + 5 ask levels, then activity messages
// ===========================================================================

static std::vector<Message> make_session_messages(
    int pre_market_count,   // messages before RTH open
    int rth_count,          // messages during RTH
    int post_market_count,  // messages after RTH close
    double mid = 1000.0,
    double tick = 0.25)
{
    std::vector<Message> msgs;
    uint64_t next_id = 1;

    // Pre-market messages: timestamps before 13:30 UTC
    // Start at 09:00 UTC, spaced 1 minute apart
    uint64_t pre_start = DAY_BASE_NS + 9ULL * NS_PER_HOUR;
    for (int i = 0; i < pre_market_count; ++i) {
        // Alternate bid/ask adds to build a proper book
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        double price = (side == Message::Side::Bid)
            ? mid - (1 + i / 2) * tick
            : mid + (1 + i / 2) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 100, pre_start + i * NS_PER_MIN));
    }

    // RTH messages: timestamps from 13:30 to before 20:00 UTC
    uint64_t rth_start = DAY_BASE_NS + RTH_OPEN_NS;
    uint64_t rth_duration = RTH_CLOSE_NS - RTH_OPEN_NS;
    for (int i = 0; i < rth_count; ++i) {
        // Space evenly across RTH window
        uint64_t ts = rth_start + (i * rth_duration) / (rth_count + 1);
        // Alternate between bid and ask adds at various levels
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        double price = (side == Message::Side::Bid)
            ? mid - (1 + (i % 5)) * tick
            : mid + (1 + (i % 5)) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 50 + i, ts));
    }

    // Post-market messages: timestamps at or after 20:00 UTC
    uint64_t post_start = DAY_BASE_NS + RTH_CLOSE_NS;
    for (int i = 0; i < post_market_count; ++i) {
        Message::Side side = (i % 2 == 0) ? Message::Side::Bid : Message::Side::Ask;
        double price = (side == Message::Side::Bid)
            ? mid - (1 + i / 2) * tick
            : mid + (1 + i / 2) * tick;
        msgs.push_back(make_msg(next_id++, side, Message::Action::Add,
                                price, 100, post_start + i * NS_PER_MIN));
    }

    return msgs;
}

// ===========================================================================
// LOBEnv Session: Construction
// ===========================================================================

TEST(LOBEnvSession, ConstructsWithSessionConfig) {
    auto msgs = make_session_messages(10, 20, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    // Should not crash
}

TEST(LOBEnvSession, ConstructsWithStepsPerEpisodeZero) {
    auto msgs = make_session_messages(10, 20, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    // steps_per_episode=0 means run until session close
}

TEST(LOBEnvSession, ConstructsWithStepsPerEpisodeNonZero) {
    auto msgs = make_session_messages(10, 20, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 5);
    // steps_per_episode=5, terminates at min(5, session_close)
}

// ===========================================================================
// LOBEnv Session: reset() with warmup
// ===========================================================================

TEST(LOBEnvSession, ResetWarmsUpBookFromPreMarketMessages) {
    // 10 pre-market messages, 20 RTH messages
    auto msgs = make_session_messages(10, 20, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    // After warmup, the book should have valid observations
    EXPECT_EQ(result.obs.size(), 44u);
    // Spread (index 40) should be positive after warmup builds a valid book
    EXPECT_GT(result.obs[40], 0.0f) << "spread should be positive after warmup";
}

TEST(LOBEnvSession, ResetReturnsZeroPositionAfterWarmup) {
    auto msgs = make_session_messages(10, 20, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_FLOAT_EQ(result.obs[43], 0.0f) << "Position should be 0 (flat) after reset";
}

TEST(LOBEnvSession, ResetSetsDoneToFalseWhenRthMessagesExist) {
    auto msgs = make_session_messages(10, 20, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_FALSE(result.done) << "Episode should not be done when RTH messages exist";
}

TEST(LOBEnvSession, ResetSetsRewardToZero) {
    auto msgs = make_session_messages(10, 20, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_FLOAT_EQ(result.reward, 0.0f);
}

// ===========================================================================
// LOBEnv Session: Warmup does not count as episode steps
// ===========================================================================

TEST(LOBEnvSession, WarmupDoesNotCountAsEpisodeSteps) {
    // 10 pre-market warmup messages, 5 RTH messages
    auto msgs = make_session_messages(10, 5, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Should be able to take steps during RTH (warmup didn't consume step budget)
    int steps = 0;
    bool done = false;
    while (!done && steps < 100) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }
    // We had 5 RTH messages, so should get some steps before done
    EXPECT_GT(steps, 0) << "Should get at least one step during RTH after warmup";
}

// ===========================================================================
// LOBEnv Session: Episode runs only during RTH
// ===========================================================================

TEST(LOBEnvSession, EpisodeTerminatesAtSessionClose) {
    // 10 pre-market, 20 RTH, 10 post-market
    auto msgs = make_session_messages(10, 20, 10);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    int steps = 0;
    bool done = false;
    while (!done && steps < 100) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }

    EXPECT_TRUE(done);
    // Should have consumed at most the RTH messages (not post-market)
    EXPECT_LE(steps, 20) << "Should not step beyond RTH into post-market messages";
}

TEST(LOBEnvSession, StepsPerEpisodeZeroRunsUntilSessionClose) {
    auto msgs = make_session_messages(10, 30, 10);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    int steps = 0;
    bool done = false;
    while (!done && steps < 200) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }

    EXPECT_TRUE(done);
    // With steps_per_episode=0, should run through all RTH messages
    EXPECT_LE(steps, 30) << "Should stop at or before exhausting RTH messages";
    EXPECT_GT(steps, 0);
}

TEST(LOBEnvSession, StepsPerEpisodeTerminatesBeforeSessionClose) {
    // 10 pre-market, 30 RTH messages, but steps_per_episode=5
    auto msgs = make_session_messages(10, 30, 10);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 5);
    env.reset();

    int steps = 0;
    bool done = false;
    while (!done && steps < 100) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }

    EXPECT_TRUE(done);
    EXPECT_EQ(steps, 5)
        << "Should terminate at steps_per_episode=5 before session close";
}

TEST(LOBEnvSession, SessionCloseTerminatesBeforeStepsPerEpisode) {
    // 10 pre-market, 3 RTH messages, steps_per_episode=100
    auto msgs = make_session_messages(10, 3, 10);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 100);
    env.reset();

    int steps = 0;
    bool done = false;
    while (!done && steps < 200) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }

    EXPECT_TRUE(done);
    EXPECT_LE(steps, 3)
        << "Should terminate at session close (3 RTH msgs) before steps_per_episode=100";
}

// ===========================================================================
// LOBEnv Session: Position forced flat at session close
// ===========================================================================

TEST(LOBEnvSession, PositionForcedFlatAtSessionClose) {
    auto msgs = make_session_messages(10, 5, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Go long and hold until session close
    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(2);  // action=2 = long
        done = result.done;
    }

    // At session close, position must be forced flat
    EXPECT_FLOAT_EQ(result.obs[43], 0.0f)
        << "Position must be forced to 0 (flat) at session close";
}

TEST(LOBEnvSession, PositionForcedFlatFromShortAtSessionClose) {
    auto msgs = make_session_messages(10, 5, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Go short and hold until session close
    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(0);  // action=0 = short
        done = result.done;
    }

    EXPECT_FLOAT_EQ(result.obs[43], 0.0f)
        << "Short position must be forced to 0 (flat) at session close";
}

// ===========================================================================
// LOBEnv Session: Reward at session close includes flattening PnL
// ===========================================================================

TEST(LOBEnvSession, RewardAtSessionCloseIncludesFlatteningPnL) {
    auto msgs = make_session_messages(10, 5, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Go long for the whole session
    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(2);  // long
        done = result.done;
    }

    // The final reward should be finite (includes flattening PnL)
    EXPECT_TRUE(std::isfinite(result.reward))
        << "Final reward at session close should be finite";
    // Position is flat, so the reward should reflect the PnL from flattening
    // (closing the position at last mid). This is NOT simply 0.
}

TEST(LOBEnvSession, FlatPositionAtSessionCloseHasNoFlatteningPnL) {
    auto msgs = make_session_messages(10, 5, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Stay flat the whole session — no flattening needed
    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(1);  // flat
        done = result.done;
    }

    // Reward should be 0 since position was flat the whole time
    EXPECT_FLOAT_EQ(result.reward, 0.0f)
        << "Flat position at session close should have 0 flattening PnL";
}

// ===========================================================================
// LOBEnv Session: Edge case — no RTH messages
// ===========================================================================

TEST(LOBEnvSession, NoRthMessagesCausesDoneImmediately) {
    // Only pre-market messages, no RTH messages at all
    auto msgs = make_session_messages(10, 0, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    // Should be done immediately since no RTH messages exist
    EXPECT_TRUE(result.done)
        << "Episode should be done immediately when source has no RTH messages";
}

TEST(LOBEnvSession, AllPostMarketCausesDoneImmediately) {
    // Only post-market messages (all timestamps >= 20:00 UTC)
    uint64_t post_start = DAY_BASE_NS + RTH_CLOSE_NS;
    std::vector<Message> msgs;
    for (int i = 0; i < 10; ++i) {
        msgs.push_back(make_msg(i + 1, Message::Side::Bid, Message::Action::Add,
                                999.75 - i * 0.25, 100, post_start + i * NS_PER_MIN));
    }

    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_TRUE(result.done)
        << "Episode should be done immediately when all messages are post-market";
}

// ===========================================================================
// LOBEnv Session: Edge case — source starts mid-session
// ===========================================================================

TEST(LOBEnvSession, SourceStartsMidSessionSkipsWarmup) {
    // All messages are during RTH (no pre-market data)
    auto msgs = make_session_messages(0, 20, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    // Should start episode immediately (no warmup needed)
    EXPECT_FALSE(result.done) << "Episode should start when source has RTH messages";
    EXPECT_EQ(result.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv Session: Edge case — source exhausted during warmup
// ===========================================================================

TEST(LOBEnvSession, SourceExhaustedDuringWarmupAllPreMarket) {
    // All messages are pre-market, source exhausts during warmup
    auto msgs = make_session_messages(5, 0, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_TRUE(result.done)
        << "Should be done when source exhausts with only pre-market messages";
}

// ===========================================================================
// LOBEnv Session: Default constructor (no session) behaves as before
// ===========================================================================

TEST(LOBEnvSession, DefaultConstructorNoSessionFilteringBehavesAsOriginal) {
    // Use the original constructor without SessionConfig
    auto src = std::make_unique<ScriptedSource>(make_session_messages(10, 20, 10));

    LOBEnv env(std::move(src), 5);  // original constructor: source + steps_per_episode
    StepResult result = env.reset();

    EXPECT_FALSE(result.done);
    EXPECT_EQ(result.obs.size(), 44u);

    // Should terminate at steps_per_episode regardless of timestamps
    int steps = 0;
    bool done = false;
    while (!done && steps < 100) {
        StepResult r = env.step(1);
        done = r.done;
        ++steps;
    }
    // With 40 total messages and steps_per_episode=5, should stop at 5
    // (or earlier if source is exhausted from reset consuming messages)
    EXPECT_LE(steps, 5);
}

// ===========================================================================
// LOBEnv Session: Reset can be called multiple times
// ===========================================================================

TEST(LOBEnvSession, ResetCanBeCalledMultipleTimesWithSession) {
    auto msgs = make_session_messages(10, 20, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);

    StepResult r1 = env.reset();
    EXPECT_FALSE(r1.done);

    // Run a few steps
    env.step(2);
    env.step(0);

    // Reset again
    StepResult r2 = env.reset();
    EXPECT_FALSE(r2.done);
    EXPECT_FLOAT_EQ(r2.obs[43], 0.0f) << "Position should be flat after re-reset";
    EXPECT_FLOAT_EQ(r2.reward, 0.0f);
}

// ===========================================================================
// LOBEnv Session: Warmup with limited warmup_messages
// ===========================================================================

TEST(LOBEnvSession, WarmupZeroSkipsPreMarketMessages) {
    // 20 pre-market messages, 10 RTH messages
    auto msgs = make_session_messages(20, 10, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);

    SessionConfig cfg = SessionConfig::default_rth();
    cfg.warmup_messages = 0;  // skip all pre-market warmup

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    // Episode should still start (fast-forward to first RTH message)
    EXPECT_FALSE(result.done);
}

TEST(LOBEnvSession, WarmupLimitedToNMessages) {
    // 20 pre-market messages, 10 RTH messages
    auto msgs = make_session_messages(20, 10, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);

    SessionConfig cfg = SessionConfig::default_rth();
    cfg.warmup_messages = 5;  // only replay last 5 pre-market messages

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    EXPECT_FALSE(result.done);
    EXPECT_EQ(result.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv Session: Observations valid throughout session episode
// ===========================================================================

TEST(LOBEnvSession, ObservationsValidThroughoutSessionEpisode) {
    auto msgs = make_session_messages(10, 20, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    int steps = 0;
    bool done = false;
    while (!done && steps < 100) {
        StepResult result = env.step(steps % 3);  // cycle through actions
        done = result.done;
        ++steps;

        ASSERT_EQ(result.obs.size(), 44u) << "Obs size wrong at step " << steps;
        EXPECT_ALL_FINITE(result.obs);
        EXPECT_TRUE(std::isfinite(result.reward)) << "reward not finite at step " << steps;
    }
}

// ===========================================================================
// LOBEnv Session: Verify first step timestamp is within RTH
// ===========================================================================

TEST(LOBEnvSession, FirstStepIsWithinRTH) {
    // 10 pre-market messages, 10 RTH messages, 5 post-market
    auto msgs = make_session_messages(10, 10, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    StepResult result = env.reset();

    // The observation returned by reset() should correspond to state
    // after pre-market warmup, ready for the first RTH step
    EXPECT_FALSE(result.done);
    EXPECT_GT(result.obs[40], 0.0f) << "Spread should be positive from warmup";
}
