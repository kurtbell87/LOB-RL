#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "lob/env.h"
#include "test_helpers.h"

// Helper: Create a session env with stable BBO and run to completion with a
// fixed action, returning the final StepResult.
static StepResult run_session_to_done(int action, double mid = 1000.0,
                                       double tick = 0.25,
                                       int steps_per_episode = 0,
                                       int warmup = 10, int rth = 6,
                                       int post = 5) {
    auto msgs = make_stable_bbo_messages(warmup, rth, post, mid, tick);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, steps_per_episode);
    env.reset();

    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(action);
        done = result.done;
    }
    return result;
}

// ===========================================================================
// FlatteningPnL: Session close with LONG position — reward penalty
// ===========================================================================

TEST(FlatteningPnL, LongPositionAtSessionCloseIncursHalfSpreadPenalty) {
    StepResult result = run_session_to_done(/*action=*/2);  // long

    // The final reward should include a flattening penalty.
    // Since mid doesn't change between steps, the normal PnL delta is ~0.
    // The flattening cost for position=+1 is -tick = -0.25.
    EXPECT_LT(result.reward, 0.0f)
        << "Long position at session close should incur a negative flattening penalty";
}

TEST(FlatteningPnL, LongPositionPenaltyIsApproximatelyHalfSpread) {
    double tick = 0.25;
    StepResult result = run_session_to_done(/*action=*/2, /*mid=*/1000.0, tick);

    // With stable BBO, the final step reward should be approximately -half_spread.
    EXPECT_NEAR(result.reward, -tick, 0.01)
        << "Flattening penalty for long should be approximately -half_spread";
}

// ===========================================================================
// FlatteningPnL: Session close with SHORT position — reward penalty
// ===========================================================================

TEST(FlatteningPnL, ShortPositionAtSessionCloseIncursHalfSpreadPenalty) {
    StepResult result = run_session_to_done(/*action=*/0);  // short

    // Flattening cost for position=-1: buying at the ask costs half_spread.
    EXPECT_LT(result.reward, 0.0f)
        << "Short position at session close should incur a negative flattening penalty";
}

TEST(FlatteningPnL, ShortPositionPenaltyIsApproximatelyHalfSpread) {
    double tick = 0.25;
    StepResult result = run_session_to_done(/*action=*/0, /*mid=*/1000.0, tick);

    EXPECT_NEAR(result.reward, -tick, 0.01)
        << "Flattening penalty for short should be approximately -half_spread";
}

// ===========================================================================
// FlatteningPnL: Session close with FLAT position — no penalty
// ===========================================================================

TEST(FlatteningPnL, FlatPositionAtSessionCloseHasNoPenalty) {
    StepResult result = run_session_to_done(/*action=*/1);  // flat

    // Flat position means no flattening cost; with stable mid, reward should be 0
    EXPECT_FLOAT_EQ(result.reward, 0.0f)
        << "Flat position at session close should have zero flattening penalty";
}

// ===========================================================================
// FlatteningPnL: Non-session mode — no flattening penalty
// ===========================================================================

TEST(FlatteningPnL, NonSessionModeNoFlatteningPenalty) {
    // Use the original (non-session) constructor with stable BBO messages
    auto msgs = make_stable_bbo_messages(10, 10, 0);
    auto src = std::make_unique<ScriptedSource>(msgs);

    LOBEnv env(std::move(src), 5);
    env.reset();

    // Go long for 5 steps until episode ends
    StepResult result;
    bool done = false;
    int steps = 0;
    while (!done && steps < 100) {
        result = env.step(2);  // long
        done = result.done;
        ++steps;
    }

    EXPECT_TRUE(done);
    // In non-session mode, no flattening penalty is applied.
    // With stable mid, reward should be ~0.
    EXPECT_NEAR(result.reward, 0.0f, 0.01)
        << "Non-session mode should have no flattening penalty";

    // Position should NOT be forced to 0 in non-session mode
    EXPECT_FLOAT_EQ(result.obs[43], 1.0f)
        << "Non-session mode should not force position flat on episode end";
}

// ===========================================================================
// FlatteningPnL: Penalty is symmetric for long and short
// ===========================================================================

TEST(FlatteningPnL, PenaltyIsSymmetricForLongAndShort) {
    float long_final_reward = run_session_to_done(/*action=*/2).reward;
    float short_final_reward = run_session_to_done(/*action=*/0).reward;

    // Both should have the same magnitude penalty (spread/2)
    EXPECT_NEAR(std::abs(long_final_reward), std::abs(short_final_reward), 0.01)
        << "Flattening penalty should be the same magnitude for long and short";
}

// ===========================================================================
// FlatteningPnL: Non-terminal steps are NOT affected
// ===========================================================================

TEST(FlatteningPnL, NonTerminalStepsHaveNoFlatteningPenalty) {
    auto msgs = make_stable_bbo_messages(10, 10, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0);
    env.reset();

    // Take non-terminal steps while long — reward should be ~0 (stable mid)
    StepResult result = env.step(2);  // first step, long
    EXPECT_FALSE(result.done);
    EXPECT_NEAR(result.reward, 0.0f, 0.01)
        << "Non-terminal step while long should have reward ~0 with stable mid (no penalty)";

    result = env.step(2);  // second step, still long
    EXPECT_FALSE(result.done);
    EXPECT_NEAR(result.reward, 0.0f, 0.01)
        << "Non-terminal step while long should have reward ~0 with stable mid (no penalty)";
}

// ===========================================================================
// FlatteningPnL: With wider spread, penalty is larger
// ===========================================================================

TEST(FlatteningPnL, WiderSpreadProducesLargerPenalty) {
    float narrow_penalty = run_session_to_done(/*action=*/2, /*mid=*/1000.0,
                                                /*tick=*/0.25).reward;
    float wide_penalty = run_session_to_done(/*action=*/2, /*mid=*/1000.0,
                                              /*tick=*/1.0).reward;

    // Both should be negative (penalties)
    EXPECT_LT(narrow_penalty, 0.0f);
    EXPECT_LT(wide_penalty, 0.0f);

    // Wider spread should produce a more negative (larger magnitude) penalty
    EXPECT_LT(wide_penalty, narrow_penalty)
        << "Wider spread should produce a larger (more negative) flattening penalty";
}

// ===========================================================================
// FlatteningPnL: steps_per_episode terminates BEFORE session close —
// position is forced flat with penalty when session mode is active
// ===========================================================================

TEST(FlatteningPnL, StepsPerEpisodeLimitWithSessionIncludesPenalty) {
    StepResult result = run_session_to_done(/*action=*/2, /*mid=*/1000.0,
                                             /*tick=*/0.25,
                                             /*steps_per_episode=*/3,
                                             /*warmup=*/10, /*rth=*/30,
                                             /*post=*/5);

    EXPECT_TRUE(result.done);

    // Position should be forced flat
    EXPECT_FLOAT_EQ(result.obs[43], 0.0f)
        << "Position should be forced flat when session-mode episode ends early";

    // Reward should include flattening penalty
    EXPECT_LT(result.reward, 0.0f)
        << "Flattening penalty should apply when session-mode episode ends via steps_per_episode";
}
