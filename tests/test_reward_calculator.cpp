#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include "lob/reward.h"
#include "lob/env.h"
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// RewardCalculator: Construction & Accessors
// ===========================================================================

TEST(RewardCalculator, DefaultConstructorUsesPnLDelta) {
    RewardCalculator rc;
    EXPECT_EQ(rc.mode(), RewardMode::PnLDelta);
}

TEST(RewardCalculator, DefaultConstructorLambdaIsZero) {
    RewardCalculator rc;
    EXPECT_FLOAT_EQ(rc.lambda(), 0.0f);
}

TEST(RewardCalculator, ConstructWithPnLDeltaMode) {
    RewardCalculator rc(RewardMode::PnLDelta);
    EXPECT_EQ(rc.mode(), RewardMode::PnLDelta);
}

TEST(RewardCalculator, ConstructWithPnLDeltaPenalizedMode) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    EXPECT_EQ(rc.mode(), RewardMode::PnLDeltaPenalized);
}

TEST(RewardCalculator, ConstructorStoresLambda) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.5f);
    EXPECT_FLOAT_EQ(rc.lambda(), 0.5f);
}

TEST(RewardCalculator, ConstructorStoresCustomLambda) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.001f);
    EXPECT_FLOAT_EQ(rc.lambda(), 0.001f);
}

// ===========================================================================
// RewardCalculator: PnLDelta mode — compute()
// ===========================================================================

TEST(RewardCalculatorPnLDelta, LongPositionPriceUp) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // position=+1, mid goes from 100 to 101 => reward = 1*(101-100) = 1.0
    float reward = rc.compute(1.0f, 101.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 1.0f);
}

TEST(RewardCalculatorPnLDelta, LongPositionPriceDown) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // position=+1, mid goes from 100 to 99 => reward = 1*(99-100) = -1.0
    float reward = rc.compute(1.0f, 99.0, 100.0);
    EXPECT_FLOAT_EQ(reward, -1.0f);
}

TEST(RewardCalculatorPnLDelta, ShortPositionPriceUp) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // position=-1, mid goes from 100 to 101 => reward = -1*(101-100) = -1.0
    float reward = rc.compute(-1.0f, 101.0, 100.0);
    EXPECT_FLOAT_EQ(reward, -1.0f);
}

TEST(RewardCalculatorPnLDelta, ShortPositionPriceDown) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // position=-1, mid goes from 100 to 99 => reward = -1*(99-100) = 1.0
    float reward = rc.compute(-1.0f, 99.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 1.0f);
}

TEST(RewardCalculatorPnLDelta, FlatPositionAlwaysZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // position=0, any price change => reward = 0
    float reward = rc.compute(0.0f, 200.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorPnLDelta, NoPriceChangeZeroReward) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // mid doesn't change => reward = 0 regardless of position
    EXPECT_FLOAT_EQ(rc.compute(1.0f, 100.0, 100.0), 0.0f);
    EXPECT_FLOAT_EQ(rc.compute(-1.0f, 100.0, 100.0), 0.0f);
}

TEST(RewardCalculatorPnLDelta, SmallPriceMove) {
    RewardCalculator rc(RewardMode::PnLDelta);
    // Typical tick-level price movement
    float reward = rc.compute(1.0f, 1000.25, 1000.0);
    EXPECT_FLOAT_EQ(reward, 0.25f);
}

// ===========================================================================
// RewardCalculator: PnLDeltaPenalized mode — compute()
// ===========================================================================

TEST(RewardCalculatorPenalized, LongPositionIncludesLambdaPenalty) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    // position=+1, mid 100->101
    // reward = 1*(101-100) - 0.1*|1| = 1.0 - 0.1 = 0.9
    float reward = rc.compute(1.0f, 101.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 0.9f);
}

TEST(RewardCalculatorPenalized, ShortPositionIncludesLambdaPenalty) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    // position=-1, mid 100->99
    // reward = -1*(99-100) - 0.1*|-1| = 1.0 - 0.1 = 0.9
    float reward = rc.compute(-1.0f, 99.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 0.9f);
}

TEST(RewardCalculatorPenalized, FlatPositionNoPenalty) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    // position=0 => no PnL, no penalty => 0
    float reward = rc.compute(0.0f, 101.0, 100.0);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorPenalized, PenaltyMakesRewardLower) {
    RewardCalculator pnl(RewardMode::PnLDelta);
    RewardCalculator penalized(RewardMode::PnLDeltaPenalized, 0.1f);

    float r_pnl = pnl.compute(1.0f, 101.0, 100.0);
    float r_pen = penalized.compute(1.0f, 101.0, 100.0);

    // Penalized should be strictly less than PnLDelta when position != 0
    EXPECT_LT(r_pen, r_pnl);
}

TEST(RewardCalculatorPenalized, LambdaZeroEquivalentToPnLDelta) {
    RewardCalculator pnl(RewardMode::PnLDelta);
    RewardCalculator penalized(RewardMode::PnLDeltaPenalized, 0.0f);

    // With lambda=0, PnLDeltaPenalized should give same result as PnLDelta
    EXPECT_FLOAT_EQ(pnl.compute(1.0f, 101.0, 100.0),
                    penalized.compute(1.0f, 101.0, 100.0));
    EXPECT_FLOAT_EQ(pnl.compute(-1.0f, 99.0, 100.0),
                    penalized.compute(-1.0f, 99.0, 100.0));
    EXPECT_FLOAT_EQ(pnl.compute(0.0f, 101.0, 100.0),
                    penalized.compute(0.0f, 101.0, 100.0));
}

TEST(RewardCalculatorPenalized, LargerLambdaLargerPenalty) {
    RewardCalculator small_lambda(RewardMode::PnLDeltaPenalized, 0.01f);
    RewardCalculator large_lambda(RewardMode::PnLDeltaPenalized, 1.0f);

    float r_small = small_lambda.compute(1.0f, 101.0, 100.0);
    float r_large = large_lambda.compute(1.0f, 101.0, 100.0);

    EXPECT_GT(r_small, r_large);
}

TEST(RewardCalculatorPenalized, PenaltyProportionalToAbsPosition) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    // Long and short with same abs value should have same penalty magnitude
    // (but different PnL signs with same price movement)
    // With no price change, reward = -lambda*|position|
    float reward_long = rc.compute(1.0f, 100.0, 100.0);
    float reward_short = rc.compute(-1.0f, 100.0, 100.0);

    EXPECT_FLOAT_EQ(reward_long, -0.1f);
    EXPECT_FLOAT_EQ(reward_short, -0.1f);
}

// ===========================================================================
// RewardCalculator: Non-finite mid values
// ===========================================================================

TEST(RewardCalculatorEdgeCases, NaNCurrentMidReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    float reward = rc.compute(1.0f, std::numeric_limits<double>::quiet_NaN(), 100.0);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorEdgeCases, NaNPrevMidReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    float reward = rc.compute(1.0f, 100.0, std::numeric_limits<double>::quiet_NaN());
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorEdgeCases, InfinityCurrentMidReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    float reward = rc.compute(1.0f, std::numeric_limits<double>::infinity(), 100.0);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorEdgeCases, InfinityPrevMidReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    float reward = rc.compute(1.0f, 100.0, std::numeric_limits<double>::infinity());
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorEdgeCases, NaNMidWithPenalizedModeReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDeltaPenalized, 0.1f);
    float reward = rc.compute(1.0f, std::numeric_limits<double>::quiet_NaN(), 100.0);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

TEST(RewardCalculatorEdgeCases, BothMidsNaNReturnsZero) {
    RewardCalculator rc(RewardMode::PnLDelta);
    double nan = std::numeric_limits<double>::quiet_NaN();
    float reward = rc.compute(1.0f, nan, nan);
    EXPECT_FLOAT_EQ(reward, 0.0f);
}

// ===========================================================================
// RewardCalculator: flattening_penalty()
// ===========================================================================

TEST(RewardCalculatorFlattening, LongPositionPenalty) {
    RewardCalculator rc;
    // flattening_penalty = -|position| * spread / 2
    // position=+1, spread=0.5 => -1 * 0.5/2 = -0.25
    float penalty = rc.flattening_penalty(1.0f, 0.5);
    EXPECT_FLOAT_EQ(penalty, -0.25f);
}

TEST(RewardCalculatorFlattening, ShortPositionPenalty) {
    RewardCalculator rc;
    // position=-1, spread=0.5 => -|-1| * 0.5/2 = -0.25
    float penalty = rc.flattening_penalty(-1.0f, 0.5);
    EXPECT_FLOAT_EQ(penalty, -0.25f);
}

TEST(RewardCalculatorFlattening, FlatPositionZeroPenalty) {
    RewardCalculator rc;
    float penalty = rc.flattening_penalty(0.0f, 0.5);
    EXPECT_FLOAT_EQ(penalty, 0.0f);
}

TEST(RewardCalculatorFlattening, WiderSpreadLargerPenalty) {
    RewardCalculator rc;
    float narrow = rc.flattening_penalty(1.0f, 0.5);
    float wide = rc.flattening_penalty(1.0f, 2.0);
    EXPECT_LT(wide, narrow);  // wide is more negative
}

TEST(RewardCalculatorFlattening, PenaltySymmetricForLongShort) {
    RewardCalculator rc;
    float long_penalty = rc.flattening_penalty(1.0f, 1.0);
    float short_penalty = rc.flattening_penalty(-1.0f, 1.0);
    EXPECT_FLOAT_EQ(long_penalty, short_penalty);
}

TEST(RewardCalculatorFlattening, PenaltyWithLargeSpread) {
    RewardCalculator rc;
    // position=1, spread=10 => -1 * 10/2 = -5.0
    float penalty = rc.flattening_penalty(1.0f, 10.0);
    EXPECT_FLOAT_EQ(penalty, -5.0f);
}

TEST(RewardCalculatorFlattening, PenaltyIndependentOfRewardMode) {
    RewardCalculator pnl(RewardMode::PnLDelta);
    RewardCalculator penalized(RewardMode::PnLDeltaPenalized, 0.5f);

    // Flattening penalty should be the same regardless of reward mode
    EXPECT_FLOAT_EQ(pnl.flattening_penalty(1.0f, 0.5),
                    penalized.flattening_penalty(1.0f, 0.5));
}

// ===========================================================================
// LOBEnv: Integration with RewardCalculator — PnLDelta (backward compat)
// ===========================================================================

// Helper: create a LOBEnv with SyntheticSource and a given reward mode
static LOBEnv make_env_with_reward(RewardMode mode, float lambda = 0.0f,
                                    int steps = 50) {
    return LOBEnv(std::make_unique<SyntheticSource>(), steps, mode, lambda);
}

TEST(LOBEnvReward, DefaultRewardModeIsPnLDelta) {
    // Default constructor should use PnLDelta (backward compatible)
    auto env = LOBEnv(std::make_unique<SyntheticSource>());
    env.reset();
    StepResult r = env.step(1);  // flat
    EXPECT_FLOAT_EQ(r.reward, 0.0f);
}

TEST(LOBEnvReward, PnLDeltaFlatPositionZeroReward) {
    auto env = make_env_with_reward(RewardMode::PnLDelta);
    env.reset();
    StepResult r = env.step(1);  // flat
    EXPECT_FLOAT_EQ(r.reward, 0.0f);
}

TEST(LOBEnvReward, PnLDeltaRewardIsFinite) {
    auto env = make_env_with_reward(RewardMode::PnLDelta);
    env.reset();
    env.step(2);  // go long
    StepResult r = env.step(2);  // still long
    EXPECT_TRUE(std::isfinite(r.reward));
}

TEST(LOBEnvReward, PnLDeltaPenalizedProducesLowerRewardThanPnLDelta) {
    // Run both modes with same SyntheticSource and same actions
    auto env_pnl = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f);
    auto env_pen = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDeltaPenalized, 0.1f);

    env_pnl.reset();
    env_pen.reset();

    float total_pnl = 0.0f;
    float total_pen = 0.0f;

    // Go long for entire episode — penalized should accumulate lower total reward
    for (int i = 0; i < 10; ++i) {
        StepResult r_pnl = env_pnl.step(2);  // long
        StepResult r_pen = env_pen.step(2);   // long
        total_pnl += r_pnl.reward;
        total_pen += r_pen.reward;
        if (r_pnl.done || r_pen.done) break;
    }

    // With position=+1 and lambda=0.1, each step has -0.1 penalty
    // So total_pen should be less than total_pnl
    EXPECT_LT(total_pen, total_pnl);
}

TEST(LOBEnvReward, PnLDeltaPenalizedFlatPositionSameAsPnLDelta) {
    // When position is flat (action=1), both modes should give 0 reward
    auto env_pnl = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDelta, 0.0f);
    auto env_pen = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDeltaPenalized, 0.5f);

    env_pnl.reset();
    env_pen.reset();

    // Stay flat — both should be zero
    for (int i = 0; i < 5; ++i) {
        StepResult r_pnl = env_pnl.step(1);  // flat
        StepResult r_pen = env_pen.step(1);   // flat
        EXPECT_FLOAT_EQ(r_pnl.reward, r_pen.reward)
            << "Flat position should give same reward in both modes at step " << i;
        if (r_pnl.done || r_pen.done) break;
    }
}

// ===========================================================================
// LOBEnv: Integration with RewardCalculator — Session mode
// ===========================================================================

TEST(LOBEnvRewardSession, PnLDeltaSessionFlatteningMatchesExistingBehavior) {
    // Session with PnLDelta should still apply flattening penalty at close
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDelta, 0.0f);
    env.reset();

    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(2);  // long
        done = result.done;
    }

    // Final reward should include flattening penalty (negative)
    EXPECT_LT(result.reward, 0.0f);
}

TEST(LOBEnvRewardSession, PnLDeltaPenalizedSessionFlatteningWorks) {
    // Session with PnLDeltaPenalized should also apply flattening penalty
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDeltaPenalized, 0.1f);
    env.reset();

    StepResult result;
    bool done = false;
    while (!done) {
        result = env.step(2);  // long
        done = result.done;
    }

    // Final reward should include both flattening penalty and lambda penalty
    EXPECT_LT(result.reward, 0.0f);
}

TEST(LOBEnvRewardSession, PenalizedModeLowerCumulativeRewardThanPnLDelta) {
    // Compare cumulative reward over a session episode
    auto msgs1 = make_stable_bbo_messages(10, 10, 5);
    auto msgs2 = msgs1;  // identical message sequence
    auto src1 = std::make_unique<ScriptedSource>(std::move(msgs1));
    auto src2 = std::make_unique<ScriptedSource>(std::move(msgs2));
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env_pnl(std::move(src1), cfg, 0, RewardMode::PnLDelta, 0.0f);
    LOBEnv env_pen(std::move(src2), cfg, 0, RewardMode::PnLDeltaPenalized, 0.1f);

    env_pnl.reset();
    env_pen.reset();

    float total_pnl = 0.0f;
    float total_pen = 0.0f;

    bool done1 = false, done2 = false;
    while (!done1 && !done2) {
        StepResult r1 = env_pnl.step(2);  // long
        StepResult r2 = env_pen.step(2);   // long
        total_pnl += r1.reward;
        total_pen += r2.reward;
        done1 = r1.done;
        done2 = r2.done;
    }

    EXPECT_LT(total_pen, total_pnl)
        << "PnLDeltaPenalized should accumulate less total reward than PnLDelta";
}

// ===========================================================================
// LOBEnv: Constructor overloads with reward mode parameters
// ===========================================================================

TEST(LOBEnvRewardConstructor, BasicConstructorAcceptsRewardMode) {
    // LOBEnv(source, steps, mode, lambda) should compile and work
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvRewardConstructor, BasicConstructorAcceptsPenalizedMode) {
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDeltaPenalized, 0.5f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvRewardConstructor, SessionConstructorAcceptsRewardMode) {
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDeltaPenalized, 0.1f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvRewardConstructor, DefaultRewardModeBackwardCompatible) {
    // Original constructors (without reward mode) should still work
    auto env1 = LOBEnv(std::make_unique<SyntheticSource>());
    auto env2 = LOBEnv(std::make_unique<SyntheticSource>(), 10);

    env1.reset();
    env2.reset();

    // Should produce valid results
    StepResult r1 = env1.step(1);
    StepResult r2 = env2.step(1);
    EXPECT_EQ(r1.obs.size(), 44u);
    EXPECT_EQ(r2.obs.size(), 44u);
}
