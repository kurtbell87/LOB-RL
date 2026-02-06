/**
 * Tests for participation bonus reward component.
 *
 * Spec: docs/participation-bonus.md
 *
 * These tests verify that:
 * - RewardCalculator::participation_bonus() computes bonus * abs(position)
 * - LOBEnv with participation_bonus > 0 adds bonus when position != 0
 * - LOBEnv with participation_bonus = 0.0 (default) preserves existing behavior
 * - participation_bonus is orthogonal to reward_mode and execution_cost
 * - Edge cases: position=0, negative bonus, combined with lambda penalty
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "lob/reward.h"
#include "lob/env.h"
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// RewardCalculator::participation_bonus() — unit tests
// ===========================================================================

TEST(ParticipationBonus, MethodExists) {
    // RewardCalculator should have a participation_bonus() static method
    double result = RewardCalculator::participation_bonus(1.0, 0.01);
    EXPECT_TRUE(std::isfinite(result));
}

TEST(ParticipationBonus, LongPositionReturnsBonus) {
    // position=+1, bonus=0.01 => 0.01 * abs(1.0) = 0.01
    double result = RewardCalculator::participation_bonus(1.0, 0.01);
    EXPECT_FLOAT_EQ(result, 0.01);
}

TEST(ParticipationBonus, ShortPositionReturnsBonus) {
    // position=-1, bonus=0.01 => 0.01 * abs(-1.0) = 0.01
    double result = RewardCalculator::participation_bonus(-1.0, 0.01);
    EXPECT_FLOAT_EQ(result, 0.01);
}

TEST(ParticipationBonus, FlatPositionReturnsZero) {
    // position=0, bonus=0.01 => 0.01 * abs(0.0) = 0.0
    double result = RewardCalculator::participation_bonus(0.0, 0.01);
    EXPECT_FLOAT_EQ(result, 0.0);
}

TEST(ParticipationBonus, ZeroBonusReturnsZero) {
    // bonus=0.0 => always 0 regardless of position
    EXPECT_FLOAT_EQ(RewardCalculator::participation_bonus(1.0, 0.0), 0.0);
    EXPECT_FLOAT_EQ(RewardCalculator::participation_bonus(-1.0, 0.0), 0.0);
    EXPECT_FLOAT_EQ(RewardCalculator::participation_bonus(0.0, 0.0), 0.0);
}

TEST(ParticipationBonus, BonusScalesWithPosition) {
    // Larger position magnitude => larger bonus
    double b1 = RewardCalculator::participation_bonus(1.0, 0.05);
    double b2 = RewardCalculator::participation_bonus(2.0, 0.05);
    EXPECT_LT(b1, b2);
    EXPECT_FLOAT_EQ(b2, 2.0 * b1);
}

TEST(ParticipationBonus, SymmetricForLongAndShort) {
    // abs(+1) == abs(-1), so bonus should be the same
    double long_bonus = RewardCalculator::participation_bonus(1.0, 0.05);
    double short_bonus = RewardCalculator::participation_bonus(-1.0, 0.05);
    EXPECT_FLOAT_EQ(long_bonus, short_bonus);
}

TEST(ParticipationBonus, NegativeBonusIsValid) {
    // Negative bonus technically valid (penalizes being in market)
    double result = RewardCalculator::participation_bonus(1.0, -0.01);
    EXPECT_FLOAT_EQ(result, -0.01);
}

TEST(ParticipationBonus, LargerBonusRateGivesLargerResult) {
    double small = RewardCalculator::participation_bonus(1.0, 0.01);
    double large = RewardCalculator::participation_bonus(1.0, 0.05);
    EXPECT_LT(small, large);
}

// ===========================================================================
// LOBEnv: Constructor accepts participation_bonus parameter
// ===========================================================================

TEST(LOBEnvParticipationBonus, ConstructorAcceptsParticipationBonusZero) {
    // LOBEnv constructor should accept participation_bonus=0.0
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f,
                       /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvParticipationBonus, ConstructorAcceptsParticipationBonusNonZero) {
    // LOBEnv constructor should accept participation_bonus=0.01
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f,
                       /*execution_cost=*/false, /*participation_bonus=*/0.01f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvParticipationBonus, DefaultConstructorBackwardCompatible) {
    // Existing constructors (without participation_bonus) should still work
    auto env1 = LOBEnv(std::make_unique<SyntheticSource>());
    auto env2 = LOBEnv(std::make_unique<SyntheticSource>(), 10);
    auto env3 = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                        RewardMode::PnLDelta, 0.0f);
    auto env4 = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                        RewardMode::PnLDelta, 0.0f, false);

    env1.reset();
    env2.reset();
    env3.reset();
    env4.reset();

    StepResult r1 = env1.step(1);
    StepResult r2 = env2.step(1);
    StepResult r3 = env3.step(1);
    StepResult r4 = env4.step(1);
    EXPECT_EQ(r1.obs.size(), 44u);
    EXPECT_EQ(r2.obs.size(), 44u);
    EXPECT_EQ(r3.obs.size(), 44u);
    EXPECT_EQ(r4.obs.size(), 44u);
}

TEST(LOBEnvParticipationBonus, SessionConstructorAcceptsParticipationBonus) {
    // Session constructor should accept participation_bonus parameter
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDelta, 0.0f,
               /*execution_cost=*/false, /*participation_bonus=*/0.01f);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv: participation_bonus=0.0 preserves existing behavior
// ===========================================================================

TEST(LOBEnvParticipationBonus, ZeroBonusMatchesDefault) {
    // participation_bonus=0.0 should produce identical rewards to the default
    auto env_default = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                               RewardMode::PnLDelta, 0.0f,
                               /*execution_cost=*/false);
    auto env_zero = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                            RewardMode::PnLDelta, 0.0f,
                            /*execution_cost=*/false, /*participation_bonus=*/0.0f);

    env_default.reset();
    env_zero.reset();

    for (int i = 0; i < 10; ++i) {
        int action = i % 3;
        StepResult r_def = env_default.step(action);
        StepResult r_zero = env_zero.step(action);
        EXPECT_FLOAT_EQ(r_def.reward, r_zero.reward)
            << "participation_bonus=0.0 should match default at step " << i;
        if (r_def.done || r_zero.done) break;
    }
}

// ===========================================================================
// LOBEnv: participation_bonus > 0 increases reward when position != 0
// ===========================================================================

TEST(LOBEnvParticipationBonus, IncreasesRewardWhenPositionNonZero) {
    // When holding a position, reward should be higher with participation bonus
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f,
                          /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    // Go long from flat
    StepResult r_off = env_off.step(2);  // long
    StepResult r_on = env_on.step(2);    // long

    // With participation bonus, reward should be strictly higher
    EXPECT_GT(r_on.reward, r_off.reward)
        << "Participation bonus should increase reward when position != 0";
}

TEST(LOBEnvParticipationBonus, NoBonusWhenFlat) {
    // When position is flat, bonus should be 0 — no difference
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f,
                          /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    // Stay flat
    StepResult r_off = env_off.step(1);  // flat
    StepResult r_on = env_on.step(1);    // flat

    EXPECT_FLOAT_EQ(r_off.reward, r_on.reward)
        << "Flat position should have no participation bonus";
}

TEST(LOBEnvParticipationBonus, BonusAppliesOnShortToo) {
    // Short position should also get the bonus
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f,
                          /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    // Go short from flat
    StepResult r_off = env_off.step(0);  // short
    StepResult r_on = env_on.step(0);    // short

    EXPECT_GT(r_on.reward, r_off.reward)
        << "Participation bonus should apply to short positions too";
}

TEST(LOBEnvParticipationBonus, HoldingLongAccumulatesBonus) {
    // Holding long for multiple steps should accumulate bonus
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f,
                          /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    float total_off = 0.0f;
    float total_on = 0.0f;

    // Go long and hold for several steps
    for (int i = 0; i < 5; ++i) {
        StepResult r_off = env_off.step(2);  // long
        StepResult r_on = env_on.step(2);    // long
        total_off += r_off.reward;
        total_on += r_on.reward;
        if (r_off.done || r_on.done) break;
    }

    // Total reward with bonus should be higher
    EXPECT_GT(total_on, total_off)
        << "Holding position should accumulate participation bonus over time";
}

// ===========================================================================
// LOBEnv: participation_bonus + execution_cost interaction
// ===========================================================================

TEST(LOBEnvParticipationBonus, WorksWithExecutionCost) {
    // participation_bonus and execution_cost are additive
    auto env_exec_only = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                 RewardMode::PnLDelta, 0.0f,
                                 /*execution_cost=*/true, /*participation_bonus=*/0.0f);
    auto env_both = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                            RewardMode::PnLDelta, 0.0f,
                            /*execution_cost=*/true, /*participation_bonus=*/0.01f);

    env_exec_only.reset();
    env_both.reset();

    // Go long: execution cost reduces, participation bonus increases
    StepResult r_exec = env_exec_only.step(2);
    StepResult r_both = env_both.step(2);

    // With both, reward should be higher than execution cost alone
    EXPECT_GT(r_both.reward, r_exec.reward)
        << "Participation bonus should offset some execution cost";
}

// ===========================================================================
// LOBEnv: participation_bonus + penalized mode interaction
// ===========================================================================

TEST(LOBEnvParticipationBonus, WorksWithPenalizedMode) {
    // participation_bonus and lambda penalty are opposing forces
    auto env_pen_only = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                RewardMode::PnLDeltaPenalized, 0.1f,
                                /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_pen_bonus = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                 RewardMode::PnLDeltaPenalized, 0.1f,
                                 /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_pen_only.reset();
    env_pen_bonus.reset();

    // Go long
    StepResult r_pen = env_pen_only.step(2);
    StepResult r_both = env_pen_bonus.step(2);

    // Bonus partially offsets the lambda penalty
    EXPECT_GT(r_both.reward, r_pen.reward)
        << "Participation bonus should partially offset lambda penalty";
}

TEST(LOBEnvParticipationBonus, FlatNoEffectEitherMode) {
    // Flat position: no PnL, no lambda penalty, no participation bonus
    auto env_pnl = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.01f);
    auto env_pen = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDeltaPenalized, 0.5f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_pnl.reset();
    env_pen.reset();

    // Stay flat
    StepResult r_pnl = env_pnl.step(1);
    StepResult r_pen = env_pen.step(1);

    EXPECT_FLOAT_EQ(r_pnl.reward, 0.0f);
    EXPECT_FLOAT_EQ(r_pen.reward, 0.0f);
}

// ===========================================================================
// LOBEnv: participation_bonus on terminal step
// ===========================================================================

TEST(LOBEnvParticipationBonus, BonusAppliesOnTerminalStep) {
    // On the final step, participation bonus still applies before flattening
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 3,
                           RewardMode::PnLDelta, 0.0f,
                           /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 3,
                          RewardMode::PnLDelta, 0.0f,
                          /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    // Step through to near the end holding long
    float total_off = 0.0f, total_on = 0.0f;
    for (int i = 0; i < 3; ++i) {
        StepResult r_off = env_off.step(2);
        StepResult r_on = env_on.step(2);
        total_off += r_off.reward;
        total_on += r_on.reward;
        if (r_off.done || r_on.done) break;
    }

    // Over the full episode, bonus should make total higher
    EXPECT_GT(total_on, total_off)
        << "Participation bonus should accumulate across all steps including terminal";
}

// ===========================================================================
// LOBEnv: participation_bonus with session mode
// ===========================================================================

TEST(LOBEnvParticipationBonusSession, SessionConstructorWorks) {
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDelta, 0.0f,
               /*execution_cost=*/false, /*participation_bonus=*/0.01f);
    env.reset();

    StepResult r = env.step(2);  // go long
    EXPECT_TRUE(std::isfinite(r.reward));
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvParticipationBonusSession, SessionBonusHigherThanWithout) {
    auto msgs1 = make_stable_bbo_messages(10, 6, 5);
    auto msgs2 = msgs1;
    auto src_off = std::make_unique<ScriptedSource>(std::move(msgs1));
    auto src_on = std::make_unique<ScriptedSource>(std::move(msgs2));
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env_off(std::move(src_off), cfg, 0, RewardMode::PnLDelta, 0.0f,
                   /*execution_cost=*/false, /*participation_bonus=*/0.0f);
    LOBEnv env_on(std::move(src_on), cfg, 0, RewardMode::PnLDelta, 0.0f,
                  /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    env_off.reset();
    env_on.reset();

    float total_off = 0.0f, total_on = 0.0f;
    bool done_off = false, done_on = false;

    while (!done_off && !done_on) {
        StepResult r_off = env_off.step(2);
        StepResult r_on = env_on.step(2);
        total_off += r_off.reward;
        total_on += r_on.reward;
        done_off = r_off.done;
        done_on = r_on.done;
    }

    EXPECT_GT(total_on, total_off)
        << "Session with participation bonus should have higher total reward when holding position";
}

// ===========================================================================
// LOBEnv: participation_bonus reset tracking
// ===========================================================================

TEST(LOBEnvParticipationBonus, ResetDoesNotAffectBonus) {
    // After reset, bonus should still apply consistently
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f,
                       /*execution_cost=*/false, /*participation_bonus=*/0.01f);

    // Episode 1: go long
    env.reset();
    StepResult r1 = env.step(2);

    // Episode 2: go long
    env.reset();
    StepResult r2 = env.step(2);

    // Both first steps should have the same reward (including bonus)
    EXPECT_FLOAT_EQ(r1.reward, r2.reward)
        << "Participation bonus should be consistent across resets";
}

// ===========================================================================
// LOBEnv: all three cost/bonus components together
// ===========================================================================

TEST(LOBEnvParticipationBonus, AllComponentsTogether) {
    // PnL + lambda penalty + execution cost + participation bonus
    // All four components should stack correctly
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDeltaPenalized, 0.1f,
                       /*execution_cost=*/true, /*participation_bonus=*/0.05f);

    env.reset();
    StepResult r = env.step(2);  // go long

    // Reward should be finite and include all components
    EXPECT_TRUE(std::isfinite(r.reward));
    EXPECT_EQ(r.obs.size(), 44u);
}
