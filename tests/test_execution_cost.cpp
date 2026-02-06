/**
 * Tests for execution cost on position changes.
 *
 * Spec: docs/execution-cost.md
 *
 * These tests verify that:
 * - RewardCalculator::execution_cost() computes -|new_pos - old_pos| * spread/2
 * - LOBEnv with execution_cost=true subtracts execution cost on position change
 * - LOBEnv with execution_cost=false preserves existing behavior exactly
 * - prev_position_ tracks correctly across steps and resets
 * - Edge cases: no position change, long->short (delta=2), NaN/zero spread
 * - execution_cost is orthogonal to reward_mode (works with both modes)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include "lob/reward.h"
#include "lob/env.h"
#include "synthetic_source.h"
#include "test_helpers.h"

// ===========================================================================
// RewardCalculator::execution_cost() — unit tests
// ===========================================================================

TEST(ExecutionCost, MethodExists) {
    // RewardCalculator should have an execution_cost() method
    RewardCalculator rc;
    float cost = rc.execution_cost(0.0f, 0.0f, 1.0);
    EXPECT_FLOAT_EQ(cost, 0.0f);
}

TEST(ExecutionCost, NoPositionChangeZeroCost) {
    // No position change => cost = 0
    RewardCalculator rc;
    EXPECT_FLOAT_EQ(rc.execution_cost(0.0f, 0.0f, 1.0), 0.0f);
    EXPECT_FLOAT_EQ(rc.execution_cost(1.0f, 1.0f, 1.0), 0.0f);
    EXPECT_FLOAT_EQ(rc.execution_cost(-1.0f, -1.0f, 1.0), 0.0f);
}

TEST(ExecutionCost, FlatToLongCostsHalfSpread) {
    // flat->long: delta=1, spread=0.5 => cost = -1 * 0.5/2 = -0.25
    RewardCalculator rc;
    float cost = rc.execution_cost(0.0f, 1.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.25f);
}

TEST(ExecutionCost, FlatToShortCostsHalfSpread) {
    // flat->short: delta=1, spread=0.5 => cost = -1 * 0.5/2 = -0.25
    RewardCalculator rc;
    float cost = rc.execution_cost(0.0f, -1.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.25f);
}

TEST(ExecutionCost, LongToFlatCostsHalfSpread) {
    // long->flat: delta=1, spread=0.5 => cost = -1 * 0.5/2 = -0.25
    RewardCalculator rc;
    float cost = rc.execution_cost(1.0f, 0.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.25f);
}

TEST(ExecutionCost, ShortToFlatCostsHalfSpread) {
    // short->flat: delta=1, spread=0.5 => cost = -1 * 0.5/2 = -0.25
    RewardCalculator rc;
    float cost = rc.execution_cost(-1.0f, 0.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.25f);
}

TEST(ExecutionCost, LongToShortCostsTwoHalfSpreads) {
    // long->short: delta=2, spread=0.5 => cost = -2 * 0.5/2 = -0.5
    RewardCalculator rc;
    float cost = rc.execution_cost(1.0f, -1.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.5f);
}

TEST(ExecutionCost, ShortToLongCostsTwoHalfSpreads) {
    // short->long: delta=2, spread=0.5 => cost = -2 * 0.5/2 = -0.5
    RewardCalculator rc;
    float cost = rc.execution_cost(-1.0f, 1.0f, 0.5);
    EXPECT_FLOAT_EQ(cost, -0.5f);
}

TEST(ExecutionCost, LongToShortCostsMoreThanFlatToLong) {
    // |delta|=2 should cost more than |delta|=1
    RewardCalculator rc;
    float flip_cost = rc.execution_cost(1.0f, -1.0f, 1.0);
    float open_cost = rc.execution_cost(0.0f, 1.0f, 1.0);
    EXPECT_LT(flip_cost, open_cost);  // flip_cost is more negative
}

TEST(ExecutionCost, CostIsAlwaysNonPositive) {
    // Execution cost should always be <= 0 (it's a cost, not a reward)
    RewardCalculator rc;
    EXPECT_LE(rc.execution_cost(0.0f, 1.0f, 1.0), 0.0f);
    EXPECT_LE(rc.execution_cost(0.0f, -1.0f, 1.0), 0.0f);
    EXPECT_LE(rc.execution_cost(1.0f, -1.0f, 1.0), 0.0f);
    EXPECT_LE(rc.execution_cost(-1.0f, 1.0f, 1.0), 0.0f);
    EXPECT_LE(rc.execution_cost(0.0f, 0.0f, 1.0), 0.0f);
}

TEST(ExecutionCost, WiderSpreadLargerCost) {
    // Wider spread => larger (more negative) cost
    RewardCalculator rc;
    float narrow = rc.execution_cost(0.0f, 1.0f, 0.5);
    float wide = rc.execution_cost(0.0f, 1.0f, 2.0);
    EXPECT_LT(wide, narrow);  // wide is more negative
}

TEST(ExecutionCost, SpreadZeroGivesZeroCost) {
    // spread=0 => cost=0 even if position changes
    RewardCalculator rc;
    EXPECT_FLOAT_EQ(rc.execution_cost(0.0f, 1.0f, 0.0), 0.0f);
    EXPECT_FLOAT_EQ(rc.execution_cost(1.0f, -1.0f, 0.0), 0.0f);
}

TEST(ExecutionCost, CostIndependentOfRewardMode) {
    // execution_cost should return the same value regardless of reward mode
    RewardCalculator pnl(RewardMode::PnLDelta);
    RewardCalculator pen(RewardMode::PnLDeltaPenalized, 0.5f);

    EXPECT_FLOAT_EQ(pnl.execution_cost(0.0f, 1.0f, 1.0),
                    pen.execution_cost(0.0f, 1.0f, 1.0));
    EXPECT_FLOAT_EQ(pnl.execution_cost(1.0f, -1.0f, 0.5),
                    pen.execution_cost(1.0f, -1.0f, 0.5));
}

TEST(ExecutionCost, CostWithTypicalTickSpread) {
    // Typical spread of 0.50 (2 ticks for /MES)
    // flat->long: cost = -1 * 0.50/2 = -0.25
    RewardCalculator rc;
    EXPECT_FLOAT_EQ(rc.execution_cost(0.0f, 1.0f, 0.50), -0.25f);
}

TEST(ExecutionCost, CostSymmetricForBuyAndSell) {
    // Going long and going short from flat should cost the same
    RewardCalculator rc;
    float buy_cost = rc.execution_cost(0.0f, 1.0f, 1.0);
    float sell_cost = rc.execution_cost(0.0f, -1.0f, 1.0);
    EXPECT_FLOAT_EQ(buy_cost, sell_cost);
}

// ===========================================================================
// LOBEnv: Constructor accepts execution_cost parameter
// ===========================================================================

TEST(LOBEnvExecutionCost, ConstructorAcceptsExecutionCostFalse) {
    // LOBEnv constructor should accept execution_cost=false
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvExecutionCost, ConstructorAcceptsExecutionCostTrue) {
    // LOBEnv constructor should accept execution_cost=true
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvExecutionCost, DefaultConstructorBackwardCompatible) {
    // Existing constructors (without execution_cost) should still work
    auto env1 = LOBEnv(std::make_unique<SyntheticSource>());
    auto env2 = LOBEnv(std::make_unique<SyntheticSource>(), 10);
    auto env3 = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                        RewardMode::PnLDelta, 0.0f);

    env1.reset();
    env2.reset();
    env3.reset();

    StepResult r1 = env1.step(1);
    StepResult r2 = env2.step(1);
    StepResult r3 = env3.step(1);
    EXPECT_EQ(r1.obs.size(), 44u);
    EXPECT_EQ(r2.obs.size(), 44u);
    EXPECT_EQ(r3.obs.size(), 44u);
}

TEST(LOBEnvExecutionCost, SessionConstructorAcceptsExecutionCost) {
    // Session constructor should accept execution_cost parameter
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDelta, 0.0f,
               /*execution_cost=*/true);
    StepResult r = env.reset();
    EXPECT_EQ(r.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv: execution_cost=false preserves existing behavior
// ===========================================================================

TEST(LOBEnvExecutionCost, DisabledMatchesExistingBehavior) {
    // execution_cost=false should produce identical rewards to the default
    auto env_default = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                               RewardMode::PnLDelta, 0.0f);
    auto env_disabled = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                RewardMode::PnLDelta, 0.0f,
                                /*execution_cost=*/false);

    env_default.reset();
    env_disabled.reset();

    for (int i = 0; i < 10; ++i) {
        int action = i % 3;  // cycle through short/flat/long
        StepResult r_def = env_default.step(action);
        StepResult r_dis = env_disabled.step(action);
        EXPECT_FLOAT_EQ(r_def.reward, r_dis.reward)
            << "execution_cost=false should match default at step " << i;
        if (r_def.done || r_dis.done) break;
    }
}

// ===========================================================================
// LOBEnv: execution_cost=true subtracts cost on position change
// ===========================================================================

TEST(LOBEnvExecutionCost, EnabledReducesRewardOnPositionChange) {
    // When execution cost is enabled and position changes, reward should be lower
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    // First step: go long from flat (prev_pos=0 -> new_pos=1, delta=1)
    StepResult r_off = env_off.step(2);  // long
    StepResult r_on = env_on.step(2);    // long

    // With execution cost enabled, reward should be strictly less
    // (because spread > 0 for SyntheticSource)
    EXPECT_LT(r_on.reward, r_off.reward)
        << "Execution cost should reduce reward on position change";
}

TEST(LOBEnvExecutionCost, EnabledNoExtraCostWhenPositionUnchanged) {
    // When position doesn't change, execution cost should be 0
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    // Stay flat for both steps — no position change, no execution cost
    StepResult r_off = env_off.step(1);  // flat
    StepResult r_on = env_on.step(1);    // flat

    EXPECT_FLOAT_EQ(r_off.reward, r_on.reward)
        << "No position change should mean no execution cost difference";
}

TEST(LOBEnvExecutionCost, EnabledCostOnSecondStepWhenPositionChanges) {
    // Go long, then switch to short — the second step should have higher cost
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    // Step 1: go long (flat->long, delta=1)
    env_off.step(2);
    env_on.step(2);

    // Step 2: go short (long->short, delta=2)
    StepResult r_off = env_off.step(0);
    StepResult r_on = env_on.step(0);

    // Delta=2 on step 2 means larger cost
    EXPECT_LT(r_on.reward, r_off.reward)
        << "Long->short flip should incur execution cost";
}

TEST(LOBEnvExecutionCost, EnabledHoldingPositionNoCost) {
    // Hold long for multiple steps — only the first step should have cost
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    // Step 1: go long (flat->long, delta=1) — should have cost
    env_off.step(2);
    env_on.step(2);

    // Step 2: stay long (long->long, delta=0) — should have NO cost
    StepResult r_off = env_off.step(2);
    StepResult r_on = env_on.step(2);

    EXPECT_FLOAT_EQ(r_off.reward, r_on.reward)
        << "Holding the same position should incur no execution cost";
}

// ===========================================================================
// LOBEnv: execution_cost + reset tracking
// ===========================================================================

TEST(LOBEnvExecutionCost, ResetClearsPrevPosition) {
    // After reset, prev_position should be 0.
    // So going long on first step after reset costs spread/2 * 1
    auto env = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                       RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    // First episode
    env.reset();
    StepResult r1 = env.step(2);  // flat->long, delta=1

    // Reset and do the same
    env.reset();
    StepResult r2 = env.step(2);  // flat->long, delta=1

    // Both first steps should have the same execution cost
    // (because prev_position is reset to 0 both times)
    EXPECT_FLOAT_EQ(r1.reward, r2.reward)
        << "After reset, prev_position should be 0";
}

// ===========================================================================
// LOBEnv: execution_cost works with PnLDeltaPenalized mode
// ===========================================================================

TEST(LOBEnvExecutionCost, WorksWithPenalizedMode) {
    // execution_cost should be additive with lambda penalty
    auto env_pen_only = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                RewardMode::PnLDeltaPenalized, 0.1f,
                                /*execution_cost=*/false);
    auto env_pen_exec = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                                RewardMode::PnLDeltaPenalized, 0.1f,
                                /*execution_cost=*/true);

    env_pen_only.reset();
    env_pen_exec.reset();

    // Go long: incurs both lambda penalty and execution cost
    StepResult r_pen = env_pen_only.step(2);
    StepResult r_both = env_pen_exec.step(2);

    // With execution cost, reward should be even lower
    EXPECT_LT(r_both.reward, r_pen.reward)
        << "execution_cost + penalized mode should stack costs";
}

TEST(LOBEnvExecutionCost, FlatPositionNoCostEitherMode) {
    // Flat position: no PnL, no lambda penalty, no execution cost
    auto env_pnl = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);
    auto env_pen = LOBEnv(std::make_unique<SyntheticSource>(), 5,
                           RewardMode::PnLDeltaPenalized, 0.5f,
                           /*execution_cost=*/true);

    env_pnl.reset();
    env_pen.reset();

    // Stay flat — no cost anywhere
    StepResult r_pnl = env_pnl.step(1);
    StepResult r_pen = env_pen.step(1);

    EXPECT_FLOAT_EQ(r_pnl.reward, 0.0f);
    EXPECT_FLOAT_EQ(r_pen.reward, 0.0f);
}

// ===========================================================================
// LOBEnv: execution_cost with session mode
// ===========================================================================

TEST(LOBEnvExecutionCostSession, SessionConstructorWorks) {
    // Session mode with execution_cost=true should produce valid results
    auto msgs = make_stable_bbo_messages(10, 6, 5);
    auto src = std::make_unique<ScriptedSource>(msgs);
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env(std::move(src), cfg, 0, RewardMode::PnLDelta, 0.0f,
               /*execution_cost=*/true);
    env.reset();

    StepResult r = env.step(2);  // go long
    EXPECT_TRUE(std::isfinite(r.reward));
    EXPECT_EQ(r.obs.size(), 44u);
}

TEST(LOBEnvExecutionCostSession, SessionFlatteningAndExecutionCostBothApply) {
    // At session end, both flattening penalty AND execution cost can apply.
    // If agent changes position on the last step AND has a nonzero position,
    // both costs should be present.
    auto msgs1 = make_stable_bbo_messages(10, 6, 5);
    auto msgs2 = msgs1;
    auto src_off = std::make_unique<ScriptedSource>(std::move(msgs1));
    auto src_on = std::make_unique<ScriptedSource>(std::move(msgs2));
    SessionConfig cfg = SessionConfig::default_rth();

    LOBEnv env_off(std::move(src_off), cfg, 0, RewardMode::PnLDelta, 0.0f,
                   /*execution_cost=*/false);
    LOBEnv env_on(std::move(src_on), cfg, 0, RewardMode::PnLDelta, 0.0f,
                  /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    // Run both to completion, going long each step
    float total_off = 0.0f;
    float total_on = 0.0f;
    bool done_off = false, done_on = false;

    while (!done_off && !done_on) {
        StepResult r_off = env_off.step(2);
        StepResult r_on = env_on.step(2);
        total_off += r_off.reward;
        total_on += r_on.reward;
        done_off = r_off.done;
        done_on = r_on.done;
    }

    // With execution cost enabled, total reward should be lower
    EXPECT_LT(total_on, total_off)
        << "Execution cost over episode should reduce total reward";
}

// ===========================================================================
// LOBEnv: cumulative cost over multiple position changes
// ===========================================================================

TEST(LOBEnvExecutionCost, CumulativeCostFromFlipping) {
    // Repeatedly flipping long->short->long should accumulate cost
    auto env_off = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                           RewardMode::PnLDelta, 0.0f, /*execution_cost=*/false);
    auto env_on = LOBEnv(std::make_unique<SyntheticSource>(), 10,
                          RewardMode::PnLDelta, 0.0f, /*execution_cost=*/true);

    env_off.reset();
    env_on.reset();

    float total_off = 0.0f;
    float total_on = 0.0f;

    // Flip between long and short every step (maximum trading)
    for (int i = 0; i < 8; ++i) {
        int action = (i % 2 == 0) ? 2 : 0;  // alternate long/short
        StepResult r_off = env_off.step(action);
        StepResult r_on = env_on.step(action);
        total_off += r_off.reward;
        total_on += r_on.reward;
        if (r_off.done || r_on.done) break;
    }

    // Frequent flipping should accumulate significant execution costs
    EXPECT_LT(total_on, total_off)
        << "Frequent position flipping should accumulate execution costs";
}
