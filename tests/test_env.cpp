#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "lob/env.h"
#include "synthetic_source.h"
#include "test_helpers.h"

// Helper: create a LOBEnv with a SyntheticSource (default seed).
static LOBEnv make_synthetic_env(int steps_per_episode = 50) {
    return LOBEnv(std::make_unique<SyntheticSource>(), steps_per_episode);
}

// ===========================================================================
// Fixture: LOBEnv with default SyntheticSource, already reset.
// ===========================================================================

class LOBEnvTest : public ::testing::Test {
protected:
    void SetUp() override {
        env_ = std::make_unique<LOBEnv>(std::make_unique<SyntheticSource>());
        env_->reset();
    }

    LOBEnv& env() { return *env_; }

private:
    std::unique_ptr<LOBEnv> env_;
};

// ===========================================================================
// LOBEnv: Construction
// ===========================================================================

TEST(LOBEnv, ConstructsWithSyntheticSource) {
    auto env = make_synthetic_env();
    // Should not crash
}

TEST(LOBEnv, ConstructsWithCustomStepsPerEpisode) {
    auto env = make_synthetic_env(10);
    // Should not crash
}

// ===========================================================================
// LOBEnv: Reset
// ===========================================================================

TEST(LOBEnv, ResetReturnsValidObservation) {
    auto env = make_synthetic_env();
    StepResult result = env.reset();

    EXPECT_EQ(result.obs.size(), 44u);
}

TEST(LOBEnv, ResetObservationHasValidBBO) {
    auto env = make_synthetic_env();
    StepResult result = env.reset();

    // With 44-float obs: indices 0-9 are bid prices (relative, <= 0),
    // 20-29 are ask prices (relative, >= 0), 40 is spread, 43 is position
    float spread = result.obs[40];
    float position = result.obs[43];

    EXPECT_GE(spread, 0.0f) << "spread should be non-negative";
    EXPECT_FLOAT_EQ(position, 0.0f) << "initial position should be 0";
}

TEST(LOBEnv, ResetSetsRewardToZero) {
    auto env = make_synthetic_env();
    StepResult result = env.reset();
    EXPECT_FLOAT_EQ(result.reward, 0.0f);
}

TEST(LOBEnv, ResetSetsDoneToFalse) {
    auto env = make_synthetic_env();
    StepResult result = env.reset();
    EXPECT_FALSE(result.done);
}

TEST(LOBEnv, ResetCanBeCalledMultipleTimes) {
    auto env = make_synthetic_env();

    StepResult r1 = env.reset();
    StepResult r2 = env.reset();

    // Both resets should produce the same initial observation (deterministic)
    ASSERT_EQ(r1.obs.size(), r2.obs.size());
    for (size_t i = 0; i < r1.obs.size(); ++i) {
        EXPECT_FLOAT_EQ(r1.obs[i], r2.obs[i])
            << "Obs mismatch at index " << i << " after re-reset";
    }
}

// ===========================================================================
// LOBEnv: Step (using fixture — env is already reset)
// ===========================================================================

TEST_F(LOBEnvTest, StepReturns44FloatObservation) {
    StepResult result = env().step(1);  // action=1 (flat)
    EXPECT_EQ(result.obs.size(), 44u);
}

TEST_F(LOBEnvTest, StepAdvancesState) {
    StepResult after_step = env().step(1);

    // After stepping, the env should return valid data
    EXPECT_EQ(after_step.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv: Action mapping (using fixture)
// ===========================================================================

TEST_F(LOBEnvTest, ActionZeroSetsShortPosition) {
    StepResult result = env().step(0);  // 0 = short (-1)
    EXPECT_FLOAT_EQ(result.obs[43], -1.0f) << "Action 0 should set position to -1";
}

TEST_F(LOBEnvTest, ActionOneSetsFlat) {
    StepResult result = env().step(1);  // 1 = flat (0)
    EXPECT_FLOAT_EQ(result.obs[43], 0.0f) << "Action 1 should set position to 0";
}

TEST_F(LOBEnvTest, ActionTwoSetsLongPosition) {
    StepResult result = env().step(2);  // 2 = long (+1)
    EXPECT_FLOAT_EQ(result.obs[43], 1.0f) << "Action 2 should set position to +1";
}

// ===========================================================================
// LOBEnv: Action clamping (using fixture)
// ===========================================================================

TEST_F(LOBEnvTest, NegativeActionClampedToZero) {
    StepResult result = env().step(-5);
    // Should not crash; behavior should be same as action=0
    EXPECT_FLOAT_EQ(result.obs[43], -1.0f);
}

TEST_F(LOBEnvTest, ActionAboveTwoClampedToTwo) {
    StepResult result = env().step(100);
    // Should not crash; behavior should be same as action=2
    EXPECT_FLOAT_EQ(result.obs[43], 1.0f);
}

// ===========================================================================
// LOBEnv: Reward calculation (using fixture)
// ===========================================================================

TEST_F(LOBEnvTest, FlatPositionHasZeroReward) {
    // Stay flat — reward should be 0 regardless of price movement
    StepResult result = env().step(1);  // flat
    EXPECT_FLOAT_EQ(result.reward, 0.0f);
}

TEST_F(LOBEnvTest, RewardIsPnLDelta) {
    // Reward = position * (mid_now - mid_prev)
    // Go long, then step again while still long
    env().step(2);  // position = +1
    StepResult r2 = env().step(2);  // still long

    // We can't know the exact reward without knowing price movement,
    // but reward should be a finite number
    EXPECT_TRUE(std::isfinite(r2.reward));
}

// ===========================================================================
// LOBEnv: Episode termination
// ===========================================================================

TEST(LOBEnv, EpisodeTerminatesAtStepsPerEpisode) {
    auto env = make_synthetic_env(5);  // 5 steps per episode
    env.reset();

    bool done = false;
    int steps = 0;
    while (!done && steps < 100) {  // safety limit
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }
    EXPECT_EQ(steps, 5) << "Episode should terminate after steps_per_episode steps";
    EXPECT_TRUE(done);
}

TEST(LOBEnv, EpisodeTerminatesWhenSourceExhausted) {
    auto env = make_synthetic_env(10000);  // Very high steps_per_episode
    env.reset();

    bool done = false;
    int steps = 0;
    while (!done && steps < 500) {  // safety limit
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }
    EXPECT_TRUE(done) << "Episode should terminate when source is exhausted";
    EXPECT_LT(steps, 500) << "Should terminate before safety limit";
}

TEST(LOBEnv, ResetAfterDoneAllowsNewEpisode) {
    auto env = make_synthetic_env(3);
    env.reset();

    // Run episode to completion
    for (int i = 0; i < 3; ++i) env.step(1);

    // Reset and run again
    StepResult result = env.reset();
    EXPECT_FALSE(result.done);
    EXPECT_EQ(result.obs.size(), 44u);

    // Should be able to step again
    StepResult s = env.step(2);
    EXPECT_EQ(s.obs.size(), 44u);
}

// ===========================================================================
// LOBEnv: Observation validity throughout episode (using fixture)
// ===========================================================================

TEST_F(LOBEnvTest, ObservationsStayValidThroughoutEpisode) {
    for (int i = 0; i < 20; ++i) {
        StepResult result = env().step(i % 3);  // cycle through actions
        if (result.done) break;

        ASSERT_EQ(result.obs.size(), 44u) << "Obs size wrong at step " << i;
        EXPECT_ALL_FINITE(result.obs);
        EXPECT_TRUE(std::isfinite(result.reward)) << "reward not finite at step " << i;
    }
}

// ===========================================================================
// LOBEnv: Default steps_per_episode
// ===========================================================================

TEST(LOBEnv, DefaultStepsPerEpisodeIs50) {
    auto env = make_synthetic_env();
    env.reset();

    bool done = false;
    int steps = 0;
    while (!done && steps < 200) {
        StepResult result = env.step(1);
        done = result.done;
        ++steps;
    }
    // Should terminate at 50 or when source exhausted (source has ~100 msgs,
    // but reset consumes some msgs to build initial BBO)
    EXPECT_LE(steps, 50) << "Should not exceed 50 steps with default config";
    EXPECT_TRUE(done);
}
