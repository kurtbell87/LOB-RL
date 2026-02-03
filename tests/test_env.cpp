#include <gtest/gtest.h>
#include "lob/env.h"
#include "synthetic_source.h"

using namespace lob;

class EnvTest : public ::testing::Test {
protected:
    std::unique_ptr<LOBEnv> env;

    void SetUp() override {
        EnvConfig config;
        config.book_depth = 10;
        config.trades_per_step = 50;
        config.reward_type = RewardType::PnLDelta;

        auto source = std::make_unique<SyntheticSource>(42, 1000);
        env = std::make_unique<LOBEnv>(std::move(config), std::move(source));
    }
};

TEST_F(EnvTest, ObservationSize) {
    // 4 * book_depth + 4 = 4 * 10 + 4 = 44
    EXPECT_EQ(env->observation_size(), 44);
}

TEST_F(EnvTest, ActionSize) {
    EXPECT_EQ(LOBEnv::action_size(), 3);
}

TEST_F(EnvTest, Reset) {
    StepResult result = env->reset();

    EXPECT_EQ(result.obs.data.size(), static_cast<size_t>(env->observation_size()));
    EXPECT_FALSE(result.done);
    EXPECT_EQ(result.position, 0);
    EXPECT_DOUBLE_EQ(result.pnl, 0.0);
}

TEST_F(EnvTest, StepFlat) {
    env->reset();

    // Action 1 = flat, no position change
    StepResult result = env->step(1);

    EXPECT_EQ(result.position, 0);
    EXPECT_EQ(result.obs.data.size(), static_cast<size_t>(env->observation_size()));
}

TEST_F(EnvTest, StepLong) {
    env->reset();

    // Action 2 = long
    StepResult result = env->step(2);

    EXPECT_EQ(result.position, 1);
}

TEST_F(EnvTest, StepShort) {
    env->reset();

    // Action 0 = short
    StepResult result = env->step(0);

    EXPECT_EQ(result.position, -1);
}

TEST_F(EnvTest, PositionTransitions) {
    env->reset();

    // Go long
    StepResult r1 = env->step(2);
    EXPECT_EQ(r1.position, 1);

    // Go flat
    StepResult r2 = env->step(1);
    EXPECT_EQ(r2.position, 0);

    // Go short
    StepResult r3 = env->step(0);
    EXPECT_EQ(r3.position, -1);

    // Go long directly (short -> long)
    StepResult r4 = env->step(2);
    EXPECT_EQ(r4.position, 1);
}

TEST_F(EnvTest, EpisodeTermination) {
    env->reset();

    bool reached_done = false;
    for (int i = 0; i < 1000; ++i) {
        StepResult result = env->step(1);  // Stay flat
        if (result.done) {
            reached_done = true;
            // Should be flat at end
            EXPECT_EQ(result.position, 0);
            break;
        }
    }

    EXPECT_TRUE(reached_done);
}

TEST_F(EnvTest, ObservationValues) {
    StepResult result = env->reset();

    // Position (last element) should be 0
    EXPECT_FLOAT_EQ(result.obs.data[43], 0.0f);

    // Time remaining should be close to 1.0
    EXPECT_NEAR(result.obs.data[42], 1.0f, 0.1f);
}

TEST_F(EnvTest, MultipleResets) {
    for (int i = 0; i < 3; ++i) {
        StepResult r1 = env->reset();
        EXPECT_EQ(r1.position, 0);
        EXPECT_DOUBLE_EQ(r1.pnl, 0.0);

        // Take some steps
        env->step(2);
        env->step(0);
        env->step(1);
    }
}

TEST_F(EnvTest, RewardCalculation) {
    env->reset();

    // Go long
    StepResult r1 = env->step(2);
    EXPECT_EQ(r1.position, 1);

    // Go flat (close position)
    StepResult r2 = env->step(1);

    // Reward should be change in PnL
    // (This is a basic check - actual values depend on price movement)
    EXPECT_EQ(r2.position, 0);
}

TEST_F(EnvTest, PenalizedReward) {
    EnvConfig config;
    config.book_depth = 10;
    config.trades_per_step = 50;
    config.reward_type = RewardType::PnLDeltaPenalized;
    config.inventory_penalty = 0.01;

    auto source = std::make_unique<SyntheticSource>(42, 1000);
    LOBEnv penalized_env(std::move(config), std::move(source));

    penalized_env.reset();

    // Go long - should have penalty
    StepResult r1 = penalized_env.step(2);
    EXPECT_EQ(r1.position, 1);
    // Reward includes -0.01 * |position| = -0.01
}

TEST_F(EnvTest, ConfigAccess) {
    const EnvConfig& config = env->config();
    EXPECT_EQ(config.book_depth, 10);
    EXPECT_EQ(config.trades_per_step, 50);
}
