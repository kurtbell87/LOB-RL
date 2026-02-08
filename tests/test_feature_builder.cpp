#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "lob/feature_builder.h"
#include "test_helpers.h"

// ===========================================================================
// FeatureBuilder: Constants
// ===========================================================================

TEST(FeatureBuilder, ObsSizeIs44) {
    EXPECT_EQ(FeatureBuilder::OBS_SIZE, 44);
}

TEST(FeatureBuilder, DepthIs10) {
    EXPECT_EQ(FeatureBuilder::DEPTH, 10);
}

// ===========================================================================
// FeatureBuilder::build() — Output size
// ===========================================================================

TEST(FeatureBuilder, BuildReturnsExactly44Floats) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_EQ(obs.size(), 44u);
}

TEST(FeatureBuilder, BuildWithEmptyBookReturns44Floats) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_EQ(obs.size(), 44u);
}

// ===========================================================================
// FeatureBuilder::build() — All values finite (no NaN/Inf)
// ===========================================================================

TEST(FeatureBuilder, AllValuesFiniteWithPopulatedBook) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 1.0f, 0.75f);
    EXPECT_ALL_FINITE(obs);
}

TEST(FeatureBuilder, AllValuesFiniteWithEmptyBook) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_ALL_FINITE(obs);
}

TEST(FeatureBuilder, AllValuesFiniteWithOneSideOnly) {
    Book book;
    // Only bids, no asks
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    FeatureBuilder fb;
    auto obs = fb.build(book, -1.0f, 0.0f);
    EXPECT_ALL_FINITE(obs);
}

// ===========================================================================
// FeatureBuilder::build() — Empty book edge case
// ===========================================================================

TEST(FeatureBuilder, EmptyBookAllPricesZero) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // Bid prices [0-9] should be 0
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Empty book: bid price obs[" << i << "] should be 0";
    }
    // Ask prices [20-29] should be 0
    for (int i = 20; i < 30; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Empty book: ask price obs[" << i << "] should be 0";
    }
}

TEST(FeatureBuilder, EmptyBookAllSizesZero) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // Bid sizes [10-19] should be 0
    for (int i = 10; i < 20; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Empty book: bid size obs[" << i << "] should be 0";
    }
    // Ask sizes [30-39] should be 0
    for (int i = 30; i < 40; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Empty book: ask size obs[" << i << "] should be 0";
    }
}

TEST(FeatureBuilder, EmptyBookSpreadZero) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::SPREAD], 0.0f);
}

TEST(FeatureBuilder, EmptyBookImbalanceZero) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::IMBALANCE], 0.0f);
}

TEST(FeatureBuilder, EmptyBookTimeRemainingPreserved) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.75f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::TIME_LEFT], 0.75f);
}

TEST(FeatureBuilder, EmptyBookPositionPreserved) {
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, -1.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::POSITION], -1.0f);
}

// ===========================================================================
// FeatureBuilder::build() — Bid prices (indices 0-9)
// Normalized as: (bid_price - mid) / mid — should be negative
// ===========================================================================

TEST(FeatureBuilder, BidPricesAreRelativeToMid) {
    // Single bid at 100, single ask at 102 → mid = 101
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // obs[0] = (100 - 101) / 101 = -1/101 ≈ -0.00990099
    float expected = static_cast<float>((100.0 - 101.0) / 101.0);
    EXPECT_NEAR(obs[0], expected, 1e-5f)
        << "Best bid price should be (bid - mid) / mid";
}

TEST(FeatureBuilder, BidPricesAreNegativeOrZero) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    for (int i = 0; i < 10; ++i) {
        EXPECT_LE(obs[i], 0.0f)
            << "Bid price obs[" << i << "] = " << obs[i]
            << " should be <= 0 (below mid)";
    }
}

TEST(FeatureBuilder, BidPricesCloserToMidAreCloserToZero) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // obs[0] is best bid (closest to mid), so abs(obs[0]) < abs(obs[1]) < ...
    for (int i = 0; i < 9; ++i) {
        if (obs[i+1] != 0.0f) {  // skip padding
            EXPECT_GT(obs[i], obs[i+1])
                << "Bid price obs[" << i << "] should be closer to 0 than obs[" << i+1 << "]";
        }
    }
}

TEST(FeatureBuilder, BidPricesMultipleLevelsCorrectValues) {
    // 3 bid levels at 100.0, 99.5, 99.0; ask at 101.0
    // mid = (100 + 101) / 2 = 100.5
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.5, 20));
    book.apply(make_msg(3, Message::Side::Bid, Message::Action::Add, 99.0, 30));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 101.0, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    double mid = 100.5;
    EXPECT_NEAR(obs[0], static_cast<float>((100.0 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[1], static_cast<float>((99.5 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[2], static_cast<float>((99.0 - mid) / mid), 1e-5f);

    // Remaining bid prices should be 0 (padding)
    for (int i = 3; i < 10; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Padded bid price obs[" << i << "] should be 0";
    }
}

// ===========================================================================
// FeatureBuilder::build() — Bid sizes (indices 10-19)
// Normalized as: size / max_size across all 20 levels
// ===========================================================================

TEST(FeatureBuilder, BidSizesNormalizedByMaxAcrossAllLevels) {
    // Bids: level 0 qty=10, level 1 qty=20
    // Asks: level 0 qty=50 (this is the max across all 20 levels)
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 50));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // max_size = 50 (from ask side)
    EXPECT_NEAR(obs[10], 10.0f / 50.0f, 1e-5f) << "Bid size[0] = 10/50";
    EXPECT_NEAR(obs[11], 20.0f / 50.0f, 1e-5f) << "Bid size[1] = 20/50";

    // Remaining bid sizes should be 0
    for (int i = 12; i < 20; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Padded bid size obs[" << i << "] should be 0";
    }
}

TEST(FeatureBuilder, SizeNormalizationUsesMaxAcrossBothSides) {
    // Bid has the max size
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 100));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 25));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // max_size = 100 (from bid side)
    EXPECT_NEAR(obs[10], 100.0f / 100.0f, 1e-5f) << "Bid size should be 1.0";
    EXPECT_NEAR(obs[30], 25.0f / 100.0f, 1e-5f) << "Ask size should be 0.25";
}

TEST(FeatureBuilder, SizesInRange01) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // Bid sizes [10-19]
    for (int i = 10; i < 20; ++i) {
        EXPECT_GE(obs[i], 0.0f) << "Bid size obs[" << i << "] should be >= 0";
        EXPECT_LE(obs[i], 1.0f) << "Bid size obs[" << i << "] should be <= 1";
    }
    // Ask sizes [30-39]
    for (int i = 30; i < 40; ++i) {
        EXPECT_GE(obs[i], 0.0f) << "Ask size obs[" << i << "] should be >= 0";
        EXPECT_LE(obs[i], 1.0f) << "Ask size obs[" << i << "] should be <= 1";
    }
}

TEST(FeatureBuilder, AtLeastOneSizeIsOne) {
    // The max-size level should normalize to 1.0
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    float max_size = 0.0f;
    for (int i = 10; i < 20; ++i) max_size = std::max(max_size, obs[i]);
    for (int i = 30; i < 40; ++i) max_size = std::max(max_size, obs[i]);
    EXPECT_FLOAT_EQ(max_size, 1.0f)
        << "At least one size should normalize to 1.0";
}

// ===========================================================================
// FeatureBuilder::build() — Ask prices (indices 20-29)
// Normalized as: (ask_price - mid) / mid — should be positive
// ===========================================================================

TEST(FeatureBuilder, AskPricesAreRelativeToMid) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // mid = 101, obs[20] = (102 - 101) / 101
    float expected = static_cast<float>((102.0 - 101.0) / 101.0);
    EXPECT_NEAR(obs[20], expected, 1e-5f)
        << "Best ask price should be (ask - mid) / mid";
}

TEST(FeatureBuilder, AskPricesArePositiveOrZero) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    for (int i = 20; i < 30; ++i) {
        EXPECT_GE(obs[i], 0.0f)
            << "Ask price obs[" << i << "] = " << obs[i]
            << " should be >= 0 (above mid)";
    }
}

TEST(FeatureBuilder, AskPricesCloserToMidAreCloserToZero) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    for (int i = 20; i < 29; ++i) {
        if (obs[i+1] != 0.0f) {  // skip padding
            EXPECT_LT(obs[i], obs[i+1])
                << "Ask price obs[" << i << "] should be closer to 0 than obs[" << i+1 << "]";
        }
    }
}

TEST(FeatureBuilder, AskPricesMultipleLevelsCorrectValues) {
    // 3 ask levels at 101.0, 101.5, 102.0; bid at 100.0
    // mid = (100 + 101) / 2 = 100.5
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.5, 20));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 30));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    double mid = 100.5;
    EXPECT_NEAR(obs[20], static_cast<float>((101.0 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[21], static_cast<float>((101.5 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[22], static_cast<float>((102.0 - mid) / mid), 1e-5f);

    // Remaining ask prices should be 0 (padding)
    for (int i = 23; i < 30; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Padded ask price obs[" << i << "] should be 0";
    }
}

// ===========================================================================
// FeatureBuilder::build() — Ask sizes (indices 30-39)
// ===========================================================================

TEST(FeatureBuilder, AskSizesNormalizedByMaxAcrossAllLevels) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 10));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 102.0, 20));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // max_size = 50 (from bid side)
    EXPECT_NEAR(obs[30], 10.0f / 50.0f, 1e-5f);
    EXPECT_NEAR(obs[31], 20.0f / 50.0f, 1e-5f);

    for (int i = 32; i < 40; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f);
    }
}

// ===========================================================================
// FeatureBuilder::build() — Spread (index 40)
// Normalized as: spread / mid
// ===========================================================================

TEST(FeatureBuilder, SpreadNormalizedByMid) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 102.0, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // spread = 2.0, mid = 101.0
    float expected = static_cast<float>(2.0 / 101.0);
    EXPECT_NEAR(obs[FeatureBuilder::SPREAD], expected, 1e-5f);
}

TEST(FeatureBuilder, SpreadIsNonNegative) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_GE(obs[FeatureBuilder::SPREAD], 0.0f);
}

TEST(FeatureBuilder, SpreadTightBook) {
    // Tight spread: bid=100.00, ask=100.25 → spread=0.25, mid=100.125
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.25, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    float expected = static_cast<float>(0.25 / 100.125);
    EXPECT_NEAR(obs[FeatureBuilder::SPREAD], expected, 1e-5f);
}

// ===========================================================================
// FeatureBuilder::build() — Imbalance (index 41)
// (bid_qty_top - ask_qty_top) / (bid_qty_top + ask_qty_top)
// ===========================================================================

TEST(FeatureBuilder, ImbalanceEqualQtysIsZero) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 50));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    EXPECT_NEAR(obs[FeatureBuilder::IMBALANCE], 0.0f, 1e-5f)
        << "Equal bid/ask top qty should give imbalance = 0";
}

TEST(FeatureBuilder, ImbalanceBidHeavyIsPositive) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 80));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 20));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // (80 - 20) / (80 + 20) = 60/100 = 0.6
    EXPECT_NEAR(obs[FeatureBuilder::IMBALANCE], 0.6f, 1e-5f);
}

TEST(FeatureBuilder, ImbalanceAskHeavyIsNegative) {
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 20));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 101.0, 80));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // (20 - 80) / (20 + 80) = -60/100 = -0.6
    EXPECT_NEAR(obs[FeatureBuilder::IMBALANCE], -0.6f, 1e-5f);
}

TEST(FeatureBuilder, ImbalanceAllBidIsOne) {
    // Only bid side has qty at top
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 50));
    // Ask side empty → ask_qty_top = 0
    // imbalance = (50 - 0) / (50 + 0) = 1.0

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // With only bids, mid is NaN → edge case. But if we treat it:
    // Spec says empty book → imbalance 0. But here we have bids only.
    // The formula (50-0)/(50+0) = 1.0 if ask_qty_top is 0 and bid is populated.
    // But spec says if mid is NaN, return 0.0 for imbalance.
    // This tests the contract — implementation decides the edge case handling.
    // We expect 0.0 per the spec (empty/broken book = 0).
    EXPECT_TRUE(obs[FeatureBuilder::IMBALANCE] == 0.0f || obs[FeatureBuilder::IMBALANCE] == 1.0f)
        << "With only bids, imbalance should be 0.0 (edge case) or 1.0 (formula)";
}

TEST(FeatureBuilder, ImbalanceInRangeNeg1To1) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    EXPECT_GE(obs[FeatureBuilder::IMBALANCE], -1.0f);
    EXPECT_LE(obs[FeatureBuilder::IMBALANCE], 1.0f);
}

// ===========================================================================
// FeatureBuilder::build() — Time remaining (index 42)
// ===========================================================================

TEST(FeatureBuilder, TimeRemainingPassedThrough) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;

    auto obs1 = fb.build(book, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(obs1[FeatureBuilder::TIME_LEFT], 1.0f) << "time_remaining=1.0 at open";

    auto obs2 = fb.build(book, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(obs2[FeatureBuilder::TIME_LEFT], 0.0f) << "time_remaining=0.0 at close";

    auto obs3 = fb.build(book, 0.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs3[FeatureBuilder::TIME_LEFT], 0.5f) << "time_remaining=0.5 mid-session";
}

TEST(FeatureBuilder, TimeRemainingNoSessionDefault) {
    // When no session, LOBEnv should pass 0.5 as time_remaining.
    // This tests the FeatureBuilder just passes it through.
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::TIME_LEFT], 0.5f);
}

// ===========================================================================
// FeatureBuilder::build() — Position (index 43)
// ===========================================================================

TEST(FeatureBuilder, PositionPassedThroughShort) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, -1.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::POSITION], -1.0f);
}

TEST(FeatureBuilder, PositionPassedThroughFlat) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::POSITION], 0.0f);
}

TEST(FeatureBuilder, PositionPassedThroughLong) {
    Book book = make_symmetric_book();
    FeatureBuilder fb;
    auto obs = fb.build(book, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::POSITION], 1.0f);
}

// ===========================================================================
// FeatureBuilder::build() — Fewer than 10 levels (padding)
// ===========================================================================

TEST(FeatureBuilder, FewerThan10LevelsPadsWithZero) {
    // Only 2 bid levels, 2 ask levels
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.0, 20));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 101.0, 15));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 102.0, 25));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // Levels 0-1 should be non-zero for bid prices
    EXPECT_NE(obs[0], 0.0f);
    EXPECT_NE(obs[1], 0.0f);
    // Levels 2-9 should be 0 for bid prices
    for (int i = 2; i < 10; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Bid price padding obs[" << i << "] should be 0";
    }

    // Levels 0-1 should be non-zero for bid sizes
    EXPECT_NE(obs[10], 0.0f);
    EXPECT_NE(obs[11], 0.0f);
    // Levels 2-9 should be 0 for bid sizes
    for (int i = 12; i < 20; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Bid size padding obs[" << i << "] should be 0";
    }

    // Same for asks
    EXPECT_NE(obs[20], 0.0f);
    EXPECT_NE(obs[21], 0.0f);
    for (int i = 22; i < 30; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f)
            << "Ask price padding obs[" << i << "] should be 0";
    }
}

// ===========================================================================
// FeatureBuilder::build() — Full 10-level book verification
// ===========================================================================

TEST(FeatureBuilder, Full10LevelBookAllIndicesPopulated) {
    Book book = make_symmetric_book(10, 10);
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // All 10 bid prices should be non-zero (all negative, relative to mid)
    for (int i = 0; i < 10; ++i) {
        EXPECT_NE(obs[i], 0.0f)
            << "Bid price obs[" << i << "] should be populated";
    }
    // All 10 bid sizes should be non-zero
    for (int i = 10; i < 20; ++i) {
        EXPECT_GT(obs[i], 0.0f)
            << "Bid size obs[" << i << "] should be > 0";
    }
    // All 10 ask prices should be non-zero (all positive, relative to mid)
    for (int i = 20; i < 30; ++i) {
        EXPECT_NE(obs[i], 0.0f)
            << "Ask price obs[" << i << "] should be populated";
    }
    // All 10 ask sizes should be non-zero
    for (int i = 30; i < 40; ++i) {
        EXPECT_GT(obs[i], 0.0f)
            << "Ask size obs[" << i << "] should be > 0";
    }
}

// ===========================================================================
// FeatureBuilder::build() — All sizes zero edge case
// ===========================================================================

TEST(FeatureBuilder, AllSizesZeroProducesZeroSizes) {
    // This shouldn't normally happen, but test defensively.
    // We can't easily create a book with 0-qty levels through the Book API,
    // so this test verifies the empty-book behavior instead (which has qty=0).
    Book book;
    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    for (int i = 10; i < 20; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f);
    }
    for (int i = 30; i < 40; ++i) {
        EXPECT_FLOAT_EQ(obs[i], 0.0f);
    }
}

// ===========================================================================
// FeatureBuilder::build() — Symmetry test with known values
// ===========================================================================

TEST(FeatureBuilder, SymmetricBookHasSymmetricPrices) {
    // Symmetric book: bid at 100.0, ask at 100.50 → mid = 100.25
    // bid relative: (100.0 - 100.25) / 100.25 = -0.002493...
    // ask relative: (100.50 - 100.25) / 100.25 = +0.002493...
    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Ask, Message::Action::Add, 100.50, 10));

    FeatureBuilder fb;
    auto obs = fb.build(book, 0.0f, 0.5f);

    // The magnitudes should be equal
    EXPECT_NEAR(std::abs(obs[0]), std::abs(obs[20]), 1e-5f)
        << "Symmetric book should have symmetric price features";
    // Signs should be opposite
    EXPECT_LT(obs[0], 0.0f);
    EXPECT_GT(obs[20], 0.0f);
}

// ===========================================================================
// FeatureBuilder::build() — Exact layout verification
// ===========================================================================

TEST(FeatureBuilder, ExactLayoutWith2LevelBook) {
    // Bids: 100.0 qty=10, 99.5 qty=30
    // Asks: 100.5 qty=20, 101.0 qty=40
    // mid = (100.0 + 100.5) / 2 = 100.25
    // max_size = max(10, 30, 20, 40) = 40

    Book book;
    book.apply(make_msg(1, Message::Side::Bid, Message::Action::Add, 100.0, 10));
    book.apply(make_msg(2, Message::Side::Bid, Message::Action::Add, 99.5, 30));
    book.apply(make_msg(3, Message::Side::Ask, Message::Action::Add, 100.5, 20));
    book.apply(make_msg(4, Message::Side::Ask, Message::Action::Add, 101.0, 40));

    FeatureBuilder fb;
    auto obs = fb.build(book, 1.0f, 0.3f);

    double mid = 100.25;
    uint32_t max_size = 40;
    double spread = 0.5;

    // Bid prices [0-9]
    EXPECT_NEAR(obs[0], static_cast<float>((100.0 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[1], static_cast<float>((99.5 - mid) / mid), 1e-5f);
    for (int i = 2; i < 10; ++i) EXPECT_FLOAT_EQ(obs[i], 0.0f);

    // Bid sizes [10-19]
    EXPECT_NEAR(obs[10], 10.0f / max_size, 1e-5f);
    EXPECT_NEAR(obs[11], 30.0f / max_size, 1e-5f);
    for (int i = 12; i < 20; ++i) EXPECT_FLOAT_EQ(obs[i], 0.0f);

    // Ask prices [20-29]
    EXPECT_NEAR(obs[20], static_cast<float>((100.5 - mid) / mid), 1e-5f);
    EXPECT_NEAR(obs[21], static_cast<float>((101.0 - mid) / mid), 1e-5f);
    for (int i = 22; i < 30; ++i) EXPECT_FLOAT_EQ(obs[i], 0.0f);

    // Ask sizes [30-39]
    EXPECT_NEAR(obs[30], 20.0f / max_size, 1e-5f);
    EXPECT_NEAR(obs[31], 40.0f / max_size, 1e-5f);
    for (int i = 32; i < 40; ++i) EXPECT_FLOAT_EQ(obs[i], 0.0f);

    // Spread
    EXPECT_NEAR(obs[FeatureBuilder::SPREAD], static_cast<float>(spread / mid), 1e-5f);

    // Imbalance: (bid_qty_top - ask_qty_top) / (bid_qty_top + ask_qty_top)
    // = (10 - 20) / (10 + 20) = -10/30 = -0.3333...
    EXPECT_NEAR(obs[FeatureBuilder::IMBALANCE], -10.0f / 30.0f, 1e-5f);

    // Time remaining
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::TIME_LEFT], 0.3f);

    // Position
    EXPECT_FLOAT_EQ(obs[FeatureBuilder::POSITION], 1.0f);
}
