#include <gtest/gtest.h>
#include "lob/source.h"
#include "lob/message.h"
#include "synthetic_source.h"

// ===========================================================================
// SyntheticSource: Determinism
// ===========================================================================

TEST(SyntheticSource, ProducesDeterministicSequence) {
    SyntheticSource src1;
    SyntheticSource src2;

    Message m1, m2;
    int count = 0;
    while (src1.next(m1) && src2.next(m2)) {
        EXPECT_EQ(m1.order_id, m2.order_id) << "Mismatch at message " << count;
        EXPECT_EQ(static_cast<int>(m1.side), static_cast<int>(m2.side))
            << "Side mismatch at message " << count;
        EXPECT_EQ(static_cast<int>(m1.action), static_cast<int>(m2.action))
            << "Action mismatch at message " << count;
        EXPECT_DOUBLE_EQ(m1.price, m2.price) << "Price mismatch at message " << count;
        EXPECT_EQ(m1.qty, m2.qty) << "Qty mismatch at message " << count;
        EXPECT_EQ(m1.ts_ns, m2.ts_ns) << "Timestamp mismatch at message " << count;
        ++count;
    }
    EXPECT_GT(count, 0) << "Source should produce at least one message";
}

TEST(SyntheticSource, SameSequenceWithSameSeed) {
    SyntheticSource src1(42);
    SyntheticSource src2(42);

    Message m1, m2;
    while (src1.next(m1) && src2.next(m2)) {
        EXPECT_EQ(m1.order_id, m2.order_id);
        EXPECT_EQ(static_cast<int>(m1.action), static_cast<int>(m2.action));
        EXPECT_DOUBLE_EQ(m1.price, m2.price);
        EXPECT_EQ(m1.qty, m2.qty);
    }
    // Both should be exhausted at same time
    EXPECT_FALSE(src1.next(m1));
    EXPECT_FALSE(src2.next(m2));
}

TEST(SyntheticSource, DifferentSeedsProduceDifferentSequences) {
    SyntheticSource src1(1);
    SyntheticSource src2(999);

    Message m1, m2;
    bool any_differ = false;
    while (src1.next(m1) && src2.next(m2)) {
        if (m1.price != m2.price || m1.qty != m2.qty) {
            any_differ = true;
            break;
        }
    }
    EXPECT_TRUE(any_differ) << "Different seeds should produce different sequences";
}

// ===========================================================================
// SyntheticSource: Reset
// ===========================================================================

TEST(SyntheticSource, ResetRewindsToBeginning) {
    SyntheticSource src;

    // Read first message
    Message first_pass;
    ASSERT_TRUE(src.next(first_pass));

    // Read a few more to advance state
    Message discard;
    for (int i = 0; i < 5; ++i) src.next(discard);

    // Reset
    src.reset();

    // First message should be identical
    Message after_reset;
    ASSERT_TRUE(src.next(after_reset));
    EXPECT_EQ(first_pass.order_id, after_reset.order_id);
    EXPECT_EQ(static_cast<int>(first_pass.action), static_cast<int>(after_reset.action));
    EXPECT_DOUBLE_EQ(first_pass.price, after_reset.price);
    EXPECT_EQ(first_pass.qty, after_reset.qty);
}

TEST(SyntheticSource, ResetProducesFullSequenceAgain) {
    SyntheticSource src;

    // Drain entire source
    Message m;
    int count1 = 0;
    while (src.next(m)) ++count1;

    // Reset and drain again
    src.reset();
    int count2 = 0;
    while (src.next(m)) ++count2;

    EXPECT_EQ(count1, count2);
    EXPECT_GT(count1, 0);
}

// ===========================================================================
// SyntheticSource: Message count and structure
// ===========================================================================

TEST(SyntheticSource, ProducesApproximately100Messages) {
    SyntheticSource src;
    Message m;
    int count = 0;
    while (src.next(m)) ++count;

    // Spec says ~100 messages
    EXPECT_GE(count, 50) << "Should produce at least 50 messages";
    EXPECT_LE(count, 200) << "Should produce at most 200 messages";
}

TEST(SyntheticSource, ReturnsFalseWhenExhausted) {
    SyntheticSource src;
    Message m;
    while (src.next(m)) {}

    // Subsequent calls should also return false
    EXPECT_FALSE(src.next(m));
    EXPECT_FALSE(src.next(m));
}

TEST(SyntheticSource, AllMessagesHaveValidFields) {
    SyntheticSource src;
    Message m;
    uint64_t prev_ts = 0;
    while (src.next(m)) {
        // Side must be Bid or Ask
        EXPECT_TRUE(m.side == Message::Side::Bid || m.side == Message::Side::Ask);

        // Action must be one of the four valid actions
        EXPECT_TRUE(m.action == Message::Action::Add ||
                    m.action == Message::Action::Cancel ||
                    m.action == Message::Action::Modify ||
                    m.action == Message::Action::Trade);

        // Price must be positive
        EXPECT_GT(m.price, 0.0);

        // Qty must be positive for Add messages
        if (m.action == Message::Action::Add) {
            EXPECT_GT(m.qty, 0u);
        }

        // Timestamps should be non-decreasing
        EXPECT_GE(m.ts_ns, prev_ts) << "Timestamps must be non-decreasing";
        prev_ts = m.ts_ns;
    }
}

TEST(SyntheticSource, Phase1BuildsBothSides) {
    // First 10 messages should add 5 bid + 5 ask levels
    SyntheticSource src;
    Message m;
    int bid_adds = 0, ask_adds = 0;

    for (int i = 0; i < 10 && src.next(m); ++i) {
        if (m.action == Message::Action::Add) {
            if (m.side == Message::Side::Bid) ++bid_adds;
            else ++ask_adds;
        }
    }

    EXPECT_EQ(bid_adds, 5) << "Phase 1 should add 5 bid levels";
    EXPECT_EQ(ask_adds, 5) << "Phase 1 should add 5 ask levels";
}

TEST(SyntheticSource, ImplementsIMessageSourceInterface) {
    // SyntheticSource should be usable through the IMessageSource interface
    std::unique_ptr<IMessageSource> src = std::make_unique<SyntheticSource>();
    Message m;
    EXPECT_TRUE(src->next(m));
    src->reset();
    EXPECT_TRUE(src->next(m));
}

TEST(SyntheticSource, PricesAroundMid1000WithQuarterTick) {
    SyntheticSource src;
    Message m;

    // Read phase 1 messages (first 10)
    for (int i = 0; i < 10 && src.next(m); ++i) {
        if (m.action == Message::Action::Add) {
            // Prices should be near 1000.00
            EXPECT_GT(m.price, 990.0) << "Prices should be near mid=1000";
            EXPECT_LT(m.price, 1010.0) << "Prices should be near mid=1000";

            // Prices should be on 0.25 tick grid
            double ticks = m.price / 0.25;
            EXPECT_DOUBLE_EQ(ticks, std::round(ticks))
                << "Price " << m.price << " should be on 0.25 tick grid";
        }
    }
}
