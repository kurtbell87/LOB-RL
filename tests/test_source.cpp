#include <gtest/gtest.h>
#include "synthetic_source.h"

using namespace lob;

TEST(SyntheticSourceTest, Construction) {
    SyntheticSource source(42, 100);
    EXPECT_EQ(source.message_count(), 100);
    EXPECT_TRUE(source.has_next());
}

TEST(SyntheticSourceTest, Determinism) {
    SyntheticSource source1(42, 100);
    SyntheticSource source2(42, 100);

    while (source1.has_next() && source2.has_next()) {
        MBOMessage msg1 = source1.next();
        MBOMessage msg2 = source2.next();

        EXPECT_EQ(msg1.timestamp_ns, msg2.timestamp_ns);
        EXPECT_EQ(msg1.order_id, msg2.order_id);
        EXPECT_EQ(msg1.price, msg2.price);
        EXPECT_EQ(msg1.quantity, msg2.quantity);
        EXPECT_EQ(msg1.side, msg2.side);
        EXPECT_EQ(msg1.action, msg2.action);
    }

    EXPECT_FALSE(source1.has_next());
    EXPECT_FALSE(source2.has_next());
}

TEST(SyntheticSourceTest, DifferentSeedsProduceDifferentSequences) {
    SyntheticSource source1(42, 100);
    SyntheticSource source2(43, 100);

    bool found_difference = false;
    while (source1.has_next() && source2.has_next()) {
        MBOMessage msg1 = source1.next();
        MBOMessage msg2 = source2.next();

        if (msg1.order_id != msg2.order_id || msg1.price != msg2.price) {
            found_difference = true;
            break;
        }
    }

    EXPECT_TRUE(found_difference);
}

TEST(SyntheticSourceTest, Reset) {
    SyntheticSource source(42, 100);

    // Consume some messages
    for (int i = 0; i < 50 && source.has_next(); ++i) {
        source.next();
    }

    // Reset
    source.reset();
    EXPECT_TRUE(source.has_next());

    // Should get the same first message
    SyntheticSource fresh(42, 100);
    MBOMessage msg1 = source.next();
    MBOMessage msg2 = fresh.next();

    EXPECT_EQ(msg1.timestamp_ns, msg2.timestamp_ns);
    EXPECT_EQ(msg1.order_id, msg2.order_id);
}

TEST(SyntheticSourceTest, MessageValidity) {
    SyntheticSource source(42, 1000);

    uint64_t prev_timestamp = 0;
    while (source.has_next()) {
        MBOMessage msg = source.next();

        // Timestamps should be non-decreasing
        EXPECT_GE(msg.timestamp_ns, prev_timestamp);
        prev_timestamp = msg.timestamp_ns;

        // Order IDs should be positive
        EXPECT_GT(msg.order_id, 0ULL);

        // Side and action should be valid
        EXPECT_TRUE(msg.side == Side::Bid || msg.side == Side::Ask);
        EXPECT_TRUE(
            msg.action == Action::Add ||
            msg.action == Action::Cancel ||
            msg.action == Action::Modify ||
            msg.action == Action::Trade ||
            msg.action == Action::Clear
        );
    }
}

TEST(SyntheticSourceTest, StartsWithAdds) {
    SyntheticSource source(42, 100);

    // First 20 messages should be adds (10 bids + 10 asks)
    for (int i = 0; i < 20 && source.has_next(); ++i) {
        MBOMessage msg = source.next();
        EXPECT_EQ(msg.action, Action::Add);
    }
}
