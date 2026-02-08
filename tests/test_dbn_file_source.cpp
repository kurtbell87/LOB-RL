#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>
#include "lob/source.h"
#include "lob/message.h"
#include "lob/book.h"
#include "lob/precompute.h"
#include "dbn_file_source.h"
#include "test_helpers.h"

// ===========================================================================
// DbnFileSource: Construction
//
// DbnFileSource(path, instrument_id) reads .dbn.zst files via databento-cpp.
// ===========================================================================

TEST(DbnFileSource, ConstructsFromDbnZstPath) {
    // This test requires a real .dbn.zst fixture file.
    // The fixture should be placed at tests/fixtures/test_data.mbo.dbn.zst
    // by the GREEN phase implementation.
    //
    // For now, we test construction with a known path pattern.
    // If the file doesn't exist, the constructor should throw.
    EXPECT_THROW(
        DbnFileSource src("/nonexistent/path/no_such_file.dbn.zst", 12345),
        std::runtime_error
    );
}

TEST(DbnFileSource, ThrowsOnInvalidPath) {
    EXPECT_THROW(
        DbnFileSource src("/tmp/this_does_not_exist_at_all.dbn.zst", 0),
        std::runtime_error
    );
}

// ===========================================================================
// DbnFileSource: IMessageSource Interface
// ===========================================================================

TEST(DbnFileSource, ImplementsIMessageSourceInterface) {
    // DbnFileSource should be usable through IMessageSource pointer.
    // This is a compile-time check that DbnFileSource inherits IMessageSource.
    // At runtime, we just verify the type relationship.
    //
    // We can't construct a real one without a fixture file,
    // so we test the interface through a different path:
    // unique_ptr<IMessageSource> should accept DbnFileSource*.
    // This test will fail to link/compile if DbnFileSource doesn't inherit IMessageSource.
    static_assert(std::is_base_of_v<IMessageSource, DbnFileSource>,
                  "DbnFileSource must inherit from IMessageSource");
}

// ===========================================================================
// DbnFileSource with fixture file tests
//
// These tests require a .dbn.zst fixture file at:
//   tests/fixtures/test_mes.mbo.dbn.zst
//
// The fixture should contain a realistic sequence of MBO messages:
//   - Multiple Add messages building a book (bids + asks)
//   - Some Cancel, Modify, Trade, Fill actions
//   - A mix of instrument IDs (some matching, some not)
//   - Some Clear ('R') and None ('N') actions that should be filtered
//
// The GREEN phase must create this fixture (either a real tiny .dbn.zst
// or by building it programmatically using databento-cpp's writer).
//
// For the fixture we assume:
//   - instrument_id = 12345 for matching records
//   - At least 10 matching records after filtering
//   - Contains at least one Fill mapped to Trade
//   - Contains at least one Side::None mapped to Bid
// ===========================================================================

static std::string dbn_fixture_path() {
    return fixture_path("test_mes.mbo.dbn.zst");
}

class DbnFileSourceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip tests if fixture doesn't exist (can't test without data)
        // The GREEN phase must create this fixture.
        if (!std::filesystem::exists(dbn_fixture_path())) {
            GTEST_SKIP() << "Fixture file not found: " << dbn_fixture_path()
                         << " (GREEN phase must create it)";
        }
    }
};

// --- Iteration Tests ---

TEST_F(DbnFileSourceFixture, NextReturnsMessages) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    int count = 0;
    while (src.next(m)) ++count;
    EXPECT_GT(count, 0) << "DbnFileSource should produce at least one message";
}

TEST_F(DbnFileSourceFixture, NextReturnsFalseWhenExhausted) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    while (src.next(m)) {}

    // Subsequent calls should also return false
    EXPECT_FALSE(src.next(m));
    EXPECT_FALSE(src.next(m));
}

TEST_F(DbnFileSourceFixture, NextSkipsClearAndNoneActions) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    while (src.next(m)) {
        // Should never see actions other than Add, Cancel, Modify, Trade
        EXPECT_TRUE(m.action == Message::Action::Add ||
                    m.action == Message::Action::Cancel ||
                    m.action == Message::Action::Modify ||
                    m.action == Message::Action::Trade)
            << "DbnFileSource should filter out Clear and None actions";
    }
}

TEST_F(DbnFileSourceFixture, NextFiltersByInstrumentId) {
    // Read with instrument_id=12345 — should get filtered results
    DbnFileSource src_filtered(dbn_fixture_path(), 12345);
    Message m;
    int count_filtered = 0;
    while (src_filtered.next(m)) ++count_filtered;

    // Read with instrument_id=0 — should get all instruments
    DbnFileSource src_all(dbn_fixture_path(), 0);
    int count_all = 0;
    while (src_all.next(m)) ++count_all;

    EXPECT_LE(count_filtered, count_all)
        << "Filtered count should be <= unfiltered count";
}

TEST_F(DbnFileSourceFixture, AllMessagesHaveValidFields) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    uint64_t prev_ts = 0;
    while (src.next(m)) {
        // Side must be Bid or Ask (never None after mapping)
        EXPECT_TRUE(m.side == Message::Side::Bid || m.side == Message::Side::Ask);

        // Action must be one of the four valid actions
        EXPECT_TRUE(m.action == Message::Action::Add ||
                    m.action == Message::Action::Cancel ||
                    m.action == Message::Action::Modify ||
                    m.action == Message::Action::Trade);

        // Price must be finite (could be 0 for some edge cases)
        EXPECT_TRUE(std::isfinite(m.price));

        // Timestamps should be non-decreasing
        EXPECT_GE(m.ts_ns, prev_ts) << "Timestamps must be non-decreasing";
        prev_ts = m.ts_ns;
    }
}

TEST_F(DbnFileSourceFixture, FillActionMappedToTrade) {
    // The fixture should contain at least one Fill action.
    // After mapping, we should see it as Trade.
    // We can't directly detect that a Trade was originally a Fill,
    // but we can verify no messages have an action outside the valid set.
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    int trade_count = 0;
    while (src.next(m)) {
        if (m.action == Message::Action::Trade) ++trade_count;
    }
    EXPECT_GT(trade_count, 0) << "Fixture should contain at least one Trade/Fill";
}

// --- Reset Tests ---

TEST_F(DbnFileSourceFixture, ResetRewindsToBeginning) {
    DbnFileSource src(dbn_fixture_path(), 12345);

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
    EXPECT_EQ(static_cast<int>(first_pass.side), static_cast<int>(after_reset.side));
    EXPECT_EQ(static_cast<int>(first_pass.action), static_cast<int>(after_reset.action));
    EXPECT_DOUBLE_EQ(first_pass.price, after_reset.price);
    EXPECT_EQ(first_pass.qty, after_reset.qty);
    EXPECT_EQ(first_pass.ts_ns, after_reset.ts_ns);
}

TEST_F(DbnFileSourceFixture, ResetProducesFullSequenceAgain) {
    DbnFileSource src(dbn_fixture_path(), 12345);

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

TEST_F(DbnFileSourceFixture, ResetProducesDeterministicSequence) {
    DbnFileSource src(dbn_fixture_path(), 12345);

    // First pass: collect all messages
    std::vector<Message> pass1;
    Message m;
    while (src.next(m)) pass1.push_back(m);

    // Reset
    src.reset();

    // Second pass: compare
    std::vector<Message> pass2;
    while (src.next(m)) pass2.push_back(m);

    ASSERT_EQ(pass1.size(), pass2.size());
    for (size_t i = 0; i < pass1.size(); ++i) {
        EXPECT_EQ(pass1[i].order_id, pass2[i].order_id) << "Mismatch at " << i;
        EXPECT_DOUBLE_EQ(pass1[i].price, pass2[i].price) << "Mismatch at " << i;
        EXPECT_EQ(pass1[i].qty, pass2[i].qty) << "Mismatch at " << i;
        EXPECT_EQ(pass1[i].ts_ns, pass2[i].ts_ns) << "Mismatch at " << i;
    }
}

// --- Price Conversion Tests ---

TEST_F(DbnFileSourceFixture, PricesAreOnQuarterTickGrid) {
    // /MES tick size is $0.25 — all prices should be multiples of 0.25
    DbnFileSource src(dbn_fixture_path(), 12345);
    Message m;
    while (src.next(m)) {
        if (m.price > 0.0) {
            double ticks = m.price / 0.25;
            double rounded = std::round(ticks);
            EXPECT_NEAR(ticks, rounded, 1e-6)
                << "Price " << m.price << " is not on 0.25 tick grid";
        }
    }
}

// --- Integration with Book ---

TEST_F(DbnFileSourceFixture, IntegrationWithBookProducesValidBBO) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Book book;
    Message m;

    while (src.next(m)) {
        book.apply(m);
    }

    // After processing all messages, book should have valid BBO
    // (assuming the fixture contains Add messages for both sides)
    EXPECT_FALSE(std::isnan(book.best_bid()));
    EXPECT_FALSE(std::isnan(book.best_ask()));
    EXPECT_GT(book.best_bid(), 0.0);
    EXPECT_GT(book.best_ask(), 0.0);
    EXPECT_LE(book.best_bid(), book.best_ask()) << "Bid should not exceed ask";
}

TEST_F(DbnFileSourceFixture, IntegrationResetAndReplayMatchesOriginal) {
    DbnFileSource src(dbn_fixture_path(), 12345);
    Book book1, book2;
    Message m;

    // First pass
    while (src.next(m)) book1.apply(m);

    // Reset source and replay into a fresh book
    src.reset();
    while (src.next(m)) book2.apply(m);

    EXPECT_DOUBLE_EQ(book1.best_bid(), book2.best_bid());
    EXPECT_DOUBLE_EQ(book1.best_ask(), book2.best_ask());
    EXPECT_DOUBLE_EQ(book1.spread(), book2.spread());
}

// ===========================================================================
// DbnFileSource: Precompute Integration
//
// Spec: precompute() string overload should work with .dbn.zst paths.
// The string overload creates a DbnFileSource internally.
// ===========================================================================

TEST_F(DbnFileSourceFixture, PrecomputeStringOverloadWorkWithDbnZst) {
    // The string overload of precompute() should detect .dbn.zst extension
    // and create a DbnFileSource instead of BinaryFileSource.
    SessionConfig cfg = SessionConfig::default_rth();

    // This call should NOT throw — it should create a DbnFileSource.
    PrecomputedDay day = precompute(dbn_fixture_path(), cfg);

    // The fixture may or may not have RTH data, so num_steps could be 0.
    // But the call should succeed without errors.
    EXPECT_GE(day.num_steps, 0);

    // If we got steps, verify structural integrity
    if (day.num_steps > 0) {
        EXPECT_EQ(day.obs.size(), static_cast<size_t>(day.num_steps * 43));
        EXPECT_EQ(day.mid.size(), static_cast<size_t>(day.num_steps));
        EXPECT_EQ(day.spread.size(), static_cast<size_t>(day.num_steps));

        for (size_t i = 0; i < day.obs.size(); ++i) {
            EXPECT_TRUE(std::isfinite(day.obs[i]))
                << "obs[" << i << "] from .dbn.zst is not finite";
        }
    }
}

// ===========================================================================
// DbnFileSource: precompute() with instrument_id
//
// Spec: precompute(path, cfg, instrument_id) — the new signature
// ===========================================================================

TEST_F(DbnFileSourceFixture, PrecomputeWithInstrumentIdOverload) {
    SessionConfig cfg = SessionConfig::default_rth();

    // Call with explicit instrument_id
    PrecomputedDay day = precompute(dbn_fixture_path(), cfg, 12345);

    EXPECT_GE(day.num_steps, 0);

    if (day.num_steps > 0) {
        EXPECT_EQ(day.obs.size(), static_cast<size_t>(day.num_steps * 43));
        EXPECT_EQ(day.mid.size(), static_cast<size_t>(day.num_steps));
        EXPECT_EQ(day.spread.size(), static_cast<size_t>(day.num_steps));
    }
}

TEST_F(DbnFileSourceFixture, PrecomputeWithWrongInstrumentIdProducesFewerSteps) {
    SessionConfig cfg = SessionConfig::default_rth();

    // Correct instrument_id
    PrecomputedDay day_correct = precompute(dbn_fixture_path(), cfg, 12345);

    // Wrong instrument_id — should filter out all records → 0 steps
    PrecomputedDay day_wrong = precompute(dbn_fixture_path(), cfg, 99999);

    EXPECT_LE(day_wrong.num_steps, day_correct.num_steps)
        << "Wrong instrument_id should produce fewer (or equal) steps";
}
