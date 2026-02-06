#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "lob/session.h"
#include "test_helpers.h"

// ===========================================================================
// SessionConfig: default_rth()
// ===========================================================================

TEST(SessionConfig, DefaultRthReturnsCorrectOpenTime) {
    SessionConfig cfg = SessionConfig::default_rth();
    EXPECT_EQ(cfg.rth_open_ns, RTH_OPEN_NS)
        << "default RTH open should be 13:30 UTC in nanoseconds-since-midnight";
}

TEST(SessionConfig, DefaultRthReturnsCorrectCloseTime) {
    SessionConfig cfg = SessionConfig::default_rth();
    EXPECT_EQ(cfg.rth_close_ns, RTH_CLOSE_NS)
        << "default RTH close should be 20:00 UTC in nanoseconds-since-midnight";
}

TEST(SessionConfig, DefaultRthWarmupIsAllPreMarket) {
    SessionConfig cfg = SessionConfig::default_rth();
    EXPECT_EQ(cfg.warmup_messages, -1)
        << "default warmup should be -1 (all pre-RTH messages)";
}

// ===========================================================================
// SessionConfig: Custom values
// ===========================================================================

TEST(SessionConfig, CustomConfigStoresValues) {
    SessionConfig cfg;
    cfg.rth_open_ns = 14ULL * NS_PER_HOUR;       // 14:00 UTC
    cfg.rth_close_ns = 21ULL * NS_PER_HOUR;      // 21:00 UTC
    cfg.warmup_messages = 500;

    EXPECT_EQ(cfg.rth_open_ns, 14ULL * NS_PER_HOUR);
    EXPECT_EQ(cfg.rth_close_ns, 21ULL * NS_PER_HOUR);
    EXPECT_EQ(cfg.warmup_messages, 500);
}

TEST(SessionConfig, CustomConfigZeroWarmup) {
    SessionConfig cfg;
    cfg.rth_open_ns = RTH_OPEN_NS;
    cfg.rth_close_ns = RTH_CLOSE_NS;
    cfg.warmup_messages = 0;

    EXPECT_EQ(cfg.warmup_messages, 0);
}

// ===========================================================================
// SessionFilter: Construction
// ===========================================================================

TEST(SessionFilter, ConstructsWithDefaultConfig) {
    SessionFilter filter;
    // Should not crash, uses default_rth()
    EXPECT_EQ(filter.rth_open_ns(), RTH_OPEN_NS);
    EXPECT_EQ(filter.rth_close_ns(), RTH_CLOSE_NS);
}

TEST(SessionFilter, ConstructsWithCustomConfig) {
    SessionConfig cfg;
    cfg.rth_open_ns = 14ULL * NS_PER_HOUR;
    cfg.rth_close_ns = 21ULL * NS_PER_HOUR;
    cfg.warmup_messages = 0;

    SessionFilter filter(cfg);
    EXPECT_EQ(filter.rth_open_ns(), 14ULL * NS_PER_HOUR);
    EXPECT_EQ(filter.rth_close_ns(), 21ULL * NS_PER_HOUR);
}

TEST(SessionFilter, RthDurationIsCorrect) {
    SessionFilter filter;
    EXPECT_EQ(filter.rth_duration_ns(), RTH_DURATION_NS);
}

// ===========================================================================
// SessionFilter: classify()
// ===========================================================================

TEST(SessionFilter, ClassifyPreMarketBeforeOpen) {
    SessionFilter filter;
    // 10:00 UTC = well before RTH open
    uint64_t ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::PreMarket);
}

TEST(SessionFilter, ClassifyExactlyAtOpenIsRTH) {
    SessionFilter filter;
    // Exactly 13:30:00.000000000 UTC
    uint64_t ts = DAY_BASE_NS + RTH_OPEN_NS;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::RTH)
        << "Timestamp exactly at open should be classified as RTH";
}

TEST(SessionFilter, ClassifyDuringRTH) {
    SessionFilter filter;
    // 15:00 UTC = mid-session
    uint64_t ts = DAY_BASE_NS + 15ULL * NS_PER_HOUR;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::RTH);
}

TEST(SessionFilter, ClassifyOneNanosecondBeforeClose) {
    SessionFilter filter;
    // 19:59:59.999999999 UTC = last nanosecond of RTH
    uint64_t ts = DAY_BASE_NS + RTH_CLOSE_NS - 1;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::RTH);
}

TEST(SessionFilter, ClassifyExactlyAtCloseIsPostMarket) {
    SessionFilter filter;
    // Exactly 20:00:00.000000000 UTC
    uint64_t ts = DAY_BASE_NS + RTH_CLOSE_NS;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::PostMarket)
        << "Timestamp exactly at close should be PostMarket";
}

TEST(SessionFilter, ClassifyPostMarketAfterClose) {
    SessionFilter filter;
    // 22:00 UTC = well after RTH close
    uint64_t ts = DAY_BASE_NS + 22ULL * NS_PER_HOUR;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::PostMarket);
}

TEST(SessionFilter, ClassifyMidnightIsPreMarket) {
    SessionFilter filter;
    // Exactly midnight UTC
    uint64_t ts = DAY_BASE_NS;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::PreMarket);
}

TEST(SessionFilter, ClassifyOneNanosecondBeforeOpenIsPreMarket) {
    SessionFilter filter;
    uint64_t ts = DAY_BASE_NS + RTH_OPEN_NS - 1;
    EXPECT_EQ(filter.classify(ts), SessionFilter::Phase::PreMarket);
}

// ===========================================================================
// SessionFilter: time_of_day_ns()
// ===========================================================================

TEST(SessionFilter, TimeOfDayAtMidnight) {
    // Midnight = 0 ns since midnight
    uint64_t ts = DAY_BASE_NS;
    EXPECT_EQ(SessionFilter::time_of_day_ns(ts), 0ULL);
}

TEST(SessionFilter, TimeOfDayAtNoon) {
    uint64_t ts = DAY_BASE_NS + 12ULL * NS_PER_HOUR;
    EXPECT_EQ(SessionFilter::time_of_day_ns(ts), 12ULL * NS_PER_HOUR);
}

TEST(SessionFilter, TimeOfDayAtRthOpen) {
    uint64_t ts = DAY_BASE_NS + RTH_OPEN_NS;
    EXPECT_EQ(SessionFilter::time_of_day_ns(ts), RTH_OPEN_NS);
}

TEST(SessionFilter, TimeOfDayAtEndOfDay) {
    // 23:59:59.999999999 UTC
    uint64_t ts = DAY_BASE_NS + 24ULL * NS_PER_HOUR - 1;
    EXPECT_EQ(SessionFilter::time_of_day_ns(ts), 24ULL * NS_PER_HOUR - 1);
}

TEST(SessionFilter, TimeOfDayDifferentDaysSameTime) {
    // Two different days at the same time should give the same time_of_day
    uint64_t day1_ts = DAY_BASE_NS + 15ULL * NS_PER_HOUR + 42ULL * NS_PER_MIN;
    uint64_t day2_ts = day1_ts + 24ULL * NS_PER_HOUR;  // next day, same time

    EXPECT_EQ(SessionFilter::time_of_day_ns(day1_ts),
              SessionFilter::time_of_day_ns(day2_ts));
}

TEST(SessionFilter, TimeOfDayWithSubSecondPrecision) {
    // 13:30:00.123456789 UTC
    uint64_t sub_second = 123456789ULL;
    uint64_t ts = DAY_BASE_NS + RTH_OPEN_NS + sub_second;
    EXPECT_EQ(SessionFilter::time_of_day_ns(ts), RTH_OPEN_NS + sub_second);
}

// ===========================================================================
// SessionFilter: session_progress()
// ===========================================================================

TEST(SessionFilter, SessionProgressAtOpenIsZero) {
    SessionFilter filter;
    uint64_t ts = DAY_BASE_NS + RTH_OPEN_NS;
    EXPECT_FLOAT_EQ(filter.session_progress(ts), 0.0f);
}

TEST(SessionFilter, SessionProgressAtCloseIsOne) {
    SessionFilter filter;
    uint64_t ts = DAY_BASE_NS + RTH_CLOSE_NS;
    EXPECT_FLOAT_EQ(filter.session_progress(ts), 1.0f);
}

TEST(SessionFilter, SessionProgressAtMidpointIsHalf) {
    SessionFilter filter;
    uint64_t midpoint = RTH_OPEN_NS + RTH_DURATION_NS / 2;
    uint64_t ts = DAY_BASE_NS + midpoint;
    EXPECT_NEAR(filter.session_progress(ts), 0.5f, 1e-5f);
}

TEST(SessionFilter, SessionProgressBeforeOpenClampsToZero) {
    SessionFilter filter;
    uint64_t ts = DAY_BASE_NS + 10ULL * NS_PER_HOUR;  // well before open
    EXPECT_FLOAT_EQ(filter.session_progress(ts), 0.0f);
}

TEST(SessionFilter, SessionProgressAfterCloseClampsToOne) {
    SessionFilter filter;
    uint64_t ts = DAY_BASE_NS + 23ULL * NS_PER_HOUR;  // well after close
    EXPECT_FLOAT_EQ(filter.session_progress(ts), 1.0f);
}

TEST(SessionFilter, SessionProgressQuarterPoint) {
    SessionFilter filter;
    uint64_t quarter = RTH_OPEN_NS + RTH_DURATION_NS / 4;
    uint64_t ts = DAY_BASE_NS + quarter;
    EXPECT_NEAR(filter.session_progress(ts), 0.25f, 1e-5f);
}

TEST(SessionFilter, SessionProgressThreeQuarterPoint) {
    SessionFilter filter;
    uint64_t three_quarter = RTH_OPEN_NS + 3 * RTH_DURATION_NS / 4;
    uint64_t ts = DAY_BASE_NS + three_quarter;
    EXPECT_NEAR(filter.session_progress(ts), 0.75f, 1e-5f);
}

// ===========================================================================
// SessionFilter: Custom config classify
// ===========================================================================

TEST(SessionFilter, CustomConfigClassifyCorrectly) {
    SessionConfig cfg;
    cfg.rth_open_ns = 14ULL * NS_PER_HOUR;   // 14:00 UTC
    cfg.rth_close_ns = 21ULL * NS_PER_HOUR;  // 21:00 UTC
    cfg.warmup_messages = 0;

    SessionFilter filter(cfg);

    // 13:30 should be PreMarket with this custom config
    uint64_t ts_1330 = DAY_BASE_NS + 13ULL * NS_PER_HOUR + 30ULL * NS_PER_MIN;
    EXPECT_EQ(filter.classify(ts_1330), SessionFilter::Phase::PreMarket);

    // 14:00 should be RTH
    uint64_t ts_1400 = DAY_BASE_NS + 14ULL * NS_PER_HOUR;
    EXPECT_EQ(filter.classify(ts_1400), SessionFilter::Phase::RTH);

    // 21:00 should be PostMarket
    uint64_t ts_2100 = DAY_BASE_NS + 21ULL * NS_PER_HOUR;
    EXPECT_EQ(filter.classify(ts_2100), SessionFilter::Phase::PostMarket);
}

// ===========================================================================
// M1: SessionConfig validation — prevent rth_close_ns <= rth_open_ns underflow
// ===========================================================================

TEST(SessionConfig, IsValidReturnsTrueForDefaultRth) {
    SessionConfig cfg = SessionConfig::default_rth();
    EXPECT_TRUE(cfg.is_valid())
        << "default_rth() should be a valid configuration";
}

TEST(SessionConfig, IsValidReturnsFalseWhenCloseEqualsOpen) {
    SessionConfig cfg;
    cfg.rth_open_ns = 13ULL * NS_PER_HOUR;
    cfg.rth_close_ns = 13ULL * NS_PER_HOUR;  // Equal to open — invalid
    cfg.warmup_messages = 0;

    EXPECT_FALSE(cfg.is_valid())
        << "Config with rth_close_ns == rth_open_ns should be invalid";
}

TEST(SessionConfig, IsValidReturnsFalseWhenCloseLessThanOpen) {
    SessionConfig cfg;
    cfg.rth_open_ns = 20ULL * NS_PER_HOUR;   // 20:00 UTC
    cfg.rth_close_ns = 13ULL * NS_PER_HOUR;  // 13:00 UTC — before open!
    cfg.warmup_messages = 0;

    EXPECT_FALSE(cfg.is_valid())
        << "Config with rth_close_ns < rth_open_ns should be invalid";
}

TEST(SessionConfig, IsValidReturnsTrueForValidCustomConfig) {
    SessionConfig cfg;
    cfg.rth_open_ns = 14ULL * NS_PER_HOUR;   // 14:00 UTC
    cfg.rth_close_ns = 21ULL * NS_PER_HOUR;  // 21:00 UTC
    cfg.warmup_messages = 100;

    EXPECT_TRUE(cfg.is_valid())
        << "Valid custom config should pass is_valid()";
}

TEST(SessionFilter, ConstructorThrowsForInvalidConfig) {
    SessionConfig invalid_cfg;
    invalid_cfg.rth_open_ns = 20ULL * NS_PER_HOUR;
    invalid_cfg.rth_close_ns = 13ULL * NS_PER_HOUR;  // Close before open
    invalid_cfg.warmup_messages = 0;

    EXPECT_THROW(SessionFilter filter(invalid_cfg), std::invalid_argument)
        << "SessionFilter constructor should throw for invalid config";
}

TEST(SessionFilter, ConstructorThrowsForEqualOpenAndClose) {
    SessionConfig invalid_cfg;
    invalid_cfg.rth_open_ns = 15ULL * NS_PER_HOUR;
    invalid_cfg.rth_close_ns = 15ULL * NS_PER_HOUR;  // Equal — invalid
    invalid_cfg.warmup_messages = 0;

    EXPECT_THROW(SessionFilter filter(invalid_cfg), std::invalid_argument)
        << "SessionFilter constructor should throw when open == close";
}

TEST(SessionFilter, DefaultConstructorUsesValidConfig) {
    // Default constructor should not throw and should use default_rth()
    EXPECT_NO_THROW({
        SessionFilter filter;
        EXPECT_EQ(filter.rth_open_ns(), RTH_OPEN_NS);
        EXPECT_EQ(filter.rth_close_ns(), RTH_CLOSE_NS);
    });
}
