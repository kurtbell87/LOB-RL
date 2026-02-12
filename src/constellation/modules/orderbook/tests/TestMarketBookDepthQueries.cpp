#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <optional>
#include <cmath>

#include "databento/record.hpp"
#include "databento/enums.hpp"
#include "databento/constants.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include "interfaces/orderbook/IMarketBookDataSource.hpp"

/**
 * @file   TestMarketBookDepthQueries.cpp
 * @brief  Tests for new IMarketBookDataSource depth query methods on MarketBook:
 *         GetLevel, TotalDepth, WeightedMidPrice, VolumeAdjustedMidPrice.
 *
 *         These test the contract specified in docs/depth-queries.md.
 *         All methods are tested through MarketBook (the only concrete
 *         implementation of IMarketBookDataSource).
 */

namespace constellation::modules::orderbook {

using interfaces::orderbook::BookSide;
using interfaces::orderbook::PriceLevel;

// Helper: create an MboMsg for a given instrument
static databento::MboMsg make_mbo(uint32_t instrument_id, uint64_t order_id,
                                   int64_t price_nanos, uint32_t size,
                                   databento::Side side, databento::Action action) {
    databento::MboMsg mbo{};
    mbo.hd.instrument_id = instrument_id;
    mbo.order_id = order_id;
    mbo.price    = price_nanos;
    mbo.size     = size;
    mbo.side     = side;
    mbo.action   = action;
    return mbo;
}

// Price conversion helper: real currency -> nanos
static int64_t to_nanos(double price) {
    return static_cast<int64_t>(price * 1e9);
}

// Convenience: add a bid order to a MarketBook
static void add_bid(MarketBook& mb, uint32_t inst, uint64_t oid,
                    double price, uint32_t qty) {
    mb.OnMboUpdate(make_mbo(inst, oid, to_nanos(price), qty,
                            databento::Side::Bid, databento::Action::Add));
}

// Convenience: add an ask order to a MarketBook
static void add_ask(MarketBook& mb, uint32_t inst, uint64_t oid,
                    double price, uint32_t qty) {
    mb.OnMboUpdate(make_mbo(inst, oid, to_nanos(price), qty,
                            databento::Side::Ask, databento::Action::Add));
}

// =========================================================================
// GetLevel
// =========================================================================

TEST_CASE("MarketBook::GetLevel returns bid at depth 0 (best bid)", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);

    auto lvl = mb.GetLevel(1, BookSide::Bid, 0);
    REQUIRE(lvl.has_value());
    CHECK(lvl->price == to_nanos(99.50));
    CHECK(lvl->total_quantity == 10);
}

TEST_CASE("MarketBook::GetLevel returns ask at depth 0 (best ask)", "[depth_queries]") {
    MarketBook mb;
    add_ask(mb, 1, 200, 100.50, 15);
    add_ask(mb, 1, 201, 101.00, 25);

    auto lvl = mb.GetLevel(1, BookSide::Ask, 0);
    REQUIRE(lvl.has_value());
    CHECK(lvl->price == to_nanos(100.50));
    CHECK(lvl->total_quantity == 15);
}

TEST_CASE("MarketBook::GetLevel returns deeper bid levels in descending price order", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_bid(mb, 1, 101, 99.50,  20);
    add_bid(mb, 1, 102, 99.00,  30);

    auto lvl0 = mb.GetLevel(1, BookSide::Bid, 0);
    auto lvl1 = mb.GetLevel(1, BookSide::Bid, 1);
    auto lvl2 = mb.GetLevel(1, BookSide::Bid, 2);

    REQUIRE(lvl0.has_value());
    REQUIRE(lvl1.has_value());
    REQUIRE(lvl2.has_value());

    CHECK(lvl0->price == to_nanos(100.00));
    CHECK(lvl1->price == to_nanos(99.50));
    CHECK(lvl2->price == to_nanos(99.00));

    CHECK(lvl0->total_quantity == 10);
    CHECK(lvl1->total_quantity == 20);
    CHECK(lvl2->total_quantity == 30);
}

TEST_CASE("MarketBook::GetLevel returns deeper ask levels in ascending price order", "[depth_queries]") {
    MarketBook mb;
    add_ask(mb, 1, 200, 101.00, 10);
    add_ask(mb, 1, 201, 102.00, 20);
    add_ask(mb, 1, 202, 103.00, 30);

    auto lvl0 = mb.GetLevel(1, BookSide::Ask, 0);
    auto lvl1 = mb.GetLevel(1, BookSide::Ask, 1);
    auto lvl2 = mb.GetLevel(1, BookSide::Ask, 2);

    REQUIRE(lvl0.has_value());
    REQUIRE(lvl1.has_value());
    REQUIRE(lvl2.has_value());

    CHECK(lvl0->price == to_nanos(101.00));
    CHECK(lvl1->price == to_nanos(102.00));
    CHECK(lvl2->price == to_nanos(103.00));
}

TEST_CASE("MarketBook::GetLevel returns nullopt for unknown instrument", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);

    auto lvl = mb.GetLevel(9999, BookSide::Bid, 0);
    CHECK_FALSE(lvl.has_value());
}

TEST_CASE("MarketBook::GetLevel returns nullopt for empty book", "[depth_queries]") {
    MarketBook mb;
    // Create the instrument by adding and cancelling
    auto msg = make_mbo(1, 100, to_nanos(99.50), 10,
                        databento::Side::Bid, databento::Action::Add);
    mb.OnMboUpdate(msg);
    msg.action = databento::Action::Cancel;
    mb.OnMboUpdate(msg);

    auto lvl = mb.GetLevel(1, BookSide::Bid, 0);
    CHECK_FALSE(lvl.has_value());
}

TEST_CASE("MarketBook::GetLevel returns nullopt when depth_index exceeds available levels", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);

    // Only 2 levels, so index 2 should be nullopt
    auto lvl = mb.GetLevel(1, BookSide::Bid, 2);
    CHECK_FALSE(lvl.has_value());
}

TEST_CASE("MarketBook::GetLevel aggregates multiple orders at same price", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.50, 25);
    add_bid(mb, 1, 102, 99.50, 5);

    auto lvl = mb.GetLevel(1, BookSide::Bid, 0);
    REQUIRE(lvl.has_value());
    CHECK(lvl->price == to_nanos(99.50));
    CHECK(lvl->total_quantity == 40);
    CHECK(lvl->order_count == 3);
}

// =========================================================================
// TotalDepth
// =========================================================================

TEST_CASE("MarketBook::TotalDepth returns 0 for unknown instrument", "[depth_queries]") {
    MarketBook mb;
    CHECK(mb.TotalDepth(9999, BookSide::Bid, 5) == 0);
}

TEST_CASE("MarketBook::TotalDepth returns 0 for empty book side", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);

    // Ask side is empty
    CHECK(mb.TotalDepth(1, BookSide::Ask, 5) == 0);
}

TEST_CASE("MarketBook::TotalDepth returns 0 when n_levels is 0", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);

    CHECK(mb.TotalDepth(1, BookSide::Bid, 0) == 0);
}

TEST_CASE("MarketBook::TotalDepth sums top 1 level correctly", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);
    add_bid(mb, 1, 102, 99.00, 30);

    CHECK(mb.TotalDepth(1, BookSide::Bid, 1) == 10);
}

TEST_CASE("MarketBook::TotalDepth sums top 3 bid levels", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);
    add_bid(mb, 1, 102, 99.00, 30);

    CHECK(mb.TotalDepth(1, BookSide::Bid, 3) == 60);
}

TEST_CASE("MarketBook::TotalDepth sums top 3 ask levels", "[depth_queries]") {
    MarketBook mb;
    add_ask(mb, 1, 200, 100.50, 10);
    add_ask(mb, 1, 201, 101.00, 20);
    add_ask(mb, 1, 202, 101.50, 30);

    CHECK(mb.TotalDepth(1, BookSide::Ask, 3) == 60);
}

TEST_CASE("MarketBook::TotalDepth sums only available levels when n_levels > book depth", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);

    // Requesting 10 levels but only 2 exist
    CHECK(mb.TotalDepth(1, BookSide::Bid, 10) == 30);
}

TEST_CASE("MarketBook::TotalDepth includes aggregated orders at same price", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.50, 15);  // same price
    add_bid(mb, 1, 102, 99.25, 20);

    // Level 0: 25 (10+15), Level 1: 20. Total for 2 levels = 45
    CHECK(mb.TotalDepth(1, BookSide::Bid, 2) == 45);
}

// =========================================================================
// WeightedMidPrice
// =========================================================================

TEST_CASE("MarketBook::WeightedMidPrice returns nullopt for unknown instrument", "[depth_queries]") {
    MarketBook mb;
    CHECK_FALSE(mb.WeightedMidPrice(9999).has_value());
}

TEST_CASE("MarketBook::WeightedMidPrice returns nullopt for empty book", "[depth_queries]") {
    MarketBook mb;
    // Create instrument with an add+cancel
    auto msg = make_mbo(1, 100, to_nanos(99.50), 10,
                        databento::Side::Bid, databento::Action::Add);
    mb.OnMboUpdate(msg);
    msg.action = databento::Action::Cancel;
    mb.OnMboUpdate(msg);

    CHECK_FALSE(mb.WeightedMidPrice(1).has_value());
}

TEST_CASE("MarketBook::WeightedMidPrice returns nullopt when only bids present", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);

    CHECK_FALSE(mb.WeightedMidPrice(1).has_value());
}

TEST_CASE("MarketBook::WeightedMidPrice returns nullopt when only asks present", "[depth_queries]") {
    MarketBook mb;
    add_ask(mb, 1, 200, 100.50, 10);

    CHECK_FALSE(mb.WeightedMidPrice(1).has_value());
}

TEST_CASE("MarketBook::WeightedMidPrice equals simple mid when quantities are equal", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 50);
    add_ask(mb, 1, 200, 101.00, 50);

    auto wmid = mb.WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    // wmid = (bid_price * ask_qty + ask_price * bid_qty) / (bid_qty + ask_qty)
    //      = (100 * 50 + 101 * 50) / 100 = 100.5
    CHECK_THAT(*wmid, Catch::Matchers::WithinAbs(100.5, 1e-9));
}

TEST_CASE("MarketBook::WeightedMidPrice skews toward ask when bid quantity dominates", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 90);
    add_ask(mb, 1, 200, 101.00, 10);

    auto wmid = mb.WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    // wmid = (100 * 10 + 101 * 90) / 100 = (1000 + 9090) / 100 = 100.90
    double expected = (100.0 * 10.0 + 101.0 * 90.0) / 100.0;
    CHECK_THAT(*wmid, Catch::Matchers::WithinAbs(expected, 1e-9));

    // Verify it's above simple mid
    double simple_mid = (100.0 + 101.0) / 2.0;
    CHECK(*wmid > simple_mid);
}

TEST_CASE("MarketBook::WeightedMidPrice skews toward bid when ask quantity dominates", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_ask(mb, 1, 200, 101.00, 90);

    auto wmid = mb.WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    // wmid = (100 * 90 + 101 * 10) / 100 = (9000 + 1010) / 100 = 100.10
    double expected = (100.0 * 90.0 + 101.0 * 10.0) / 100.0;
    CHECK_THAT(*wmid, Catch::Matchers::WithinAbs(expected, 1e-9));

    double simple_mid = (100.0 + 101.0) / 2.0;
    CHECK(*wmid < simple_mid);
}

TEST_CASE("MarketBook::WeightedMidPrice returns double in real currency (not nanos)", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 30);
    add_ask(mb, 1, 200, 100.50, 70);

    auto wmid = mb.WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    // wmid = (99.50 * 70 + 100.50 * 30) / 100 = (6965 + 3015) / 100 = 99.80
    double expected = (99.50 * 70.0 + 100.50 * 30.0) / 100.0;
    CHECK_THAT(*wmid, Catch::Matchers::WithinAbs(expected, 1e-9));

    // The value should be a sensible price, not nanos
    CHECK(*wmid < 200.0);
    CHECK(*wmid > 50.0);
}

// =========================================================================
// VolumeAdjustedMidPrice (VAMP)
// =========================================================================

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns nullopt for unknown instrument", "[depth_queries]") {
    MarketBook mb;
    CHECK_FALSE(mb.VolumeAdjustedMidPrice(9999, 5).has_value());
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns nullopt for empty book", "[depth_queries]") {
    MarketBook mb;
    auto msg = make_mbo(1, 100, to_nanos(99.50), 10,
                        databento::Side::Bid, databento::Action::Add);
    mb.OnMboUpdate(msg);
    msg.action = databento::Action::Cancel;
    mb.OnMboUpdate(msg);

    CHECK_FALSE(mb.VolumeAdjustedMidPrice(1, 5).has_value());
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns nullopt when only bids present", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.50, 10);
    add_bid(mb, 1, 101, 99.25, 20);

    CHECK_FALSE(mb.VolumeAdjustedMidPrice(1, 5).has_value());
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns nullopt when only asks present", "[depth_queries]") {
    MarketBook mb;
    add_ask(mb, 1, 200, 100.50, 10);

    CHECK_FALSE(mb.VolumeAdjustedMidPrice(1, 5).has_value());
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns nullopt when n_levels is 0", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_ask(mb, 1, 200, 101.00, 10);

    CHECK_FALSE(mb.VolumeAdjustedMidPrice(1, 0).has_value());
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice with single level each side", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 30);
    add_ask(mb, 1, 200, 101.00, 70);

    auto vamp = mb.VolumeAdjustedMidPrice(1, 1);
    REQUIRE(vamp.has_value());

    // VAMP = (100*30 + 101*70) / (30+70) = (3000+7070)/100 = 100.70
    double expected = (100.0 * 30 + 101.0 * 70) / 100.0;
    CHECK_THAT(*vamp, Catch::Matchers::WithinAbs(expected, 1e-9));
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice with multiple levels hand-computed", "[depth_queries]") {
    MarketBook mb;
    // Bids: 100.00(10), 99.00(20)
    add_bid(mb, 1, 100, 100.00, 10);
    add_bid(mb, 1, 101, 99.00,  20);
    // Asks: 101.00(15), 102.00(25)
    add_ask(mb, 1, 200, 101.00, 15);
    add_ask(mb, 1, 201, 102.00, 25);

    auto vamp = mb.VolumeAdjustedMidPrice(1, 2);
    REQUIRE(vamp.has_value());

    // VAMP = (100*10 + 99*20 + 101*15 + 102*25) / (10+20+15+25)
    //      = (1000 + 1980 + 1515 + 2550) / 70 = 7045/70 = 100.642857...
    double expected = (100.0*10 + 99.0*20 + 101.0*15 + 102.0*25) / 70.0;
    CHECK_THAT(*vamp, Catch::Matchers::WithinAbs(expected, 1e-9));
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice uses all available when n_levels > book depth", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_bid(mb, 1, 101, 99.00,  20);
    add_ask(mb, 1, 200, 101.00, 15);
    add_ask(mb, 1, 201, 102.00, 25);

    // n_levels=10 but only 2 levels per side
    auto vamp10 = mb.VolumeAdjustedMidPrice(1, 10);
    auto vamp2  = mb.VolumeAdjustedMidPrice(1, 2);

    REQUIRE(vamp10.has_value());
    REQUIRE(vamp2.has_value());

    // Should be the same since only 2 levels available
    CHECK_THAT(*vamp10, Catch::Matchers::WithinAbs(*vamp2, 1e-9));
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice with asymmetric depth uses all available on each side", "[depth_queries]") {
    MarketBook mb;
    // Bids: 1 level
    add_bid(mb, 1, 100, 100.00, 10);
    // Asks: 3 levels
    add_ask(mb, 1, 200, 101.00, 10);
    add_ask(mb, 1, 201, 102.00, 20);
    add_ask(mb, 1, 202, 103.00, 30);

    auto vamp = mb.VolumeAdjustedMidPrice(1, 3);
    REQUIRE(vamp.has_value());

    // = (100*10 + 101*10 + 102*20 + 103*30) / (10+10+20+30)
    // = (1000 + 1010 + 2040 + 3090) / 70 = 7140/70 = 102.0
    double expected = (100.0*10 + 101.0*10 + 102.0*20 + 103.0*30) / 70.0;
    CHECK_THAT(*vamp, Catch::Matchers::WithinAbs(expected, 1e-9));
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice with symmetric book and equal quantities", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_bid(mb, 1, 101, 99.00,  10);
    add_ask(mb, 1, 200, 101.00, 10);
    add_ask(mb, 1, 201, 102.00, 10);

    auto vamp = mb.VolumeAdjustedMidPrice(1, 2);
    REQUIRE(vamp.has_value());

    // = (100*10 + 99*10 + 101*10 + 102*10) / 40 = (1000+990+1010+1020)/40 = 4020/40 = 100.5
    CHECK_THAT(*vamp, Catch::Matchers::WithinAbs(100.5, 1e-9));
}

TEST_CASE("MarketBook::VolumeAdjustedMidPrice returns double in real currency", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 4250.25, 10);
    add_ask(mb, 1, 200, 4250.50, 10);

    auto vamp = mb.VolumeAdjustedMidPrice(1, 1);
    REQUIRE(vamp.has_value());

    // Should be a sensible price like 4250.375, not nanos
    CHECK(*vamp > 4000.0);
    CHECK(*vamp < 5000.0);
    double expected = (4250.25 * 10 + 4250.50 * 10) / 20.0;
    CHECK_THAT(*vamp, Catch::Matchers::WithinAbs(expected, 1e-6));
}

// =========================================================================
// Multi-instrument isolation
// =========================================================================

TEST_CASE("MarketBook depth queries are isolated per instrument", "[depth_queries]") {
    MarketBook mb;

    // Instrument 1: tight spread, small qty
    add_bid(mb, 1, 100, 100.00, 10);
    add_ask(mb, 1, 200, 100.50, 10);

    // Instrument 2: wide spread, large qty
    add_bid(mb, 2, 300, 50.00, 1000);
    add_ask(mb, 2, 400, 60.00, 1000);

    // GetLevel should return instrument-specific data
    auto bid1 = mb.GetLevel(1, BookSide::Bid, 0);
    auto bid2 = mb.GetLevel(2, BookSide::Bid, 0);
    REQUIRE(bid1.has_value());
    REQUIRE(bid2.has_value());
    CHECK(bid1->price == to_nanos(100.00));
    CHECK(bid2->price == to_nanos(50.00));

    // TotalDepth should be instrument-specific
    CHECK(mb.TotalDepth(1, BookSide::Bid, 1) == 10);
    CHECK(mb.TotalDepth(2, BookSide::Bid, 1) == 1000);

    // WeightedMidPrice should differ per instrument
    auto wmid1 = mb.WeightedMidPrice(1);
    auto wmid2 = mb.WeightedMidPrice(2);
    REQUIRE(wmid1.has_value());
    REQUIRE(wmid2.has_value());
    CHECK(*wmid1 > 99.0);
    CHECK(*wmid2 > 49.0);
    CHECK(*wmid2 < 61.0);
}

// =========================================================================
// Edge: all quantity at one price level
// =========================================================================

TEST_CASE("MarketBook::WeightedMidPrice with single order each side", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 99.00, 1);
    add_ask(mb, 1, 200, 101.00, 1);

    auto wmid = mb.WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    // Equal qty => simple mid = 100.0
    CHECK_THAT(*wmid, Catch::Matchers::WithinAbs(100.0, 1e-9));
}

// =========================================================================
// IMarketBookDataSource interface compliance
// =========================================================================

TEST_CASE("MarketBook implements new IMarketBookDataSource methods through interface pointer", "[depth_queries]") {
    MarketBook mb;
    add_bid(mb, 1, 100, 100.00, 10);
    add_ask(mb, 1, 200, 101.00, 10);

    // Access through IMarketBookDataSource pointer
    interfaces::orderbook::IMarketBookDataSource* ds = &mb;

    auto lvl = ds->GetLevel(1, BookSide::Bid, 0);
    REQUIRE(lvl.has_value());
    CHECK(lvl->price == to_nanos(100.00));

    CHECK(ds->TotalDepth(1, BookSide::Bid, 1) == 10);

    auto wmid = ds->WeightedMidPrice(1);
    REQUIRE(wmid.has_value());

    auto vamp = ds->VolumeAdjustedMidPrice(1, 1);
    REQUIRE(vamp.has_value());
}

} // end namespace constellation::modules::orderbook
