#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <memory>

#include "databento/record.hpp"
#include "databento/enums.hpp"
#include "databento/constants.hpp"
#include "orderbook/MarketBook.hpp"
#include "features/primitives/MicroDepthFeature.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"

/**
 * @file   TestMicroDepthFeatureFix.cpp
 * @brief  Tests that MicroDepthFeature returns real depth data (not zeros)
 *         after the GetLevel method is added to IMarketBookDataSource.
 *
 *         Per docs/depth-queries.md, MicroDepthFeature::ComputeUpdate should
 *         call source.GetLevel(instrument_id, side, depth_index) and store
 *         the resulting price (converted from nanos to real currency) and
 *         total_quantity. Currently it returns hardcoded zeros.
 */

namespace constellation::modules::features::primitives {

using modules::orderbook::MarketBook;
using interfaces::orderbook::BookSide;

// Helper: create an MboMsg
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

static int64_t to_nanos(double price) {
    return static_cast<int64_t>(price * 1e9);
}

TEST_CASE("MicroDepthFeature returns real bid price at depth 0 after OnDataUpdate", "[micro_depth_fix]") {
    MarketBook mb;
    mb.OnMboUpdate(make_mbo(1, 100, to_nanos(99.50), 10,
                            databento::Side::Bid, databento::Action::Add));
    mb.OnMboUpdate(make_mbo(1, 200, to_nanos(100.50), 20,
                            databento::Side::Ask, databento::Action::Add));

    MicroDepthFeature::Config cfg{1, BookSide::Bid, 0};
    MicroDepthFeature feat(cfg);

    // Trigger computation through the IMarketBookDataSource interface
    feat.OnDataUpdate(mb, &mb);

    double price = feat.GetValue("micro_depth_price");
    double size  = feat.GetValue("micro_depth_size");

    // Price should be 99.50 (real currency, not nanos and not zero)
    CHECK_THAT(price, Catch::Matchers::WithinAbs(99.50, 1e-6));
    CHECK(size == 10.0);
}

TEST_CASE("MicroDepthFeature returns real ask price at depth 0 after OnDataUpdate", "[micro_depth_fix]") {
    MarketBook mb;
    mb.OnMboUpdate(make_mbo(1, 100, to_nanos(99.50), 10,
                            databento::Side::Bid, databento::Action::Add));
    mb.OnMboUpdate(make_mbo(1, 200, to_nanos(100.50), 20,
                            databento::Side::Ask, databento::Action::Add));

    MicroDepthFeature::Config cfg{1, BookSide::Ask, 0};
    MicroDepthFeature feat(cfg);
    feat.OnDataUpdate(mb, &mb);

    double price = feat.GetValue("micro_depth_price");
    double size  = feat.GetValue("micro_depth_size");

    CHECK_THAT(price, Catch::Matchers::WithinAbs(100.50, 1e-6));
    CHECK(size == 20.0);
}

TEST_CASE("MicroDepthFeature returns deeper level (depth_index=1)", "[micro_depth_fix]") {
    MarketBook mb;
    mb.OnMboUpdate(make_mbo(1, 100, to_nanos(100.00), 10,
                            databento::Side::Bid, databento::Action::Add));
    mb.OnMboUpdate(make_mbo(1, 101, to_nanos(99.50),  20,
                            databento::Side::Bid, databento::Action::Add));
    mb.OnMboUpdate(make_mbo(1, 200, to_nanos(101.00), 15,
                            databento::Side::Ask, databento::Action::Add));

    MicroDepthFeature::Config cfg{1, BookSide::Bid, 1};
    MicroDepthFeature feat(cfg);
    feat.OnDataUpdate(mb, &mb);

    double price = feat.GetValue("micro_depth_price");
    double size  = feat.GetValue("micro_depth_size");

    // Second bid level: 99.50, qty=20
    CHECK_THAT(price, Catch::Matchers::WithinAbs(99.50, 1e-6));
    CHECK(size == 20.0);
}

TEST_CASE("MicroDepthFeature returns 0 when depth exceeds available levels", "[micro_depth_fix]") {
    MarketBook mb;
    mb.OnMboUpdate(make_mbo(1, 100, to_nanos(100.00), 10,
                            databento::Side::Bid, databento::Action::Add));
    mb.OnMboUpdate(make_mbo(1, 200, to_nanos(101.00), 15,
                            databento::Side::Ask, databento::Action::Add));

    // Only 1 bid level exists, asking for depth_index=5
    MicroDepthFeature::Config cfg{1, BookSide::Bid, 5};
    MicroDepthFeature feat(cfg);
    feat.OnDataUpdate(mb, &mb);

    double price = feat.GetValue("micro_depth_price");
    double size  = feat.GetValue("micro_depth_size");

    CHECK(price == 0.0);
    CHECK(size == 0.0);
}

} // end namespace
