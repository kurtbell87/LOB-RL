/**
 * LOB Validation Test — Databento Reference Comparison
 *
 * Downloads DBEQ.BASIC GOOG/GOOGL MBO data for 2024-04-03 and verifies that
 * both Constellation's LimitOrderBook and LOB-RL's Book produce the exact
 * same BBO as Databento's reference implementation.
 *
 * Reference: https://databento.com/docs/examples/order-book/limit-order-book
 *
 * Key design notes from the reference:
 *   - Books are keyed by (instrument_id, publisher_id)
 *   - "Aggregated BBO" merges across publishers for one instrument
 *   - TOB flag (IsTob()) on Add clears the side and adds a single level
 *   - Trade/Fill/None actions are no-ops
 *   - Modify of unknown order is treated as Add
 *   - BBO is printed at every IsLast() event
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <databento/constants.hpp>
#include <databento/datetime.hpp>
#include <databento/dbn_file_store.hpp>
#include <databento/enums.hpp>
#include <databento/flag_set.hpp>
#include <databento/historical.hpp>
#include <databento/pretty.hpp>
#include <databento/record.hpp>
#include <databento/symbol_map.hpp>

// Constellation
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"

// LOB-RL
#include "lob/book.h"
#include "lob/message.h"
#include "constellation/adapters/message_adapter.h"

namespace db = databento;
namespace cst_ob = constellation::modules::orderbook;

// ── Databento Reference Implementation (from their docs) ────────────────

struct RefPriceLevel {
    int64_t price{db::kUndefPrice};
    uint32_t size{0};
    uint32_t count{0};
    bool IsEmpty() const { return price == db::kUndefPrice; }
    operator bool() const { return !IsEmpty(); }
};

class RefBook {
public:
    std::pair<RefPriceLevel, RefPriceLevel> Bbo() const {
        return {GetBidLevel(), GetAskLevel()};
    }

    RefPriceLevel GetBidLevel(std::size_t idx = 0) const {
        if (bids_.size() > idx) {
            auto level_it = bids_.rbegin();
            std::advance(level_it, idx);
            return GetPriceLevel(level_it->first, level_it->second);
        }
        return RefPriceLevel{};
    }

    RefPriceLevel GetAskLevel(std::size_t idx = 0) const {
        if (offers_.size() > idx) {
            auto level_it = offers_.begin();
            std::advance(level_it, idx);
            return GetPriceLevel(level_it->first, level_it->second);
        }
        return RefPriceLevel{};
    }

    void Apply(const db::MboMsg& mbo) {
        switch (mbo.action) {
            case db::Action::Clear: Clear(); break;
            case db::Action::Add: Add(mbo); break;
            case db::Action::Cancel: Cancel(mbo); break;
            case db::Action::Modify: Modify(mbo); break;
            case db::Action::Trade:
            case db::Action::Fill:
            case db::Action::None: break;
            default: break;
        }
    }

private:
    using LevelOrders = std::vector<db::MboMsg>;
    struct PriceAndSide {
        int64_t price;
        db::Side side;
    };
    using Orders = std::unordered_map<uint64_t, PriceAndSide>;
    using SideLevels = std::map<int64_t, LevelOrders>;

    static RefPriceLevel GetPriceLevel(int64_t price, const LevelOrders level) {
        RefPriceLevel res{price};
        for (const auto& order : level) {
            if (!order.flags.IsTob()) {
                ++res.count;
            }
            res.size += order.size;
        }
        return res;
    }

    static LevelOrders::iterator GetLevelOrder(LevelOrders& level, uint64_t order_id) {
        return std::find_if(level.begin(), level.end(),
            [order_id](const db::MboMsg& order) {
                return order.order_id == order_id;
            });
    }

    void Clear() {
        orders_by_id_.clear();
        offers_.clear();
        bids_.clear();
    }

    void Add(db::MboMsg mbo) {
        if (mbo.flags.IsTob()) {
            SideLevels& levels = GetSideLevels(mbo.side);
            levels.clear();
            if (mbo.price != db::kUndefPrice) {
                LevelOrders level = {mbo};
                levels.emplace(mbo.price, level);
            }
        } else {
            LevelOrders& level = GetOrInsertLevel(mbo.side, mbo.price);
            level.emplace_back(mbo);
            orders_by_id_.emplace(mbo.order_id, PriceAndSide{mbo.price, mbo.side});
        }
    }

    void Cancel(db::MboMsg mbo) {
        auto& levels = GetSideLevels(mbo.side);
        auto level_it = levels.find(mbo.price);
        if (level_it == levels.end()) return;
        auto& level = level_it->second;
        auto order_it = GetLevelOrder(level, mbo.order_id);
        if (order_it == level.end()) return;
        if (order_it->size >= mbo.size) {
            order_it->size -= mbo.size;
        }
        if (order_it->size == 0) {
            orders_by_id_.erase(mbo.order_id);
            level.erase(order_it);
            if (level.empty()) {
                levels.erase(mbo.price);
            }
        }
    }

    void Modify(db::MboMsg mbo) {
        auto price_side_it = orders_by_id_.find(mbo.order_id);
        if (price_side_it == orders_by_id_.end()) {
            Add(mbo);
            return;
        }
        auto prev_price = price_side_it->second.price;
        auto& prev_levels = GetSideLevels(mbo.side);
        auto prev_level_it = prev_levels.find(prev_price);
        if (prev_level_it == prev_levels.end()) {
            Add(mbo);
            return;
        }
        auto& prev_level = prev_level_it->second;
        auto level_order_it = GetLevelOrder(prev_level, mbo.order_id);
        if (level_order_it == prev_level.end()) {
            Add(mbo);
            return;
        }
        if (prev_price != mbo.price) {
            price_side_it->second.price = mbo.price;
            prev_level.erase(level_order_it);
            if (prev_level.empty()) {
                prev_levels.erase(prev_price);
            }
            LevelOrders& level = GetOrInsertLevel(mbo.side, mbo.price);
            level.emplace_back(mbo);
        } else if (level_order_it->size < mbo.size) {
            auto& level = prev_level;
            level.erase(level_order_it);
            level.emplace_back(mbo);
        } else {
            level_order_it->size = mbo.size;
        }
    }

    SideLevels& GetSideLevels(db::Side side) {
        return side == db::Side::Ask ? offers_ : bids_;
    }

    LevelOrders& GetOrInsertLevel(db::Side side, int64_t price) {
        return GetSideLevels(side)[price];
    }

    Orders orders_by_id_;
    SideLevels offers_;
    SideLevels bids_;
};

class RefMarket {
public:
    struct PublisherBook {
        uint16_t publisher_id;
        RefBook book;
    };

    void Apply(const db::MboMsg& mbo) {
        auto& instrument_books = books_[mbo.hd.instrument_id];
        auto book_it = std::find_if(instrument_books.begin(), instrument_books.end(),
            [&mbo](const PublisherBook& pb) {
                return pb.publisher_id == mbo.hd.publisher_id;
            });
        if (book_it == instrument_books.end()) {
            instrument_books.emplace_back(PublisherBook{mbo.hd.publisher_id, {}});
            book_it = std::prev(instrument_books.end());
        }
        book_it->book.Apply(mbo);
    }

    std::pair<RefPriceLevel, RefPriceLevel> AggregatedBbo(uint32_t instrument_id) {
        RefPriceLevel agg_bid;
        RefPriceLevel agg_ask;
        for (const auto& pub_book : books_[instrument_id]) {
            auto bbo = pub_book.book.Bbo();
            const auto& bid = bbo.first;
            const auto& ask = bbo.second;
            if (bid) {
                if (agg_bid.IsEmpty() || bid.price > agg_bid.price) {
                    agg_bid = bid;
                } else if (bid.price == agg_bid.price) {
                    agg_bid.count += bid.count;
                    agg_bid.size += bid.size;
                }
            }
            if (ask) {
                if (agg_ask.IsEmpty() || ask.price < agg_ask.price) {
                    agg_ask = ask;
                } else if (ask.price == agg_ask.price) {
                    agg_ask.count += ask.count;
                    agg_ask.size += ask.size;
                }
            }
        }
        return {agg_bid, agg_ask};
    }

private:
    std::unordered_map<uint32_t, std::vector<PublisherBook>> books_;
};

// ── Constellation per-publisher books ────────────────────────────────────

struct CstPublisherBook {
    uint16_t publisher_id;
    std::unique_ptr<cst_ob::LimitOrderBook> lob;
    CstPublisherBook(uint16_t pub_id, uint32_t inst_id)
        : publisher_id(pub_id), lob(std::make_unique<cst_ob::LimitOrderBook>(inst_id)) {}
};

class CstMarket {
public:
    void Apply(const db::MboMsg& mbo) {
        auto& instrument_books = books_[mbo.hd.instrument_id];
        auto book_it = std::find_if(instrument_books.begin(), instrument_books.end(),
            [&mbo](const CstPublisherBook& pb) {
                return pb.publisher_id == mbo.hd.publisher_id;
            });
        if (book_it == instrument_books.end()) {
            instrument_books.emplace_back(
                CstPublisherBook{mbo.hd.publisher_id, mbo.hd.instrument_id});
            book_it = std::prev(instrument_books.end());
        }
        book_it->lob->OnMboUpdate(mbo);
    }

    std::pair<RefPriceLevel, RefPriceLevel> AggregatedBbo(uint32_t instrument_id) {
        RefPriceLevel agg_bid;
        RefPriceLevel agg_ask;
        for (auto& pub_book : books_[instrument_id]) {
            auto bid_opt = pub_book.lob->BestBid();
            auto ask_opt = pub_book.lob->BestAsk();
            if (bid_opt) {
                RefPriceLevel bid{bid_opt->price,
                    static_cast<uint32_t>(bid_opt->total_quantity),
                    bid_opt->order_count};
                if (agg_bid.IsEmpty() || bid.price > agg_bid.price) {
                    agg_bid = bid;
                } else if (bid.price == agg_bid.price) {
                    agg_bid.count += bid.count;
                    agg_bid.size += bid.size;
                }
            }
            if (ask_opt) {
                RefPriceLevel ask{ask_opt->price,
                    static_cast<uint32_t>(ask_opt->total_quantity),
                    ask_opt->order_count};
                if (agg_ask.IsEmpty() || ask.price < agg_ask.price) {
                    agg_ask = ask;
                } else if (ask.price == agg_ask.price) {
                    agg_ask.count += ask.count;
                    agg_ask.size += ask.size;
                }
            }
        }
        return {agg_bid, agg_ask};
    }

private:
    std::unordered_map<uint32_t, std::vector<CstPublisherBook>> books_;
};

// ── Test fixture ─────────────────────────────────────────────────────────

static const std::string kDataFile = "data/dbeq-basic-20240403.mbo.dbn.zst";

class LobValidationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (std::filesystem::exists(kDataFile)) {
            return;  // Already cached
        }

        const char* api_key = std::getenv("DATABENTO_API_KEY");
        if (!api_key || std::string(api_key).empty()) {
            return;  // Will be skipped in individual tests
        }

        // Download via databento-cpp Historical API
        std::cerr << "[LobValidation] Downloading DBEQ.BASIC GOOG/GOOGL 2024-04-03...\n";
        auto client = db::Historical::Builder()
            .SetKey(api_key)
            .Build();

        client.TimeseriesGetRangeToFile(
            db::dataset::kDbeqBasic,
            {"2024-04-03T08:00:00", "2024-04-03T14:00:00"},
            {"GOOG", "GOOGL"},
            db::Schema::Mbo,
            kDataFile);

        std::cerr << "[LobValidation] Downloaded to " << kDataFile << "\n";
    }

    bool HasDataFile() const {
        return std::filesystem::exists(kDataFile);
    }
};

// ── First 7 known reference BBOs for GOOGL ───────────────────────────────
// Exact output from Databento LOB reference implementation:
//
//   [0] 2024-04-03T11:00:00.903881953Z  ask=empty              bid=200@152.80 (1)
//   [1] 2024-04-03T11:00:00.903887026Z  ask=200@155.20 (1)     bid=200@152.80 (1)
//   [2] 2024-04-03T11:00:00.914731635Z  ask=200@155.20 (1)     bid=200@152.80 (1)
//   [3] 2024-04-03T11:00:00.925739296Z  ask=200@155.20 (1)     bid=200@152.80 (1)
//   [4] 2024-04-03T11:00:00.925744379Z  ask=200@155.20 (1)     bid=200@152.80 (1)
//   [5] 2024-04-03T11:03:55.585836619Z  ask=200@155.20 (1)     bid=200@152.80 (1)
//   [6] 2024-04-03T11:03:55.585841782Z  ask=200@155.40 (1)     bid=200@152.80 (1)

static const int64_t PX_152_80 = 152'800'000'000;  // 152.800000000
static const int64_t PX_155_20 = 155'200'000'000;  // 155.200000000
static const int64_t PX_155_40 = 155'400'000'000;  // 155.400000000

// ── Test: Constellation vs Databento Reference (full file) ───────────────

TEST_F(LobValidationTest, ConstellationMatchesDabentoReference) {
    if (!HasDataFile()) {
        GTEST_SKIP() << "DBEQ data file not found. Set DATABENTO_API_KEY to download.";
    }

    db::DbnFileStore store(kDataFile);
    auto symbol_map = store.GetMetadata().CreateSymbolMap();

    RefMarket ref_market;
    CstMarket cst_market;

    uint64_t event_count = 0;
    uint64_t mismatch_count = 0;
    uint64_t bbo_checks = 0;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;

        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        event_count++;

        // Apply to both
        ref_market.Apply(*mbo);
        cst_market.Apply(*mbo);

        // Check BBO at every IsLast() event (same as reference)
        if (mbo->flags.IsLast()) {
            bbo_checks++;
            auto ref_bbo = ref_market.AggregatedBbo(mbo->hd.instrument_id);
            auto cst_bbo = cst_market.AggregatedBbo(mbo->hd.instrument_id);

            // Compare bid
            bool bid_match = (ref_bbo.first.price == cst_bbo.first.price &&
                              ref_bbo.first.size == cst_bbo.first.size);
            // Compare ask
            bool ask_match = (ref_bbo.second.price == cst_bbo.second.price &&
                              ref_bbo.second.size == cst_bbo.second.size);

            if (!bid_match || !ask_match) {
                mismatch_count++;
                if (mismatch_count <= 10) {  // Print first 10 mismatches
                    const auto& sym = symbol_map.At(*mbo);
                    std::cerr << "MISMATCH #" << mismatch_count << " at " << sym
                              << " event " << event_count << ":\n";
                    std::cerr << "  REF  bid=" << ref_bbo.first.size << "@"
                              << ref_bbo.first.price << " ask=" << ref_bbo.second.size << "@"
                              << ref_bbo.second.price << "\n";
                    std::cerr << "  CST  bid=" << cst_bbo.first.size << "@"
                              << cst_bbo.first.price << " ask=" << cst_bbo.second.size << "@"
                              << cst_bbo.second.price << "\n";
                }
            }
        }
    }

    std::cerr << "[LobValidation] Processed " << event_count << " MBO events, "
              << bbo_checks << " BBO checks, " << mismatch_count << " mismatches\n";

    EXPECT_EQ(mismatch_count, 0u)
        << "Constellation BBO mismatched Databento reference " << mismatch_count
        << " times out of " << bbo_checks << " checks";
    EXPECT_GT(bbo_checks, 0u) << "No BBO checks were performed";
}

// ── Test: Verify specific GOOGL BBO values from reference ────────────────

TEST_F(LobValidationTest, MatchesKnownGOOGLBboValues) {
    if (!HasDataFile()) {
        GTEST_SKIP() << "DBEQ data file not found. Set DATABENTO_API_KEY to download.";
    }

    db::DbnFileStore store(kDataFile);
    auto symbol_map = store.GetMetadata().CreateSymbolMap();

    RefMarket ref_market;
    CstMarket cst_market;

    // Track the sequence of GOOGL BBO events to verify against known values
    struct BboEvent {
        uint64_t ts_recv_ns;
        RefPriceLevel bid;
        RefPriceLevel ask;
    };
    std::vector<BboEvent> googl_ref_events;
    std::vector<BboEvent> googl_cst_events;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;

        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        ref_market.Apply(*mbo);
        cst_market.Apply(*mbo);

        if (mbo->flags.IsLast()) {
            const auto& sym = symbol_map.At(*mbo);
            if (sym == "GOOGL") {
                auto ref_bbo = ref_market.AggregatedBbo(mbo->hd.instrument_id);
                auto cst_bbo = cst_market.AggregatedBbo(mbo->hd.instrument_id);
                uint64_t ts = mbo->ts_recv.time_since_epoch().count();

                googl_ref_events.push_back({ts, ref_bbo.first, ref_bbo.second});
                googl_cst_events.push_back({ts, cst_bbo.first, cst_bbo.second});
            }
        }
    }

    ASSERT_GE(googl_ref_events.size(), 7u) << "Expected at least 7 GOOGL BBO events";

    // ── Verify all 7 known GOOGL BBO events from Databento reference ──

    // [0] bid=200@152.80 (1 order), ask=empty
    EXPECT_EQ(googl_ref_events[0].bid.price, PX_152_80);
    EXPECT_EQ(googl_ref_events[0].bid.size, 200u);
    EXPECT_EQ(googl_ref_events[0].bid.count, 1u);
    EXPECT_TRUE(googl_ref_events[0].ask.IsEmpty());

    // [1]-[5] bid=200@152.80 (1 order), ask=200@155.20 (1 order)
    for (size_t i = 1; i <= 5; ++i) {
        EXPECT_EQ(googl_ref_events[i].bid.price, PX_152_80) << "event " << i;
        EXPECT_EQ(googl_ref_events[i].bid.size, 200u) << "event " << i;
        EXPECT_EQ(googl_ref_events[i].bid.count, 1u) << "event " << i;
        EXPECT_EQ(googl_ref_events[i].ask.price, PX_155_20) << "event " << i;
        EXPECT_EQ(googl_ref_events[i].ask.size, 200u) << "event " << i;
        EXPECT_EQ(googl_ref_events[i].ask.count, 1u) << "event " << i;
    }

    // [6] bid=200@152.80 (1 order), ask=200@155.40 (1 order)
    EXPECT_EQ(googl_ref_events[6].bid.price, PX_152_80);
    EXPECT_EQ(googl_ref_events[6].bid.size, 200u);
    EXPECT_EQ(googl_ref_events[6].bid.count, 1u);
    EXPECT_EQ(googl_ref_events[6].ask.price, PX_155_40);
    EXPECT_EQ(googl_ref_events[6].ask.size, 200u);
    EXPECT_EQ(googl_ref_events[6].ask.count, 1u);

    // Verify Constellation matches reference for ALL GOOGL events
    ASSERT_EQ(googl_ref_events.size(), googl_cst_events.size());
    uint64_t googl_mismatches = 0;
    for (size_t i = 0; i < googl_ref_events.size(); ++i) {
        bool match = googl_ref_events[i].bid.price == googl_cst_events[i].bid.price &&
                     googl_ref_events[i].bid.size == googl_cst_events[i].bid.size &&
                     googl_ref_events[i].ask.price == googl_cst_events[i].ask.price &&
                     googl_ref_events[i].ask.size == googl_cst_events[i].ask.size;
        if (!match) googl_mismatches++;
    }
    EXPECT_EQ(googl_mismatches, 0u)
        << "Constellation mismatched " << googl_mismatches << " / "
        << googl_ref_events.size() << " GOOGL BBO events";
}

// ── Test: LOB-RL Book via adapter path (single publisher) ────────────
//
// Exercises the full adapter round-trip:
//   MboMsg → to_message() → Message → Book::apply() → to_mbo_msg() → LimitOrderBook::OnMboUpdate()
//
// Book is single-instrument/single-publisher, so we filter to one
// (instrument_id, publisher_id) pair and compare against RefBook.
//
// For messages the adapter can't convert (Clear, None), we apply the raw
// MboMsg directly to the underlying LimitOrderBook to keep books in sync.
// This isolates adapter conversion fidelity from action-coverage gaps.

TEST_F(LobValidationTest, BookAdapterPathSinglePublisher) {
    if (!HasDataFile()) {
        GTEST_SKIP() << "DBEQ data file not found. Set DATABENTO_API_KEY to download.";
    }

    // First pass: find the first GOOGL (instrument_id, publisher_id)
    uint32_t target_iid = 0;
    uint16_t target_pid = 0;
    {
        db::DbnFileStore scan(kDataFile);
        auto sym_map = scan.GetMetadata().CreateSymbolMap();
        while (true) {
            const db::Record* rec = scan.NextRecord();
            if (!rec) break;
            const auto* mbo = rec->GetIf<db::MboMsg>();
            if (!mbo) continue;
            if (mbo->flags.IsLast()) {
                const auto& sym = sym_map.At(*mbo);
                if (sym == "GOOGL") {
                    target_iid = mbo->hd.instrument_id;
                    target_pid = mbo->hd.publisher_id;
                    break;
                }
            }
        }
    }
    ASSERT_NE(target_iid, 0u) << "Could not find GOOGL in data file";

    // Second pass: process all events for target (instrument, publisher)
    db::DbnFileStore store(kDataFile);
    auto symbol_map = store.GetMetadata().CreateSymbolMap();

    RefBook ref_book;
    Book lob_book;

    uint64_t event_count = 0;
    uint64_t adapter_converted = 0;
    uint64_t adapter_fallback = 0;
    uint64_t bbo_checks = 0;
    uint64_t mismatch_count = 0;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;
        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        // Filter to target publisher only
        if (mbo->hd.instrument_id != target_iid ||
            mbo->hd.publisher_id != target_pid) continue;

        event_count++;

        // Apply to reference book (handles all actions)
        ref_book.Apply(*mbo);

        // Try adapter path: MboMsg → Message → Book::apply()
        Message msg;
        if (constellation::adapters::to_message(*mbo, msg) && msg.is_valid()) {
            lob_book.apply(msg);
            adapter_converted++;
        } else {
            // Fallback: apply raw MboMsg to underlying LimitOrderBook
            lob_book.constellation_lob().OnMboUpdate(*mbo);
            adapter_fallback++;
        }

        // Compare at IsLast() events
        if (mbo->flags.IsLast()) {
            bbo_checks++;
            auto ref_bbo = ref_book.Bbo();

            double book_bid = lob_book.best_bid();
            double book_ask = lob_book.best_ask();
            uint32_t book_bid_qty = lob_book.best_bid_qty();
            uint32_t book_ask_qty = lob_book.best_ask_qty();

            // Compare bid
            bool bid_match = true;
            if (ref_bbo.first.IsEmpty()) {
                bid_match = std::isnan(book_bid);
            } else {
                double ref_bid_px = static_cast<double>(ref_bbo.first.price)
                                    / db::kFixedPriceScale;
                bid_match = !std::isnan(book_bid) &&
                            std::abs(ref_bid_px - book_bid) < 1e-6 &&
                            ref_bbo.first.size == book_bid_qty;
            }

            // Compare ask
            bool ask_match = true;
            if (ref_bbo.second.IsEmpty()) {
                ask_match = std::isnan(book_ask);
            } else {
                double ref_ask_px = static_cast<double>(ref_bbo.second.price)
                                    / db::kFixedPriceScale;
                ask_match = !std::isnan(book_ask) &&
                            std::abs(ref_ask_px - book_ask) < 1e-6 &&
                            ref_bbo.second.size == book_ask_qty;
            }

            if (!bid_match || !ask_match) {
                mismatch_count++;
                if (mismatch_count <= 5) {
                    double ref_bid_px = ref_bbo.first.IsEmpty() ? 0.0
                        : static_cast<double>(ref_bbo.first.price) / db::kFixedPriceScale;
                    double ref_ask_px = ref_bbo.second.IsEmpty() ? 0.0
                        : static_cast<double>(ref_bbo.second.price) / db::kFixedPriceScale;
                    std::cerr << "ADAPTER MISMATCH #" << mismatch_count
                              << " at event " << event_count << ":\n"
                              << "  REF  bid=" << ref_bbo.first.size << "@" << ref_bid_px
                              << " ask=" << ref_bbo.second.size << "@" << ref_ask_px << "\n"
                              << "  BOOK bid=" << book_bid_qty << "@" << book_bid
                              << " ask=" << book_ask_qty << "@" << book_ask << "\n";
                }
            }
        }
    }

    std::cerr << "[BookAdapter] " << event_count << " events for GOOGL publisher "
              << target_pid << ": " << adapter_converted << " via adapter, "
              << adapter_fallback << " direct fallback, "
              << bbo_checks << " BBO checks, " << mismatch_count << " mismatches\n";

    EXPECT_GT(bbo_checks, 0u) << "No BBO checks were performed";
    EXPECT_EQ(mismatch_count, 0u)
        << "Book adapter path mismatched reference " << mismatch_count
        << " times out of " << bbo_checks << " checks";
}

// ── Test: Constellation MarketBook (per-instrument, not per-publisher) ──
//
// MarketBook auto-creates one LimitOrderBook per instrument_id, but does NOT
// separate by publisher_id.  For CME futures (single publisher per instrument)
// this is fine.  For DBEQ equities (multiple exchanges/publishers per symbol)
// orders from different publishers are mixed into one book, causing incorrect
// BBO due to TOB events from one publisher clearing another publisher's levels.
//
// This test documents the divergence rather than asserting correctness.

TEST_F(LobValidationTest, MarketBookDivergenceFromReference) {
    if (!HasDataFile()) {
        GTEST_SKIP() << "DBEQ data file not found. Set DATABENTO_API_KEY to download.";
    }

    db::DbnFileStore store(kDataFile);
    auto symbol_map = store.GetMetadata().CreateSymbolMap();

    RefMarket ref_market;
    cst_ob::MarketBook market_book;

    uint64_t event_count = 0;
    uint64_t bbo_checks = 0;
    uint64_t bid_mismatches = 0;
    uint64_t ask_mismatches = 0;
    uint64_t crossed_book_count = 0;

    // Track per-instrument publisher counts to show multi-publisher situation
    std::unordered_map<uint32_t, std::set<uint16_t>> instrument_publishers;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;
        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        event_count++;
        instrument_publishers[mbo->hd.instrument_id].insert(mbo->hd.publisher_id);

        // Apply to both
        ref_market.Apply(*mbo);
        market_book.OnMboUpdate(*mbo);

        if (mbo->flags.IsLast()) {
            bbo_checks++;
            uint32_t iid = mbo->hd.instrument_id;
            auto ref_bbo = ref_market.AggregatedBbo(iid);

            auto mb_bid = market_book.GetBestBid(iid);
            auto mb_ask = market_book.GetBestAsk(iid);

            // Convert MarketBook BBO to RefPriceLevel for comparison
            RefPriceLevel mb_bid_lvl;
            if (mb_bid) {
                mb_bid_lvl = {mb_bid->price,
                              static_cast<uint32_t>(mb_bid->total_quantity),
                              mb_bid->order_count};
            }
            RefPriceLevel mb_ask_lvl;
            if (mb_ask) {
                mb_ask_lvl = {mb_ask->price,
                              static_cast<uint32_t>(mb_ask->total_quantity),
                              mb_ask->order_count};
            }

            if (ref_bbo.first.price != mb_bid_lvl.price ||
                ref_bbo.first.size != mb_bid_lvl.size) {
                bid_mismatches++;
            }
            if (ref_bbo.second.price != mb_ask_lvl.price ||
                ref_bbo.second.size != mb_ask_lvl.size) {
                ask_mismatches++;
            }

            // Detect crossed books (bid >= ask)
            if (mb_bid && mb_ask && mb_bid->price >= mb_ask->price) {
                crossed_book_count++;
            }
        }
    }

    // Report multi-publisher situation
    for (const auto& [iid, pubs] : instrument_publishers) {
        if (pubs.size() > 1) {
            std::cerr << "[MarketBook] instrument " << iid << " has "
                      << pubs.size() << " publishers: ";
            for (auto p : pubs) std::cerr << p << " ";
            std::cerr << "\n";
        }
    }

    std::cerr << "[MarketBook] " << event_count << " events, " << bbo_checks
              << " BBO checks\n"
              << "  Bid mismatches: " << bid_mismatches << "\n"
              << "  Ask mismatches: " << ask_mismatches << "\n"
              << "  Crossed books:  " << crossed_book_count << "\n";

    // This test is diagnostic — it documents the divergence.
    // MarketBook is expected to diverge for multi-publisher equities data.
    // We verify it ran and report statistics without asserting zero mismatches.
    EXPECT_GT(bbo_checks, 0u) << "No BBO checks were performed";

    // If there are multi-publisher instruments, we expect mismatches
    bool has_multi_publisher = std::any_of(
        instrument_publishers.begin(), instrument_publishers.end(),
        [](const auto& kv) { return kv.second.size() > 1; });

    if (has_multi_publisher) {
        // Document that mismatches exist for multi-publisher data
        std::cerr << "[MarketBook] Multi-publisher instruments detected — "
                  << "MarketBook does not separate by publisher, so mismatches are "
                  << "expected. This is an architectural limitation, not a bug, for "
                  << "equities data.\n";
        // We expect at least some mismatches
        EXPECT_GT(bid_mismatches + ask_mismatches, 0u)
            << "Expected mismatches for multi-publisher data, but got none";
    }
}

// ── MES Futures Validation ──────────────────────────────────────────────
//
// CME MES futures are single-publisher (GLBX.MDP3), so a direct
// per-instrument comparison between the Databento reference Book and
// Constellation's LimitOrderBook is sufficient.

static const std::string kMesDataFile =
    "data/mes/glbx-mdp3-20220103.mbo.dbn.zst";
static const std::string kMesRefOutput =
    "references/known_working/mes_reference_bbo_20220103.txt";
static const std::string kMesCstOutput =
    "references/known_working/mes_constellation_bbo_20220103.txt";

// Helper: format a RefPriceLevel line in the Databento reference output style
static std::string FormatLevel(const RefPriceLevel& lvl) {
    if (lvl.IsEmpty()) {
        return "    0 @ kUndefPrice | 0 order(s)";
    }
    std::ostringstream oss;
    oss << "    " << lvl.size << " @ " << db::pretty::Px{lvl.price}
        << " | " << lvl.count << " order(s)";
    return oss.str();
}

TEST_F(LobValidationTest, MesConstellationMatchesReference) {
    if (!std::filesystem::exists(kMesDataFile)) {
        GTEST_SKIP() << "MES data file not found at " << kMesDataFile;
    }

    db::DbnFileStore store(kMesDataFile);
    db::TsSymbolMap symbol_map;
    try {
        symbol_map = store.GetMetadata().CreateSymbolMap();
    } catch (...) {
        // Older DBN files may lack symbology
    }

    RefMarket ref_market;
    CstMarket cst_market;

    uint64_t event_count = 0;
    uint64_t mismatch_count = 0;
    uint64_t bbo_checks = 0;

    // Write first N BBO events to reference and constellation output files
    std::ofstream ref_out(kMesRefOutput);
    std::ofstream cst_out(kMesCstOutput);
    constexpr size_t kMaxOutput = 100;
    size_t output_count = 0;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;
        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        event_count++;
        ref_market.Apply(*mbo);
        cst_market.Apply(*mbo);

        if (mbo->flags.IsLast()) {
            bbo_checks++;
            auto ref_bbo = ref_market.AggregatedBbo(mbo->hd.instrument_id);
            auto cst_bbo = cst_market.AggregatedBbo(mbo->hd.instrument_id);

            // Resolve symbol name
            std::string sym;
            try { sym = symbol_map.At(*mbo); }
            catch (...) { sym = "inst_" + std::to_string(mbo->hd.instrument_id); }

            std::string ts = db::ToIso8601(mbo->ts_recv);

            // Save first N events to output files
            if (output_count < kMaxOutput) {
                ref_out << sym << " BBO | " << ts << "\n"
                        << FormatLevel(ref_bbo.second) << "\n"
                        << FormatLevel(ref_bbo.first) << "\n";

                cst_out << sym << " BBO | " << ts << "\n"
                        << FormatLevel(cst_bbo.second) << "\n"
                        << FormatLevel(cst_bbo.first) << "\n";
                output_count++;
            }

            // Compare bid
            bool bid_match = (ref_bbo.first.price == cst_bbo.first.price &&
                              ref_bbo.first.size == cst_bbo.first.size);
            // Compare ask
            bool ask_match = (ref_bbo.second.price == cst_bbo.second.price &&
                              ref_bbo.second.size == cst_bbo.second.size);

            if (!bid_match || !ask_match) {
                mismatch_count++;
                if (mismatch_count <= 10) {
                    std::cerr << "MES MISMATCH #" << mismatch_count
                              << " at " << sym << " event " << event_count
                              << " (" << ts << "):\n"
                              << "  REF  bid=" << ref_bbo.first.size << "@"
                              << ref_bbo.first.price << " ask="
                              << ref_bbo.second.size << "@"
                              << ref_bbo.second.price << "\n"
                              << "  CST  bid=" << cst_bbo.first.size << "@"
                              << cst_bbo.first.price << " ask="
                              << cst_bbo.second.size << "@"
                              << cst_bbo.second.price << "\n";
                }
            }
        }
    }

    ref_out.close();
    cst_out.close();

    std::cerr << "[MES Validation] Processed " << event_count << " MBO events, "
              << bbo_checks << " BBO checks, " << mismatch_count << " mismatches\n"
              << "[MES Validation] Reference output:     " << kMesRefOutput
              << " (" << output_count << " events)\n"
              << "[MES Validation] Constellation output:  " << kMesCstOutput
              << " (" << output_count << " events)\n";

    EXPECT_EQ(mismatch_count, 0u)
        << "Constellation BBO mismatched Databento reference " << mismatch_count
        << " times out of " << bbo_checks << " checks";
    EXPECT_GT(bbo_checks, 0u) << "No BBO checks were performed";
}

TEST_F(LobValidationTest, MesBookAdapterPathMatchesReference) {
    if (!std::filesystem::exists(kMesDataFile)) {
        GTEST_SKIP() << "MES data file not found at " << kMesDataFile;
    }

    // MES is single-publisher. Process through the LOB-RL Book adapter path
    // (MboMsg → to_message() → Book::apply()) and compare against RefBook.
    // For actions the adapter can't convert, fall back to raw OnMboUpdate.

    db::DbnFileStore store(kMesDataFile);
    db::TsSymbolMap symbol_map;
    try {
        symbol_map = store.GetMetadata().CreateSymbolMap();
    } catch (...) {}

    // Find the first instrument_id
    std::set<uint32_t> instrument_ids;
    RefMarket ref_market;

    // We'll build one Book per instrument, using the adapter path
    std::unordered_map<uint32_t, Book> lob_books;

    uint64_t event_count = 0;
    uint64_t bbo_checks = 0;
    uint64_t outright_mismatches = 0;
    uint64_t spread_mismatches = 0;
    uint64_t adapter_converted = 0;
    uint64_t adapter_fallback = 0;

    // Track which instruments are spreads (contain '-' in symbol)
    std::set<uint32_t> spread_instruments;

    while (true) {
        const db::Record* rec = store.NextRecord();
        if (!rec) break;
        const auto* mbo = rec->GetIf<db::MboMsg>();
        if (!mbo) continue;

        event_count++;
        uint32_t iid = mbo->hd.instrument_id;
        instrument_ids.insert(iid);

        // Detect spread instruments by symbol name
        if (mbo->flags.IsLast() && spread_instruments.find(iid) == spread_instruments.end()) {
            try {
                std::string sym = symbol_map.At(*mbo);
                if (sym.find('-') != std::string::npos) {
                    spread_instruments.insert(iid);
                }
            } catch (...) {}
        }

        // Apply to reference
        ref_market.Apply(*mbo);

        // Apply to LOB-RL Book via adapter
        auto& book = lob_books[iid];
        Message msg;
        if (constellation::adapters::to_message(*mbo, msg) && msg.is_valid()) {
            book.apply(msg);
            adapter_converted++;
        } else {
            book.constellation_lob().OnMboUpdate(*mbo);
            adapter_fallback++;
        }

        if (mbo->flags.IsLast()) {
            bbo_checks++;
            auto ref_bbo = ref_market.AggregatedBbo(iid);

            double book_bid = book.best_bid();
            double book_ask = book.best_ask();
            uint32_t book_bid_qty = book.best_bid_qty();
            uint32_t book_ask_qty = book.best_ask_qty();

            bool bid_match = true;
            if (ref_bbo.first.IsEmpty()) {
                bid_match = std::isnan(book_bid);
            } else {
                double ref_bid_px = static_cast<double>(ref_bbo.first.price)
                                    / db::kFixedPriceScale;
                bid_match = !std::isnan(book_bid) &&
                            std::abs(ref_bid_px - book_bid) < 1e-6 &&
                            ref_bbo.first.size == book_bid_qty;
            }

            bool ask_match = true;
            if (ref_bbo.second.IsEmpty()) {
                ask_match = std::isnan(book_ask);
            } else {
                double ref_ask_px = static_cast<double>(ref_bbo.second.price)
                                    / db::kFixedPriceScale;
                ask_match = !std::isnan(book_ask) &&
                            std::abs(ref_ask_px - book_ask) < 1e-6 &&
                            ref_bbo.second.size == book_ask_qty;
            }

            if (!bid_match || !ask_match) {
                bool is_spread = spread_instruments.count(iid) > 0;
                if (is_spread) {
                    spread_mismatches++;
                } else {
                    outright_mismatches++;
                    if (outright_mismatches <= 5) {
                        double ref_bid_px = ref_bbo.first.IsEmpty() ? 0.0
                            : static_cast<double>(ref_bbo.first.price)
                              / db::kFixedPriceScale;
                        double ref_ask_px = ref_bbo.second.IsEmpty() ? 0.0
                            : static_cast<double>(ref_bbo.second.price)
                              / db::kFixedPriceScale;
                        std::cerr << "MES ADAPTER MISMATCH #" << outright_mismatches
                                  << " at event " << event_count << ":\n"
                                  << "  REF  bid=" << ref_bbo.first.size << "@"
                                  << ref_bid_px << " ask=" << ref_bbo.second.size
                                  << "@" << ref_ask_px << "\n"
                                  << "  BOOK bid=" << book_bid_qty << "@" << book_bid
                                  << " ask=" << book_ask_qty << "@" << book_ask
                                  << "\n";
                    }
                }
            }
        }
    }

    std::cerr << "[MES BookAdapter] " << event_count << " events, "
              << instrument_ids.size() << " instruments ("
              << spread_instruments.size() << " spreads), "
              << adapter_converted << " via adapter, "
              << adapter_fallback << " direct fallback, "
              << bbo_checks << " BBO checks\n"
              << "  Outright mismatches: " << outright_mismatches << "\n"
              << "  Spread mismatches:   " << spread_mismatches
              << " (expected — adapter does not handle spread instruments)\n";

    EXPECT_GT(bbo_checks, 0u) << "No BBO checks were performed";
    EXPECT_EQ(outright_mismatches, 0u)
        << "MES Book adapter path mismatched reference on outright contracts "
        << outright_mismatches << " times out of " << bbo_checks << " checks";
}
