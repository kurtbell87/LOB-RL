#include "lob/book.h"
#include "constellation/adapters/message_adapter.h"
#include "interfaces/orderbook/IMarketStateView.hpp"
#include "databento/constants.hpp"
#include <limits>
#include <cmath>

namespace cst_ob = constellation::modules::orderbook;
namespace cst_if = constellation::interfaces::orderbook;

// Convert Constellation int64 nano-price to double.
static double to_double_price(std::int64_t p) {
    return static_cast<double>(p) / databento::kFixedPriceScale;
}

// Cap uint64 quantity to uint32 max (saturation).
static uint32_t cap_qty(uint64_t q) {
    return q > std::numeric_limits<uint32_t>::max()
        ? std::numeric_limits<uint32_t>::max()
        : static_cast<uint32_t>(q);
}

Book::Book()
    : lob_(std::make_unique<cst_ob::LimitOrderBook>(kInstrumentId)) {}

Book::~Book() = default;
Book::Book(Book&&) noexcept = default;
Book& Book::operator=(Book&&) noexcept = default;

void Book::apply(const Message& msg) {
    // Preserve LOB-RL behavior: Trade messages are no-ops for the book.
    if (msg.action == Message::Action::Trade) return;

    auto mbo = constellation::adapters::to_mbo_msg(msg, kInstrumentId);

    // Preserve LOB-RL behavior: Modify of unknown order is treated as Add.
    if (msg.action == Message::Action::Modify &&
        known_orders_.find(msg.order_id) == known_orders_.end()) {
        mbo.action = databento::Action::Add;
    }

    // Track order lifecycle
    if (mbo.action == databento::Action::Add) {
        known_orders_.insert(msg.order_id);
    } else if (mbo.action == databento::Action::Cancel) {
        known_orders_.erase(msg.order_id);
    }

    lob_->OnMboUpdate(mbo);
}

void Book::reset() {
    lob_ = std::make_unique<cst_ob::LimitOrderBook>(kInstrumentId);
    known_orders_.clear();
}

double Book::best_bid() const {
    auto lvl = lob_->BestBid();
    if (!lvl) return std::numeric_limits<double>::quiet_NaN();
    return to_double_price(lvl->price);
}

double Book::best_ask() const {
    auto lvl = lob_->BestAsk();
    if (!lvl) return std::numeric_limits<double>::quiet_NaN();
    return to_double_price(lvl->price);
}

double Book::mid_price() const {
    auto bb = lob_->BestBid();
    auto ba = lob_->BestAsk();
    if (!bb || !ba) return std::numeric_limits<double>::quiet_NaN();
    return (to_double_price(bb->price) + to_double_price(ba->price)) / 2.0;
}

double Book::spread() const {
    auto bb = lob_->BestBid();
    auto ba = lob_->BestAsk();
    if (!bb || !ba) return std::numeric_limits<double>::quiet_NaN();
    return to_double_price(ba->price) - to_double_price(bb->price);
}

size_t Book::bid_depth() const {
    return lob_->GetBids().size();
}

size_t Book::ask_depth() const {
    return lob_->GetAsks().size();
}

std::vector<Book::PriceLevel> Book::top_bids(int k) const {
    std::vector<PriceLevel> result;
    if (k <= 0) return result;
    result.reserve(static_cast<size_t>(k));

    auto bids = lob_->GetBids();  // sorted best-first
    for (size_t i = 0; i < bids.size() && static_cast<int>(i) < k; ++i) {
        result.push_back({to_double_price(bids[i].price),
                          cap_qty(bids[i].total_quantity)});
    }

    while (static_cast<int>(result.size()) < k) {
        result.push_back({std::numeric_limits<double>::quiet_NaN(), 0});
    }
    return result;
}

std::vector<Book::PriceLevel> Book::top_asks(int k) const {
    std::vector<PriceLevel> result;
    if (k <= 0) return result;
    result.reserve(static_cast<size_t>(k));

    auto asks = lob_->GetAsks();  // sorted best-first (lowest price first)
    for (size_t i = 0; i < asks.size() && static_cast<int>(i) < k; ++i) {
        result.push_back({to_double_price(asks[i].price),
                          cap_qty(asks[i].total_quantity)});
    }

    while (static_cast<int>(result.size()) < k) {
        result.push_back({std::numeric_limits<double>::quiet_NaN(), 0});
    }
    return result;
}

uint32_t Book::best_bid_qty() const {
    auto lvl = lob_->BestBid();
    if (!lvl) return 0;
    return cap_qty(lvl->total_quantity);
}

uint32_t Book::best_ask_qty() const {
    auto lvl = lob_->BestAsk();
    if (!lvl) return 0;
    return cap_qty(lvl->total_quantity);
}

uint32_t Book::total_bid_depth(int n) const {
    if (n <= 0) return 0;
    auto bids = lob_->GetBids();
    uint32_t total = 0;
    for (size_t i = 0; i < bids.size() && static_cast<int>(i) < n; ++i) {
        total += cap_qty(bids[i].total_quantity);
    }
    return total;
}

uint32_t Book::total_ask_depth(int n) const {
    if (n <= 0) return 0;
    auto asks = lob_->GetAsks();
    uint32_t total = 0;
    for (size_t i = 0; i < asks.size() && static_cast<int>(i) < n; ++i) {
        total += cap_qty(asks[i].total_quantity);
    }
    return total;
}

double Book::weighted_mid() const {
    auto bb = lob_->BestBid();
    auto ba = lob_->BestAsk();
    if (!bb || !ba) return std::numeric_limits<double>::quiet_NaN();

    double bq = static_cast<double>(bb->total_quantity);
    double aq = static_cast<double>(ba->total_quantity);
    if (bq + aq == 0.0) return std::numeric_limits<double>::quiet_NaN();

    double bp = to_double_price(bb->price);
    double ap = to_double_price(ba->price);
    return (bq * ap + aq * bp) / (bq + aq);
}

double Book::vamp(int n) const {
    if (n <= 0) return std::numeric_limits<double>::quiet_NaN();

    auto bids = lob_->GetBids();
    auto asks = lob_->GetAsks();
    if (bids.empty() || asks.empty())
        return std::numeric_limits<double>::quiet_NaN();

    double sum_pq = 0.0;
    double sum_q = 0.0;

    for (size_t i = 0; i < bids.size() && static_cast<int>(i) < n; ++i) {
        double p = to_double_price(bids[i].price);
        double q = static_cast<double>(bids[i].total_quantity);
        sum_pq += p * q;
        sum_q += q;
    }

    for (size_t i = 0; i < asks.size() && static_cast<int>(i) < n; ++i) {
        double p = to_double_price(asks[i].price);
        double q = static_cast<double>(asks[i].total_quantity);
        sum_pq += p * q;
        sum_q += q;
    }

    if (sum_q == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return sum_pq / sum_q;
}

cst_ob::LimitOrderBook& Book::constellation_lob() {
    return *lob_;
}

const cst_ob::LimitOrderBook& Book::constellation_lob() const {
    return *lob_;
}
