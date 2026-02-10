"""Tests for LOB reconstructor, dead book feature fix, and N_FEATURES constant.

Spec: docs/lob-features-fix.md

Tests cover:
  1. OrderBook class: Add, Cancel, Modify, Trade, Fill, Clear actions + BBO/depth/spread queries.
  2. Book feature integration: compute_bar_features with mbo_data produces correct col 1/2/10/11.
  3. N_FEATURES constant: exists, equals 13, used consistently.
  4. Precompute wiring: process_session stores n_features, load checks version.
  5. extract_all_mbo: returns correct DataFrame schema.
"""

import numpy as np
import pandas as pd
import pytest

from lob_rl.barrier.bar_pipeline import TradeBar

from .conftest import (
    TICK_SIZE,
    _RTH_OPEN_NS,
    _RTH_CLOSE_NS,
    _RTH_DURATION_NS,
    make_bar as _make_bar,
    make_session_bars as _make_session_bars,
)


# ===========================================================================
# Helpers — synthetic MBO data construction (test-only, no implementation logic)
# ===========================================================================

def _make_mbo_df(records):
    """Build a DataFrame matching extract_all_mbo() schema from a list of dicts.

    Each record: {action, side, price, size, order_id, ts_event}.
    """
    df = pd.DataFrame(records)
    # Ensure correct dtypes
    df["action"] = df["action"].astype(str)
    df["side"] = df["side"].astype(str)
    df["price"] = df["price"].astype(np.float64)
    df["size"] = df["size"].astype(np.int32)
    df["order_id"] = df["order_id"].astype(np.int64)
    df["ts_event"] = df["ts_event"].astype(np.int64)
    return df.sort_values("ts_event").reset_index(drop=True)


def _make_simple_book_mbo(t_start, t_end, bid_price=4000.0, ask_price=4000.25,
                           bid_qty=100, ask_qty=50, n_bid_levels=1, n_ask_levels=1):
    """Create MBO records that build a simple order book within a bar's time range.

    Returns a list of dicts (Add actions only) that establish a known book state.
    """
    records = []
    oid = 1000
    # Bid levels
    for lvl in range(n_bid_levels):
        price = bid_price - lvl * TICK_SIZE
        records.append({
            "action": "A", "side": "B", "price": price,
            "size": bid_qty, "order_id": oid,
            "ts_event": t_start + lvl + 1,
        })
        oid += 1
    # Ask levels
    for lvl in range(n_ask_levels):
        price = ask_price + lvl * TICK_SIZE
        records.append({
            "action": "A", "side": "A", "price": price,
            "size": ask_qty, "order_id": oid,
            "ts_event": t_start + n_bid_levels + lvl + 1,
        })
        oid += 1
    return records


# ===========================================================================
# 1. OrderBook — Imports
# ===========================================================================


class TestOrderBookImports:
    """OrderBook class should be importable from lob_reconstructor."""

    def test_order_book_importable(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook
        assert callable(OrderBook)


# ===========================================================================
# 2. OrderBook — Add action
# ===========================================================================


class TestOrderBookAdd:
    """OrderBook.apply('A', ...) adds orders and creates price levels."""

    def test_add_order_creates_level(self):
        """Add a single bid → best_bid is correct, best_bid_qty is correct."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        assert book.best_bid() == 4000.0
        assert book.best_bid_qty() == 10

    def test_add_multiple_orders_same_level(self):
        """Two adds at same price → qty sums."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "B", 4000.0, 20, order_id=2)
        assert book.best_bid() == 4000.0
        assert book.best_bid_qty() == 30

    def test_add_orders_different_levels_bid(self):
        """Multiple bid levels → best_bid is highest, depth sorted descending."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 3999.75, 5, order_id=1)
        book.apply("A", "B", 4000.0, 10, order_id=2)
        book.apply("A", "B", 3999.50, 3, order_id=3)
        assert book.best_bid() == 4000.0
        depth = book.bid_depth(3)
        assert len(depth) == 3
        assert depth[0][0] == 4000.0   # highest first
        assert depth[1][0] == 3999.75
        assert depth[2][0] == 3999.50

    def test_add_orders_ask_side(self):
        """Add ask orders → best_ask is lowest."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.25, 15, order_id=1)
        book.apply("A", "A", 4000.50, 20, order_id=2)
        assert book.best_ask() == 4000.25
        assert book.best_ask_qty() == 15


# ===========================================================================
# 3. OrderBook — Cancel action
# ===========================================================================


class TestOrderBookCancel:
    """OrderBook.apply('C', ...) cancels orders."""

    def test_cancel_known_order(self):
        """Cancel reduces qty; removes level when qty reaches 0."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("C", "B", 4000.0, 10, order_id=1)
        assert book.best_bid() == 0.0  # empty
        assert book.best_bid_qty() == 0

    def test_cancel_partial(self):
        """Two orders at same level, cancel one → remaining qty."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "B", 4000.0, 20, order_id=2)
        book.apply("C", "B", 4000.0, 10, order_id=1)
        assert book.best_bid_qty() == 20

    def test_cancel_unknown_order(self):
        """Cancel unknown order_id → no crash, no state change."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        # Cancel an order_id that was never added
        book.apply("C", "B", 4000.0, 10, order_id=999)
        # Original order still exists
        assert book.best_bid_qty() == 10


# ===========================================================================
# 4. OrderBook — Modify action
# ===========================================================================


class TestOrderBookModify:
    """OrderBook.apply('M', ...) modifies orders."""

    def test_modify_existing_order_size(self):
        """Modify changes size for an existing order."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("M", "B", 4000.0, 25, order_id=1)
        assert book.best_bid_qty() == 25

    def test_modify_existing_order_price(self):
        """Modify moves an order to a new price level."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("M", "B", 4000.25, 10, order_id=1)
        # Old level should be gone, new level has the order
        assert book.best_bid() == 4000.25
        assert book.best_bid_qty() == 10

    def test_modify_unknown_order_acts_as_add(self):
        """Modify unknown order_id → treated as Add (defensive)."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("M", "B", 4000.0, 15, order_id=999)
        assert book.best_bid() == 4000.0
        assert book.best_bid_qty() == 15


# ===========================================================================
# 5. OrderBook — Trade action
# ===========================================================================


class TestOrderBookTrade:
    """OrderBook.apply('T', ...) processes trade executions."""

    def test_trade_decrements_qty(self):
        """Trade reduces order qty. Removes level when fully filled."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.25, 20, order_id=1)
        book.apply("T", "A", 4000.25, 10, order_id=1)
        assert book.best_ask_qty() == 10

    def test_trade_fully_fills(self):
        """Trade consuming full size → level removed."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.25, 10, order_id=1)
        book.apply("T", "A", 4000.25, 10, order_id=1)
        assert book.best_ask() == 0.0
        assert book.best_ask_qty() == 0

    def test_trade_unknown_order(self):
        """Trade with unknown order_id → decrements price level directly."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Create a level with known qty
        book.apply("A", "A", 4000.25, 30, order_id=1)
        # Trade from unknown order_id — should still decrement the level
        book.apply("T", "A", 4000.25, 10, order_id=999)
        assert book.best_ask_qty() == 20


# ===========================================================================
# 6. OrderBook — Fill and Clear
# ===========================================================================


class TestOrderBookFillAndClear:
    """OrderBook.apply('F', ...) and apply('R', ...)."""

    def test_fill_same_as_trade(self):
        """'F' action behaves like 'T'."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 20, order_id=1)
        book.apply("F", "B", 4000.0, 5, order_id=1)
        assert book.best_bid_qty() == 15

    def test_clear_removes_order(self):
        """'R' action removes tracked order."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("R", "B", 4000.0, 10, order_id=1)
        assert book.best_bid() == 0.0
        assert book.best_bid_qty() == 0

    def test_clear_unknown_noop(self):
        """'R' on unknown order → no-op."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("R", "B", 4000.0, 10, order_id=999)
        # Original order still there
        assert book.best_bid_qty() == 10


# ===========================================================================
# 7. OrderBook — Spread and mid-price queries
# ===========================================================================


class TestOrderBookSpread:
    """Spread, spread_ticks, mid_price, weighted_mid_price."""

    def test_spread(self):
        """Spread = best_ask - best_bid."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "A", 4000.50, 10, order_id=2)
        assert book.spread() == pytest.approx(0.50)

    def test_spread_empty_book(self):
        """Spread returns 0.0 when book is empty."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        assert book.spread() == pytest.approx(0.0)

    def test_spread_one_side_empty(self):
        """Spread returns 0.0 when one side is empty."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        assert book.spread() == pytest.approx(0.0)

    def test_spread_ticks(self):
        """Spread in tick units: spread / TICK_SIZE."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "A", 4000.50, 10, order_id=2)
        expected = 0.50 / TICK_SIZE  # = 2.0
        assert book.spread_ticks() == pytest.approx(expected)

    def test_mid_price(self):
        """Mid = (best_bid + best_ask) / 2."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "A", 4000.50, 10, order_id=2)
        assert book.mid_price() == pytest.approx(4000.25)

    def test_mid_price_empty(self):
        """mid_price returns 0.0 when either side empty."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        assert book.mid_price() == pytest.approx(0.0)

    def test_weighted_mid_price(self):
        """Imbalance-weighted mid: (bid_qty * ask + ask_qty * bid) / (bid_qty + ask_qty)."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 100, order_id=1)  # heavy bid
        book.apply("A", "A", 4000.50, 50, order_id=2)   # light ask
        # weighted_mid = (100 * 4000.50 + 50 * 4000.0) / (100 + 50)
        expected = (100 * 4000.50 + 50 * 4000.0) / 150
        assert book.weighted_mid_price() == pytest.approx(expected)

    def test_weighted_mid_zero_qty(self):
        """weighted_mid returns mid_price when either BBO qty is 0."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Only bid side — weighted_mid should fall back to mid_price (which is 0)
        book.apply("A", "B", 4000.0, 10, order_id=1)
        assert book.weighted_mid_price() == book.mid_price()


# ===========================================================================
# 8. OrderBook — Depth queries
# ===========================================================================


class TestOrderBookDepth:
    """bid_depth, ask_depth, total_bid_depth, total_ask_depth."""

    def test_bid_depth_sorted_descending(self):
        """Bids sorted descending by price."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 3999.50, 5, order_id=1)
        book.apply("A", "B", 4000.0, 10, order_id=2)
        book.apply("A", "B", 3999.75, 8, order_id=3)

        depth = book.bid_depth(10)
        prices = [p for p, q in depth]
        assert prices == sorted(prices, reverse=True)

    def test_ask_depth_sorted_ascending(self):
        """Asks sorted ascending by price."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.50, 5, order_id=1)
        book.apply("A", "A", 4000.25, 10, order_id=2)
        book.apply("A", "A", 4001.0, 8, order_id=3)

        depth = book.ask_depth(10)
        prices = [p for p, q in depth]
        assert prices == sorted(prices)

    def test_total_bid_depth(self):
        """Cumulative quantity across top n bid levels."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "B", 3999.75, 20, order_id=2)
        book.apply("A", "B", 3999.50, 30, order_id=3)
        assert book.total_bid_depth(2) == 30   # top 2 levels: 10 + 20
        assert book.total_bid_depth(10) == 60  # all 3 levels

    def test_total_ask_depth(self):
        """Cumulative quantity across top n ask levels."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "A", 4000.25, 15, order_id=1)
        book.apply("A", "A", 4000.50, 25, order_id=2)
        assert book.total_ask_depth(1) == 15
        assert book.total_ask_depth(10) == 40

    def test_depth_n_limits(self):
        """Only returns top N levels."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        for i in range(20):
            book.apply("A", "B", 4000.0 - i * TICK_SIZE, 5, order_id=i)
        depth = book.bid_depth(5)
        assert len(depth) == 5

    def test_depth_empty_book(self):
        """Empty book → empty depth list."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        assert book.bid_depth(5) == []
        assert book.ask_depth(5) == []
        assert book.total_bid_depth(5) == 0
        assert book.total_ask_depth(5) == 0


# ===========================================================================
# 9. OrderBook — is_empty
# ===========================================================================


class TestOrderBookIsEmpty:
    """OrderBook.is_empty()."""

    def test_empty_on_init(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        assert book.is_empty() is True

    def test_not_empty_after_add(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        assert book.is_empty() is False

    def test_empty_after_cancel_all(self):
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("C", "B", 4000.0, 10, order_id=1)
        assert book.is_empty() is True


# ===========================================================================
# 10. OrderBook — Edge cases
# ===========================================================================


class TestOrderBookEdgeCases:
    """Edge cases: zero price, negative qty, replay sequence."""

    def test_price_zero_ignored(self):
        """Zero-price orders silently skipped."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 0.0, 10, order_id=1)
        assert book.is_empty() is True
        assert book.best_bid() == 0.0
        assert book.best_bid_qty() == 0

    def test_negative_qty_clamped(self):
        """Over-cancel doesn't produce negative qty at a level."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        book.apply("A", "B", 4000.0, 10, order_id=1)
        # Trade for more than available
        book.apply("T", "B", 4000.0, 15, order_id=1)
        # Should be 0 or removed, not negative
        assert book.best_bid_qty() >= 0

    def test_replay_sequence(self):
        """Multi-message sequence produces correct final state."""
        from lob_rl.barrier.lob_reconstructor import OrderBook

        book = OrderBook()
        # Build up a book
        book.apply("A", "B", 4000.0, 10, order_id=1)
        book.apply("A", "B", 3999.75, 20, order_id=2)
        book.apply("A", "A", 4000.25, 15, order_id=3)
        book.apply("A", "A", 4000.50, 25, order_id=4)
        # Modify order 1: change size
        book.apply("M", "B", 4000.0, 30, order_id=1)
        # Trade against order 3
        book.apply("T", "A", 4000.25, 5, order_id=3)
        # Cancel order 2
        book.apply("C", "B", 3999.75, 20, order_id=2)

        assert book.best_bid() == 4000.0
        assert book.best_bid_qty() == 30
        assert book.best_ask() == 4000.25
        assert book.best_ask_qty() == 10  # 15 - 5
        assert book.spread() == pytest.approx(0.25)
        assert book.total_bid_depth(10) == 30  # only one bid level remains
        assert book.total_ask_depth(10) == 35  # 10 + 25


# ===========================================================================
# 11. Book Feature Integration — BBO imbalance (col 1)
# ===========================================================================


class TestBBOImbalanceWithMBO:
    """compute_bar_features with mbo_data produces correct BBO imbalance."""

    def test_bbo_imbalance_known_values(self):
        """Synthetic MBO with bid_qty=100, ask_qty=50 → imbalance = 100/150 ≈ 0.667."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        bar = bars[1]  # Use middle bar
        t_start = bar.t_start
        t_end = bar.t_end

        # Build MBO with known bid/ask at bar close
        records = _make_simple_book_mbo(t_start, t_end,
                                         bid_price=4000.0, ask_price=4000.25,
                                         bid_qty=100, ask_qty=50)
        # Add some pre-bar messages to establish the book for bar 0
        records_bar0 = _make_simple_book_mbo(
            bars[0].t_start, bars[0].t_end,
            bid_price=4000.0, ask_price=4000.25,
            bid_qty=100, ask_qty=50
        )
        # Offset order IDs
        for r in records_bar0:
            r["order_id"] += 5000

        all_records = records_bar0 + records
        mbo_data = _make_mbo_df(all_records)

        features = compute_bar_features(bars, mbo_data=mbo_data)

        # For bar 1, BBO imbalance = bid_qty / (bid_qty + ask_qty) = 100/150
        expected_imbalance = 100.0 / (100.0 + 50.0)
        assert features[1, 1] == pytest.approx(expected_imbalance, abs=0.1)

    def test_bbo_imbalance_range(self):
        """BBO imbalance output in [0, 1]."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        records = []
        for bar in bars:
            records.extend(_make_simple_book_mbo(
                bar.t_start, bar.t_end,
                bid_qty=np.random.randint(10, 200),
                ask_qty=np.random.randint(10, 200),
            ))
        # Unique order IDs
        for i, r in enumerate(records):
            r["order_id"] = 10000 + i
        mbo_data = _make_mbo_df(records)

        features = compute_bar_features(bars, mbo_data=mbo_data)
        col1 = features[:, 1]
        assert np.all(col1 >= 0.0), f"BBO imbalance < 0: {col1.min()}"
        assert np.all(col1 <= 1.0), f"BBO imbalance > 1: {col1.max()}"

    def test_equal_bbo_gives_half(self):
        """Equal bid/ask qty → BBO imbalance = 0.5."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        records = _make_simple_book_mbo(
            bar.t_start, bar.t_end,
            bid_qty=100, ask_qty=100,
        )
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)
        assert features[0, 1] == pytest.approx(0.5, abs=0.05)


# ===========================================================================
# 12. Book Feature Integration — Depth imbalance (col 2)
# ===========================================================================


class TestDepthImbalanceWithMBO:
    """compute_bar_features with mbo_data produces correct depth imbalance."""

    def test_depth_imbalance_known_values(self):
        """5-level bid total=300, 5-level ask total=150 → depth_imbal = 300/450 ≈ 0.667."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        records = _make_simple_book_mbo(
            bar.t_start, bar.t_end,
            bid_qty=60, ask_qty=30,
            n_bid_levels=5, n_ask_levels=5,
        )
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # total_bid = 5 * 60 = 300, total_ask = 5 * 30 = 150
        expected = 300.0 / (300.0 + 150.0)
        assert features[0, 2] == pytest.approx(expected, abs=0.1)

    def test_depth_imbalance_range(self):
        """Depth imbalance output in [0, 1]."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        records = []
        for i, bar in enumerate(bars):
            records.extend(_make_simple_book_mbo(
                bar.t_start, bar.t_end,
                bid_qty=50 + i * 20, ask_qty=30 + i * 10,
                n_bid_levels=5, n_ask_levels=5,
            ))
        for i, r in enumerate(records):
            r["order_id"] = 20000 + i
        mbo_data = _make_mbo_df(records)

        features = compute_bar_features(bars, mbo_data=mbo_data)
        col2 = features[:, 2]
        assert np.all(col2 >= 0.0)
        assert np.all(col2 <= 1.0)


# ===========================================================================
# 13. Book Feature Integration — Cancel rate asymmetry (col 10)
# ===========================================================================


class TestCancelAsymmetryWithMBO:
    """compute_bar_features with mbo_data produces correct cancel asymmetry."""

    def test_cancel_asymmetry_known_values(self):
        """bid_cancels=8, ask_cancels=2 → asym = (8-2)/(8+2+1e-10) ≈ 0.6."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        t_start = bar.t_start
        t_end = bar.t_end
        oid = 3000

        # First, add orders on both sides
        records = []
        for i in range(10):
            records.append({
                "action": "A", "side": "B", "price": 4000.0,
                "size": 5, "order_id": oid,
                "ts_event": t_start + i + 1,
            })
            oid += 1
        for i in range(5):
            records.append({
                "action": "A", "side": "A", "price": 4000.25,
                "size": 5, "order_id": oid,
                "ts_event": t_start + 20 + i + 1,
            })
            oid += 1

        # Now cancel: 8 bid cancels, 2 ask cancels
        bid_cancel_oids = list(range(3000, 3008))  # First 8 bid orders
        ask_cancel_oids = list(range(3010, 3012))   # First 2 ask orders
        ts_cancel = t_start + 100
        for oid_c in bid_cancel_oids:
            records.append({
                "action": "C", "side": "B", "price": 4000.0,
                "size": 5, "order_id": oid_c,
                "ts_event": ts_cancel,
            })
            ts_cancel += 1
        for oid_c in ask_cancel_oids:
            records.append({
                "action": "C", "side": "A", "price": 4000.25,
                "size": 5, "order_id": oid_c,
                "ts_event": ts_cancel,
            })
            ts_cancel += 1

        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # cancel_asym = (bid_cancels - ask_cancels) / (bid_cancels + ask_cancels + 1e-10)
        expected = (8 - 2) / (8 + 2 + 1e-10)
        assert features[0, 10] == pytest.approx(expected, abs=0.1)

    def test_cancel_asymmetry_range(self):
        """Cancel asymmetry output in [-1, +1]."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        records = []
        oid = 5000
        for bar in bars:
            for i in range(5):
                records.append({
                    "action": "A", "side": "B", "price": 4000.0,
                    "size": 5, "order_id": oid,
                    "ts_event": bar.t_start + i + 1,
                })
                oid += 1
                records.append({
                    "action": "A", "side": "A", "price": 4000.25,
                    "size": 5, "order_id": oid,
                    "ts_event": bar.t_start + 10 + i + 1,
                })
                oid += 1
            # Cancel a few
            records.append({
                "action": "C", "side": "B", "price": 4000.0,
                "size": 5, "order_id": oid - 10,
                "ts_event": bar.t_start + 50,
            })
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)
        col10 = features[:, 10]
        assert np.all(col10 >= -1.0)
        assert np.all(col10 <= 1.0)

    def test_no_cancels_gives_zero(self):
        """No cancel actions in bar → cancel asymmetry = 0.0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        # Only add actions, no cancels
        records = _make_simple_book_mbo(bar.t_start, bar.t_end)
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)
        assert features[0, 10] == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# 14. Book Feature Integration — Mean spread (col 11)
# ===========================================================================


class TestMeanSpreadWithMBO:
    """compute_bar_features with mbo_data produces correct mean spread."""

    def test_mean_spread_known_value(self):
        """Constant 1-tick spread → mean_spread = 1.0 tick."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        # Build book with 1-tick spread (bid=4000.0, ask=4000.25)
        records = _make_simple_book_mbo(
            bar.t_start, bar.t_end,
            bid_price=4000.0, ask_price=4000.25,
        )
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # Spread = 0.25 / 0.25 = 1 tick
        assert features[0, 11] == pytest.approx(1.0, abs=0.5)

    def test_mean_spread_positive(self):
        """Mean spread output > 0."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        records = []
        for i, bar in enumerate(bars):
            records.extend(_make_simple_book_mbo(
                bar.t_start, bar.t_end,
                bid_price=4000.0, ask_price=4000.25 + i * TICK_SIZE,
            ))
        for i, r in enumerate(records):
            r["order_id"] = 30000 + i
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)
        col11 = features[:, 11]
        assert np.all(col11 > 0), f"Mean spread not positive: {col11}"


# ===========================================================================
# 15. Book Feature Integration — Backward compatibility and edge cases
# ===========================================================================


class TestBookFeaturesBackwardCompat:
    """Backward compatibility: no mbo_data → neutral defaults unchanged."""

    def test_features_without_mbo_unchanged(self):
        """No mbo_data → same neutral defaults as before."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(10)
        features = compute_bar_features(bars, mbo_data=None)
        # Col 1: BBO imbalance default 0.5
        np.testing.assert_allclose(features[:, 1], 0.5)
        # Col 2: Depth imbalance default 0.5
        np.testing.assert_allclose(features[:, 2], 0.5)
        # Col 10: Cancel asymmetry default 0.0
        np.testing.assert_allclose(features[:, 10], 0.0)
        # Col 11: Mean spread default 1.0
        np.testing.assert_allclose(features[:, 11], 1.0)

    def test_features_shape_unchanged(self):
        """Output shape is still (N, 13) with or without mbo_data."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features_no_mbo = compute_bar_features(bars)
        assert features_no_mbo.shape == (5, 13)

        # With minimal mbo_data
        records = _make_simple_book_mbo(bars[0].t_start, bars[-1].t_end)
        mbo_data = _make_mbo_df(records)
        features_with_mbo = compute_bar_features(bars, mbo_data=mbo_data)
        assert features_with_mbo.shape == (5, 13)

    def test_empty_mbo_falls_back(self):
        """Empty DataFrame → neutral defaults."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        empty_mbo = pd.DataFrame(columns=["action", "side", "price", "size", "order_id", "ts_event"])
        features = compute_bar_features(bars, mbo_data=empty_mbo)
        np.testing.assert_allclose(features[:, 1], 0.5)
        np.testing.assert_allclose(features[:, 2], 0.5)
        np.testing.assert_allclose(features[:, 10], 0.0)
        np.testing.assert_allclose(features[:, 11], 1.0)

    def test_book_empty_at_close_uses_defaults(self):
        """If book empties during bar (all cancelled), use neutral defaults for BBO/depth."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        t_start = bar.t_start
        t_end = bar.t_end

        # Add an order then immediately cancel it — book empty at close
        records = [
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": t_start + 1},
            {"action": "C", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 1, "ts_event": t_start + 2},
        ]
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # Book empty at close → neutral defaults
        assert features[0, 1] == pytest.approx(0.5)  # BBO imbalance
        assert features[0, 2] == pytest.approx(0.5)  # Depth imbalance

    def test_mbo_timestamps_align_with_bars(self):
        """Messages are assigned to correct bars based on bar boundaries."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        # Place different book states in each bar
        records = []
        # Bar 0: heavy bid (bid_qty=200, ask_qty=50)
        records.extend([
            {"action": "A", "side": "B", "price": 4000.0, "size": 200,
             "order_id": 100, "ts_event": bars[0].t_start + 1},
            {"action": "A", "side": "A", "price": 4000.25, "size": 50,
             "order_id": 101, "ts_event": bars[0].t_start + 2},
        ])
        # Bar 1: add more ask (cumulative: bid=200, ask=250)
        records.extend([
            {"action": "A", "side": "A", "price": 4000.25, "size": 200,
             "order_id": 102, "ts_event": bars[1].t_start + 1},
        ])
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # Bar 0: BBO imbalance = 200/(200+50) = 0.8
        assert features[0, 1] > 0.6  # heavy bid side
        # Bar 1: Book persists across bars. After bar 1 messages: bid=200, ask=250
        # BBO imbalance = 200/(200+250) < 0.5
        assert features[1, 1] < 0.6  # ask side now heavier

    def test_book_features_multiple_bars_persistence(self):
        """Book state persists across bars correctly."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(3)
        # Add bid in bar 0, never cancel — should persist through bars 1 and 2
        records = [
            {"action": "A", "side": "B", "price": 4000.0, "size": 100,
             "order_id": 200, "ts_event": bars[0].t_start + 1},
            {"action": "A", "side": "A", "price": 4000.25, "size": 100,
             "order_id": 201, "ts_event": bars[0].t_start + 2},
        ]
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # All three bars should have the same BBO imbalance (100/200 = 0.5)
        for i in range(3):
            assert features[i, 1] == pytest.approx(0.5, abs=0.05)

    def test_spread_samples_within_bar(self):
        """Mean spread computed from all events within bar."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(1)
        bar = bars[0]
        # Multiple events creating different spreads within the bar
        records = [
            # Initial book: spread = 1 tick
            {"action": "A", "side": "B", "price": 4000.0, "size": 10,
             "order_id": 300, "ts_event": bar.t_start + 1},
            {"action": "A", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 301, "ts_event": bar.t_start + 2},
            # Wider spread: add ask at 4000.50, cancel original ask
            {"action": "C", "side": "A", "price": 4000.25, "size": 10,
             "order_id": 301, "ts_event": bar.t_start + 3},
            {"action": "A", "side": "A", "price": 4000.50, "size": 10,
             "order_id": 302, "ts_event": bar.t_start + 4},
        ]
        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # Mean spread should be between 1 and 2 ticks (combination of 1-tick and 2-tick spreads)
        assert features[0, 11] >= 0.5  # at least some positive spread
        assert features[0, 11] <= 3.0  # not unreasonably large

    def test_cancel_count_per_bar(self):
        """Cancels counted only within bar boundaries."""
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(2)
        records = []
        oid = 400
        # Bar 0: 3 bid cancels
        for i in range(3):
            records.append({
                "action": "A", "side": "B", "price": 4000.0, "size": 5,
                "order_id": oid, "ts_event": bars[0].t_start + i + 1,
            })
            records.append({
                "action": "C", "side": "B", "price": 4000.0, "size": 5,
                "order_id": oid, "ts_event": bars[0].t_start + 100 + i,
            })
            oid += 1
        # Bar 1: 1 ask cancel
        records.append({
            "action": "A", "side": "A", "price": 4000.25, "size": 5,
            "order_id": oid, "ts_event": bars[1].t_start + 1,
        })
        records.append({
            "action": "C", "side": "A", "price": 4000.25, "size": 5,
            "order_id": oid, "ts_event": bars[1].t_start + 100,
        })
        oid += 1

        mbo_data = _make_mbo_df(records)
        features = compute_bar_features(bars, mbo_data=mbo_data)

        # Bar 0: bid_cancels=3, ask_cancels=0 → asym > 0
        assert features[0, 10] > 0
        # Bar 1: bid_cancels=0, ask_cancels=1 → asym < 0
        assert features[1, 10] < 0


# ===========================================================================
# 16. N_FEATURES constant
# ===========================================================================


class TestNFeaturesConstant:
    """N_FEATURES constant exists and is used correctly."""

    def test_n_features_constant_exists(self):
        """N_FEATURES is importable from lob_rl.barrier."""
        from lob_rl.barrier import N_FEATURES
        assert isinstance(N_FEATURES, int)

    def test_n_features_value(self):
        """N_FEATURES == 13."""
        from lob_rl.barrier import N_FEATURES
        assert N_FEATURES == 13

    def test_feature_pipeline_uses_n_features(self):
        """compute_bar_features output shape matches N_FEATURES columns."""
        from lob_rl.barrier import N_FEATURES
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(5)
        features = compute_bar_features(bars)
        assert features.shape[1] == N_FEATURES

    def test_barrier_env_uses_n_features(self):
        """BarrierEnv computes h using N_FEATURES, not hardcoded 13."""
        from lob_rl.barrier import N_FEATURES
        from lob_rl.barrier.barrier_env import BarrierEnv
        from lob_rl.barrier.feature_pipeline import build_feature_matrix
        from lob_rl.barrier.label_pipeline import compute_labels

        bars = _make_session_bars(40)
        labels = compute_labels(bars, a=20, b=10, t_max=40)
        features = build_feature_matrix(bars, h=10)
        env = BarrierEnv(bars, labels, features)

        # The env should have _h = feature_dim // N_FEATURES
        assert env._h == features.shape[1] // N_FEATURES

    def test_conftest_default_dim(self):
        """DEFAULT_FEATURE_DIM should equal N_FEATURES * DEFAULT_H."""
        from lob_rl.barrier import N_FEATURES
        from .conftest import DEFAULT_FEATURE_DIM, DEFAULT_H

        assert DEFAULT_FEATURE_DIM == N_FEATURES * DEFAULT_H


# ===========================================================================
# 17. extract_all_mbo — function contract
# ===========================================================================


class TestExtractAllMBO:
    """extract_all_mbo() function should be importable with correct signature."""

    def test_extract_all_mbo_importable(self):
        """extract_all_mbo is importable from bar_pipeline."""
        from lob_rl.barrier.bar_pipeline import extract_all_mbo
        assert callable(extract_all_mbo)

    def test_extract_all_mbo_returns_dataframe(self):
        """extract_all_mbo returns a DataFrame (tested via signature, actual file test skipped)."""
        # This test verifies the function exists and is callable.
        # A full integration test with .dbn.zst files would require fixtures.
        from lob_rl.barrier.bar_pipeline import extract_all_mbo
        import inspect
        sig = inspect.signature(extract_all_mbo)
        # Should accept filepath and optional instrument_id
        params = list(sig.parameters.keys())
        assert "filepath" in params


# ===========================================================================
# 18. Precompute wiring — process_session and load_session_from_cache
# ===========================================================================


class TestPrecomputeWiring:
    """Precompute stores n_features, load_session checks version."""

    def test_process_session_stores_n_features(self, tmp_path):
        """process_session result dict includes 'n_features' key."""
        # We can't call process_session directly (needs .dbn.zst files),
        # but we can test the contract: result dicts from process_session
        # should include 'n_features'. We test this by checking the
        # load_session_from_cache version check works.

        # Create a synthetic .npz that mimics what process_session would produce
        from lob_rl.barrier import N_FEATURES

        n_bars = 20
        lookback = 10
        n_usable = n_bars - lookback + 1

        npz_path = tmp_path / "test_session.npz"
        np.savez_compressed(
            npz_path,
            features=np.random.randn(n_usable, N_FEATURES * lookback).astype(np.float32),
            bar_open=np.full(n_bars, 4000.0),
            bar_high=np.full(n_bars, 4001.0),
            bar_low=np.full(n_bars, 3999.0),
            bar_close=np.full(n_bars, 4000.0),
            bar_volume=np.full(n_bars, 100, dtype=np.int32),
            bar_vwap=np.full(n_bars, 4000.0),
            bar_t_start=np.arange(n_bars, dtype=np.int64) * 1000,
            bar_t_end=np.arange(1, n_bars + 1, dtype=np.int64) * 1000,
            trade_prices=np.full(n_bars, 4000.0),
            trade_sizes=np.ones(n_bars, dtype=np.int32),
            bar_trade_offsets=np.arange(n_bars + 1, dtype=np.int64),
            label_values=np.zeros(n_bars, dtype=np.int8),
            label_tau=np.full(n_bars, 10, dtype=np.int32),
            label_resolution_bar=np.arange(n_bars, dtype=np.int32),
            n_bars=np.array(n_bars, dtype=np.int32),
            n_usable=np.array(n_usable, dtype=np.int32),
            n_features=np.array(N_FEATURES, dtype=np.int32),
        )

        # Loading should succeed when n_features matches
        from scripts.precompute_barrier_cache import load_session_from_cache
        session = load_session_from_cache(str(npz_path))
        assert "bars" in session
        assert "labels" in session
        assert "features" in session

    def test_load_session_version_check(self, tmp_path):
        """Mismatched n_features raises ValueError."""
        n_bars = 20
        lookback = 10
        n_usable = n_bars - lookback + 1

        npz_path = tmp_path / "bad_version.npz"
        np.savez_compressed(
            npz_path,
            features=np.random.randn(n_usable, 999).astype(np.float32),  # wrong feature count
            bar_open=np.full(n_bars, 4000.0),
            bar_high=np.full(n_bars, 4001.0),
            bar_low=np.full(n_bars, 3999.0),
            bar_close=np.full(n_bars, 4000.0),
            bar_volume=np.full(n_bars, 100, dtype=np.int32),
            bar_vwap=np.full(n_bars, 4000.0),
            bar_t_start=np.arange(n_bars, dtype=np.int64) * 1000,
            bar_t_end=np.arange(1, n_bars + 1, dtype=np.int64) * 1000,
            trade_prices=np.full(n_bars, 4000.0),
            trade_sizes=np.ones(n_bars, dtype=np.int32),
            bar_trade_offsets=np.arange(n_bars + 1, dtype=np.int64),
            label_values=np.zeros(n_bars, dtype=np.int8),
            label_tau=np.full(n_bars, 10, dtype=np.int32),
            label_resolution_bar=np.arange(n_bars, dtype=np.int32),
            n_bars=np.array(n_bars, dtype=np.int32),
            n_usable=np.array(n_usable, dtype=np.int32),
            n_features=np.array(999, dtype=np.int32),  # wrong!
        )

        from scripts.precompute_barrier_cache import load_session_from_cache
        with pytest.raises(ValueError, match="n_features"):
            load_session_from_cache(str(npz_path))

    def test_load_session_backward_compat(self, tmp_path):
        """Missing n_features key doesn't crash (old caches still loadable)."""
        n_bars = 20
        lookback = 10
        n_usable = n_bars - lookback + 1

        npz_path = tmp_path / "old_cache.npz"
        # Old-style cache: no n_features key
        np.savez_compressed(
            npz_path,
            features=np.random.randn(n_usable, 130).astype(np.float32),
            bar_open=np.full(n_bars, 4000.0),
            bar_high=np.full(n_bars, 4001.0),
            bar_low=np.full(n_bars, 3999.0),
            bar_close=np.full(n_bars, 4000.0),
            bar_volume=np.full(n_bars, 100, dtype=np.int32),
            bar_vwap=np.full(n_bars, 4000.0),
            bar_t_start=np.arange(n_bars, dtype=np.int64) * 1000,
            bar_t_end=np.arange(1, n_bars + 1, dtype=np.int64) * 1000,
            trade_prices=np.full(n_bars, 4000.0),
            trade_sizes=np.ones(n_bars, dtype=np.int32),
            bar_trade_offsets=np.arange(n_bars + 1, dtype=np.int64),
            label_values=np.zeros(n_bars, dtype=np.int8),
            label_tau=np.full(n_bars, 10, dtype=np.int32),
            label_resolution_bar=np.arange(n_bars, dtype=np.int32),
            n_bars=np.array(n_bars, dtype=np.int32),
            n_usable=np.array(n_usable, dtype=np.int32),
            # No n_features key!
        )

        from scripts.precompute_barrier_cache import load_session_from_cache
        # Should not crash — old caches are still loadable
        # May emit a warning but should return valid data
        session = load_session_from_cache(str(npz_path))
        assert "bars" in session
        assert len(session["bars"]) == n_bars


# ===========================================================================
# 19. build_feature_matrix integration with mbo_data
# ===========================================================================


class TestBuildFeatureMatrixWithMBO:
    """build_feature_matrix passes mbo_data through correctly."""

    def test_build_feature_matrix_accepts_mbo_data(self):
        """build_feature_matrix(bars, mbo_data=df) doesn't crash."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix

        bars = _make_session_bars(50)
        records = _make_simple_book_mbo(
            bars[0].t_start, bars[-1].t_end,
            bid_qty=100, ask_qty=50,
        )
        mbo_data = _make_mbo_df(records)
        result = build_feature_matrix(bars, mbo_data=mbo_data)
        assert result.ndim == 2
        assert np.all(np.isfinite(result))

    def test_build_feature_matrix_mbo_affects_output(self):
        """Features with mbo_data differ from features without (cols 1, 2, 10, 11)."""
        from lob_rl.barrier.feature_pipeline import build_feature_matrix
        from lob_rl.barrier.feature_pipeline import compute_bar_features

        bars = _make_session_bars(50)

        # Build MBO with asymmetric book (bid-heavy)
        records = []
        oid = 8000
        for bar in bars:
            records.append({
                "action": "A", "side": "B", "price": 4000.0, "size": 200,
                "order_id": oid, "ts_event": bar.t_start + 1,
            })
            oid += 1
            records.append({
                "action": "A", "side": "A", "price": 4000.25, "size": 50,
                "order_id": oid, "ts_event": bar.t_start + 2,
            })
            oid += 1
        mbo_data = _make_mbo_df(records)

        raw_without = compute_bar_features(bars, mbo_data=None)
        raw_with = compute_bar_features(bars, mbo_data=mbo_data)

        # Cols 1, 2 should differ (not both 0.5)
        assert not np.allclose(raw_with[:, 1], 0.5), \
            "BBO imbalance with mbo_data should not all be 0.5"
        # Non-book columns (0, 3-9, 12) should be the same
        for col in [0, 3, 4, 5, 6, 7, 8, 9, 12]:
            np.testing.assert_allclose(
                raw_with[:, col], raw_without[:, col],
                err_msg=f"Col {col} should be unchanged with mbo_data"
            )
