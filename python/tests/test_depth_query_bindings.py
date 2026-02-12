"""Tests for depth query Python bindings on MarketBook.

Tests the new methods added per docs/depth-queries.md:
- mb.get_level(instrument_id, side, depth_index)
- mb.total_depth(instrument_id, side, n_levels)
- mb.weighted_mid_price(instrument_id)
- mb.vamp(instrument_id, n_levels)
- BookSide enum availability
"""

import pytest
import lob_rl_core as core


# ── Helpers ──────────────────────────────────────────────────────────────

def make_mbo_msg(instrument_id: int, order_id: int, price: float,
                 size: int, side: str, action: str):
    """Create a databento MboMsg dict for OnMboUpdate.

    Note: MarketBook.on_mbo_update is expected to be exposed or
    we build the book via BatchBacktestEngine. Since the spec says
    MarketBook methods should be directly testable, we need a way
    to populate the book from Python.

    The current bindings don't expose OnMboUpdate on MarketBook,
    so these tests assume that binding is added (or test via
    BatchBacktestEngine if available).

    For now, tests are written against the expected API contract.
    If on_mbo_update isn't exposed, tests will fail with AttributeError
    (which is a valid RED failure).
    """
    pass


def populate_book(mb, instrument_id, bids, asks):
    """Add bid and ask orders to a MarketBook.

    bids: list of (order_id, price, qty)
    asks: list of (order_id, price, qty)

    Uses the on_mbo_update method which must be bound.
    """
    for oid, price, qty in bids:
        mb.on_mbo_update(instrument_id, oid, price, qty, core.BookSide.Bid)
    for oid, price, qty in asks:
        mb.on_mbo_update(instrument_id, oid, price, qty, core.BookSide.Ask)


# ── BookSide Enum Tests ─────────────────────────────────────────────────


class TestBookSideEnum:
    def test_bid_exists(self):
        assert core.BookSide.Bid is not None

    def test_ask_exists(self):
        assert core.BookSide.Ask is not None

    def test_bid_and_ask_are_distinct(self):
        assert core.BookSide.Bid != core.BookSide.Ask


# ── get_level Tests ──────────────────────────────────────────────────────


class TestGetLevel:
    def test_unknown_instrument_returns_none(self):
        mb = core.MarketBook()
        result = mb.get_level(9999, core.BookSide.Bid, 0)
        assert result is None

    def test_returns_dict_with_price_quantity_order_count(self):
        """get_level should return a dict with keys: price, quantity, order_count."""
        mb = core.MarketBook()
        # Need to populate the book first — this will fail if on_mbo_update
        # isn't bound, which is a valid RED failure
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[(200, 100.50, 20)])

        result = mb.get_level(1, core.BookSide.Bid, 0)
        assert result is not None
        assert "price" in result
        assert "quantity" in result
        assert "order_count" in result

    def test_bid_level_0_returns_best_bid(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10), (101, 99.25, 20)],
                      asks=[(200, 100.50, 15)])

        result = mb.get_level(1, core.BookSide.Bid, 0)
        assert result is not None
        assert abs(result["price"] - 99.50) < 1e-6
        assert result["quantity"] == 10

    def test_ask_level_0_returns_best_ask(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[(200, 100.50, 15), (201, 101.00, 25)])

        result = mb.get_level(1, core.BookSide.Ask, 0)
        assert result is not None
        assert abs(result["price"] - 100.50) < 1e-6
        assert result["quantity"] == 15

    def test_deeper_level_returns_correct_data(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10), (101, 99.50, 20), (102, 99.00, 30)],
                      asks=[])

        result = mb.get_level(1, core.BookSide.Bid, 2)
        assert result is not None
        assert abs(result["price"] - 99.00) < 1e-6
        assert result["quantity"] == 30

    def test_beyond_max_depth_returns_none(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[])

        result = mb.get_level(1, core.BookSide.Bid, 5)
        assert result is None


# ── total_depth Tests ────────────────────────────────────────────────────


class TestTotalDepth:
    def test_unknown_instrument_returns_zero(self):
        mb = core.MarketBook()
        assert mb.total_depth(9999, core.BookSide.Bid, 5) == 0

    def test_n_levels_zero_returns_zero(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[])

        assert mb.total_depth(1, core.BookSide.Bid, 0) == 0

    def test_sums_top_n_bid_levels(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10), (101, 99.50, 20), (102, 99.00, 30)],
                      asks=[])

        assert mb.total_depth(1, core.BookSide.Bid, 2) == 30

    def test_sums_top_n_ask_levels(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[],
                      asks=[(200, 100.50, 10), (201, 101.00, 20)])

        assert mb.total_depth(1, core.BookSide.Ask, 2) == 30

    def test_returns_int(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[])

        result = mb.total_depth(1, core.BookSide.Bid, 1)
        assert isinstance(result, int)


# ── weighted_mid_price Tests ─────────────────────────────────────────────


class TestWeightedMidPrice:
    def test_unknown_instrument_returns_none(self):
        mb = core.MarketBook()
        assert mb.weighted_mid_price(9999) is None

    def test_only_bids_returns_none(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[])

        assert mb.weighted_mid_price(1) is None

    def test_only_asks_returns_none(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[],
                      asks=[(200, 100.50, 10)])

        assert mb.weighted_mid_price(1) is None

    def test_equal_quantities_returns_simple_mid(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 50)],
                      asks=[(200, 101.00, 50)])

        wmid = mb.weighted_mid_price(1)
        assert wmid is not None
        assert abs(wmid - 100.5) < 1e-9

    def test_asymmetric_quantities_skew(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 90)],
                      asks=[(200, 101.00, 10)])

        wmid = mb.weighted_mid_price(1)
        assert wmid is not None
        # wmid = (100*10 + 101*90) / 100 = 100.9
        expected = (100.0 * 10 + 101.0 * 90) / 100.0
        assert abs(wmid - expected) < 1e-9

    def test_returns_float(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10)],
                      asks=[(200, 101.00, 10)])

        wmid = mb.weighted_mid_price(1)
        assert isinstance(wmid, float)


# ── vamp Tests ───────────────────────────────────────────────────────────


class TestVamp:
    def test_unknown_instrument_returns_none(self):
        mb = core.MarketBook()
        assert mb.vamp(9999, 5) is None

    def test_only_bids_returns_none(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 99.50, 10)],
                      asks=[])

        assert mb.vamp(1, 5) is None

    def test_n_levels_zero_returns_none(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10)],
                      asks=[(200, 101.00, 10)])

        assert mb.vamp(1, 0) is None

    def test_single_level_each_side(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 30)],
                      asks=[(200, 101.00, 70)])

        v = mb.vamp(1, 1)
        assert v is not None
        # VAMP = (100*30 + 101*70) / 100 = 100.70
        expected = (100.0 * 30 + 101.0 * 70) / 100.0
        assert abs(v - expected) < 1e-9

    def test_multi_level_hand_computed(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10), (101, 99.00, 20)],
                      asks=[(200, 101.00, 15), (201, 102.00, 25)])

        v = mb.vamp(1, 2)
        assert v is not None
        # = (100*10 + 99*20 + 101*15 + 102*25) / 70 = 7045/70
        expected = (100.0*10 + 99.0*20 + 101.0*15 + 102.0*25) / 70.0
        assert abs(v - expected) < 1e-6

    def test_returns_float(self):
        mb = core.MarketBook()
        populate_book(mb, 1,
                      bids=[(100, 100.00, 10)],
                      asks=[(200, 101.00, 10)])

        v = mb.vamp(1, 1)
        assert isinstance(v, float)
