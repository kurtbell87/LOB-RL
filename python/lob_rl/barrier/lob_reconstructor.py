"""Limit order book reconstruction from MBO (Market By Order) messages.

Pure Python LOB that replays Add/Cancel/Modify/Trade/Fill/Clear actions
to maintain a stateful order book. Used by feature_pipeline to compute
book-derived features (BBO imbalance, depth imbalance, cancel asymmetry,
mean spread).
"""

from lob_rl.barrier import TICK_SIZE


class OrderBook:
    """Stateful limit order book reconstructed from MBO messages.

    Tracks individual orders by order_id. Maintains price levels on
    bid and ask sides as dict[price -> total_qty].
    """

    def __init__(self):
        self._bids = {}   # price -> total_qty
        self._asks = {}   # price -> total_qty
        self._orders = {}  # order_id -> (side, price, size)

    def apply(self, action, side, price, size, order_id=0):
        """Process a single MBO message.

        Parameters
        ----------
        action : str
            'A' (Add), 'C' (Cancel), 'M' (Modify), 'T' (Trade),
            'R' (Clear), 'F' (Fill).
        side : str
            'B' (Bid) or 'A' (Ask).
        price : float
            Order price.
        size : int
            Order size.
        order_id : int
            Unique order identifier.
        """
        if action == "A":
            self._add(side, price, size, order_id)
        elif action == "C":
            self._cancel(side, price, size, order_id)
        elif action == "M":
            self._modify(side, price, size, order_id)
        elif action == "T" or action == "F":
            self._trade(side, price, size, order_id)
        elif action == "R":
            self._clear(side, price, size, order_id)

    def _add(self, side, price, size, order_id):
        if price == 0.0:
            return
        book = self._bids if side == "B" else self._asks
        book[price] = book.get(price, 0) + size
        self._orders[order_id] = (side, price, size)

    def _cancel(self, side, price, size, order_id):
        if order_id not in self._orders:
            return
        stored_side, stored_price, stored_size = self._orders[order_id]
        book = self._bids if stored_side == "B" else self._asks
        if stored_price in book:
            book[stored_price] -= stored_size
            if book[stored_price] <= 0:
                del book[stored_price]
        del self._orders[order_id]

    def _modify(self, side, price, size, order_id):
        if order_id in self._orders:
            # Remove old order
            old_side, old_price, old_size = self._orders[order_id]
            old_book = self._bids if old_side == "B" else self._asks
            if old_price in old_book:
                old_book[old_price] -= old_size
                if old_book[old_price] <= 0:
                    del old_book[old_price]
            # Add at new location
            if price == 0.0:
                del self._orders[order_id]
                return
            new_book = self._bids if side == "B" else self._asks
            new_book[price] = new_book.get(price, 0) + size
            self._orders[order_id] = (side, price, size)
        else:
            # Unknown order_id — treat as Add
            self._add(side, price, size, order_id)

    def _trade(self, side, price, size, order_id):
        if order_id in self._orders:
            stored_side, stored_price, stored_size = self._orders[order_id]
            book = self._bids if stored_side == "B" else self._asks
            new_size = stored_size - size
            if new_size <= 0:
                # Fully filled — remove order and adjust level
                if stored_price in book:
                    book[stored_price] -= stored_size
                    if book[stored_price] <= 0:
                        del book[stored_price]
                del self._orders[order_id]
            else:
                # Partial fill
                if stored_price in book:
                    book[stored_price] -= size
                    if book[stored_price] <= 0:
                        del book[stored_price]
                self._orders[order_id] = (stored_side, stored_price, new_size)
        else:
            # Unknown order_id — decrement the price level directly
            book = self._bids if side == "B" else self._asks
            if price in book:
                book[price] -= size
                if book[price] <= 0:
                    del book[price]

    def _clear(self, side, price, size, order_id):
        if order_id not in self._orders:
            return
        stored_side, stored_price, stored_size = self._orders[order_id]
        book = self._bids if stored_side == "B" else self._asks
        if stored_price in book:
            book[stored_price] -= stored_size
            if book[stored_price] <= 0:
                del book[stored_price]
        del self._orders[order_id]

    def best_bid(self):
        """Return best bid price, or 0.0 if no bids."""
        if not self._bids:
            return 0.0
        return max(self._bids.keys())

    def best_ask(self):
        """Return best ask price, or 0.0 if no asks."""
        if not self._asks:
            return 0.0
        return min(self._asks.keys())

    def best_bid_qty(self):
        """Return total quantity at best bid, or 0 if no bids."""
        if not self._bids:
            return 0
        return self._bids[max(self._bids.keys())]

    def best_ask_qty(self):
        """Return total quantity at best ask, or 0 if no asks."""
        if not self._asks:
            return 0
        return self._asks[min(self._asks.keys())]

    def spread(self):
        """Return spread in price units, or 0.0 if either side empty."""
        if not self._bids or not self._asks:
            return 0.0
        return self.best_ask() - self.best_bid()

    def spread_ticks(self):
        """Return spread in tick units."""
        return self.spread() / TICK_SIZE

    def mid_price(self):
        """Return (best_bid + best_ask) / 2, or 0.0 if either side empty."""
        if not self._bids or not self._asks:
            return 0.0
        return (self.best_bid() + self.best_ask()) / 2.0

    def weighted_mid_price(self):
        """Imbalance-weighted mid price.

        (bid_qty * ask + ask_qty * bid) / (bid_qty + ask_qty).
        Falls back to mid_price() if either BBO qty is 0.
        """
        if not self._bids or not self._asks:
            return self.mid_price()
        bid_qty = self.best_bid_qty()
        ask_qty = self.best_ask_qty()
        if bid_qty == 0 or ask_qty == 0:
            return self.mid_price()
        bid = self.best_bid()
        ask = self.best_ask()
        return (bid_qty * ask + ask_qty * bid) / (bid_qty + ask_qty)

    def bid_depth(self, n=10):
        """Top n bid levels as [(price, qty)], sorted descending by price."""
        if not self._bids:
            return []
        sorted_levels = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)
        return [(p, q) for p, q in sorted_levels[:n]]

    def ask_depth(self, n=10):
        """Top n ask levels as [(price, qty)], sorted ascending by price."""
        if not self._asks:
            return []
        sorted_levels = sorted(self._asks.items(), key=lambda x: x[0])
        return [(p, q) for p, q in sorted_levels[:n]]

    def total_bid_depth(self, n=10):
        """Cumulative quantity across top n bid levels."""
        depth = self.bid_depth(n)
        return sum(q for _, q in depth)

    def total_ask_depth(self, n=10):
        """Cumulative quantity across top n ask levels."""
        depth = self.ask_depth(n)
        return sum(q for _, q in depth)

    def vamp(self, n=3):
        """Volume-adjusted mid-price using top n levels on each side.

        Returns 0.0 if either side has no levels.
        """
        bid_levels = self.bid_depth(n)
        ask_levels = self.ask_depth(n)
        if not bid_levels or not ask_levels:
            return 0.0
        total_qty = sum(q for _, q in bid_levels) + sum(q for _, q in ask_levels)
        if total_qty == 0:
            return 0.0
        weighted = (sum(p * q for p, q in bid_levels) + sum(p * q for p, q in ask_levels))
        return weighted / total_qty

    def is_empty(self):
        """Return True if both sides are empty."""
        return not self._bids and not self._asks
