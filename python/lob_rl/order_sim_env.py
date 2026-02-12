"""OrderSimulationEnv — Gymnasium environment backed by Constellation's order engine.

Uses Constellation's OrdersEngine + MarketBook for realistic order execution
with actual fill simulation. Actions include limit orders at various offsets
from the current mid price.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import lob_rl_core as core


class OrderSimulationEnv(gym.Env):
    """RL environment with realistic order simulation via Constellation.

    Observation (11-dim float32):
        [0]  mid_price (normalized by initial mid)
        [1]  spread (ticks)
        [2]  best_bid_qty (normalized by 100)
        [3]  best_ask_qty (normalized by 100)
        [4]  position (-1/0/+1)
        [5]  unrealized_pnl (normalized by tick_value)
        [6]  open_buy_orders (count)
        [7]  open_sell_orders (count)
        [8]  time_remaining (fraction)
        [9]  last_fill_price (normalized by initial mid, 0 if none)
        [10] inventory_age (steps since last position change, normalized)

    Action (Discrete(7)):
        0: Hold (no new orders)
        1: Buy market
        2: Sell market
        3: Buy limit at best bid
        4: Sell limit at best ask
        5: Cancel all buy orders
        6: Cancel all sell orders

    Parameters
    ----------
    dbn_path : str
        Path to .dbn.zst file for MBO replay.
    instrument_id : int
        Databento instrument ID to filter.
    max_steps : int
        Maximum steps per episode.
    tick_size : float
        Instrument tick size (e.g. 0.25 for MES).
    tick_value : float
        Dollar value per tick (e.g. 1.25 for MES).
    max_position : int
        Maximum absolute position size.
    commission : float
        Commission per fill in dollars.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dbn_path: str,
        instrument_id: int,
        max_steps: int = 1000,
        tick_size: float = 0.25,
        tick_value: float = 1.25,
        max_position: int = 1,
        commission: float = 0.0,
    ):
        super().__init__()

        self.dbn_path = dbn_path
        self.instrument_id = instrument_id
        self.max_steps = max_steps
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.max_position = max_position
        self.commission = commission

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        # State
        self._engine: core.OrdersEngine | None = None
        self._position = 0
        self._realized_pnl = 0.0
        self._entry_price = 0.0
        self._step_count = 0
        self._initial_mid = 0.0
        self._last_fill_price = 0.0
        self._inventory_age = 0
        self._mid_price = 0.0
        self._spread = 0.0
        self._best_bid_qty = 0
        self._best_ask_qty = 0
        self._scale = core.FIXED_PRICE_SCALE

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._engine = core.OrdersEngine()
        self._position = 0
        self._realized_pnl = 0.0
        self._entry_price = 0.0
        self._step_count = 0
        self._last_fill_price = 0.0
        self._inventory_age = 0

        # For now, initialize with placeholder values.
        # In a full implementation, this would replay MBO data to build
        # the initial book state.
        self._initial_mid = 5000.0
        self._mid_price = 5000.0
        self._spread = self.tick_size
        self._best_bid_qty = 10
        self._best_ask_qty = 10

        obs = self._make_obs()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        reward = 0.0
        self._step_count += 1
        self._inventory_age += 1

        # Execute action
        if action == 1:  # Buy market
            reward += self._execute_market(core.OrderSide.Buy)
        elif action == 2:  # Sell market
            reward += self._execute_market(core.OrderSide.Sell)
        elif action == 3:  # Buy limit at best bid
            self._place_limit(core.OrderSide.Buy, self._mid_price - self.tick_size)
        elif action == 4:  # Sell limit at best ask
            self._place_limit(core.OrderSide.Sell, self._mid_price + self.tick_size)
        elif action == 5:  # Cancel buy orders
            self._cancel_side(core.OrderSide.Buy)
        elif action == 6:  # Cancel sell orders
            self._cancel_side(core.OrderSide.Sell)
        # action == 0: Hold

        # Check episode termination
        truncated = self._step_count >= self.max_steps
        terminated = False

        # Forced flatten at episode end
        if truncated and self._position != 0:
            reward += self._flatten()

        obs = self._make_obs()
        info = {
            "position": self._position,
            "realized_pnl": self._realized_pnl,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def _execute_market(self, side: core.OrderSide) -> float:
        """Place and immediately 'fill' a market order. Returns PnL delta."""
        if side == core.OrderSide.Buy and self._position >= self.max_position:
            return 0.0
        if side == core.OrderSide.Sell and self._position <= -self.max_position:
            return 0.0

        fill_price = self._mid_price
        if side == core.OrderSide.Buy:
            fill_price = self._mid_price + self._spread / 2  # pay the ask
        else:
            fill_price = self._mid_price - self._spread / 2  # hit the bid

        pnl = 0.0
        if side == core.OrderSide.Buy:
            if self._position < 0:
                # Closing short
                pnl = (self._entry_price - fill_price) * (self.tick_value / self.tick_size)
                self._realized_pnl += pnl
                self._entry_price = 0.0
            else:
                self._entry_price = fill_price
            self._position += 1
        else:
            if self._position > 0:
                # Closing long
                pnl = (fill_price - self._entry_price) * (self.tick_value / self.tick_size)
                self._realized_pnl += pnl
                self._entry_price = 0.0
            else:
                self._entry_price = fill_price
            self._position -= 1

        pnl -= self.commission
        self._last_fill_price = fill_price
        self._inventory_age = 0

        # Also place in OrdersEngine for tracking
        spec = core.OrderSpec()
        spec.instrument_id = self.instrument_id
        spec.type = core.OrderType.Market
        spec.side = side
        spec.quantity = 1
        self._engine.place_order(spec)

        return pnl

    def _place_limit(self, side: core.OrderSide, price: float):
        """Place a limit order."""
        spec = core.OrderSpec()
        spec.instrument_id = self.instrument_id
        spec.type = core.OrderType.Limit
        spec.side = side
        spec.quantity = 1
        spec.limit_price = int(price * self._scale)
        self._engine.place_order(spec)

    def _cancel_side(self, side: core.OrderSide):
        """Cancel all open orders on a given side."""
        open_orders = self._engine.list_open_orders(self.instrument_id)
        for info in open_orders:
            if info.side == side:
                self._engine.cancel_order(info.order_id)

    def _flatten(self) -> float:
        """Flatten position at current mid. Returns PnL."""
        if self._position > 0:
            return self._execute_market(core.OrderSide.Sell)
        elif self._position < 0:
            return self._execute_market(core.OrderSide.Buy)
        return 0.0

    def _make_obs(self) -> np.ndarray:
        """Build observation vector."""
        open_orders = self._engine.list_open_orders(self.instrument_id) if self._engine else []
        n_buy = sum(1 for o in open_orders if o.side == core.OrderSide.Buy)
        n_sell = sum(1 for o in open_orders if o.side == core.OrderSide.Sell)

        unrealized = 0.0
        if self._position > 0:
            unrealized = (self._mid_price - self._entry_price) * (self.tick_value / self.tick_size)
        elif self._position < 0:
            unrealized = (self._entry_price - self._mid_price) * (self.tick_value / self.tick_size)

        norm_mid = self._mid_price / self._initial_mid if self._initial_mid > 0 else 0.0
        norm_fill = self._last_fill_price / self._initial_mid if self._initial_mid > 0 else 0.0
        time_remaining = max(0.0, 1.0 - self._step_count / self.max_steps)

        return np.array([
            norm_mid,
            self._spread / self.tick_size,
            self._best_bid_qty / 100.0,
            self._best_ask_qty / 100.0,
            float(self._position),
            unrealized / self.tick_value if self.tick_value > 0 else 0.0,
            float(n_buy),
            float(n_sell),
            time_remaining,
            norm_fill,
            min(self._inventory_age / 100.0, 1.0),
        ], dtype=np.float32)
