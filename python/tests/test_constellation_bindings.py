"""Integration tests for Constellation bindings exposed via lob_rl_core.

Tests order simulation (OrdersEngine), replay (BatchBacktestEngine),
market book (MarketBook), and feature system bindings.
"""

import pytest
import lob_rl_core as core


# ── OrdersEngine Tests ────────────────────────────────────────────────


class TestOrdersEngine:
    def test_place_market_order(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Market
        spec.side = core.OrderSide.Buy
        spec.quantity = 10
        oid = engine.place_order(spec)
        assert oid > 0

    def test_place_limit_order(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Limit
        spec.side = core.OrderSide.Buy
        spec.quantity = 5
        spec.limit_price = 5000 * core.FIXED_PRICE_SCALE
        oid = engine.place_order(spec)
        assert oid > 0

    def test_list_open_orders(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Limit
        spec.side = core.OrderSide.Buy
        spec.quantity = 5
        spec.limit_price = 5000 * core.FIXED_PRICE_SCALE
        engine.place_order(spec)

        open_orders = engine.list_open_orders(1)
        assert len(open_orders) == 1
        assert open_orders[0].side == core.OrderSide.Buy

    def test_cancel_order(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Limit
        spec.side = core.OrderSide.Sell
        spec.quantity = 3
        spec.limit_price = 5100 * core.FIXED_PRICE_SCALE
        oid = engine.place_order(spec)

        result = engine.cancel_order(oid)
        assert result is True

        open_orders = engine.list_open_orders(1)
        assert len(open_orders) == 0

    def test_get_order_info(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Limit
        spec.side = core.OrderSide.Buy
        spec.quantity = 7
        spec.limit_price = 4900 * core.FIXED_PRICE_SCALE
        oid = engine.place_order(spec)

        info = engine.get_order_info(oid)
        assert info.order_id == oid
        assert info.instrument_id == 1
        assert info.side == core.OrderSide.Buy
        assert info.original_quantity == 7
        assert info.status == core.OrderStatus.New

    def test_modify_order(self):
        engine = core.OrdersEngine()
        spec = core.OrderSpec()
        spec.instrument_id = 1
        spec.type = core.OrderType.Limit
        spec.side = core.OrderSide.Buy
        spec.quantity = 5
        spec.limit_price = 5000 * core.FIXED_PRICE_SCALE
        oid = engine.place_order(spec)

        update = core.OrderUpdate()
        update.new_quantity = 10
        result = engine.modify_order(oid, update)
        assert result is True

    def test_multiple_instruments(self):
        engine = core.OrdersEngine()
        for inst_id in [1, 2, 3]:
            spec = core.OrderSpec()
            spec.instrument_id = inst_id
            spec.type = core.OrderType.Limit
            spec.side = core.OrderSide.Buy
            spec.quantity = 1
            spec.limit_price = 5000 * core.FIXED_PRICE_SCALE
            engine.place_order(spec)

        assert len(engine.list_open_orders(1)) == 1
        assert len(engine.list_open_orders(2)) == 1
        assert len(engine.list_open_orders(3)) == 1


# ── Enum Tests ────────────────────────────────────────────────────────


class TestEnums:
    def test_order_type_values(self):
        assert core.OrderType.Market is not None
        assert core.OrderType.Limit is not None
        assert core.OrderType.Stop is not None
        assert core.OrderType.StopLimit is not None

    def test_order_side_values(self):
        assert core.OrderSide.Buy is not None
        assert core.OrderSide.Sell is not None

    def test_order_status_values(self):
        assert core.OrderStatus.New is not None
        assert core.OrderStatus.PartiallyFilled is not None
        assert core.OrderStatus.Filled is not None
        assert core.OrderStatus.Canceled is not None
        assert core.OrderStatus.Rejected is not None

    def test_time_in_force_values(self):
        assert core.TimeInForce.Day is not None
        assert core.TimeInForce.GTC is not None
        assert core.TimeInForce.IOC is not None
        assert core.TimeInForce.FOK is not None

    def test_book_side_values(self):
        assert core.BookSide.Bid is not None
        assert core.BookSide.Ask is not None


# ── MarketBook Tests ──────────────────────────────────────────────────


class TestMarketBook:
    def test_empty_market_book(self):
        mb = core.MarketBook()
        assert mb.instrument_count() == 0
        assert mb.get_instrument_ids() == []

    def test_no_bid_ask_for_unknown_instrument(self):
        mb = core.MarketBook()
        assert mb.best_bid_price(1) is None
        assert mb.best_ask_price(1) is None

    def test_global_counters_start_at_zero(self):
        mb = core.MarketBook()
        assert mb.get_global_add_count() == 0
        assert mb.get_global_cancel_count() == 0
        assert mb.get_global_modify_count() == 0
        assert mb.get_global_trade_count() == 0
        assert mb.get_global_total_event_count() == 0


# ── BatchBacktestEngine Tests ────────────────────────────────────────


class TestBatchBacktestEngine:
    def test_create_and_configure(self):
        engine = core.BatchBacktestEngine()
        cfg = core.BatchAggregatorConfig()
        cfg.batch_size = 10000
        engine.set_aggregator_config(cfg)

    def test_empty_fills(self):
        engine = core.BatchBacktestEngine()
        cfg = core.BatchAggregatorConfig()
        engine.set_aggregator_config(cfg)
        fills = engine.get_fills()
        assert fills == []

    def test_reset_stats(self):
        engine = core.BatchBacktestEngine()
        cfg = core.BatchAggregatorConfig()
        engine.set_aggregator_config(cfg)
        engine.reset_stats()


class TestBatchAggregatorConfig:
    def test_default_values(self):
        cfg = core.BatchAggregatorConfig()
        assert cfg.batch_size == 50000
        assert cfg.enable_logging is True

    def test_custom_values(self):
        cfg = core.BatchAggregatorConfig()
        cfg.batch_size = 100000
        cfg.enable_logging = False
        cfg.enable_instrument_boundary = True
        cfg.boundary_instrument_id = 42
        assert cfg.batch_size == 100000
        assert cfg.enable_logging is False
        assert cfg.enable_instrument_boundary is True
        assert cfg.boundary_instrument_id == 42


class TestFillRecord:
    def test_default_values(self):
        fill = core.FillRecord()
        assert fill.timestamp == 0
        assert fill.order_id == 0
        assert fill.instrument_id == 0
        assert fill.fill_price == 0
        assert fill.fill_qty == 0
        assert fill.is_buy is True
        assert fill.fill_price_float == 0.0


# ── Feature System Tests ─────────────────────────────────────────────


class TestFeatureSystem:
    def test_create_all_feature_types(self):
        features = [
            core.create_best_bid_price_feature(instrument_id=1),
            core.create_best_ask_price_feature(instrument_id=1),
            core.create_spread_feature(instrument_id=1),
            core.create_micro_price_feature(instrument_id=1),
            core.create_order_imbalance_feature(instrument_id=1),
            core.create_log_return_feature(instrument_id=1),
            core.create_mid_price_feature(instrument_id=1),
            core.create_cancel_add_ratio_feature(),
            core.create_rolling_volatility_feature(instrument_id=1, window_size=20),
            core.create_micro_depth_feature(instrument_id=1, side=core.BookSide.Bid, depth_index=0),
            core.create_volume_at_price_feature(instrument_id=1),
        ]
        assert len(features) == 11
        for f in features:
            assert isinstance(f, core.IFeature)

    def test_feature_has_feature(self):
        bid = core.create_best_bid_price_feature(instrument_id=1)
        assert bid.has_feature("best_bid_price") is True
        assert bid.has_feature("nonexistent") is False

    def test_feature_manager_register_and_query(self):
        fm = core.create_feature_manager()
        bid = core.create_best_bid_price_feature(instrument_id=1)
        fm.register_feature(bid)
        val = fm.get_value("best_bid_price")
        assert isinstance(val, float)

    def test_feature_manager_multiple_features(self):
        fm = core.create_feature_manager()
        fm.register_feature(core.create_best_bid_price_feature(instrument_id=1))
        fm.register_feature(core.create_best_ask_price_feature(instrument_id=1))
        fm.register_feature(core.create_spread_feature(instrument_id=1))
        fm.register_feature(core.create_micro_price_feature(instrument_id=1))
        fm.register_feature(core.create_order_imbalance_feature(instrument_id=1))

        # All should be queryable
        for name in ["best_bid_price", "best_ask_price", "bid_ask_spread",
                     "micro_price", "order_imbalance"]:
            val = fm.get_value(name)
            assert isinstance(val, float)

    def test_micro_depth_feature(self):
        feat = core.create_micro_depth_feature(
            instrument_id=1, side=core.BookSide.Ask, depth_index=2)
        assert feat.has_feature("micro_depth_price") is True
        assert feat.has_feature("micro_depth_size") is True

    def test_rolling_volatility_feature(self):
        feat = core.create_rolling_volatility_feature(
            instrument_id=1, window_size=50)
        assert feat.has_feature("rolling_volatility") is True


# ── OrderSimulationEnv Tests ─────────────────────────────────────────


class TestOrderSimulationEnv:
    def test_create_env(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(dbn_path="dummy.dbn.zst", instrument_id=1)
        assert env.observation_space.shape == (11,)
        assert env.action_space.n == 7

    def test_reset(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(dbn_path="dummy.dbn.zst", instrument_id=1)
        obs, info = env.reset()
        assert obs.shape == (11,)
        assert obs.dtype.name == "float32"

    def test_step_hold(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(dbn_path="dummy.dbn.zst", instrument_id=1)
        env.reset()
        obs, reward, term, trunc, info = env.step(0)
        assert reward == 0.0
        assert info["position"] == 0

    def test_buy_then_sell(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(dbn_path="dummy.dbn.zst", instrument_id=1)
        env.reset()
        # Buy market
        obs, reward, term, trunc, info = env.step(1)
        assert info["position"] == 1
        # Sell market (close position)
        obs, reward, term, trunc, info = env.step(2)
        assert info["position"] == 0

    def test_max_position_capped(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(
            dbn_path="dummy.dbn.zst", instrument_id=1, max_position=1)
        env.reset()
        env.step(1)  # Buy -> pos=1
        obs, reward, term, trunc, info = env.step(1)  # Buy again -> still 1
        assert info["position"] == 1

    def test_forced_flatten_at_episode_end(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(
            dbn_path="dummy.dbn.zst", instrument_id=1, max_steps=3)
        env.reset()
        env.step(1)  # Buy -> pos=1
        env.step(0)  # Hold
        obs, reward, term, trunc, info = env.step(0)  # Last step -> flatten
        assert trunc is True
        assert info["position"] == 0

    def test_limit_order_and_cancel(self):
        from lob_rl.order_sim_env import OrderSimulationEnv
        env = OrderSimulationEnv(dbn_path="dummy.dbn.zst", instrument_id=1)
        env.reset()
        # Place buy limit
        env.step(3)
        # Place sell limit
        env.step(4)
        # Cancel buy orders
        env.step(5)
        # Cancel sell orders
        obs, reward, term, trunc, info = env.step(6)
        assert info["position"] == 0

    def test_price_scale_constant(self):
        assert core.FIXED_PRICE_SCALE == 1_000_000_000
