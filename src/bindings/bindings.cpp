#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lob/env.h"
#include "lob/session.h"
#include "lob/reward.h"
#include "lob/precompute.h"
#include "lob/feature_builder.h"
#include "lob/barrier/barrier_precompute.h"
#include "synthetic_source.h"
#include "dbn_file_source.h"
// Constellation
#include "orders/OrdersEngine.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/orders/IOrderEvents.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"
#include "replay/BatchBacktestEngine.hpp"
#include "replay/BatchAggregator.hpp"
#include "features/FeatureFactory.hpp"
#include "features/FeatureManager.hpp"
#include "interfaces/features/IFeature.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include "databento/constants.hpp"

namespace py = pybind11;

// Copy a std::vector<T> into a 1-D numpy array of shape (N,) with output dtype Out.
template<typename Out, typename T>
static py::array_t<Out> vec_to_numpy(const std::vector<T>& vec, int N) {
    py::array_t<Out> arr(N);
    if (N > 0) {
        auto buf = arr.template mutable_unchecked<1>();
        for (int i = 0; i < N; ++i) buf(i) = static_cast<Out>(vec[i]);
    }
    return arr;
}

// Convenience: same-type overload for double vectors.
static py::array_t<double> to_numpy_1d(const std::vector<double>& vec, int N) {
    return vec_to_numpy<double>(vec, N);
}

// Copy a flat std::vector<float> into a 2-D numpy array of shape (rows, cols).
static py::array_t<float> to_numpy_2d(const std::vector<float>& vec, int rows, int cols) {
    py::array_t<float> arr({rows, cols});
    if (rows > 0) {
        auto buf = arr.mutable_unchecked<2>();
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                buf(r, c) = vec[r * cols + c];
    }
    return arr;
}

static RewardMode parse_reward_mode(const std::string& mode_str) {
    if (mode_str.empty() || mode_str == "pnl_delta") return RewardMode::PnLDelta;
    if (mode_str == "pnl_delta_penalized") return RewardMode::PnLDeltaPenalized;
    throw std::invalid_argument("Invalid reward_mode: '" + mode_str +
        "'. Expected 'pnl_delta' or 'pnl_delta_penalized'.");
}

static LOBEnv make_synthetic_env(int steps_per_episode,
                                  const std::string& reward_mode, float lambda_,
                                  bool execution_cost, float participation_bonus) {
    return LOBEnv(std::make_unique<SyntheticSource>(), steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost,
                  participation_bonus);
}

static LOBEnv make_file_env(const std::string& path, int steps_per_episode,
                             const std::string& reward_mode, float lambda_,
                             bool execution_cost, float participation_bonus,
                             uint32_t instrument_id) {
    return LOBEnv(std::make_unique<DbnFileSource>(path, instrument_id), steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost,
                  participation_bonus);
}

static LOBEnv make_session_env(const std::string& path, const SessionConfig& cfg,
                                int steps_per_episode,
                                const std::string& reward_mode, float lambda_,
                                bool execution_cost, float participation_bonus,
                                uint32_t instrument_id) {
    return LOBEnv(std::make_unique<DbnFileSource>(path, instrument_id), cfg, steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost,
                  participation_bonus);
}

PYBIND11_MODULE(lob_rl_core, m) {
    m.doc() = "LOB-RL core C++ bindings";

    // SessionConfig
    py::class_<SessionConfig>(m, "SessionConfig")
        .def(py::init<>())
        .def_readwrite("rth_open_ns", &SessionConfig::rth_open_ns)
        .def_readwrite("rth_close_ns", &SessionConfig::rth_close_ns)
        .def_readwrite("warmup_messages", &SessionConfig::warmup_messages)
        .def_static("default_rth", &SessionConfig::default_rth);

    // LOBEnv
    py::class_<LOBEnv>(m, "LOBEnv")
        // Constructor with file path + SessionConfig + steps_per_episode
        // (must be before file_path overloads to avoid ambiguity)
        .def(py::init(&make_session_env),
            py::arg("file_path"), py::arg("session_config"), py::arg("steps_per_episode"),
            py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false, py::arg("participation_bonus") = 0.0f,
            py::arg("instrument_id") = 0u)
        // Constructor with file path + steps_per_episode
        .def(py::init(&make_file_env),
            py::arg("file_path"), py::arg("steps_per_episode"),
            py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false, py::arg("participation_bonus") = 0.0f,
            py::arg("instrument_id") = 0u)
        // Constructor with file path only (default steps=50)
        .def(py::init([](const std::string& path, const std::string& reward_mode,
                         float lambda_, bool execution_cost, float participation_bonus,
                         uint32_t instrument_id) {
            return make_file_env(path, 50, reward_mode, lambda_, execution_cost,
                                 participation_bonus, instrument_id);
        }), py::arg("file_path"), py::arg("reward_mode") = "",
            py::arg("lambda_") = 0.0f, py::arg("execution_cost") = false,
            py::arg("participation_bonus") = 0.0f, py::arg("instrument_id") = 0u)
        // Constructor with steps_per_episode (int) -> SyntheticSource
        .def(py::init(&make_synthetic_env),
            py::arg("steps_per_episode"),
            py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false, py::arg("participation_bonus") = 0.0f)
        // Default constructor: SyntheticSource, steps_per_episode=50
        // No positional args — only kwargs
        .def(py::init([](const std::string& reward_mode, float lambda_, bool execution_cost,
                         float participation_bonus) {
            return make_synthetic_env(50, reward_mode, lambda_, execution_cost, participation_bonus);
        }), py::kw_only(), py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false, py::arg("participation_bonus") = 0.0f)
        // steps_per_episode read-only property
        .def_property_readonly("steps_per_episode", &LOBEnv::steps_per_episode)
        .def("reset", [](LOBEnv& env) {
            StepResult r = env.reset();
            return r.obs;
        })
        .def("step", [](LOBEnv& env, int action) {
            StepResult r = env.step(action);
            return py::make_tuple(r.obs, static_cast<double>(r.reward), r.done);
        });

    // precompute(path, cfg, instrument_id=0) -> (obs, mid, spread, num_steps)
    m.def("precompute", [](const std::string& path, const SessionConfig& cfg,
                            uint32_t instrument_id) {
        PrecomputedDay day = precompute(path, cfg, instrument_id);

        const int N = day.num_steps;
        constexpr int COLS = FeatureBuilder::POSITION;  // 43: obs without position

        auto obs    = to_numpy_2d(day.obs, N, COLS);
        auto mid    = to_numpy_1d(day.mid, N);
        auto spread = to_numpy_1d(day.spread, N);

        return py::make_tuple(obs, mid, spread, N);
    }, py::arg("path"), py::arg("session_config"), py::arg("instrument_id") = 0u);

    // barrier_precompute(path, instrument_id, ...) -> dict or None
    m.def("barrier_precompute", [](const std::string& path,
                                    uint32_t instrument_id,
                                    int bar_size, int lookback,
                                    int a, int b, int t_max) -> py::object {
        BarrierPrecomputedDay day = barrier_precompute(
            path, instrument_id, bar_size, lookback, a, b, t_max);

        // Return None if insufficient data
        if (day.n_usable == 0 && day.n_bars < lookback + 1) {
            return py::none();
        }

        py::dict result;
        int n = day.n_bars;

        // Bar OHLCV arrays (float64)
        result["bar_open"] = to_numpy_1d(day.bar_open, n);
        result["bar_high"] = to_numpy_1d(day.bar_high, n);
        result["bar_low"] = to_numpy_1d(day.bar_low, n);
        result["bar_close"] = to_numpy_1d(day.bar_close, n);
        result["bar_vwap"] = to_numpy_1d(day.bar_vwap, n);

        result["bar_volume"] = vec_to_numpy<int32_t>(day.bar_volume, n);
        result["bar_t_start"] = vec_to_numpy<int64_t>(day.bar_t_start, n);
        result["bar_t_end"] = vec_to_numpy<int64_t>(day.bar_t_end, n);

        int nt = static_cast<int>(day.trade_prices.size());
        result["trade_prices"] = to_numpy_1d(day.trade_prices, nt);
        result["trade_sizes"] = vec_to_numpy<int32_t>(day.trade_sizes, nt);

        int no = static_cast<int>(day.bar_trade_offsets.size());
        result["bar_trade_offsets"] = vec_to_numpy<int64_t>(day.bar_trade_offsets, no);

        result["label_values"] = vec_to_numpy<int8_t>(day.label_values, n);
        result["label_tau"] = vec_to_numpy<int32_t>(day.label_tau, n);
        result["label_resolution_bar"] = vec_to_numpy<int32_t>(day.label_resolution_bar, n);

        // Short-direction labels
        result["short_label_values"] = vec_to_numpy<int8_t>(day.short_label_values, n);
        result["short_label_tau"] = vec_to_numpy<int32_t>(day.short_label_tau, n);
        result["short_label_resolution_bar"] = vec_to_numpy<int32_t>(day.short_label_resolution_bar, n);

        // Features (float32, 2D)
        int feat_cols = N_FEATURES * lookback;
        result["features"] = to_numpy_2d(day.features, day.n_usable, feat_cols);

        // Per-bar normalized features (float32, 2D)
        result["bar_features"] = to_numpy_2d(day.bar_features, day.n_trimmed, N_FEATURES);
        result["n_trimmed"] = day.n_trimmed;

        // Scalar metadata
        result["bar_size"] = bar_size;
        result["lookback"] = lookback;
        result["a"] = a;
        result["b"] = b;
        result["t_max"] = t_max;
        result["n_bars"] = day.n_bars;
        result["n_usable"] = day.n_usable;
        result["n_features"] = day.n_features;

        return result;
    }, py::arg("path"), py::arg("instrument_id"),
       py::arg("bar_size") = 500, py::arg("lookback") = 10,
       py::arg("a") = 20, py::arg("b") = 10, py::arg("t_max") = 40);

    // ── Constellation Order Simulation Bindings ──────────────────────────
    namespace cst_orders = constellation::interfaces::orders;

    // Enums
    py::enum_<cst_orders::OrderType>(m, "OrderType")
        .value("Market", cst_orders::OrderType::Market)
        .value("Limit", cst_orders::OrderType::Limit)
        .value("Stop", cst_orders::OrderType::Stop)
        .value("StopLimit", cst_orders::OrderType::StopLimit);

    py::enum_<cst_orders::OrderSide>(m, "OrderSide")
        .value("Buy", cst_orders::OrderSide::Buy)
        .value("Sell", cst_orders::OrderSide::Sell);

    py::enum_<cst_orders::OrderStatus>(m, "OrderStatus")
        .value("New", cst_orders::OrderStatus::New)
        .value("PartiallyFilled", cst_orders::OrderStatus::PartiallyFilled)
        .value("Filled", cst_orders::OrderStatus::Filled)
        .value("Canceled", cst_orders::OrderStatus::Canceled)
        .value("Rejected", cst_orders::OrderStatus::Rejected)
        .value("Expired", cst_orders::OrderStatus::Expired)
        .value("Unknown", cst_orders::OrderStatus::Unknown);

    py::enum_<cst_orders::TimeInForce>(m, "TimeInForce")
        .value("Day", cst_orders::TimeInForce::Day)
        .value("GTC", cst_orders::TimeInForce::GTC)
        .value("IOC", cst_orders::TimeInForce::IOC)
        .value("FOK", cst_orders::TimeInForce::FOK);

    // OrderSpec
    py::class_<cst_orders::OrderSpec>(m, "OrderSpec")
        .def(py::init<>())
        .def_readwrite("instrument_id", &cst_orders::OrderSpec::instrument_id)
        .def_readwrite("type", &cst_orders::OrderSpec::type)
        .def_readwrite("side", &cst_orders::OrderSpec::side)
        .def_readwrite("quantity", &cst_orders::OrderSpec::quantity)
        .def_readwrite("limit_price", &cst_orders::OrderSpec::limit_price)
        .def_readwrite("stop_price", &cst_orders::OrderSpec::stop_price)
        .def_readwrite("tif", &cst_orders::OrderSpec::tif)
        .def_readwrite("strategy_tag", &cst_orders::OrderSpec::strategy_tag);

    // OrderUpdate
    py::class_<cst_orders::OrderUpdate>(m, "OrderUpdate")
        .def(py::init<>())
        .def_readwrite("new_limit_price", &cst_orders::OrderUpdate::new_limit_price)
        .def_readwrite("new_quantity", &cst_orders::OrderUpdate::new_quantity);

    // OrderInfo
    py::class_<cst_orders::OrderInfo>(m, "OrderInfo")
        .def(py::init<>())
        .def_readonly("order_id", &cst_orders::OrderInfo::order_id)
        .def_readonly("instrument_id", &cst_orders::OrderInfo::instrument_id)
        .def_readonly("type", &cst_orders::OrderInfo::type)
        .def_readonly("side", &cst_orders::OrderInfo::side)
        .def_readonly("original_quantity", &cst_orders::OrderInfo::original_quantity)
        .def_readonly("filled_quantity", &cst_orders::OrderInfo::filled_quantity)
        .def_readonly("avg_fill_price", &cst_orders::OrderInfo::avg_fill_price)
        .def_readonly("limit_price", &cst_orders::OrderInfo::limit_price)
        .def_readonly("status", &cst_orders::OrderInfo::status);

    // OrdersEngine
    py::class_<constellation::orders::OrdersEngine>(m, "OrdersEngine")
        .def(py::init<>())
        .def("place_order", &constellation::orders::OrdersEngine::PlaceOrder,
             py::arg("order_spec"))
        .def("modify_order", &constellation::orders::OrdersEngine::ModifyOrder,
             py::arg("order_id"), py::arg("update"))
        .def("cancel_order", &constellation::orders::OrdersEngine::CancelOrder,
             py::arg("order_id"))
        .def("get_order_status", &constellation::orders::OrdersEngine::GetOrderStatus,
             py::arg("order_id"))
        .def("get_order_info", &constellation::orders::OrdersEngine::GetOrderInfo,
             py::arg("order_id"))
        .def("list_open_orders", &constellation::orders::OrdersEngine::ListOpenOrders,
             py::arg("instrument_id"));

    // Price scale constant
    m.attr("FIXED_PRICE_SCALE") = databento::kFixedPriceScale;

    // ── Constellation Replay & Market Bindings ──────────────────────────
    namespace cst_replay = constellation::applications::replay;
    namespace cst_ob = constellation::modules::orderbook;
    using cst_replay::BatchAggregatorConfig;
    using cst_replay::BatchAggregatorStats;

    // BatchAggregatorConfig
    py::class_<BatchAggregatorConfig>(m, "BatchAggregatorConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &BatchAggregatorConfig::batch_size)
        .def_readwrite("enable_logging", &BatchAggregatorConfig::enable_logging)
        .def_readwrite("release_gil_during_aggregation", &BatchAggregatorConfig::release_gil_during_aggregation)
        .def_readwrite("enable_instrument_boundary", &BatchAggregatorConfig::enable_instrument_boundary)
        .def_readwrite("boundary_instrument_id", &BatchAggregatorConfig::boundary_instrument_id)
        .def_readwrite("boundary_instrument_trades", &BatchAggregatorConfig::boundary_instrument_trades)
        .def_readwrite("enable_event_count_boundary", &BatchAggregatorConfig::enable_event_count_boundary)
        .def_readwrite("boundary_event_type", &BatchAggregatorConfig::boundary_event_type)
        .def_readwrite("boundary_event_count", &BatchAggregatorConfig::boundary_event_count);

    // BatchAggregatorStats
    py::class_<BatchAggregatorStats>(m, "BatchAggregatorStats")
        .def(py::init<>())
        .def_property_readonly("total_records", [](const BatchAggregatorStats& s) {
            return s.total_records.load();
        })
        .def_property_readonly("total_mbo_messages", [](const BatchAggregatorStats& s) {
            return s.total_mbo_messages.load();
        })
        .def_property_readonly("total_microseconds", [](const BatchAggregatorStats& s) {
            return s.total_microseconds.load();
        });

    // FillRecord
    py::class_<cst_replay::FillRecord>(m, "FillRecord")
        .def(py::init<>())
        .def_readonly("timestamp", &cst_replay::FillRecord::timestamp)
        .def_readonly("order_id", &cst_replay::FillRecord::order_id)
        .def_readonly("instrument_id", &cst_replay::FillRecord::instrument_id)
        .def_readonly("fill_price", &cst_replay::FillRecord::fill_price)
        .def_readonly("fill_qty", &cst_replay::FillRecord::fill_qty)
        .def_readonly("is_buy", &cst_replay::FillRecord::is_buy)
        .def_property_readonly("fill_price_float", [](const cst_replay::FillRecord& f) {
            return static_cast<double>(f.fill_price) / databento::kFixedPriceScale;
        });

    // MarketBook
    py::class_<cst_ob::MarketBook, std::shared_ptr<cst_ob::MarketBook>>(m, "MarketBook")
        .def(py::init<>())
        .def("instrument_count", &cst_ob::MarketBook::InstrumentCount)
        .def("get_instrument_ids", &cst_ob::MarketBook::GetInstrumentIds)
        .def("best_bid_price", [](const cst_ob::MarketBook& mb, uint32_t instrument_id) -> py::object {
            auto p = mb.BestBidPrice(instrument_id);
            if (!p) return py::none();
            return py::cast(static_cast<double>(*p) / databento::kFixedPriceScale);
        }, py::arg("instrument_id"))
        .def("best_ask_price", [](const cst_ob::MarketBook& mb, uint32_t instrument_id) -> py::object {
            auto p = mb.BestAskPrice(instrument_id);
            if (!p) return py::none();
            return py::cast(static_cast<double>(*p) / databento::kFixedPriceScale);
        }, py::arg("instrument_id"))
        .def("get_global_add_count", &cst_ob::MarketBook::GetGlobalAddCount)
        .def("get_global_cancel_count", &cst_ob::MarketBook::GetGlobalCancelCount)
        .def("get_global_modify_count", &cst_ob::MarketBook::GetGlobalModifyCount)
        .def("get_global_trade_count", &cst_ob::MarketBook::GetGlobalTradeCount)
        .def("get_global_total_event_count", &cst_ob::MarketBook::GetGlobalTotalEventCount);

    // BatchBacktestEngine
    py::class_<cst_replay::BatchBacktestEngine>(m, "BatchBacktestEngine")
        .def(py::init<>())
        .def("set_aggregator_config", &cst_replay::BatchBacktestEngine::SetAggregatorConfig,
             py::arg("config"))
        .def("process_files", &cst_replay::BatchBacktestEngine::ProcessFiles,
             py::arg("dbn_files"))
        .def("process_single_file", &cst_replay::BatchBacktestEngine::ProcessSingleFile,
             py::arg("dbn_file"))
        .def("get_fills", &cst_replay::BatchBacktestEngine::GetFills)
        .def("reset_stats", &cst_replay::BatchBacktestEngine::ResetStats)
        .def("get_market_view", [](const cst_replay::BatchBacktestEngine& engine) -> py::object {
            auto mv = engine.GetMarketView();
            if (!mv) return py::none();
            // Return a dict with instrument-level market state
            py::dict result;
            auto ids = mv->GetInstrumentIds();
            result["instrument_count"] = mv->InstrumentCount();
            result["instrument_ids"] = ids;

            // Per-instrument BBO
            py::dict bbo;
            for (auto id : ids) {
                auto bid = mv->GetBestBid(id);
                auto ask = mv->GetBestAsk(id);
                py::dict inst;
                inst["best_bid_price"] = bid ? py::cast(
                    static_cast<double>(bid->price) / databento::kFixedPriceScale)
                    : py::none();
                inst["best_bid_qty"] = bid ? py::cast(bid->total_quantity) : py::none();
                inst["best_ask_price"] = ask ? py::cast(
                    static_cast<double>(ask->price) / databento::kFixedPriceScale)
                    : py::none();
                inst["best_ask_qty"] = ask ? py::cast(ask->total_quantity) : py::none();
                bbo[py::cast(id)] = inst;
            }
            result["instruments"] = bbo;

            // Global counters
            result["global_add_count"] = mv->GetGlobalAddCount();
            result["global_cancel_count"] = mv->GetGlobalCancelCount();
            result["global_modify_count"] = mv->GetGlobalModifyCount();
            result["global_trade_count"] = mv->GetGlobalTradeCount();
            result["global_total_event_count"] = mv->GetGlobalTotalEventCount();

            return result;
        });

    // ── Constellation Feature System Bindings ───────────────────────────
    namespace cst_feat = constellation::modules::features;
    namespace cst_if_feat = constellation::interfaces::features;
    namespace cst_if_ob = constellation::interfaces::orderbook;

    // BookSide enum (used by MicroDepthFeature, VolumeAtPriceFeature)
    py::enum_<cst_if_ob::BookSide>(m, "BookSide")
        .value("Bid", cst_if_ob::BookSide::Bid)
        .value("Ask", cst_if_ob::BookSide::Ask);

    // IFeature (abstract base — exposed for shared_ptr holder)
    py::class_<cst_if_feat::IFeature, std::shared_ptr<cst_if_feat::IFeature>>(m, "IFeature")
        .def("get_value", &cst_if_feat::IFeature::GetValue, py::arg("name"))
        .def("has_feature", &cst_if_feat::IFeature::HasFeature, py::arg("name"));

    // FeatureManager (single-instrument)
    py::class_<cst_if_feat::IFeatureManager, std::shared_ptr<cst_if_feat::IFeatureManager>>(m, "IFeatureManager")
        .def("register_feature", &cst_if_feat::IFeatureManager::Register, py::arg("feature"))
        .def("get_value", &cst_if_feat::IFeatureManager::GetValue, py::arg("feature_name"));

    // Factory functions — create features
    m.def("create_feature_manager", []() {
        return cst_feat::CreateFeatureManager(nullptr);
    });

    m.def("create_best_bid_price_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateBestBidPriceFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_best_ask_price_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateBestAskPriceFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_spread_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateSpreadFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_micro_price_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateMicroPriceFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_order_imbalance_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateOrderImbalanceFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_log_return_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateLogReturnFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_mid_price_feature", [](uint32_t instrument_id) {
        return cst_feat::CreateMidPriceFeature({instrument_id});
    }, py::arg("instrument_id") = 0u);

    m.def("create_cancel_add_ratio_feature", &cst_feat::CreateCancelAddRatioFeature);

    m.def("create_rolling_volatility_feature", [](uint32_t instrument_id, size_t window_size) {
        return cst_feat::CreateRollingVolatilityFeature({instrument_id, window_size});
    }, py::arg("instrument_id") = 0u, py::arg("window_size") = 1u);

    m.def("create_micro_depth_feature", [](uint32_t instrument_id,
                                            cst_if_ob::BookSide side, size_t depth_index) {
        return cst_feat::CreateMicroDepthFeature({instrument_id, side, depth_index});
    }, py::arg("instrument_id") = 0u,
       py::arg("side") = cst_if_ob::BookSide::Bid,
       py::arg("depth_index") = 0u);

    m.def("create_volume_at_price_feature", [](uint32_t instrument_id,
                                                cst_if_ob::BookSide side, int64_t price) {
        return cst_feat::CreateVolumeAtPriceFeature({instrument_id, side, price});
    }, py::arg("instrument_id") = 0u,
       py::arg("side") = cst_if_ob::BookSide::Bid,
       py::arg("price") = 0);
}
