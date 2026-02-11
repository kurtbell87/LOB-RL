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

        // Features (float32, 2D)
        int feat_cols = N_FEATURES * lookback;
        result["features"] = to_numpy_2d(day.features, day.n_usable, feat_cols);

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
}
