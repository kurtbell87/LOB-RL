#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lob/env.h"
#include "lob/session.h"
#include "lob/reward.h"
#include "lob/precompute.h"
#include "lob/feature_builder.h"
#include "synthetic_source.h"
#include "binary_file_source.h"

namespace py = pybind11;

static RewardMode parse_reward_mode(const std::string& mode_str) {
    if (mode_str.empty() || mode_str == "pnl_delta") return RewardMode::PnLDelta;
    if (mode_str == "pnl_delta_penalized") return RewardMode::PnLDeltaPenalized;
    throw std::invalid_argument("Invalid reward_mode: '" + mode_str +
        "'. Expected 'pnl_delta' or 'pnl_delta_penalized'.");
}

static LOBEnv make_synthetic_env(int steps_per_episode,
                                  const std::string& reward_mode, float lambda_,
                                  bool execution_cost) {
    return LOBEnv(std::make_unique<SyntheticSource>(), steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost);
}

static LOBEnv make_file_env(const std::string& path, int steps_per_episode,
                             const std::string& reward_mode, float lambda_,
                             bool execution_cost) {
    return LOBEnv(std::make_unique<BinaryFileSource>(path), steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost);
}

static LOBEnv make_session_env(const std::string& path, const SessionConfig& cfg,
                                int steps_per_episode,
                                const std::string& reward_mode, float lambda_,
                                bool execution_cost) {
    return LOBEnv(std::make_unique<BinaryFileSource>(path), cfg, steps_per_episode,
                  parse_reward_mode(reward_mode), lambda_, execution_cost);
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
            py::arg("execution_cost") = false)
        // Constructor with file path + steps_per_episode
        .def(py::init(&make_file_env),
            py::arg("file_path"), py::arg("steps_per_episode"),
            py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false)
        // Constructor with file path only (default steps=50)
        .def(py::init([](const std::string& path, const std::string& reward_mode,
                         float lambda_, bool execution_cost) {
            return make_file_env(path, 50, reward_mode, lambda_, execution_cost);
        }), py::arg("file_path"), py::arg("reward_mode") = "",
            py::arg("lambda_") = 0.0f, py::arg("execution_cost") = false)
        // Constructor with steps_per_episode (int) -> SyntheticSource
        .def(py::init(&make_synthetic_env),
            py::arg("steps_per_episode"),
            py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false)
        // Default constructor: SyntheticSource, steps_per_episode=50
        // No positional args — only kwargs
        .def(py::init([](const std::string& reward_mode, float lambda_, bool execution_cost) {
            return make_synthetic_env(50, reward_mode, lambda_, execution_cost);
        }), py::kw_only(), py::arg("reward_mode") = "", py::arg("lambda_") = 0.0f,
            py::arg("execution_cost") = false)
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

    // precompute(path, cfg) -> (obs, mid, spread, num_steps)
    m.def("precompute", [](const std::string& path, const SessionConfig& cfg) {
        PrecomputedDay day = precompute(path, cfg);

        const int N = day.num_steps;
        constexpr int COLS = FeatureBuilder::POSITION;  // 43: obs without position

        // obs: float32 array of shape (N, COLS)
        py::array_t<float> obs({N, COLS});
        if (N > 0) {
            auto buf = obs.mutable_unchecked<2>();
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < COLS; ++c) {
                    buf(r, c) = day.obs[r * COLS + c];
                }
            }
        }

        // mid: float64 array of shape (N,)
        py::array_t<double> mid(N);
        if (N > 0) {
            auto buf = mid.mutable_unchecked<1>();
            for (int i = 0; i < N; ++i) {
                buf(i) = day.mid[i];
            }
        }

        // spread: float64 array of shape (N,)
        py::array_t<double> spread(N);
        if (N > 0) {
            auto buf = spread.mutable_unchecked<1>();
            for (int i = 0; i < N; ++i) {
                buf(i) = day.spread[i];
            }
        }

        return py::make_tuple(obs, mid, spread, N);
    }, py::arg("path"), py::arg("session_config"));
}
