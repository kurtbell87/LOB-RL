#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "lob/env.h"
#include "lob/book.h"
#include "lob/message.h"
#include "lob/source.h"
#include "synthetic_source.h"

namespace py = pybind11;

namespace lob {

// Helper to convert Observation to numpy array
py::array_t<float> observation_to_numpy(const Observation& obs) {
    py::array_t<float> result(obs.data.size());
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    std::copy(obs.data.begin(), obs.data.end(), ptr);
    return result;
}

// Wrapper for StepResult that returns numpy observation
struct PyStepResult {
    py::array_t<float> obs;
    double reward;
    bool done;
    int position;
    double pnl;
    uint64_t timestamp_ns;

    explicit PyStepResult(StepResult&& result)
        : obs(observation_to_numpy(result.obs))
        , reward(result.reward)
        , done(result.done)
        , position(result.position)
        , pnl(result.pnl)
        , timestamp_ns(result.timestamp_ns) {}
};

PYBIND11_MODULE(lob_rl_core, m) {
    m.doc() = "LOB-RL: Limit Order Book Reinforcement Learning Environment";

    // Enums
    py::enum_<Side>(m, "Side")
        .value("Bid", Side::Bid)
        .value("Ask", Side::Ask);

    py::enum_<Action>(m, "Action")
        .value("Add", Action::Add)
        .value("Cancel", Action::Cancel)
        .value("Modify", Action::Modify)
        .value("Trade", Action::Trade)
        .value("Clear", Action::Clear);

    py::enum_<RewardType>(m, "RewardType")
        .value("PnLDelta", RewardType::PnLDelta)
        .value("PnLDeltaPenalized", RewardType::PnLDeltaPenalized);

    // Session struct
    py::class_<Session>(m, "Session")
        .def(py::init<int, int, int, int>(),
             py::arg("start_hour_utc"), py::arg("start_minute_utc"),
             py::arg("end_hour_utc"), py::arg("end_minute_utc"))
        .def_readwrite("start_hour_utc", &Session::start_hour_utc)
        .def_readwrite("start_minute_utc", &Session::start_minute_utc)
        .def_readwrite("end_hour_utc", &Session::end_hour_utc)
        .def_readwrite("end_minute_utc", &Session::end_minute_utc);

    // Predefined sessions
    m.attr("US_RTH_EST") = sessions::US_RTH_EST;
    m.attr("US_RTH_EDT") = sessions::US_RTH_EDT;

    // EnvConfig struct
    py::class_<EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("data_path", &EnvConfig::data_path)
        .def_readwrite("book_depth", &EnvConfig::book_depth)
        .def_readwrite("trades_per_step", &EnvConfig::trades_per_step)
        .def_readwrite("reward_type", &EnvConfig::reward_type)
        .def_readwrite("inventory_penalty", &EnvConfig::inventory_penalty)
        .def_readwrite("session", &EnvConfig::session);

    // PyStepResult (Python-friendly version with numpy array)
    py::class_<PyStepResult>(m, "StepResult")
        .def_readonly("obs", &PyStepResult::obs)
        .def_readonly("reward", &PyStepResult::reward)
        .def_readonly("done", &PyStepResult::done)
        .def_readonly("position", &PyStepResult::position)
        .def_readonly("pnl", &PyStepResult::pnl)
        .def_readonly("timestamp_ns", &PyStepResult::timestamp_ns);

    // Level struct
    py::class_<Level>(m, "Level")
        .def_readonly("price", &Level::price)
        .def_readonly("quantity", &Level::quantity);

    // Book class
    py::class_<Book>(m, "Book")
        .def(py::init<>())
        .def("clear", &Book::clear)
        .def("bid", &Book::bid, py::arg("depth") = 0)
        .def("ask", &Book::ask, py::arg("depth") = 0)
        .def("bid_depth", &Book::bid_depth)
        .def("ask_depth", &Book::ask_depth)
        .def("mid_price", &Book::mid_price)
        .def("spread", &Book::spread)
        .def("imbalance", &Book::imbalance, py::arg("levels") = 1);

    // SyntheticSource
    py::class_<SyntheticSource>(m, "SyntheticSource")
        .def(py::init<uint32_t, uint64_t>(),
             py::arg("seed") = 42, py::arg("num_messages") = 1000)
        .def("has_next", &SyntheticSource::has_next)
        .def("reset", &SyntheticSource::reset)
        .def("message_count", &SyntheticSource::message_count);

    // LOBEnv class
    py::class_<LOBEnv>(m, "LOBEnv")
        .def(py::init([](EnvConfig config, uint32_t seed, uint64_t num_messages) {
            auto source = std::make_unique<SyntheticSource>(seed, num_messages);
            return std::make_unique<LOBEnv>(std::move(config), std::move(source));
        }), py::arg("config") = EnvConfig{}, py::arg("seed") = 42, py::arg("num_messages") = 1000)
        .def("reset", [](LOBEnv& env) {
            return PyStepResult(env.reset());
        })
        .def("step", [](LOBEnv& env, int action) {
            return PyStepResult(env.step(action));
        }, py::arg("action"))
        .def("observation_size", &LOBEnv::observation_size)
        .def_property_readonly_static("action_size", [](py::object) { return LOBEnv::action_size(); })
        .def_property_readonly("config", &LOBEnv::config, py::return_value_policy::reference);
}

}  // namespace lob
