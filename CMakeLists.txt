cmake_minimum_required(VERSION 3.20)
project(lob_rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Debug sanitizers (applied to test binary only, not pybind11 module)
# ASAN cannot be loaded via dlopen into Python, so we apply it per-target below.

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -march=native -DNDEBUG)
endif()

# ── Dependencies via FetchContent ──────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.16.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
    databento
    GIT_REPOSITORY https://github.com/databento/databento-cpp.git
    GIT_TAG        v0.47.0
)
FetchContent_MakeAvailable(databento)

# ── Core library ───────────────────────────────────────────────────────
add_library(lob_core STATIC
    src/engine/book.cpp
    src/data/synthetic_source.cpp
    src/data/dbn_file_source.cpp
    src/data/dbn_message_map.cpp
    src/env/lob_env.cpp
    src/env/feature_builder.cpp
    src/env/precompute.cpp
    src/barrier/bar_builder.cpp
    src/barrier/feature_compute.cpp
    src/barrier/barrier_label.cpp
    src/barrier/barrier_precompute.cpp
)

target_include_directories(lob_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/data
)

target_link_libraries(lob_core PUBLIC databento::databento)

# ── Tests ──────────────────────────────────────────────────────────────
enable_testing()

add_executable(lob_tests
    tests/test_source.cpp
    tests/test_book.cpp
    tests/test_book_depth.cpp
    tests/test_env.cpp
    tests/test_session.cpp
    tests/test_session_env.cpp
    tests/test_flattening_pnl.cpp
    tests/test_feature_builder.cpp
    tests/test_env_obs44.cpp
    tests/test_reward_calculator.cpp
    tests/test_execution_cost.cpp
    tests/test_participation_bonus.cpp
    tests/test_dbn_message_map.cpp
    tests/test_dbn_file_source.cpp
    tests/test_barrier_bar_builder.cpp
    tests/test_barrier_features.cpp
    tests/test_barrier_labels.cpp
    tests/test_barrier_precompute.cpp
    tests/test_per_bar_features.cpp
)

target_link_libraries(lob_tests PRIVATE lob_core GTest::gtest_main)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(lob_tests PRIVATE -fsanitize=address,undefined)
    target_link_options(lob_tests PRIVATE -fsanitize=address,undefined)
endif()
target_include_directories(lob_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/data
    ${CMAKE_SOURCE_DIR}/tests
)

include(GoogleTest)
gtest_discover_tests(lob_tests)

# ── Test fixture generator ────────────────────────────────────────────
add_executable(gen_fixture tools/gen_fixture.cpp)
target_link_libraries(gen_fixture PRIVATE databento::databento)
target_compile_definitions(gen_fixture PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# ── Python bindings ────────────────────────────────────────────────────
pybind11_add_module(lob_rl_core src/bindings/bindings.cpp)
target_link_libraries(lob_rl_core PRIVATE lob_core)
target_include_directories(lob_rl_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/data
)
