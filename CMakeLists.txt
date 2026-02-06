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
    GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ── Core library ───────────────────────────────────────────────────────
add_library(lob_core STATIC
    src/engine/book.cpp
    src/data/synthetic_source.cpp
    src/data/binary_file_source.cpp
    src/env/lob_env.cpp
    src/env/feature_builder.cpp
    src/env/precompute.cpp
)

target_include_directories(lob_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/data
)

# ── Tests ──────────────────────────────────────────────────────────────
enable_testing()

add_executable(lob_tests
    tests/test_source.cpp
    tests/test_book.cpp
    tests/test_book_depth.cpp
    tests/test_env.cpp
    tests/test_binary_file_source.cpp
    tests/test_session.cpp
    tests/test_session_env.cpp
    tests/test_flattening_pnl.cpp
    tests/test_feature_builder.cpp
    tests/test_env_obs44.cpp
    tests/test_reward_calculator.cpp
    tests/test_data_integrity.cpp
    tests/test_precompute.cpp
    tests/test_execution_cost.cpp
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

# ── Python bindings ────────────────────────────────────────────────────
pybind11_add_module(lob_rl_core src/bindings/bindings.cpp)
target_link_libraries(lob_rl_core PRIVATE lob_core)
target_include_directories(lob_rl_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/data
)
