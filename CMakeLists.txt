cmake_minimum_required(VERSION 3.20)
project(lob_rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

# Compiler warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Debug builds: sanitizers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# Release builds: optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

# Include paths
include_directories(${CMAKE_SOURCE_DIR}/include)

# Core library
add_library(lob_core STATIC
    src/engine/book.cpp
    src/data/synthetic_source.cpp
    src/env/lob_env.cpp
    src/env/feature_builder.cpp
    src/env/reward_calculator.cpp
)

# Internal headers for implementation files
target_include_directories(lob_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src/data
    ${CMAKE_SOURCE_DIR}/src/env
)

# Tests
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(lob_tests
        tests/test_source.cpp
        tests/test_book.cpp
        tests/test_env.cpp
    )

    target_link_libraries(lob_tests PRIVATE
        lob_core
        GTest::gtest_main
    )

    target_include_directories(lob_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src/data
        ${CMAKE_SOURCE_DIR}/src/env
    )

    include(GoogleTest)
    gtest_discover_tests(lob_tests)
endif()

# Python bindings
if(BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    include(FetchContent)
    set(PYBIND11_FINDPYTHON ON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(lob_rl_core
        src/bindings/bindings.cpp
    )

    target_link_libraries(lob_rl_core PRIVATE lob_core)

    target_include_directories(lob_rl_core PRIVATE
        ${CMAKE_SOURCE_DIR}/src/data
        ${CMAKE_SOURCE_DIR}/src/env
    )
endif()
