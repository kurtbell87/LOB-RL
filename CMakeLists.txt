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

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.3.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

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
    ${CMAKE_SOURCE_DIR}/src  # for constellation/adapters/
)

target_link_libraries(lob_core PUBLIC constellation_core)

# ── Constellation core library ────────────────────────────────────────
add_library(constellation_core STATIC
    # Interfaces
    src/constellation/interfaces/src/batch/RecordBatch.cpp
    # Modules: orderbook
    src/constellation/modules/orderbook/src/LimitOrderBook.cpp
    src/constellation/modules/orderbook/src/LimitOrderBookSnapshot.cpp
    src/constellation/modules/orderbook/src/MarketBook.cpp
    src/constellation/modules/orderbook/src/MarketBookSnapshot.cpp
    src/constellation/modules/orderbook/src/OrderBookFactory.cpp
    src/constellation/modules/orderbook/src/commands/OrderCommands.cpp
    # Modules: features
    src/constellation/modules/features/src/FeatureFactory.cpp
    src/constellation/modules/features/src/FeatureManager.cpp
    src/constellation/modules/features/src/FeatureRegistry.cpp
    src/constellation/modules/features/src/MultiInstrumentFeatureManager.cpp
    src/constellation/modules/features/src/primitives/BestAskPriceFeature.cpp
    src/constellation/modules/features/src/primitives/BestBidPriceFeature.cpp
    src/constellation/modules/features/src/primitives/LogReturnFeature.cpp
    src/constellation/modules/features/src/primitives/MicroDepthFeature.cpp
    src/constellation/modules/features/src/primitives/MicroPriceFeature.cpp
    src/constellation/modules/features/src/primitives/OrderImbalanceFeature.cpp
    src/constellation/modules/features/src/primitives/SpreadFeature.cpp
    src/constellation/modules/features/src/primitives/VolumeAtPriceFeature.cpp
    src/constellation/modules/features/src/derived/CancelAddRatioFeature.cpp
    src/constellation/modules/features/src/derived/MidPriceFeature.cpp
    src/constellation/modules/features/src/derived/RollingVolatilityFeature.cpp
    # Modules: market_data
    src/constellation/modules/market_data/src/DataBentoMboFeed.cpp
    src/constellation/modules/market_data/src/DbnFileFeed.cpp
    src/constellation/modules/market_data/src/MarketDataFactory.cpp
    # Modules: orders
    src/constellation/modules/orders/src/OrdersEngine.cpp
    src/constellation/modules/orders/src/OrdersFactory.cpp
    # Modules: strategy
    src/constellation/modules/strategy/src/SampleBatchStrategy.cpp
    src/constellation/modules/strategy/src/StrategyFactory.cpp
    # Applications: replay
    src/constellation/applications/replay/src/BatchAggregator.cpp
    src/constellation/applications/replay/src/BatchBacktestEngine.cpp
    src/constellation/applications/replay/src/buffers/BatchBuffer.cpp
    src/constellation/applications/replay/src/buffers/InstrumentBuffer.cpp
    # Applications: orchestrator
    src/constellation/applications/orchestrator/src/Orchestrator.cpp
    src/constellation/applications/orchestrator/src/SpscRingBuffer.cpp
    # Applications: engine
    src/constellation/applications/engine/src/Engine.cpp
)

target_include_directories(constellation_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/constellation/interfaces/include
    ${CMAKE_SOURCE_DIR}/src/constellation/modules/orderbook/include
    ${CMAKE_SOURCE_DIR}/src/constellation/modules/features/include
    ${CMAKE_SOURCE_DIR}/src/constellation/modules/market_data/include
    ${CMAKE_SOURCE_DIR}/src/constellation/modules/orders/include
    ${CMAKE_SOURCE_DIR}/src/constellation/modules/strategy/include
    ${CMAKE_SOURCE_DIR}/src/constellation/applications/replay/include
    ${CMAKE_SOURCE_DIR}/src/constellation/applications/orchestrator/include
    ${CMAKE_SOURCE_DIR}/src/constellation/applications/engine/include
)

target_link_libraries(constellation_core PUBLIC databento::databento)

# Constellation tests (Catch2-based, separate from lob_tests)
add_executable(constellation_tests
    # Interfaces tests
    src/constellation/interfaces/tests/TestBatchAggregator.cpp
    # Orderbook tests
    src/constellation/modules/orderbook/tests/TestOrderBook.cpp
    src/constellation/modules/orderbook/tests/TestOrderBookAdvanced.cpp
    src/constellation/modules/orderbook/tests/TestOrderBookExtended.cpp
    src/constellation/modules/orderbook/tests/TestOrderBookSnapshot.cpp
    src/constellation/modules/orderbook/tests/TestOrderBookConcurrency.cpp
    src/constellation/modules/orderbook/tests/TestMarketBookDepthQueries.cpp
    # Features tests
    src/constellation/modules/features/tests/TestFeatures.cpp
    src/constellation/modules/features/tests/TestExtendedFeatures.cpp
    src/constellation/modules/features/tests/TestFeatureEdgeCases.cpp
    src/constellation/modules/features/tests/TestFeaturesAdvanced.cpp
    src/constellation/modules/features/tests/TestMultiInstrumentFeatures.cpp
    src/constellation/modules/features/tests/TestMicroDepthFeatureFix.cpp
    # Market data tests
    src/constellation/modules/market_data/tests/TestMarketDataIntegration.cpp
    src/constellation/modules/market_data/tests/TestMarketDataAdvanced.cpp
    src/constellation/modules/market_data/tests/TestMarketDataEdgeCases.cpp
    # Orders tests
    src/constellation/modules/orders/tests/TestOrders.cpp
    src/constellation/modules/orders/tests/TestOrdersPartialFill.cpp
    # Strategy tests
    src/constellation/modules/strategy/tests/TestStrategy.cpp
    # Engine tests
    src/constellation/applications/engine/tests/TestEngine.cpp
    src/constellation/applications/engine/tests/TestEngineIntegration.cpp
    src/constellation/applications/engine/tests/TestEngineStress.cpp
    # Replay tests
    src/constellation/applications/replay/tests/TestBatchAggregatorTradeBoundary.cpp
)

target_link_libraries(constellation_tests PRIVATE
    constellation_core
    Catch2::Catch2WithMain
)

target_compile_definitions(constellation_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/src/constellation/test_data"
)

include(Catch)
catch_discover_tests(constellation_tests)

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

# ── LOB Validation Test (Databento reference comparison) ──────────────
add_executable(lob_validation_tests tests/test_lob_vs_databento_reference.cpp)
target_link_libraries(lob_validation_tests PRIVATE lob_core GTest::gtest_main)
target_include_directories(lob_validation_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/data
)
# Run from project root so data/ paths resolve
set_target_properties(lob_validation_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
gtest_discover_tests(lob_validation_tests)

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
