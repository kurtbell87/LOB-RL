# tests — C++ Test Suite

GoogleTest-based tests. Run with: `cd build-release && ./lob_tests`

**418 tests** across 30 test suites. (15 skipped: need `.dbn.zst` fixture.)

## Test files

| File | Tests |
|---|---|
| `test_book.cpp` | Book add/cancel/modify/trade operations |
| `test_book_depth.cpp` | `top_bids(k)`/`top_asks(k)` depth queries |
| `test_data_integrity.cpp` | Validation, overflow, truncation handling |
| `test_dbn_file_source.cpp` | DbnFileSource reading native `.dbn.zst` files |
| `test_dbn_message_map.cpp` | Databento MBO → internal Message mapping |
| `test_env.cpp` | LOBEnv step/reset, episode termination |
| `test_env_obs44.cpp` | 44-float FeatureBuilder observation space |
| `test_execution_cost.cpp` | RewardCalculator::execution_cost, LOBEnv integration |
| `test_feature_builder.cpp` | FeatureBuilder edge cases, layout indices |
| `test_fix_precompute_events.cpp` | Flag-aware precompute, F_LAST/F_SNAPSHOT handling |
| `test_flattening_pnl.cpp` | Position flattening penalty at session close |
| `test_participation_bonus.cpp` | RewardCalculator::participation_bonus, LOBEnv integration |
| `test_precompute.cpp` | Precompute arrays match step-by-step execution |
| `test_reward_calculator.cpp` | PnLDelta and PnLDeltaPenalized modes |
| `test_session.cpp` | SessionConfig, SessionFilter, RTH boundaries |
| `test_session_env.cpp` | Session-aware LOBEnv (warmup, flattening) |
| `test_source.cpp` | SyntheticSource determinism |

## Fixtures

`tests/fixtures/` — binary test fixtures and Python generators that create them.

Key fixtures: `valid_10records.bin`, `mixed_actions.bin`, `precompute_rth.bin`, `precision_test.bin`.

## Helpers

`test_helpers.h` — shared `fixture_path()`, `make_stable_bbo_messages()`, `make_session_config()`, flag constants (`F_LAST`, `F_SNAPSHOT`, `F_MBP`).

## Adding new tests

1. Create `tests/test_<name>.cpp`
2. Add to `CMakeLists.txt` in the `add_executable(lob_tests ...)` block
3. Include `#include "test_helpers.h"` for fixtures/helpers
4. Rebuild: `cd build-release && cmake --build .`
