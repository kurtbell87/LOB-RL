# tests — C++ Test Suite

GoogleTest-based tests. Run with: `cd build-release && ./lob_tests`

**489 tests** across 32 test suites.

## Key test files

| File | Tests | Count |
|---|---|---|
| `test_book.cpp` | Book add/cancel/modify/trade operations | ~60 |
| `test_book_depth.cpp` | `top_bids(k)`/`top_asks(k)` depth queries | ~30 |
| `test_env.cpp` | LOBEnv step/reset, episode termination | ~30 |
| `test_env_obs44.cpp` | 44-float FeatureBuilder observation space | ~20 |
| `test_feature_builder.cpp` | FeatureBuilder edge cases, layout indices | ~25 |
| `test_reward_calculator.cpp` | PnLDelta and PnLDeltaPenalized modes | ~20 |
| `test_execution_cost.cpp` | RewardCalculator::execution_cost, LOBEnv integration | 30 |
| `test_session.cpp` | SessionConfig, SessionFilter, RTH boundaries | ~25 |
| `test_session_env.cpp` | Session-aware LOBEnv (warmup, flattening) | ~30 |
| `test_flattening_pnl.cpp` | Position flattening penalty at session close | ~15 |
| `test_source.cpp` | SyntheticSource determinism | ~15 |
| `test_binary_file_source.cpp` | BinaryFileSource reading, edge cases | ~25 |
| `test_precompute.cpp` | Precompute arrays match step-by-step execution | ~20 |
| `test_participation_bonus.cpp` | RewardCalculator::participation_bonus, LOBEnv integration | ~15 |
| `test_fix_precompute_events.cpp` | Flag-aware precompute, F_LAST/F_SNAPSHOT handling | ~29 |
| `test_data_integrity.cpp` | Validation, overflow, truncation handling | ~20 |

## Fixtures

`tests/fixtures/` — binary test fixtures and Python generators that create them.

Key fixtures: `valid_10records.bin`, `mixed_actions.bin`, `precompute_rth.bin`, `precision_test.bin`.

## Helpers

`test_helpers.h` — shared `fixture_path()`, `make_stable_bbo_messages()`, `make_session_config()`.

## Adding new tests

1. Create `tests/test_<name>.cpp`
2. Add to `CMakeLists.txt` in the `add_executable(lob_tests ...)` block
3. Include `#include "test_helpers.h"` for fixtures/helpers
4. Rebuild: `cd build-release && cmake --build .`
