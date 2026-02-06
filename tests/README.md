# tests — C++ Test Suite

GoogleTest-based tests. Run with: `cd build-release && ./lob_tests`

## Key test files

| File | Tests |
|---|---|
| `test_book.cpp` | Book add/cancel/modify/trade operations |
| `test_book_depth.cpp` | `top_bids(k)`/`top_asks(k)` depth queries |
| `test_env.cpp` | LOBEnv step/reset, episode termination |
| `test_env_obs44.cpp` | 44-float FeatureBuilder observation space |
| `test_feature_builder.cpp` | FeatureBuilder edge cases |
| `test_reward_calculator.cpp` | PnLDelta and PnLDeltaPenalized modes |
| `test_session.cpp` | SessionConfig, SessionFilter, RTH boundaries |
| `test_session_env.cpp` | Session-aware LOBEnv (warmup, flattening) |
| `test_flattening_pnl.cpp` | Position flattening penalty at session close |
| `test_source.cpp` | SyntheticSource determinism |
| `test_binary_file_source.cpp` | BinaryFileSource reading, edge cases |
| `test_precompute.cpp` | Precompute arrays match step-by-step execution |
| `test_data_integrity.cpp` | Validation, overflow, truncation handling |

## Fixtures

`tests/fixtures/` — binary test fixtures and Python generators that create them.

## Helpers

`test_helpers.h` — shared `fixture_path()`, `make_stable_bbo_messages()`, etc.
