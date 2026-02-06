# python/tests — Python Test Suite

pytest-based tests. Run with: `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`

## Key test files

| File | Tests |
|---|---|
| `test_gym.py` | LOBEnv Python bindings (step/reset/obs shape) |
| `test_bindings_b1.py` | BinaryFileSource + SessionConfig bindings |
| `test_gym_wrapper.py` | LOBGymEnv gymnasium wrapper + `check_env()` |
| `test_obs44.py` | 44-float observation from Python |
| `test_reward_mode.py` | Reward mode bindings (pnl_delta, pnl_delta_penalized) |
| `test_convert_dbn.py` | DBN converter unit tests |
| `test_multi_day_env.py` | MultiDayEnv sequential/shuffle/seed |
| `test_precomputed_env.py` | PrecomputedEnv correctness |
| `test_precomputed_multi_day.py` | Precomputed MultiDayEnv integration |
| `test_precompute_binding.py` | `lob_rl_core.precompute()` binding |

## Fixtures

`python/tests/fixtures/` — binary test data. `conftest.py` provides shared pytest fixtures.
