# python/tests — Python Test Suite

pytest-based tests. Run with: `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`

**696 tests**, all passing.

## Key test files

| File | Tests |
|---|---|
| `test_gym.py` | LOBEnv Python bindings (step/reset/obs shape) |
| `test_bindings_b1.py` | BinaryFileSource + SessionConfig bindings |
| `test_gym_wrapper.py` | LOBGymEnv gymnasium wrapper + `check_env()` |
| `test_obs44.py` | 44-float observation from Python |
| `test_reward_mode.py` | Reward mode bindings (pnl_delta, pnl_delta_penalized) |
| `test_convert_dbn.py` | DBN converter unit tests |
| `test_multi_day_env.py` | MultiDayEnv sequential/shuffle/seed (54-dim obs) |
| `test_precomputed_env.py` | PrecomputedEnv correctness (54-dim obs) |
| `test_precomputed_multi_day.py` | Precomputed MultiDayEnv integration (54-dim obs) |
| `test_precompute_binding.py` | `lob_rl_core.precompute()` binding |
| `test_execution_cost.py` | Execution cost for both PrecomputedEnv (54-dim) and LOBGymEnv (44-dim) |
| `test_participation_bonus.py` | Participation bonus for both PrecomputedEnv (54-dim) and LOBGymEnv (44-dim) |
| `test_temporal_features.py` | 70 tests: all 10 temporal features, obs layout, edge cases, multi-day integration |
| `test_training_pipeline_v2.py` | train.py integration: VecNormalize, SubprocVecEnv, CLI flags, make_env |

## Fixtures

`python/tests/fixtures/` — binary test data. `conftest.py` provides shared pytest fixtures.
