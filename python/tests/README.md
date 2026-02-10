# python/tests — Python Test Suite

pytest-based tests. Run with: `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`

**1664 tests** (1308 core + 356 barrier), all passing (4 core + 8 barrier skipped: fixture-dependent).

## Test files

| File | Tests |
|---|---|
| `test_bar_aggregation.py` | `aggregate_bars()` tick → bar feature aggregation |
| `test_bar_level_env.py` | `BarLevelEnv` 21-dim obs, from_cache/from_file |
| `test_bar_multi_day.py` | `MultiDayEnv` with `bar_size > 0` |
| `test_bar_train.py` | `train.py --bar-size` integration |
| `test_bindings_b1.py` | BinaryFileSource + SessionConfig bindings |
| `test_cache_multi_day.py` | MultiDayEnv with cache_dir/cache_files |
| `test_contract_boundary_guard.py` | Forced flatten on terminal step, instrument_id tracking |
| `test_execution_cost.py` | Execution cost for PrecomputedEnv (54-dim) and LOBGymEnv (44-dim) |
| `test_frame_stacking.py` | 40 tests: `--frame-stack`, VecFrameStack wrapping in train/eval |
| `test_from_cache.py` | PrecomputedEnv.from_cache() loading |
| `test_gym.py` | LOBEnv Python bindings (step/reset/obs shape) |
| `test_gym_wrapper.py` | LOBGymEnv gymnasium wrapper + `check_env()` |
| `test_lazy_loading.py` | MultiDayEnv lazy-load .npz on reset() |
| `test_multi_day_env.py` | MultiDayEnv sequential/shuffle/seed (54-dim obs) |
| `test_native_dbn_source.py` | DbnFileSource Python integration |
| `test_obs44.py` | 44-float observation from Python |
| `test_participation_bonus.py` | Participation bonus for PrecomputedEnv (54-dim) and LOBGymEnv (44-dim) |
| `test_precompute_binding.py` | `lob_rl_core.precompute()` binding |
| `test_precomputed_env.py` | PrecomputedEnv correctness (54-dim obs) |
| `test_precomputed_multi_day.py` | Precomputed MultiDayEnv integration (54-dim obs) |
| `test_recurrent_ppo.py` | 59 tests: `--recurrent`, mutual exclusivity, LSTM state tracking |
| `test_reward_mode.py` | Reward mode bindings (pnl_delta, pnl_delta_penalized) |
| `test_shuffle_split.py` | 42 tests: `--shuffle-split`, `--seed`, determinism, date printing |
| `test_step_interval.py` | `--step-interval N` subsampling |
| `test_temporal_features.py` | 70 tests: all 10 temporal features, obs layout, edge cases |
| `test_train_cache_dir.py` | train.py --cache-dir integration |
| `test_training_pipeline_v2.py` | train.py integration: VecNormalize, SubprocVecEnv, CLI flags |
| `test_checkpointing.py` | 72 tests: `--checkpoint-freq`, `--resume`, VecNormalize checkpoint saving |
| `barrier/test_bar_pipeline.py` | 59 tests: TradeBar dataclass, `build_bars_from_trades()`, `filter_rth_trades()`, `build_session_bars()`, `build_dataset()` |
| `barrier/test_label_pipeline.py` | 65 tests: BarrierLabel, `compute_labels()`, barrier hit detection, tiebreaking, short direction, T_max calibration |
| `barrier/test_feature_pipeline.py` | 92 tests: 13-feature computation, z-score normalization, lookback assembly, `build_feature_matrix()` |
| `barrier/test_gamblers_ruin.py` | 81 tests: `gamblers_ruin_analytic()`, `generate_random_walk()`, `validate_drift_level()`, `run_validation()`, 5 drift levels |
| `barrier/test_regime_switch.py` | 51 tests: `generate_regime_switch_trades()`, `validate_regime_switch()`, KS tests, normalization adaptation, regime boundary detection |
| `barrier/conftest.py` | Shared helpers: `make_bar()`, `make_flat_bars()`, `make_session_bars()`, `TICK_SIZE`, RTH constants |

## Fixtures

`python/tests/fixtures/` — binary test data. `conftest.py` provides shared pytest fixtures including `make_tick_data()`, `save_cache_with_instrument_id()`, `save_cache_without_instrument_id()`, and train.py source helpers (`TRAIN_SCRIPT`, `load_train_source()`, `extract_main_body()`, `extract_evaluate_sortino_body()`). Array factories use named constants from `lob_rl._obs_layout` (`BASE_OBS_SIZE`, `BID_PRICES`, etc.) instead of magic numbers.
