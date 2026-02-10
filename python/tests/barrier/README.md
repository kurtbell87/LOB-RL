# python/tests/barrier/

Tests for the barrier pipeline package (`python/lob_rl/barrier/`).

## Files

| File | Role | Test Count |
|------|------|-----------|
| `conftest.py` | Shared test helpers: `make_bar()`, `make_flat_bars()`, `make_session_bars()`, `make_session_data()`, `make_session_data_list()`, `run_episode()`, `make_mbo_dataframe()`, `make_mbo_df()`, `make_simple_book_mbo()`, RTH timestamp constants, dimension constants. | — |
| `test_bar_pipeline.py` | Bar construction: OHLCV, VWAP, timestamps, trade retention, RTH filtering, DST, batch processing. | 67 |
| `test_label_pipeline.py` | Barrier labels: upper/lower/timeout, tiebreaking, short direction, calibration, diagnostics. | 65 |
| `test_feature_pipeline.py` | Features: 22-column layout, normalization, lookback assembly, end-to-end builder. | 92 |
| `test_gamblers_ruin.py` | Gambler's ruin validation: analytic formula, random walk generation, drift levels. | 81 |
| `test_regime_switch.py` | Regime-switch validation: synthetic data, label distributions, KS tests, normalization adaptation. | 51 |
| `test_supervised_diagnostic.py` | Supervised diagnostic: dataset construction, MLP architecture, overfit test, training, evaluation, random forest, full pipeline. | 56 |
| `test_reward_accounting.py` | Reward accounting: hand-computed reward sequences for long/short, barrier hits, timeouts, MTM normalization, transaction costs, position state transitions, unrealized PnL, action masking. | 46 |
| `test_barrier_env.py` | Gymnasium env: API compliance, observation content, action masking, episode lifecycle, reward integration, position state transitions, random agent, info dict, edge cases, factory method. | 41 |
| `test_multi_session_env.py` | Multi-session wrapper: reset/API, session cycling, shuffle/determinism, skip short sessions, action mask delegation, single session, from_bar_lists factory, random agent stress. | 12 |
| `test_barrier_vec_env.py` | Vectorized env: env_fn callable, DummyVecEnv creation, num_envs, step/reset shapes, random agent on vec env. | 6 |
| `test_training_diagnostics.py` | Training callback: init, metric tracking (episode reward, flat action rate, trade win rate), NaN detection, red flag detection (entropy collapse, flat rate), CSV output, snapshot accumulation. | 12 |
| `test_ppo_training.py` | PPO training: linear schedule, model creation, prediction with masks, short training run, checkpointing, end-to-end integration, action masking enforcement, eval callback, policy architecture. | 11 |
| `test_train_barrier_script.py` | Training CLI script: arg parsing, session loading, model building, training smoke test, diagnostics output. | 22 |
| `test_lob_features_fix.py` | LOB reconstructor, dead book feature fix, N_FEATURES constant, precompute wiring, extract_all_mbo. | 64 |
| `test_microstructure_features.py` | Phase 1 microstructure features (cols 13-16): OFI, depth ratio, weighted mid displacement, spread dynamics. | 60 |
| `test_phase2_microstructure_features.py` | Phase 2 microstructure features (cols 17-21): VAMP displacement, aggressor imbalance, trade arrival, cancel-to-trade ratio, price impact. | 76 |

**Total: ~762 tests** (some skipped: need `.dbn.zst` fixture data).

## Shared Test Helpers (`conftest.py`)

```python
# RTH constants for 2022-06-15 CDT
_RTH_OPEN_NS = 1655296200_000_000_000   # 13:30 UTC
_RTH_CLOSE_NS = 1655319600_000_000_000  # 20:00 UTC
_RTH_DURATION_NS = _RTH_CLOSE_NS - _RTH_OPEN_NS

# Dimension constants
DEFAULT_H = 10              # default lookback
DEFAULT_FEATURE_DIM = 220   # N_FEATURES (22) * DEFAULT_H
DEFAULT_OBS_DIM = 222       # DEFAULT_FEATURE_DIM + 2

make_bar(bar_index, open_price, high, low, close, volume=100, vwap=None,
         t_start=0, t_end=1, session_date="2022-06-15",
         trade_prices=None, trade_sizes=None) -> TradeBar

make_flat_bars(n, base_price=4000.0, spread=1.0) -> list[TradeBar]
    # Bars that stay within a narrow range (won't trigger barriers)

make_session_bars(n, base_price=4000.0, spread=2.0, volume=100) -> list[TradeBar]
    # Bars spanning RTH session with increasing timestamps

make_session_data(n_bars=40, base_price=4000.0, h=DEFAULT_H) -> dict
    # Returns {bars, labels, features} for env construction

make_session_data_list(n_sessions=5, n_bars=40, h=DEFAULT_H) -> list[dict]
    # Returns n_sessions session dicts with slightly different prices

run_episode(env, rng=None) -> tuple[float, int]
    # Run one full episode with random valid actions, returns (total_reward, steps)

make_mbo_dataframe(records) -> pd.DataFrame
    # Minimal MBO DataFrame from list of dicts (asserts required columns present)

make_mbo_df(records) -> pd.DataFrame
    # MBO DataFrame with enforced dtypes, sorted by ts_event

make_simple_book_mbo(t_start, t_end, bid_price=4000.0, ask_price=4000.25,
                     bid_qty=100, ask_qty=50, n_bid_levels=1, n_ask_levels=1) -> list[dict]
    # Creates Add-only MBO records that establish a known book state
```

## Cross-File Dependencies

- All test files import from `lob_rl.barrier.*` modules.
- `test_label_pipeline.py` and `test_feature_pipeline.py` use helpers from `conftest.py`.
- `test_gamblers_ruin.py` and `test_regime_switch.py` are self-contained (call validation module functions directly).
- `test_supervised_diagnostic.py` uses helpers from `conftest.py` and imports from `lob_rl.barrier.supervised_diagnostic`.
- `test_bar_pipeline.py` has its own helpers (`_make_trades`, `_make_trades_with_prices`, `_utc_ns`).
- `test_reward_accounting.py` uses `make_bar` from `conftest.py` and imports from `lob_rl.barrier.reward_accounting`.
- `test_barrier_env.py` uses `make_bar`, `make_flat_bars`, `make_session_bars`, `make_session_data`, dimension constants from `conftest.py`.
- `test_multi_session_env.py` uses `make_session_bars`, `make_session_data`, `make_session_data_list`, `run_episode`, dimension constants from `conftest.py`.
- `test_barrier_vec_env.py` uses `make_session_data_list`, dimension constants from `conftest.py`.
- `test_training_diagnostics.py` uses `make_session_data_list` from `conftest.py`, imports from `lob_rl.barrier.training_diagnostics` and `lob_rl.barrier.barrier_vec_env`.
- `test_ppo_training.py` uses `make_session_data_list`, `DEFAULT_OBS_DIM` from `conftest.py`, imports from `lob_rl.barrier.barrier_vec_env` and `sb3_contrib`.
- `test_train_barrier_script.py` uses `make_session_bars` from `conftest.py`, imports from `scripts.train_barrier`.
- `test_lob_features_fix.py` uses `make_bar`, `make_session_bars`, `make_mbo_df`, `make_simple_book_mbo` from `conftest.py`, imports from `lob_rl.barrier.feature_pipeline`, `lob_rl.barrier.lob_reconstructor`, `scripts.precompute_barrier_cache`.
- `test_microstructure_features.py` uses `make_bar`, `make_session_bars`, `make_mbo_dataframe` from `conftest.py`, imports from `lob_rl.barrier.feature_pipeline`.
- `test_phase2_microstructure_features.py` uses `make_bar`, `make_session_bars`, `make_mbo_dataframe` from `conftest.py`, imports from `lob_rl.barrier.feature_pipeline`, `lob_rl.barrier.lob_reconstructor`.

## Running

```bash
cd build-release
PYTHONPATH=.:../python uv run --with pytest --with gymnasium --with numpy --with pandas --with stable-baselines3 --with sb3-contrib --with scipy --with torch --with scikit-learn pytest ../python/tests/barrier/ -v
```
