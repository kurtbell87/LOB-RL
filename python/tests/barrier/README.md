# python/tests/barrier/

Tests for the barrier pipeline package (`python/lob_rl/barrier/`).

## Files

| File | Role | Test Count |
|------|------|-----------|
| `conftest.py` | Shared test helpers: `make_bar()`, `make_flat_bars()`, `make_session_bars()`, RTH timestamp constants. | — |
| `test_bar_pipeline.py` | Bar construction: OHLCV, VWAP, timestamps, trade retention, RTH filtering, DST, batch processing. | ~80 |
| `test_label_pipeline.py` | Barrier labels: upper/lower/timeout, tiebreaking, short direction, calibration, diagnostics. | ~80 |
| `test_feature_pipeline.py` | Features: 13-column layout, normalization, lookback assembly, end-to-end builder. | ~55 |
| `test_gamblers_ruin.py` | Gambler's ruin validation: analytic formula, random walk generation, drift levels. | 81 |
| `test_regime_switch.py` | Regime-switch validation: synthetic data, label distributions, KS tests, normalization adaptation. | 51 |
| `test_supervised_diagnostic.py` | Supervised diagnostic: dataset construction, MLP architecture, overfit test, training, evaluation, random forest, full pipeline. | 56 |
| `test_reward_accounting.py` | Reward accounting: hand-computed reward sequences for long/short, barrier hits, timeouts, MTM normalization, transaction costs, position state transitions, unrealized PnL, action masking. | 46 |
| `test_barrier_env.py` | Gymnasium env: API compliance, observation content, action masking, episode lifecycle, reward integration, position state transitions, random agent, info dict, edge cases, factory method. | 41 |
| `test_multi_session_env.py` | Multi-session wrapper: reset/API, session cycling, shuffle/determinism, skip short sessions, action mask delegation, single session, from_bar_lists factory, random agent stress. | 12 |
| `test_barrier_vec_env.py` | Vectorized env: env_fn callable, DummyVecEnv creation, num_envs, step/reset shapes, random agent on vec env. | 5 |
| `test_training_diagnostics.py` | Training callback: init, metric tracking (episode reward, flat action rate, trade win rate), NaN detection, red flag detection (entropy collapse, flat rate), CSV output, snapshot accumulation. | 11 |
| `test_ppo_training.py` | PPO training integration: MaskablePPO training smoke test, eval callback. | 13 |

**Total: ~532 tests** (8 skipped: need `.dbn.zst` fixture data).

## Shared Test Helpers (`conftest.py`)

```python
# RTH constants for 2022-06-15 CDT
_RTH_OPEN_NS = 1655296200_000_000_000   # 13:30 UTC
_RTH_CLOSE_NS = 1655319600_000_000_000  # 20:00 UTC
_RTH_DURATION_NS = _RTH_CLOSE_NS - _RTH_OPEN_NS

# Dimension constants
DEFAULT_H = 10              # default lookback
DEFAULT_FEATURE_DIM = 130   # 13 * DEFAULT_H
DEFAULT_OBS_DIM = 132       # DEFAULT_FEATURE_DIM + 2

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
- `test_training_diagnostics.py` uses `make_session_bars` from `conftest.py`, imports from `lob_rl.barrier.training_diagnostics` and `lob_rl.barrier.barrier_vec_env`.
- `test_ppo_training.py` uses `make_session_bars` from `conftest.py`, imports from `lob_rl.barrier.barrier_vec_env` and `sb3_contrib`.

## Running

```bash
cd build-release
PYTHONPATH=.:../python uv run --with pytest --with gymnasium --with numpy --with pandas --with stable-baselines3 --with sb3-contrib --with scipy --with torch --with scikit-learn pytest ../python/tests/barrier/ -v
```
