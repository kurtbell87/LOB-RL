# python/lob_rl/barrier/

Offline barrier-label construction pipeline for LOB-RL. Reads raw MBO data, builds fixed-count trade bars, computes triple-barrier labels, extracts 22 bar-level features with trailing z-score normalization, and provides validation modules.

## Files

| File | Role |
|------|------|
| `__init__.py` | Package init. Exports `TICK_SIZE`, `N_FEATURES`, `RTH_OPEN_NS`, `RTH_DURATION_NS`, `build_synthetic_trades()`. |
| `bar_pipeline.py` | Bar construction: `TradeBar`, `build_bars_from_trades()`, `filter_rth_trades()`, `extract_trades_from_mbo()`, `build_session_bars()`, `build_dataset()`. |
| `label_pipeline.py` | Barrier labels: `BarrierLabel`, `compute_labels()`, `calibrate_t_max()`, `compute_tiebreak_frequency()`, `compute_label_distribution()`. |
| `feature_pipeline.py` | Features: `compute_bar_features()`, `normalize_features()`, `assemble_lookback()`, `build_feature_matrix()`. |
| `lob_reconstructor.py` | LOB reconstruction: `OrderBook` — maintains bid/ask levels from MBO messages. Used by `feature_pipeline.py` for book-derived features. |
| `gamblers_ruin.py` | Validation: `gamblers_ruin_analytic()`, `generate_random_walk()`, `validate_drift_level()`, `run_validation()`. |
| `regime_switch.py` | Validation: `generate_regime_switch_trades()`, `validate_regime_switch()`, `compute_segment_stats()`, `ks_test_features()`, `measure_normalization_adaptation()`. |
| `supervised_diagnostic.py` | MLP classifier diagnostic: `build_labeled_dataset()`, `BarrierMLP`, `overfit_test()`, `train_mlp()`, `evaluate_classifier()`, `train_random_forest()`, `run_diagnostic()`. |
| `reward_accounting.py` | Reward computation: `RewardConfig`, `PositionState`, `compute_entry()`, `compute_hold_reward()`, `compute_mtm_reward()`, `classify_exit()`, `compute_unrealized_pnl()`, `compute_reward_sequence()`, `get_action_mask()`. Action constants: `ACTION_LONG`, `ACTION_SHORT`, `ACTION_FLAT`, `ACTION_HOLD`. |
| `barrier_env.py` | Gymnasium RL env: `BarrierEnv` — combines bars, labels, features, and reward accounting. One episode = one RTH session. `from_bars()` factory. `action_masks()` for SB3. |
| `multi_session_env.py` | Multi-session wrapper: `MultiSessionBarrierEnv` — cycles through pre-built sessions, round-robin or shuffled. `from_bar_lists()` factory. |
| `barrier_vec_env.py` | Vectorized env helpers: `make_barrier_env_fn()`, `make_barrier_vec_env()` — creates SB3-compatible DummyVecEnv/SubprocVecEnv. |
| `training_diagnostics.py` | Training callback: `BarrierDiagnosticCallback` — monitors entropy, value loss, flat action rate, episode reward, trade win rate, NaN detection, red flags. Also `linear_schedule()`. |
| `first_passage_analysis.py` | Brier scores, temporal splits, bootstrap tests, calibration curves, model fitting, null calibration, signal detection, label loading. |
| `sequence_models.py` | Sequence-aware models (LSTM, Transformer) for barrier prediction: `BarEmbedding`, `LinearBarEmbedding`, `BarrierLSTM`, `BarrierTransformer`, `collate_sessions()`, `train_sequence_model()`, `predict_sessions()`, `evaluate_sequence_model()`. |
| `_sb3_compat.py` | SB3 compatibility shim: patches `MaskableActorCriticPolicy` and `MlpExtractor` for legacy `net_arch` format. Auto-applied on import. |

## API Signatures

### `__init__.py`

```python
TICK_SIZE = 0.25           # /MES tick size
N_FEATURES = 22            # Number of bar-level feature columns
RTH_OPEN_NS = 1655296200_000_000_000   # 2022-06-15 13:30 UTC
RTH_DURATION_NS = 23400_000_000_000    # 6.5 hours in ns

build_synthetic_trades(prices: np.ndarray, timestamps: np.ndarray) -> np.ndarray
    # Returns structured array: (price float64, size int32, side U1, ts_event int64, timestamp int64)
```

### `bar_pipeline.py`

```python
@dataclass
class TradeBar:
    bar_index: int; open: float; high: float; low: float; close: float
    volume: int; vwap: float; t_start: int; t_end: int
    session_date: str; trade_prices: np.ndarray; trade_sizes: np.ndarray

build_bars_from_trades(trades, n=500, session_date="") -> list[TradeBar]
filter_rth_trades(trades) -> np.ndarray
extract_trades_from_mbo(filepath, instrument_id=None) -> np.ndarray
build_session_bars(filepath, n=500, instrument_id=None) -> list[TradeBar]
build_dataset(filepaths, n=500, roll_calendar=None, output_path=None) -> pd.DataFrame
```

### `label_pipeline.py`

```python
@dataclass
class BarrierLabel:
    bar_index: int; label: int; tau: int; resolution_type: str
    entry_price: float; resolution_bar: int

compute_labels(bars, a=20, b=10, t_max=40, direction="long") -> list[BarrierLabel]
calibrate_t_max(bars, a=20, b=10) -> int
compute_tiebreak_frequency(labels) -> float
compute_label_distribution(labels) -> dict  # {p_plus, p_minus, p_zero}
```

### `feature_pipeline.py`

```python
# Module constants
_REALIZED_VOL_WARMUP = 19    # bars 0..18 have NaN in col 8
_SESSION_AGE_PERIOD = 20.0   # col 12 saturates at bar_index / 20
_BOOK_COL_MAP = [1, 2, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    # Maps book_features col index -> compute_bar_features col index
_BOOK_DEFAULTS = (0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Neutral defaults when mbo_data is None

compute_bar_features(bars, mbo_data=None) -> np.ndarray  # shape (N, 22)
normalize_features(raw, window=2000) -> np.ndarray        # z-score, clipped [-5, +5]
assemble_lookback(normed, h=10) -> np.ndarray             # shape (N-h+1, 22*h)
build_feature_matrix(bars, h=10, window=2000, mbo_data=None) -> np.ndarray
```

Feature column layout (22 columns):
| Col | Feature | Range |
|-----|---------|-------|
| 0 | Trade flow imbalance | [-1, +1] |
| 1 | BBO imbalance | [0, 1] (0.5 default) |
| 2 | Depth imbalance | [0, 1] (0.5 default) |
| 3 | Bar range (ticks) | >= 0 |
| 4 | Bar body (ticks) | signed |
| 5 | Body/range ratio | [-1, +1] |
| 6 | VWAP displacement | [-1, +1] |
| 7 | Volume (log) | finite |
| 8 | Trailing realized vol | NaN for first 19 bars |
| 9 | Normalized session time | [0, 1] |
| 10 | Cancel rate asymmetry | [-1, +1] (0.0 default) |
| 11 | Mean spread | > 0 (1.0 default) |
| 12 | Session age | [0, 1] |
| 13 | Order Flow Imbalance (OFI) | [-1, +1] (0.0 default) |
| 14 | Multi-level depth ratio | [0, 1] (0.5 default) |
| 15 | Weighted mid displacement | signed (0.0 default) |
| 16 | Spread dynamics (std) | >= 0 (0.0 default) |
| 17 | VAMP displacement | signed (0.0 default) |
| 18 | Aggressor imbalance | [-1, +1] (0.0 default) |
| 19 | Trade arrival rate | >= 0 (0.0 default) |
| 20 | Cancel-to-trade ratio | >= 0 (0.0 default) |
| 21 | Price impact per trade | signed (0.0 default) |

### `first_passage_analysis.py`

```python
brier_score(y_true, y_pred) -> float
constant_brier(y_true) -> float        # Brier score of constant predictor ybar
brier_skill_score(y_true, y_pred) -> float  # BSS = 1 - Brier(model) / Brier(constant)
temporal_split(n_sessions, train_frac=0.6, val_frac=0.2) -> tuple[ndarray, ndarray, ndarray]
temporal_cv_folds(n_sessions, n_folds=5) -> list[tuple[ndarray, ndarray]]
calibration_curve(y_true, y_pred, n_bins=10) -> tuple[ndarray, ndarray]
paired_bootstrap_brier(y_true, pred_model, pred_baseline, n_boot=10000,
                        block_size=50, seed=42) -> dict
    # {delta, ci_lower, ci_upper, p_value}
null_calibration_report(Y_long, Y_short, tau_long, tau_short,
                         timeout_long, timeout_short, session_boundaries) -> dict
fit_logistic(X_train, y_train, max_iter=1000) -> LogisticRegression
fit_gbt(X_train, y_train, seed=42) -> model  # LightGBM or sklearn fallback
load_binary_labels(cache_dir, lookback=10) -> dict
    # {X, Y_long, Y_short, timeout_long, timeout_short, tau_long, tau_short,
    #  session_boundaries, dates}
REALIZED_VOL_WARMUP = 19
load_session_features(cache_dir) -> list[dict]
    # Each: {features: (n,22) float32, Y_long: (n,) bool, Y_short: (n,) bool, date: str}
verify_lattice(tick_size=0.25, a=20, b=10) -> dict  # {lattice_ok, R_ticks}
signal_detection_report(X, Y_long, Y_short, session_boundaries, seed=42) -> dict
```

### `sequence_models.py`

```python
class BarEmbedding(nn.Module):
    d_model: int  # set by subclass

class LinearBarEmbedding(BarEmbedding):
    __init__(n_features=22, d_model=64, max_len=2048, dropout=0.1)
    forward(x: Tensor[B, T, n_features]) -> Tensor[B, T, d_model]

class _DualHeadModel(nn.Module):
    # Internal base: shared head_long/head_short Linear(d, 1) + _apply_heads()

class BarrierLSTM(_DualHeadModel):
    __init__(n_features=22, d_model=64, n_layers=2, dropout=0.1,
             embedding: BarEmbedding | None = None)
    forward(x: Tensor[B, T, 22]) -> tuple[Tensor[B, T], Tensor[B, T]]

class BarrierTransformer(_DualHeadModel):
    __init__(n_features=22, d_model=64, n_layers=2, n_heads=4, dropout=0.1,
             embedding: BarEmbedding | None = None)
    forward(x: Tensor[B, T, 22]) -> tuple[Tensor[B, T], Tensor[B, T]]

collate_sessions(sessions: list[dict]) -> tuple[Tensor, Tensor, Tensor, Tensor]
    # (features[B,T,22], Y_long[B,T], Y_short[B,T], mask[B,T])

train_sequence_model(model, train_sessions, val_sessions, epochs=50,
                      batch_size=4, lr=1e-3, weight_decay=1e-4, patience=10,
                      seed=42, device="cpu") -> dict
    # {train_loss_history, val_brier_history, best_epoch, best_val_brier, stopped_early}

predict_sessions(model, sessions, batch_size=8, device="cpu") -> tuple[ndarray, ndarray]
    # (p_long[N_total], p_short[N_total]) float64, padding removed

evaluate_sequence_model(model, sessions, batch_size=8, device="cpu",
                         n_bootstrap=1000, seed=42) -> dict
    # Keys per label: brier_constant_{l}, brier_model_{l}, bss_{l},
    #   bootstrap_{l}, p_hat_mean_{l}, p_hat_std_{l}
```

### `gamblers_ruin.py`

```python
gamblers_ruin_analytic(a, b, p) -> float
generate_random_walk(n_trades, p=0.5, start_price=4000.0, tick_size=0.25, seed=None) -> np.ndarray
validate_drift_level(p, a=20, b=10, n_bars=10000, bar_size=500, t_max=40, seed=None) -> dict
run_validation(drift_levels=None, a=20, b=10, n_bars=10000, bar_size=500, t_max=40, seed=42) -> list[dict]
```

### `regime_switch.py`

```python
generate_regime_switch_trades(n_bars_low=5000, n_bars_high=5000, bar_size=500,
                               p=0.5, start_price=4000.0, tick_size=0.25,
                               seed=None) -> tuple[np.ndarray, list[TradeBar]]
validate_regime_switch(n_bars_low=5000, n_bars_high=5000, bar_size=500,
                        a=100, b=100, t_max=40, seed=42) -> dict
compute_segment_stats(labels, start, end) -> dict  # {p_plus, p_minus, p_zero, mean_tau, median_tau}
ks_test_features(features, boundary, window=500) -> dict  # {ks_p_col_0..ks_p_col_N}
measure_normalization_adaptation(normed_features, boundary, col=8, threshold_sigma=1.0) -> int
```

### `supervised_diagnostic.py`

```python
# Module constants
_OVERFIT_MAX_SAMPLES = 256   # max samples for overfit sanity check

build_labeled_dataset(bars, labels, h=10) -> tuple[np.ndarray, np.ndarray]
    # X: shape (n_usable, N_FEATURES*h), float32; y: shape (n_usable,), int64
    # Label mapping: -1 -> 0, 0 -> 1, +1 -> 2

class BarrierMLP(nn.Module):
    __init__(input_dim, hidden_dim=256, n_classes=3)
    forward(x) -> Tensor

overfit_test(X, y, epochs=500, batch_size=256, seed=42) -> dict
    # {train_accuracy: float, passed: bool}  (passed = accuracy > 0.95)
train_mlp(X, y, epochs=100, batch_size=256, hidden_dim=256, lr=1e-3, seed=42)
    -> tuple[BarrierMLP, dict]  # dict: {train_accuracy, train_loss}
evaluate_classifier(model, X_test, y_test) -> dict
    # {accuracy, balanced_accuracy, majority_class, majority_baseline,
    #  beats_baseline, confusion_matrix, per_class}
train_random_forest(X_train, y_train, X_test, y_test, seed=42, n_estimators=100) -> dict
run_diagnostic(bars, labels, h=10, train_frac=0.8, epochs=100, seed=42) -> dict
    # {n_samples, n_train, n_test, label_distribution, overfit_test, mlp, random_forest, passed}
```

### `reward_accounting.py`

```python
# Action constants
ACTION_LONG = 0; ACTION_SHORT = 1; ACTION_FLAT = 2; ACTION_HOLD = 3

@dataclass
class RewardConfig:
    a: int = 20          # profit barrier width in ticks
    b: int = 10          # stop barrier width in ticks
    G: float = 2.0       # profit gain reward
    L: float = 1.0       # stop loss penalty magnitude
    C: float = 0.2       # round-trip transaction cost
    T_max: int = 40      # max holding period in bars
    tick_size: float = 0.25  # /MES tick size

@dataclass
class PositionState:
    position: int = 0           # 0=flat, +1=long, -1=short
    entry_price: float = 0.0
    profit_barrier: float = 0.0
    stop_barrier: float = 0.0
    hold_counter: int = 0

get_action_mask(position: int) -> list[bool]  # [long, short, flat, hold]
compute_entry(bar: TradeBar, action: int, config: RewardConfig) -> PositionState
compute_hold_reward(bar: TradeBar, state: PositionState, config: RewardConfig)
    -> tuple[float, PositionState]  # (reward, new_state)
classify_exit(reward: float, hold_counter: int, config: RewardConfig) -> str
    # "profit", "stop", or "timeout"
compute_mtm_reward(state: PositionState, close_price: float, config: RewardConfig) -> float
    # MTM = position * (close - entry) / (b * tick_size) - C
compute_unrealized_pnl(state: PositionState, current_price: float) -> float  # ticks
compute_reward_sequence(bars: list[TradeBar], action: int, start_bar_idx: int,
    config: RewardConfig) -> list[dict]
    # Each dict: {reward, position, hold_counter, exit_type, unrealized_pnl}
    # exit_type: "profit", "stop", "timeout", or None
```

### `barrier_env.py`

```python
class BarrierEnv(gymnasium.Env):
    __init__(bars: list[TradeBar], labels: list[BarrierLabel],
             features: np.ndarray, config: RewardConfig = None)
    from_bars(cls, bars: list[TradeBar], h: int = 10,
              config: RewardConfig = None) -> BarrierEnv  # classmethod
    reset(seed=None, options=None) -> tuple[np.ndarray, dict]
    step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]
    action_masks() -> np.ndarray  # shape (4,), dtype int8
    # Observation: shape (N_FEATURES*h + 2,) = features | position | unrealized_pnl
    # Action space: Discrete(4) = [long, short, flat, hold]
    # Info dict keys: position, bar_idx, exit_type, entry_price, n_trades
```

### `multi_session_env.py`

```python
class MultiSessionBarrierEnv(gymnasium.Env):
    __init__(sessions: list[dict], config: RewardConfig = None,
             shuffle: bool = False, seed: int = None)
    # sessions: each dict has keys {bars, labels, features}
    from_bar_lists(cls, bar_lists: list[list[TradeBar]], h: int = 10,
                   config: RewardConfig = None, **kwargs) -> MultiSessionBarrierEnv
    reset(seed=None, options=None) -> tuple[np.ndarray, dict]
    step(action: int) -> tuple[np.ndarray, float, bool, bool, dict]
    action_masks() -> np.ndarray
    current_session_index: int  # property
```

### `barrier_vec_env.py`

```python
make_barrier_env_fn(sessions, config=None, shuffle=False, seed=None) -> callable
make_barrier_vec_env(sessions, n_envs=1, use_subprocess=False,
                     config=None, shuffle=False, seed=None) -> VecEnv
```

### `training_diagnostics.py`

```python
# Red-flag thresholds
_ENTROPY_COLLAPSE_THRESHOLD = 0.3
_ENTROPY_COLLAPSE_WINDOW = 100
_FLAT_RATE_LOW = 0.10
_FLAT_RATE_HIGH = 0.90

linear_schedule(initial_value: float) -> callable
    # Returns f(progress_remaining) -> current_value

class BarrierDiagnosticCallback(BaseCallback):
    __init__(check_freq: int = 1, output_dir: str = None, verbose: int = 0)
    diagnostics: list[dict]  # accumulated snapshots
    check_red_flags() -> list[str]  # red flag descriptions
    # Snapshot keys: entropy_flat, value_loss, policy_loss,
    #   episode_reward_mean, flat_action_rate, trade_win_rate,
    #   n_trades, has_nan
```

## Cross-File Dependencies

- `bar_pipeline.py` depends on: `pandas`, `numpy`, `zoneinfo`, `__init__.py` (none from barrier)
- `label_pipeline.py` depends on: `__init__.py` (`TICK_SIZE`)
- `feature_pipeline.py` depends on: `__init__.py` (`TICK_SIZE`), `lob_reconstructor.py` (`OrderBook`)
- `gamblers_ruin.py` depends on: `__init__.py` (`TICK_SIZE`, `RTH_OPEN_NS`, `RTH_DURATION_NS`, `build_synthetic_trades`), `bar_pipeline.py`, `label_pipeline.py`
- `regime_switch.py` depends on: `__init__.py` (same as gamblers_ruin), `bar_pipeline.py`, `label_pipeline.py`, `feature_pipeline.py`, `scipy.stats`
- `supervised_diagnostic.py` depends on: `feature_pipeline.py`, `torch`, `numpy`, `sklearn.ensemble` (lazy import in `train_random_forest`)
- `reward_accounting.py` depends on: `__init__.py` (`TICK_SIZE`)
- `barrier_env.py` depends on: `__init__.py` (`TICK_SIZE`), `feature_pipeline.py`, `label_pipeline.py`, `reward_accounting.py`, `gymnasium`
- `multi_session_env.py` depends on: `barrier_env.py`, `feature_pipeline.py`, `label_pipeline.py`, `reward_accounting.py`, `gymnasium`
- `barrier_vec_env.py` depends on: `multi_session_env.py`, `_sb3_compat.py`, `stable_baselines3`
- `training_diagnostics.py` depends on: `stable_baselines3`, `numpy`
- `first_passage_analysis.py` depends on: `numpy`, `sklearn` (lazy), `glob`, `os`
- `sequence_models.py` depends on: `first_passage_analysis.py` (`brier_score`, `brier_skill_score`, `constant_brier`, `paired_bootstrap_brier`), `torch`, `numpy`
- `_sb3_compat.py` depends on: `sb3_contrib`, `stable_baselines3`, `torch`

## Modification Hints

- To add a new feature column: update `compute_bar_features()` in `feature_pipeline.py`, increment `N_FEATURES` in `__init__.py` (currently 22), add to `_BOOK_COL_MAP` and `_BOOK_DEFAULTS` if book-derived, and update the column layout table above.
- To add a new validation module: create `python/lob_rl/barrier/<name>.py`, use `build_synthetic_trades()` from `__init__.py` for trade generation, add tests in `python/tests/barrier/test_<name>.py`.
- To add a new label resolution type: update `_label_single_bar()` and `_tiebreak()` in `label_pipeline.py`, handle the new type in the short-direction flip logic in `compute_labels()`.
- To add a new sequence model architecture: subclass `_DualHeadModel`, accept `embedding: BarEmbedding | None` in constructor, implement `forward(x: [B, T, 22]) -> (logit_long: [B, T], logit_short: [B, T])`. Call `self._init_heads(d)` in `__init__` and `self._apply_heads(h)` in `forward`.
- To add a new embedding strategy: subclass `BarEmbedding`, set `self.d_model`, implement `forward(x: [B, T, n_features]) -> [B, T, d_model]`. Pass to any model constructor.
