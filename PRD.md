# LOB-RL: Limit Order Book RL Environment

## Product Requirements Document

---

## 1. Goal

Train RL agents on Databento MBO futures data and evaluate trading performance on properly held-out data.

**Primary success metric:** Positive Sortino ratio on held-out OOS data across multiple walk-forward windows.

**Stretch goal:** Stable paper-trading performance over months, leading to live deployment with real capital.

---

## 2. Constraints

| Constraint | Decision |
|------------|----------|
| Strategy | Directional (not market-making) |
| Position | {-1, 0, +1} target (long, flat, short) |
| Orders | Market only, instant fill at BBO |
| Market impact | Zero (small size assumption, <10 contracts) |
| Holding period | Minutes to ~1 hour |
| Episode | One session, flat at open and close |
| Initial session | US RTH (13:30-20:00 UTC / 14:30-21:00 UTC DST) |
| Data | ~250 trading days /MES MBO (Jan 2022 – Jan 2023, Databento) |
| Train/Val/Test split | 170 / 40 / 40 days |

---

## 3. Non-Goals

- Market making / limit orders
- Fill simulation / queue position modeling
- Multi-instrument
- GPU acceleration (CPU at ~5,650 FPS is sufficient for current scale)
- Hyperparameter AutoML / Optuna (manual sweep was sufficient)
- Live trading / broker connectivity (until Phase 4)

---

## 4. Architecture

```
data/mes/*.dbn.zst → precompute_cache.py → cache/mes/*.npz (one-time)
                                                 ↓
                            ┌─── bar_size=0: PrecomputedEnv (54-dim, tick-level)
                            │
       MultiDayEnv ─────────┤
                            │
                            └─── bar_size>0: BarLevelEnv (21-dim, bar-level)  ← current best
                                     ↑
                                aggregate_bars() + cross-bar temporal
                                                 ↓
                                        SB3 PPO (scripts/train.py)
```

**Layers:**

| Layer | Role |
|-------|------|
| C++ LOBEnv (pybind11) | Replays raw MBO data, maintains 10-level book, computes tick-level features (54-dim), reward, position tracking. 489 tests. |
| `precompute_cache.py` | One-time pass: replays each day through C++ engine, saves tick arrays to `.npz`. Prevents look-ahead by design. |
| `PrecomputedEnv` | Tick-level gymnasium env (54-dim obs). Loads from `.npz` cache. Supports `step_interval`. |
| `BarLevelEnv` | Aggregates ticks into N-tick bars via `aggregate_bars()`. 21-dim obs. Current winning approach. |
| `MultiDayEnv` | Wraps multiple single-day envs. Supports `cache_dir`, `bar_size`, `train_days` split. |
| `train.py` | Training entry point. SB3 PPO with SubprocVecEnv, VecNormalize, TensorBoard logging. |

---

## 5. Observation Space

### Tick-level (54 dims, K=10 book depth)

| Index | Feature |
|-------|---------|
| 0-9 | Bid prices (relative to mid, normalized) |
| 10-19 | Bid sizes (normalized) |
| 20-29 | Ask prices (relative to mid, normalized) |
| 30-39 | Ask sizes (normalized) |
| 40 | Spread (normalized) |
| 41 | Imbalance [-1, 1] |
| 42 | Time remaining in session [0, 1] |
| 43 | Current position {-1, 0, 1} |
| 44-53 | Temporal features (EMA imbalance, returns, volatility, trade intensity, book pressure at multiple horizons) |

### Bar-level (21 dims) — current winning approach

| Index | Feature |
|-------|---------|
| 0-3 | Mid price OHLC (normalized relative to bar open) |
| 4-7 | Spread OHLC |
| 8-11 | Imbalance OHLC |
| 12 | Volume (tick count in bar) |
| 13 | VWAP (volume-weighted avg mid) |
| 14 | Bar duration (seconds) |
| 15-18 | Cross-bar temporal features (EMA mid return, EMA spread, EMA imbalance, EMA volatility) |
| 19 | Time remaining in session [0, 1] |
| 20 | Current position {-1, 0, 1} |

Bar-level with `bar_size=1000` significantly outperforms tick-level and finer bars.

---

## 6. Action Space

Discrete, 3 actions:

| Action | Meaning |
|--------|---------|
| 0 | Target short (-1) |
| 1 | Target flat (0) |
| 2 | Target long (+1) |

Agent outputs target position. Environment executes market orders to reach target.

---

## 7. Reward

| Mode | Description | Status |
|------|-------------|--------|
| PnLDelta | Change in mark-to-market PnL per step | Base reward, used in best config |
| PnLDeltaPenalized | PnL - λ\|position\| (inventory penalty) | Implemented but rejected — unnecessary once entropy is healthy |
| Execution cost | Half-spread cost applied on each trade | Active in best config (`--execution-cost`) |
| Participation bonus | Small reward for being in a position | Rejected — agent exploits it |

Reward is computed in C++ (PnLDelta/PnLDeltaPenalized). Execution cost is applied in the Python wrapper.

---

## 8. Overfitting Mitigation

| Strategy | Implementation |
|----------|----------------|
| Temporal split | 170 train / 40 val / 40 test days (strict chronological) |
| Walk-forward | Train on rolling windows, validate on next window, final eval on held-out test |
| Precomputed cache | Cache prevents look-ahead by design (audited clean) |
| VecNormalize | Separate normalization stats per split (train stats frozen for val/test) |
| Feature simplicity | Start minimal, add features only if they help OOS |
| Entropy regularization | `ent_coef=0.05` prevents policy collapse |
| Baseline | Must beat buy-and-hold and random on test set |

---

## 9. Current Best Configuration

From hyperparameter sweep (7 configs, 2M steps each):

| Parameter | Value |
|-----------|-------|
| `bar_size` | 1000 |
| `ent_coef` | 0.05 |
| `learning_rate` | 1e-3 |
| `policy_arch` | 256, 256 |
| `activation` | ReLU |
| `execution_cost` | enabled |
| `total_timesteps` | 2,000,000 |

**Results:** Return 139.5, entropy -0.48 (stable), explained_var 0.98, 21/21 days positive. Note: these results are almost entirely in-sample (only 1 OOS day in the 21-day dataset). Proper OOS validation requires the larger dataset.

---

## 10. Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1. MVP | **DONE** | C++ engine, Python envs, training pipeline, precompute cache, bar-level env, hyperparameter sweep. 489 C++ + 1001 Python tests (1490 total). |
| 2. Validation | **NEXT** | Ingest 250-day dataset, cache, retrain, proper OOS eval. |
| 3. Iteration | Planned | If OOS fails: iterate on features, architecture, reward shaping. |
| 4. Paper Trading | Planned | Live data feed, paper trading loop, monitoring, risk controls. |
| 5. Live | Planned | Real money, small size, after months of stable paper trading. |

### Phase 2: Validation (current)

1. Download and ingest ~250 days of /MES MBO data (Jan 2022 – Jan 2023)
2. Update `data/mes/manifest.json`, precompute cache for all days
3. Retrain winning config (possibly 5–10M steps for larger dataset)
4. Evaluate on 40-day val set, tune hyperparameters if needed
5. Final eval on 40-day test set (touch once)
6. Walk-forward validation across multiple windows

### Phase 3: Iteration (if OOS underperforms)

- **Feature engineering:** order flow, volume profiles, multi-horizon imbalance, regime indicators
- **Architecture:** LSTM/GRU for temporal patterns, attention mechanisms
- **Reward shaping:** Sortino-based rewards, drawdown penalties
- No strong prior — let OOS diagnostics guide direction

### Phase 4: Paper Trading

- Live Databento data feed → real-time bar construction → agent inference
- Paper trading loop with simulated fills
- Monitoring dashboard (P&L, position, drawdown, Sortino)
- Risk controls (max position, daily loss limit, kill switch)

### Phase 5: Live Trading

- Real money with small size (<10 contracts)
- Only after months of stable paper trading performance
- Broker connectivity (likely via CME FIX or broker API)

---

## 11. Success Criteria

| Phase | Criterion |
|-------|-----------|
| Phase 2 | Positive Sortino ratio on 40-day OOS test set |
| Phase 3 | Consistent positive Sortino across walk-forward windows |
| Phase 4 | Stable paper-trading P&L over 2+ months |
| Phase 5 | Real P&L covers transaction costs with acceptable drawdowns |

---

## 12. Lessons Learned

Key findings from the MVP phase and hyperparameter sweep:

- **Entropy collapse is the #1 failure mode.** `ent_coef=0.05` fixes it. Without sufficient entropy regularization, the agent collapses to always-flat.
- **Coarser bars outperform fine bars.** `bar_size=1000` >> `bar_size=500` >> `bar_size=200`. Finer bars are too noisy for the agent to learn from.
- **Higher learning rate needed for short runs.** `lr=1e-3` learns effectively in 2M steps; `lr=1e-4` is too slow; default `lr=3e-4` is mediocre.
- **Participation bonus is exploitable.** The agent games it by holding positions regardless of signal. Removed.
- **Inventory penalty is unnecessary.** Once entropy is healthy, the agent naturally manages inventory without explicit penalty.
- **Apple silicon CPU is sufficient.** ~5,650 FPS, ~7 min per 2M-step run. GPU acceleration is not needed at current scale.
- **Precomputed cache is essential.** Eliminates C++ replay overhead during training, enables fast iteration.
