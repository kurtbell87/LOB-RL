# Survey: Does the observation space contain any predictive signal?

## Prior Internal Experiments

Nine experiments have been completed, all producing negative OOS returns. The pattern is remarkably consistent:

| Experiment | Val Return | Test Return | Key Insight |
|---|---|---|---|
| pre-001 (hyperparam sweep) | N/A | N/A | bar_size=1000, ent=0.05, lr=1e-3 are optimal in-sample |
| pre-002 (chrono split) | -53.8 | -36.6 | First OOS failure — massive overfit |
| pre-003 (shuffle split MLP) | -51.5 | -62.5 | Regime shift not the explanation |
| pre-004 (frame-stack local) | -48.4 | -50.2 | Marginal temporal context helps nothing |
| pre-005 (LSTM GPU) | -36.7 | -33.4 | Best OOS — but still deeply negative |
| pre-006 (MLP GPU 5M) | -62.9 | -44.0 | More steps = more memorization |
| pre-007 (frame-stack GPU) | -82.3 | -49.4 | Frame-stacking is actively harmful |
| exp-001 (199 train days) | -59.95 | -41.99 | Data quantity NOT the bottleneck |
| exp-002 (no exec cost) | -4.43 | -5.03 | Exec cost explains ~35pt of loss but gross alpha still slightly negative |

**The most critical finding:** exp-001 showed MLP val return is identical at 20d (-75.82) and 199d (-75.53) despite explained_variance dropping from 0.99 to 0.21. The model learns very different things with different amounts of data, yet OOS performance is unchanged. This strongly suggests the observation space lacks exploitable signal.

**The closest to positive OOS:** exp-002 Run C (199d, no exec cost, 2M steps MLP) showed val +10.93 but was catastrophically undertrained (explained_variance=0.174). This is the only configuration that ever produced a positive val return, but it's unreliable evidence — the model hadn't converged.

## Current Infrastructure

### What exists and is relevant:
- **Supervised diagnostic script**: `scripts/supervised_diagnostic.py` — already built to test MLP capacity on oracle-label classification. Uses 53-dim tick-level features from `PrecomputedEnv`. **Has never been run.** Claims `--bar-size` support in `scripts/README.md` but does NOT actually implement bar-level features — it only operates on tick-level (53-dim) obs.
- **Bar-level env**: `python/lob_rl/bar_level_env.py` — 21-dim obs, the representation used in all recent training. The obs-to-label pipeline would need to extract features from this env.
- **Bar aggregation**: `python/lob_rl/bar_aggregation.py` — converts tick-level to bar-level features (13 intra-bar).
- **Temporal features**: `bar_level_env.py:_precompute_temporal()` — 7 cross-bar temporal features.
- **Cached data**: 249 `.npz` files in `cache/mes/` (all 2022 trading days). Each has `obs`, `mid`, `spread`, `instrument_id`.
- **VecNormalize**: Applied during RL training (normalizes obs and reward). The supervised diagnostic applies its own standardization (z-score on training data).
- **Train/val/test split**: Implemented in `train.py` via `--shuffle-split --seed 42`. With 20 train days: 5 val, 224 test. With 199 train days: 5 val, 45 test.

### The gap:
The supervised diagnostic needs to work at **bar-level (21-dim)** to match RL training, not tick-level (53-dim). The script would need modification or a new version. However, the tick-level test is also informative — if 53-dim tick features lack signal, 21-dim bar features (which are derived from the same underlying data by lossy aggregation) cannot have *more* signal.

## Known Failure Modes

1. **Entropy collapse on 199-day runs.** All 199d runs collapsed below -0.60 entropy. ent_coef=0.05 is insufficient at 199d. This means the agent converges to a nearly deterministic policy before finding useful patterns.

2. **Seed sensitivity.** LSTM 199d showed a 37-point val return swing between seeds 42 and 43. Any single-seed result on 5 val episodes is unreliable. Standard error of val mean ≈ val_std / sqrt(5) ≈ 18 points.

3. **LSTM lr=1e-3 is too aggressive.** clip_fraction 40-50%, approx_kl rising toward 0.3. The LSTM may never have been properly trained — all LSTM results may be contaminated by training instability.

4. **VecNormalize running statistics.** Computed across all training days. Could leak distributional information (P1 question, unresolved). This is a confound for both RL and supervised experiments — if the supervised test uses per-day z-scoring, it tests the raw feature quality; if it uses cross-day normalization, it tests what the RL agent actually sees.

5. **Forced flatten cost always charged.** Even without `--execution-cost`, `compute_forced_flatten()` charges spread/2 * |position| on the terminal bar. This is a ~$0.625 systematic negative bias per episode (~23 bars), which is small but nonzero.

6. **Oracle label at bar level is ambiguous.** At tick level, the oracle is clear: look at next-tick mid_delta. At bar level, the oracle is next-bar mid_close delta. But within a bar, the path matters (a bar can open up and close down). The oracle action depends on whether we define "signal" as predicting bar close direction, or predicting within-bar optimal entry/exit.

## Key Codebase Entry Points

| File | Relevance |
|---|---|
| `python/lob_rl/bar_level_env.py` | 21-dim obs construction. `_precompute_temporal()` for temporal features. `_build_obs()` for per-step observation assembly. |
| `python/lob_rl/bar_aggregation.py` | `aggregate_bars()` — tick→bar aggregation. 13 intra-bar features: bar_return, bar_range, bar_volatility, spread_mean, spread_close, imbalance_mean, imbalance_close, bid_volume_mean, ask_volume_mean, volume_imbalance, microprice_offset, time_remaining, n_ticks_norm. |
| `python/lob_rl/bar_level_env.py:61-118` | 7 temporal features: return_lag1, return_lag3, return_lag5, cumulative_return_5, rolling_vol_5, imb_delta_3, spread_delta_3. |
| `python/lob_rl/_reward.py` | `compute_step_reward()` — reward = position * (mid_now - mid_prev). Oracle label derivable from `mid_close[t+1] - mid_close[t]`. |
| `scripts/supervised_diagnostic.py` | Existing supervised diagnostic (tick-level only). Template for bar-level version. Key methods: `load_day_features()`, `overfit_small_batch()`, `train_and_evaluate()`. |
| `scripts/train.py:86-178` | `evaluate_sortino()` — evaluation loop. Shows how VecNormalize is loaded and applied during eval. |
| `python/lob_rl/_obs_layout.py` | C++ observation layout. 43 tick-level features: 10 bid prices, 10 bid sizes, 10 ask prices, 10 ask sizes, rel_spread, imbalance, time_left. |

## Architectural Priors

### The 21-dim bar-level feature set

The observation consists of:

**13 intra-bar features (from `aggregate_bars()`):**
0. `bar_return` — (close - open) / open
1. `bar_range` — (high - low) / open
2. `bar_volatility` — std(mid) / open
3. `spread_mean` — mean bid-ask spread
4. `spread_close` — closing spread
5. `imbalance_mean` — mean order book imbalance
6. `imbalance_close` — closing imbalance
7. `bid_volume_mean` — mean total bid depth
8. `ask_volume_mean` — mean total ask depth
9. `volume_imbalance` — (bid - ask) / (bid + ask) mean
10. `microprice_offset` — microprice / mid - 1 at close
11. `time_remaining` — fraction of session left
12. `n_ticks_norm` — ticks in bar / bar_size (usually 1.0)

**7 cross-bar temporal features (from `_precompute_temporal()`):**
13. `return_lag1` — bar_return[t-1]
14. `return_lag3` — bar_return[t-3]
15. `return_lag5` — bar_return[t-5]
16. `cumulative_return_5` — sum of bar_return[t-5:t]
17. `rolling_vol_5` — std of bar_return[t-5:t]
18. `imb_delta_3` — imbalance_close[t] - imbalance_close[t-3]
19. `spread_delta_3` — spread_close[t] - spread_close[t-3]

**1 agent state:**
20. `position` — current position (-1, 0, +1)

### What's notable about these features:

1. **All are derivatives of price and order book state.** No volume-clock features, no trade-flow (aggressor) information, no higher-frequency microstructure signals. The bar aggregation discards the tick-level order flow information (which 43-dim obs contained via 10-deep bid/ask sizes).

2. **Temporal features are backward-looking only.** return_lag1/3/5, cumulative_return_5, rolling_vol_5 — these are known quantities at decision time. No forward-looking leakage, which is good. But they're also low-information — past returns in liquid futures are notoriously poor predictors of future returns (weak-form efficiency).

3. **Imbalance features may carry some signal.** Order book imbalance (bid vs ask depth) has been documented in microstructure literature as a short-horizon predictor of price direction. But at bar_size=1000 (aggregating ~1000 MBO events into one bar), the predictive horizon of order book imbalance (~1-10 events) has long expired. The imbalance at bar close tells you about LOB state at one instant, but the next bar is 1000 events away.

4. **Microprice offset is potentially informative.** The microprice (size-weighted mid) diverges from the mid when the book is asymmetric. This is the single most promising feature for short-term prediction. But again, its predictive power is on the timescale of seconds, not ~1000 events.

5. **The feature set lacks trade-flow information.** No aggressor imbalance (who is crossing the spread), no trade-arrival rate, no volume-weighted average price. These are commonly used in HFT/microstructure prediction. The tick-level C++ obs has depth information but no trade-direction data.

6. **The bar_size=1000 aggregation is very aggressive.** With ~350,000 MBO events per day and ~348 bars, each bar spans ~1000 events. This is a long horizon for microstructure signals. The aggregation was chosen for RL efficiency (fewer steps per episode = faster training) but may destroy the signal that exists at higher frequency.

### Architecture fitness for this feature set:

- **MLP 256x256 is appropriate** for 21-dim tabular input. No spatial or graph structure to exploit. The question is whether the *features* contain signal, not whether the *model* can learn it.
- **LSTM may add value** if there are multi-bar temporal patterns (momentum, mean-reversion regimes). But LSTM training has been unstable (lr too high, clip fraction 40-50%).
- **The 21-dim representation is information-impoverished.** The raw tick-level data has 43 features per tick × ~1000 ticks per bar = ~43,000 data points per bar, compressed to 13 features. This is a 3,300:1 compression ratio. If signal exists in the tick-level structure (e.g., specific order flow patterns within a bar), it is lost.

## External Context

### What practitioners generally find:

1. **Short-term return prediction in liquid futures is extremely hard.** MES is one of the most liquid, efficiently priced instruments in the world. Academic literature consistently finds that simple features (past returns, book imbalance) have very weak predictive power at horizons > a few ticks.

2. **Order book imbalance has documented predictive power at ~1-10 tick horizons** (Cont, Kukanov, Stoikov 2013; Cartea, Jaimungal, Penalva 2015). At bar_size=1000 (~minutes of real time), this signal has decayed to near-zero.

3. **Trade flow (aggressor) information is typically more informative than standing orders.** The current feature set does not include any trade-flow information — it only sees the LOB state, not who is crossing the spread. This is a major gap.

4. **Feature engineering dominates model architecture** for this problem class. The consensus in quantitative finance is that better features (e.g., order flow imbalance, trade-arrival intensity, realized volatility at multiple horizons) matter far more than model complexity.

5. **Supervised classification of direction is the standard diagnostic.** The approach outlined in QUESTIONS.md (supervised classifier > 55% accuracy → signal exists) is standard practice. 55% is a reasonable threshold — in a 3-class problem with majority class ~40-50%, 55% represents meaningful improvement over the baseline.

6. **A supervised classifier on the same features cannot find signal that doesn't exist.** But it CAN find signal that RL fails to exploit (due to reward shaping, credit assignment, exploration failures). The diagnostic cleanly separates "does signal exist" from "can RL exploit it."

## Constraints and Considerations

1. **Compute budget:** The supervised diagnostic is CPU-only. With 249 days of cached data, feature extraction and MLP training should take < 30 minutes locally.

2. **Two-level test needed:** Both tick-level (53-dim, existing script) and bar-level (21-dim, needs implementation) should be tested. The tick-level test establishes an upper bound on signal; the bar-level test matches the RL agent's actual input.

3. **Oracle label definition matters:** For bar-level, the natural oracle is: given obs at time t, predict direction of mid_close[t+1] - mid_close[t]. Three classes: short (negative), flat (within half-spread), long (positive). The "flat" class for the exec-cost oracle should include moves smaller than half-spread (unprofitable to trade). Without exec cost, flat = no move.

4. **Multiple granularities should be tested.** If signal exists at bar_size=200 but not bar_size=1000, the issue is aggregation granularity, not feature quality. Testing bar_size=200, 500, 1000 would be informative.

5. **The existing `supervised_diagnostic.py` is tick-level only.** Despite `scripts/README.md` claiming `--bar-size` support, the code doesn't implement it. A bar-level version or extension is needed.

6. **Feature importance analysis would be additive.** Beyond accuracy, reporting per-feature mutual information or permutation importance would identify which features (if any) carry signal. This helps prioritize feature engineering.

7. **The position feature (dim 20) should be excluded** from the supervised test, since it's agent state, not market state. The supervised classifier should use 20-dim market features only.

## Recommendation

The FRAME agent should design an experiment that:

1. **Runs the supervised diagnostic at both tick-level (53-dim) and bar-level (20-dim market features)** on the cached data. Use the existing `supervised_diagnostic.py` as template.

2. **Tests multiple bar sizes** (200, 500, 1000) to isolate whether signal exists at higher frequency and is destroyed by aggregation.

3. **Uses both oracle definitions:**
   - With exec cost (3-class: short, flat, long; flat = |mid_delta| < half_spread)
   - Without exec cost (2-class: short, long; or 3-class with flat = zero move)

4. **Reports clear metrics:**
   - Majority-class baseline accuracy
   - Train accuracy (can the model memorize?)
   - Test accuracy (does it generalize?)
   - Delta over baseline (the key number)
   - Per-feature permutation importance on the test set

5. **Uses proper cross-validation:** Train on first 199 days (chronological or shuffle-split), test on remaining 50. Use 3+ random seeds to assess variance.

6. **Pre-commits a clear decision gate:** If test accuracy > 55% at any granularity → signal exists, RL is failing. If test accuracy ≈ baseline at all granularities → features lack predictive power, need richer features before RL can succeed.

This is the highest-leverage experiment in the program. It takes < 1 hour, requires no GPU, and determines whether all future RL experiments are addressing the right problem (RL training) or the wrong one (feature quality).
