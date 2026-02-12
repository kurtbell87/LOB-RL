# Survey: Does varying bar size B with recalibrated R produce positive BSS on barrier prediction using logistic regression?

## Prior Internal Experiments

### Directly Relevant

1. **exp-005 (CONFIRMED):** Null calibration at B=500, a=20, b=10. ȳ_long=0.320, ȳ_short=0.322. Both within 1.3pp of the Gambler's Ruin theoretical 1/3. Timeout rate <0.5%. Mean race duration ~12 bars. Constant Brier baseline established at ~0.218.

2. **exp-006 (REFUTED):** Signal detection at B=500, a=20, b=10 with 220-dim features. LR BSS = -0.0007 (long), -0.0003 (short). GBT BSS = -0.0028 (long), -0.0019 (short). All negative. Best p-value 0.751. LR is indistinguishable from the constant predictor; GBT actively overfits. 454K samples, 248 sessions, chronological 149/50/49 split.

3. **exp-007 (REFUTED):** Sequence models (LSTM, Transformer) on per-bar features at B=500. All 8 (model, label, split) pairs negative BSS. Best: transformer/short BSS = -0.0004 (p=0.791). Transformer collapses to near-constant predictions. LSTM worse than LR.

4. **exp-004 (INCONCLUSIVE):** 22-feature diagnostic. RF 49.6% vs 9-feature 47.5% on 50K subsample. Both beat 38.1% baseline but experiment was aborted.

5. **T6 (CONFIRMED weak signal):** Bidirectional framing {long, short, flat}. MLP 39.3% / RF 40.5% vs 34.5% baseline. +5pp accuracy but does NOT translate to calibrated BSS (exp-006 showed this).

6. **pre-001 hyperparam sweep:** On old 21-dim BarLevelEnv RL system, bar_size=1000 > 500 > 200 for in-sample return. But these were RL training returns, not BSS on barrier prediction, and at a different observation space.

### Key Insight from Prior Work

Six model families (LR, GBT, MLP, RF, LSTM, Transformer) all fail to beat constant Brier at B=500, a=20, b=10. The current feature set contains weak discriminative signal (+5pp accuracy) but zero calibrated probabilistic signal (BSS < 0). **All prior testing has been at a single (B, R) configuration.** The bar-size sweep is the designated next step per the Asymmetric First-Passage Trading plan (Phase 2b).

## Current Infrastructure

### Precompute Pipeline
- **Script:** `scripts/precompute_barrier_cache.py` with CLI args `--bar-size`, `--a`, `--b`, `--t-max`, `--lookback`, `--workers`
- **Backend:** C++ `lob_rl_core.barrier_precompute()` binding. ~50-100x faster than Python. 248 sessions in ~10 min with 8 workers.
- **Output:** Per-day `.npz` files in a configurable output directory. Each contains: features (N, 220), bar_features (N, 22), labels, short_labels, OHLCV bars, trade prices/sizes, summary stats.
- **Current cache:** `cache/barrier/` = B=500, a=20, b=10, t_max=40, lookback=10. 248 files, 221 MB, 454K usable bars.
- **Separate caches needed:** Different (B, a, b) combinations require separate precompute runs to separate directories.

### Signal Detection Pipeline
- **Function:** `signal_detection_report(X, Y_long, Y_short, session_boundaries, seed=42)` in `first_passage_analysis.py`
- **Models:** LR (L2 C=1.0, lbfgs) and GBT (LightGBM 200 trees, max_depth=6)
- **Metrics:** Brier scores, BSS, bootstrap p-values (block_size=50, n_boot=1000), calibration curves, 5-fold expanding CV
- **Split:** Chronological temporal 60/20/20 by session (149/50/49)
- **Runtime:** ~74 sec for full report (2 models x 2 labels + CV + bootstrap)

### Null Calibration Pipeline
- **Function:** `null_calibration_report(Y_long, Y_short, tau_long, tau_short, timeout_long, timeout_short, session_boundaries)`
- **Checks:** ȳ ∈ [0.28, 0.38], sum ≈ 2/3, P(1,1) ≈ 0, timeout rate, temporal stability

### Data Loading
- **Function:** `load_binary_labels(cache_dir, lookback=10)` returns X (N, 220), Y_long, Y_short, session_boundaries, etc.
- **Works with any cache directory.** Just point it at a different `cache_dir`.

### Raw Data
- 312 `.mbo.dbn.zst` files in `data/mes/` (57 GB, Jan-Dec 2022, 249 trading days after roll calendar filtering)
- Roll calendar at `data/mes/roll_calendar.json`

## Known Failure Modes

1. **GBT overfits at every (B, R) scale.** In exp-006, GBT was consistently worse than LR (more expressive = more noise fitting). **Recommendation: use LR only for the sweep.** GBT adds compute cost and risk of false positive from overfitting.

2. **Accuracy != BSS.** T6's +5pp accuracy did not translate to positive BSS (exp-006). The FRAME agent should use BSS as the primary metric, not accuracy. Accuracy is misleading on near-balanced binary tasks where the constant predictor is very strong.

3. **Small session instability.** Thanksgiving (20221124, 38 bars at B=500) caused C5 failure in exp-005. At larger bar sizes (B=2000), many sessions will have far fewer bars, and some may become degenerate. The experiment should track and report how many usable sessions and total bars exist at each B.

4. **Timeout rate sensitivity.** At B=500 with a=20, b=10, timeout rate was 0.46% — negligible. At different (B, R) combinations, timeout rate could increase substantially, especially if R is too large relative to per-bar volatility. If timeout > 5%, the barrier calibration is too aggressive.

5. **Cache skip bug.** The precompute script skips files that already exist in the output directory. When sweeping, each (B, a, b) combination MUST go to a separate directory, or existing files will be incorrectly reused.

6. **`n_features` version check.** The cache loader checks `n_features == N_FEATURES (22)`. This should be consistent across all bar sizes since features are computed identically.

## Key Codebase Entry Points

| File | Function | Role |
|------|----------|------|
| `scripts/precompute_barrier_cache.py` | `main()`, `process_session()` | CLI precompute with `--bar-size`, `--a`, `--b` |
| `python/lob_rl/barrier/first_passage_analysis.py` | `load_binary_labels()` | Load any barrier cache |
| `python/lob_rl/barrier/first_passage_analysis.py` | `signal_detection_report()` | LR + GBT Brier analysis |
| `python/lob_rl/barrier/first_passage_analysis.py` | `null_calibration_report()` | ȳ verification |
| `python/lob_rl/barrier/first_passage_analysis.py` | `fit_logistic()` | L2 logistic regression |
| `python/lob_rl/barrier/label_pipeline.py` | `compute_labels()` | Barrier label computation (a, b, t_max params) |
| `python/lob_rl/barrier/feature_pipeline.py` | `compute_bar_features()` | 22-feature extraction |
| `python/lob_rl/barrier/feature_pipeline.py` | `normalize_features()` | z-score normalization (window=2000) |
| `python/lob_rl/barrier/__init__.py` | `TICK_SIZE = 0.25`, `N_FEATURES = 22` | Constants |

## Architectural Priors

This is not a model architecture question — it's a **data configuration sweep**. The model is fixed (logistic regression, which exp-006 showed matches or exceeds all more complex models on BSS). The sweep variables are:

1. **Bar size B** — controls the aggregation granularity. Smaller B = more bars per session, smaller per-bar price moves, more noise. Larger B = fewer bars, larger moves, potentially better signal-to-noise.

2. **Risk unit R** — controls barrier width. Must be calibrated to the per-bar volatility at each B. The plan specifies R = {1x, 2x, 3x} median bar range. At B=500 with a=20, b=10 (i.e., R=b=10 ticks), R ≈ 2x median bar range (from the exp-005 analysis showing mean race duration of ~12 bars).

The 2:1 reward:risk asymmetry (a = 2R, b = R) is fixed — it produces the ȳ ≈ 1/3 null from Gambler's Ruin.

## External Context

**This is a well-studied question in quantitative finance.** The key insights:

1. **Bar size trades off resolution vs noise.** Smaller bars have more samples but each individual prediction is noisier. Larger bars aggregate more information but have fewer samples and longer label horizons. There is typically a "sweet spot" that depends on the instrument's microstructure.

2. **The signal-to-noise ratio in barrier prediction is generally very low.** Market microstructure predictability is concentrated at very short horizons (sub-second for MES) and at longer horizons (multi-day). The intermediate range (hundreds to thousands of trades per bar) is where microstructure signal has decayed but macro signal hasn't accumulated — a known "dead zone."

3. **Calibrating R to bar volatility is standard practice.** Using median bar range as the scaling unit ensures the barriers are "fair" relative to the price dynamics at each scale. If R is too small, everything resolves quickly (many samples but trivial predictions near 1/3). If R is too large, many races timeout (data waste) and the remaining ones may be too long-horizon for intraday features.

4. **Logistic regression is the right diagnostic tool.** It's the minimum-complexity model for signal detection. If LR can't beat the constant Brier score, the signal doesn't exist in linear form. More complex models (GBT, neural nets) will only overfit worse, as demonstrated in exp-006/007.

## Constraints and Considerations

### Compute
- **Precompute:** ~10 min per (B, R) combination with 8 workers. 4 bar sizes x 3 R calibrations = 12 combinations = ~2 hours precompute.
- **Signal detection:** ~74 sec per (B, R) combination. 12 combinations = ~15 min evaluation.
- **Total: ~2.5 hours wall clock.** Well within local compute budget.
- **Disk:** Each cache is ~150-300 MB depending on bar count. 12 caches = ~2-4 GB. Manageable.

### Data Limitations
- At B=2000, sessions will have ~4x fewer bars than B=500. A typical session at B=500 has ~1800 bars. At B=2000, this drops to ~450. With lookback=10, usable bars per session ≈ 440. 248 sessions x 440 ≈ 109K total samples (vs 454K at B=500). Statistical power decreases.
- At B=200, sessions will have ~2.5x more bars (~4500/session). 248 x 4500 ≈ 1.1M samples. More power but noisier features.

### R Calibration
- The plan says calibrate R to {1x, 2x, 3x} median bar range for each B.
- Currently a=20, b=10 at B=500. Need to empirically measure median bar range at each B to compute R values.
- R must be an integer number of ticks (lattice constraint from the Gambler's Ruin null). For TICK_SIZE=0.25, R values are always integers when measured in ticks.
- The 2:1 asymmetry means a = 2*R_ticks, b = R_ticks for each R calibration.

### Key Risk
- **The most likely outcome is that no (B, R) combination produces positive BSS.** The 22-feature set has already been tested across 6 model families at B=500. The features may simply lack the signal-to-noise ratio needed for calibrated barrier prediction at any intraday scale. This would be a legitimate negative result that closes out Phase 2b of the research plan and pivots the investigation to different features, targets, or approaches.

## Recommendation

The FRAME agent should design a **single experiment** with the following structure:

1. **Pre-step: Measure median bar range at each B ∈ {200, 500, 1000, 2000}.** This requires either loading existing data or doing a quick precompute pass. The existing B=500 cache has bar OHLCV data that gives median range directly. For other bar sizes, a lightweight measurement pass could sample a few sessions.

2. **For each B, compute R = {1x, 2x, 3x} median bar range (in ticks), then set a = 2R, b = R.** This produces up to 12 (B, R) configurations. Skip B=500/R=2x since it's already tested as exp-006.

3. **Precompute barrier caches** for each configuration to separate directories (e.g., `cache/barrier_B200_R1x/`).

4. **Run `signal_detection_report()` using LR only** (not GBT — it's been shown to overfit and add no diagnostic value). Report BSS, bootstrap p-value, and calibration for each (B, R, label).

5. **Success criterion: any (B, R, label) combination achieves BSS > 0 with p < 0.05.** Secondary: BSS >= 0.005 for meaningful magnitude.

6. **Also run `null_calibration_report()` at each (B, R)** to verify ȳ ≈ 1/3 holds across scales (sanity check that the Gambler's Ruin null is scale-invariant for MES).

**Key simplification:** Use LR only. GBT is strictly dominated by LR on BSS in all prior experiments. Dropping GBT halves the compute with no information loss.

**Key risk mitigation:** Pre-commit to reporting all 12+ (B, R) results regardless of outcome. No cherry-picking. If the best BSS across all configurations is still negative, this is a clear negative result that closes Phase 2b.

**Scope guidance:** This is a "sweep" experiment, not 12 separate experiments. A single experiment spec with one hypothesis ("At least one (B, R) configuration produces positive BSS with LR") and one success criterion is appropriate. The sweep grid is the independent variable, not 12 independent hypotheses.
