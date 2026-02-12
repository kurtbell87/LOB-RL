# Survey: Does B=2000 with realistic barrier sizes R ∈ {10,20,30,40} ticks produce positive BSS for logistic regression on barrier prediction?

## Prior Internal Experiments

### Directly relevant: exp-008 (bar-size sweep) — REFUTED

exp-008 tested B=2000 with R ∈ {12, 24, 36} ticks (calibrated as {1x, 2x, 3x} × median_bar_range = 12 ticks). All six B=2000 cells had negative BSS:

| R (ticks) | a | b | BSS_long | BSS_short | p_long | p_short | N_samples |
|-----------|---|---|----------|-----------|--------|---------|-----------|
| 12 | 24 | 12 | -0.0032 | -0.0028 | 0.999 | 0.995 | 108,449 |
| 24 | 48 | 24 | -0.0059 | -0.0019 | 0.990 | 0.769 | 108,449 |
| 36 | 72 | 36 | **-0.0114** | +0.0021 | 1.000 | 0.208 | 108,449 |

B=2000 was the **worst-performing** bar size in the sweep. BSS monotonically degrades with bar size across all (B, R) configurations. The one positive cell (R=36/short, BSS=+0.0021) is not statistically significant (p=0.208, CI spans zero).

### Overlap analysis: proposed R values vs. exp-008 tested values

The proposed experiment asks for R ∈ {10, 20, 30, 40}. Compare to exp-008's R ∈ {12, 24, 36}:

| Proposed R | Nearest tested R | Distance | a_proposed | a_tested | Overlap? |
|-----------|-----------------|----------|-----------|---------|----------|
| 10 | 12 | 2 ticks (17%) | 20 | 24 | **Partial** — below 1x median bar range |
| 20 | 24 | 4 ticks (17%) | 40 | 48 | **High** — 1.67x vs 2x median |
| 30 | 36 | 6 ticks (17%) | 60 | 72 | **High** — 2.5x vs 3x median |
| 40 | — | — | 80 | — | **None** — above exp-008's 3x max |

**Net new information:** Only R=10 (sub-1x median) and R=40 (>3x median) are genuinely novel. R=20 and R=30 are close enough to R=24 and R=36 that interpolation from existing results is reasonable (BSS is smoothly monotonic in R within each B).

### Other relevant experiments

- **exp-006 (signal detection at B=500/R=10):** LR BSS_long = -0.0007, BSS_short = -0.0003. All negative.
- **exp-007 (sequence models at B=500):** LSTM and Transformer both fail. BSS negative for all 8 cells.
- **exp-005 (null calibration at B=500):** ȳ_long = 0.320, ȳ_short = 0.322. Null holds.
- **exp-008 full sweep:** 20 of 24 cells negative. Best BSS = 0.0023 at B=200/R=4 (smallest bar size, tightest barriers, most samples). BSS degrades monotonically with B.

### Pattern at B=2000 specifically

From exp-008 metrics, the B=2000 data has:
- **108,449 samples** (fewest of any B — ~10x fewer than B=200's 1.15M)
- **248 sessions** (all pass)
- **Median bar range: 12 ticks** (sub-linear scaling: B doubles from 1000→2000, range goes 9→12)
- **ȳ ≈ 0.32** at R=12 (1x), dropping to ȳ ≈ 0.31 at R=36 (3x) — null holds
- **Timeout rates: 0.27%–0.47%** — all well below 10%
- **Mean race duration: 4.7 bars (R=12) → 34.3 bars (R=36)**

The LR model at B=2000 is **confidently worse** than constant at R=12 and R=24 (both CIs exclude zero on the wrong side). Only at R=36 is the short label borderline (CI spans zero, p=0.208).

## Current Infrastructure

### Precompute pipeline
- `scripts/precompute_barrier_cache.py` accepts `--bar-size`, `--a`, `--b`, `--t-max`, `--lookback`, `--workers`
- C++ backend (`lob_rl_core.barrier_precompute()`) provides ~50-100x speedup
- Each (B, R) config goes to a separate cache directory: `cache/barrier-b{B}-r{R}/`
- **Estimated time per config:** ~10 min with 8 workers

### Signal detection
- `first_passage_analysis.py` provides `load_binary_labels()`, `signal_detection_report()`, `fit_logistic()`, `paired_bootstrap_brier()`
- Temporal split: 60/20/20 by session (chronological)
- Block bootstrap: block_size=50, n_boot=1000
- LR: L2 regularized, C=1.0, lbfgs solver, max_iter=1000

### Existing B=2000 caches
Three caches already exist from exp-008:
- `cache/barrier-b2000-r12/` (a=24, b=12, t_max=48)
- `cache/barrier-b2000-r24/` (a=48, b=24, t_max=96)
- `cache/barrier-b2000-r36/` (a=72, b=36, t_max=100)

**New caches needed for R ∈ {10, 20, 30, 40}:**
- `cache/barrier-b2000-r10/` (a=20, b=10, t_max=40) — NEW
- `cache/barrier-b2000-r20/` (a=40, b=20, t_max=80) — NEW
- `cache/barrier-b2000-r30/` (a=60, b=30, t_max=100) — NEW
- `cache/barrier-b2000-r40/` (a=80, b=40, t_max=100, capped) — NEW

All 4 require precompute (~40 min total).

## Known Failure Modes

1. **Sample size limitation at B=2000.** Only 108K samples. This gives low statistical power — even BSS = 0.005 may not reach significance at B=2000. The bootstrap CI widths at B=2000 are ~0.001–0.002, much wider than B=200's ~0.0002. Detecting small effects is inherently harder.

2. **BSS monotonically degrades with bar size.** Established pattern across the full exp-008 sweep. B=2000 consistently produces the worst BSS at every R calibration. Moving from R={12,24,36} to R={10,20,30,40} does not change the bar size, so this structural disadvantage persists.

3. **Wider barriers → worse BSS within each B.** Also established in exp-008. R=40 (the widest proposed value) would be 3.3x the median bar range — similar territory to the worst-performing cells in the sweep.

4. **t_max cap at 100.** For R=40 (a=80, b=40), the formula t_max = 4×R = 160 exceeds the cap. Mean race duration at these wide barriers would be ~50+ bars, but t_max=100 may truncate some races. This could increase timeout rates and degrade label quality.

5. **SIGPIPE crash on long-running experiments.** exp-004 crashed twice from SIGPIPE when the parent `experiment.sh` was terminated. The experiment pipeline has been stable since, but long precompute phases are the riskiest.

## Key Codebase Entry Points

| File | Role |
|------|------|
| `scripts/precompute_barrier_cache.py` | CLI for barrier cache precompute. Args: `--bar-size`, `--a`, `--b`, `--t-max`, `--lookback`, `--workers` |
| `python/lob_rl/barrier/first_passage_analysis.py` | Signal detection: `load_binary_labels()`, `signal_detection_report()`, `fit_logistic()`, `paired_bootstrap_brier()`, `null_calibration_report()` |
| `python/lob_rl/barrier/bar_construction.py` | Bar building from MBO events |
| `python/lob_rl/barrier/feature_extraction.py` | 22-feature extraction, `N_FEATURES=22` |
| `python/lob_rl/barrier/label_pipeline.py` | Label computation, `flatten_and_bias_labels()` |
| `results/exp-008-bar-size-sweep/metrics.json` | Full sweep results (reference for cross-experiment consistency) |

## Architectural Priors

This is a supervised signal detection question, not an architecture question. The model is logistic regression (the simplest calibrated classifier). Six model families have been tested (LR, GBT, MLP, RF, LSTM, Transformer) — all fail. LR is the correct choice: it sets the tightest lower bound on linear signal.

The 22 features are computed within-bar (OHLCV, microstructure, temporal) with a 10-bar lookback. They capture local market dynamics but no cross-session or macro context.

## External Context

### What "realistic barrier sizes" means for MES
- MES tick = $1.25 (0.25 index points × $5 multiplier)
- R=10 ticks = $12.50 stop loss, $25 profit target — very tight for intraday
- R=20 ticks = $25 stop / $50 profit — moderate intraday
- R=30 ticks = $37.50 stop / $75 profit — wide intraday
- R=40 ticks = $50 stop / $100 profit — approaches swing-trade territory for MES

At B=2000 events/bar, each bar represents a substantial chunk of trading activity. The median bar range is 12 ticks. So:
- R=10 < 1x median bar range — barriers are tighter than typical per-bar movement
- R=20 ≈ 1.67x — moderate
- R=30 = 2.5x — similar to exp-008's 2x calibration
- R=40 ≈ 3.3x — beyond exp-008's 3x max

### Market microstructure literature
The efficient markets hypothesis predicts BSS ≈ 0 for any features derived from public market data. The fact that BSS < 0 (models actively worse than constant) is consistent with overfitting to noise in features that carry no signal. This is the expected outcome for well-functioning liquid markets like ES/MES.

Short-horizon microstructure predictability (sub-second) exists but is captured by high-frequency features (queue position, order flow toxicity at the microsecond level), not by bar-aggregated features at the 2000-event scale.

## Constraints and Considerations

1. **Compute:** ~40 min precompute (4 caches × 10 min), ~10 min signal detection. Local CPU only. Well within budget.

2. **Statistical power:** 108K samples at B=2000 give bootstrap CI widths of ~0.001–0.002 for Brier delta. A BSS of 0.005 corresponds to an absolute Brier delta of ~0.001, which is at the edge of detectability. This experiment may be unable to distinguish BSS=0.003 from BSS=0 at conventional significance levels.

3. **Multiple testing:** 8 new cells (4 R values × 2 labels). With Bonferroni correction at α=0.05, need raw p < 0.00625. Combined with exp-008's 24 cells, the family-wise correction across all 32 cells would be even stricter (p < 0.00156).

4. **Interpolation concern:** R=20 and R=30 are close enough to exp-008's R=24 and R=36 that the new results will likely be similar. The marginal information from these two points is low. The genuinely new points are R=10 and R=40.

## Recommendation

**This experiment has a very low probability of producing positive BSS.** The prior evidence is strong:

1. **B=2000 is the worst bar size.** BSS monotonically degrades with B across all R values.
2. **All 6 B=2000 cells in exp-008 are negative** (most significantly so).
3. **Two of four proposed R values (20, 30) overlap substantially** with already-tested R values (24, 36) that produced BSS of -0.0059 and -0.0114 (long).
4. **R=10 (the tightest proposed barrier) is sub-1x median bar range** — this is the same territory where B=200/R=4 showed BSS=0.0023 (the best result in the entire sweep), but at B=2000 the sample size is 10x smaller, eliminating the statistical power that made that result detectable.
5. **R=40 (the widest) will have the longest race durations and highest timeout rates**, placing it in the worst-performing zone of the exp-008 sweep.

**If the FRAME agent proceeds, it should:**
- Focus on R=10 and R=40 only (R=20 and R=30 add negligible new information over exp-008's R=24 and R=36)
- Set the BSS threshold at ≥ 0.005 (consistent with exp-008) — not lower
- Pre-commit to accepting the null if results match the exp-008 monotonic degradation pattern
- Consider whether this experiment changes any downstream decision: if BSS is positive at B=2000/R=10 but below 0.005, what would that motivate? The answer is likely "nothing" — the same conclusion as exp-008's B=200/R=4 result.

**Alternative recommendation:** Given that three axes (architecture, scale, data quantity) are exhausted and B=2000 was the worst performer, the FRAME agent's time would be better spent on the higher-priority open questions in QUESTIONS.md:
- **Conditional signal detection (regime-filtered)** — tests whether signal exists in specific market regimes
- **Alternative targets (1:1 barriers)** — changes the label formulation entirely
- **Feature pivot** — addresses the root cause (insufficient features) rather than re-testing at a slightly different scale
