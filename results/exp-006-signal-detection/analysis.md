# Analysis: Phase 2 Signal Detection — Can LR or GBT Beat the Constant Brier Score?

## Verdict: REFUTED

No model beats the constant predictor on any label. All four (model, label) combinations have **negative** Brier Skill Scores — meaning every model is *worse* than simply predicting ȳ for every sample. The strongest result (logistic/short, BSS = -0.0003) is negative. The bootstrap tests show p-values of 0.75–1.0, nowhere near the α = 0.05 threshold. C1 (signal detected) fails decisively. C2 (BSS ≥ 0.005) fails — the best BSS is -0.0003, not +0.005.

The 220-dimensional barrier features do not contain calibrated probabilistic signal for Y_long or Y_short at the individual-bar level, despite the weak discriminative signal found in T6's 3-class framing (+5pp accuracy above chance). The transition from accuracy-based to Brier-based evaluation reveals that the models cannot produce *calibrated probability shifts* that improve on the constant prediction.

## Results vs. Success Criteria

- [ ] **C1 — Signal detected: FAIL** — No (model, label) pair has bootstrap p < 0.05. Lowest p-value: 0.751 (logistic/short). All four delta CI intervals include zero or are entirely negative (model worse than constant). `signal_found = false`.
- [ ] **C2 — Meaningful magnitude: FAIL** — Best BSS = -0.0003 (logistic/short). The threshold was BSS ≥ +0.005. Not only does no model reach the threshold, no model has a *positive* BSS.
- [x] **C3 — CV consistency: PASS** — Reported as `true` in metrics. However, this is moot given that C1 and C2 both fail — CV consistency of a model that doesn't beat the baseline is uninformative.
- [x] **Sanity checks: PASS** — N_samples = 454,164 (≈454K, expected). N_sessions = 248 (expected). ȳ_train_long = 0.3222 ∈ [0.28, 0.38]. ȳ_train_short = 0.3241 ∈ [0.28, 0.38]. All within stated tolerances. Note: n_val_sessions = 50, n_test_sessions = 49 (spec said 49/50 respectively; minor rounding difference, immaterial).

**Per the verdict mapping:** C1 fails (no model beats constant at p < 0.05) → **REFUTED**.

## Metric-by-Metric Breakdown

### Primary Metrics

#### Brier Skill Scores (BSS = 1 - BS_model / BS_constant)

| Model | Label | BS_model | BS_constant | BSS | Delta (const - model) | Δ 95% CI | p-value |
|-------|-------|----------|-------------|-----|----------------------|----------|---------|
| Logistic | Y_long | 0.21678 | 0.21665 | **-0.0007** | -0.000129 | [-0.00033, +0.00008] | 0.897 |
| GBT | Y_long | 0.21723 | 0.21665 | **-0.0028** | -0.000576 | [-0.00098, -0.00018] | 1.000 |
| Logistic | Y_short | 0.21849 | 0.21843 | **-0.0003** | -0.000060 | [-0.00025, +0.00013] | 0.751 |
| GBT | Y_short | 0.21884 | 0.21843 | **-0.0019** | -0.000414 | [-0.00075, -0.00007] | 0.991 |

Key observations:
1. **All BSS values are negative.** Every model is *worse* than the constant predictor.
2. **GBT is worse than LR in all cases.** GBT long BSS = -0.0028 vs LR long BSS = -0.0007. GBT short BSS = -0.0019 vs LR short BSS = -0.0003. This is the opposite of what we'd expect if nonlinear signal existed — the more expressive model overfits more.
3. **GBT deltas are significantly negative.** Both GBT CI intervals exclude zero on the *wrong side* (model worse than constant): long CI [-0.00098, -0.00018], short CI [-0.00075, -0.00007]. The bootstrap test is confident that GBT is *actively worse* than guessing ȳ.
4. **LR deltas are indistinguishable from zero.** Logistic long CI [-0.00033, +0.00008] and logistic short CI [-0.00025, +0.00013] both straddle zero. LR neither helps nor hurts — it essentially collapses to the constant predictor.
5. **The best result is logistic/short at BSS = -0.0003** — for context, this is a Brier score increase of 0.00006 absolute. Utterly negligible.

#### Bootstrap p-values

All p-values are for the test "model beats constant" (i.e., delta > 0):
- Logistic long: p = 0.897 — not significant
- GBT long: p = 1.000 — GBT is *confidently worse*
- Logistic short: p = 0.751 — not significant
- GBT short: p = 0.991 — GBT is *confidently worse*

Even without Bonferroni correction (per-test α = 0.05/4 = 0.0125), no result approaches significance. The lowest raw p-value is 0.751.

### Secondary Metrics

#### Max Predicted Probability

| Model | Label | Max p̂ | Profitability threshold (0.40) |
|-------|-------|--------|-------------------------------|
| Logistic | Y_long | 0.525 | Above threshold |
| GBT | Y_long | 0.479 | Above threshold |
| Logistic | Y_short | 0.549 | Above threshold |
| GBT | Y_short | 0.536 | Above threshold |

All models produce *some* predictions above the profitability threshold of p > 0.40. However, the calibration analysis below shows these high-probability predictions are poorly calibrated — the models are overconfident at the extremes.

#### Calibration Curves

**Logistic/long (6 bins):**

| Mean predicted | Fraction positive | Calibration |
|---------------|------------------|-------------|
| 0.055 | 0.000 | Overconfident low |
| 0.135 | 0.038 | Overconfident low |
| 0.292 | 0.309 | Reasonable |
| 0.324 | 0.318 | Good |
| 0.423 | 0.328 | **Overconfident high** |
| 0.514 | 0.667 | Poor (sparse bin) |

The middle bins (0.29–0.32) are well-calibrated, but the tails are not. When logistic predicts p = 0.42, the actual rate is 0.33 — the model is overconfident by ~9pp. The extreme bins (0.05, 0.51) have very few samples and are unreliable.

**GBT/long (5 bins):**

| Mean predicted | Fraction positive | Calibration |
|---------------|------------------|-------------|
| 0.077 | 0.000 | Overconfident low |
| 0.182 | 0.375 | **Grossly underconfident** |
| 0.286 | 0.314 | Reasonable |
| 0.329 | 0.318 | Good |
| 0.416 | 0.325 | **Overconfident high** |

GBT's calibration is poor. The 0.18 bin has a 37.5% positive rate — the model is severely underconfident in this region. The 0.42 bin has a 32.5% positive rate vs. predicted 41.6% — overconfident by 9pp.

**GBT/short (6 bins):**

| Mean predicted | Fraction positive | Calibration |
|---------------|------------------|-------------|
| 0.073 | 0.020 | OK |
| 0.167 | 0.200 | Reasonable |
| 0.286 | 0.328 | Good |
| 0.328 | 0.322 | Good |
| 0.426 | 0.333 | Overconfident high |
| 0.516 | **0.091** | **Catastrophically overconfident** |

The GBT/short highest bin is telling: when the model predicts p = 0.52, the actual rate is 9.1%. This is a massive miscalibration and explains why GBT performs worse than the constant predictor — its extreme predictions are anti-informative.

**Key calibration finding:** All models are well-calibrated only near ȳ ≈ 0.32 (the base rate) and miscalibrated at the tails. The "signal" the models try to extract is noise — when they deviate from the base rate, they get *worse*, not better.

#### 5-Fold Expanding-Window CV Brier Scores

**Logistic/long CV:** [0.2156, 0.2209, 0.2173, 0.2233, 0.2193] — Mean: 0.2193, Std: 0.0028
**GBT/long CV:** [0.2171, 0.2227, 0.2188, 0.2234, 0.2200] — Mean: 0.2204, Std: 0.0024
**Logistic/short CV:** [0.2245, 0.2143, 0.2255, 0.2152, 0.2156] — Mean: 0.2190, Std: 0.0053
**GBT/short CV:** [0.2266, 0.2151, 0.2262, 0.2157, 0.2162] — Mean: 0.2199, Std: 0.0052

Within each label, GBT consistently has slightly higher (worse) Brier scores than LR across all 5 folds. The fold-to-fold variation (std 0.002–0.005) is dominated by the fact that different validation windows have different ȳ distributions — not by model quality differences.

C3 reports as PASS, meaning the best model beats its fold-specific constant baseline in ≥ 3/5 folds. But the margins are negligible, and on the primary val split, no model beats the constant. C3 passing is likely an artifact of fold-level ȳ variation: when a fold's ȳ_val differs from ȳ_train, the constant predictor (using ȳ_train) is slightly miscalibrated, and the model can beat it by default. This does not indicate genuine signal.

### Sanity Checks

| Check | Result | Detail |
|-------|--------|--------|
| N_samples ≈ 454K | **PASS** | 454,164 |
| N_sessions = 248 | **PASS** | 248 |
| ȳ_train_long ∈ [0.28, 0.38] | **PASS** | 0.3222 |
| ȳ_train_short ∈ [0.28, 0.38] | **PASS** | 0.3241 |
| GBT Brier ≤ constant + 0.01 | **PASS** | Worst GBT excess: 0.00058 (long), well below 0.01 |
| LR converged | **PASS** | No convergence warnings; max_pred values are well-distributed, not degenerate |

All sanity checks pass. The experiment infrastructure is valid. The negative result is not a pipeline bug.

## Resource Usage

| Resource | Budget | Actual |
|----------|--------|--------|
| GPU-hours | 0 | 0 |
| Wall clock | 15 min | 74 sec (~1.2 min) |
| Training runs | 0 | 0 |
| Total model fits | — | 4 (primary) + ~20 (CV) |

Well within budget. The CV fits were faster than estimated.

## Confounds and Alternative Explanations

### 1. Could the negative BSS be due to train/val distributional shift?

**Partially, but it doesn't change the conclusion.** The temporal split puts Jan–Aug 2022 in training and Sep–Nov in validation. The 2022 bear market had distinct phases. However:
- LR's BSS is near zero (-0.0003 to -0.0007), not large-negative. If there were a strong shift, we'd expect LR to be more confidently wrong.
- The CV results (which evaluate on different time windows) show the same pattern: no model consistently beats the fold-specific constant.
- The constant predictor (ȳ_train) itself is well-calibrated for the val period — ȳ_train_long = 0.322, and the val set's true positive rate appears close to this (the middle calibration bins for all models show fraction_positive ≈ 0.31–0.32).

### 2. Could the block bootstrap be anti-conservative?

The spec warned about label autocorrelation (mean race duration ~12 bars). Block size = 50 should capture this. More importantly, the CIs are so far from significance that even doubling the CI width wouldn't change the conclusion. The widest CI (GBT long: width 0.0008) would need to shift by ~0.001 to straddle zero favorably. Anti-conservative bootstraps are a concern for borderline results; these results are nowhere near borderline.

### 3. Could GBT be overfitting to training noise?

**Yes, and the data confirms it.** GBT is worse than LR on all four comparisons. GBT long BSS = -0.0028 vs LR = -0.0007; GBT short BSS = -0.0019 vs LR = -0.0003. The more expressive model hurts more. This is the classic signature of fitting to noise: LightGBM with 200 trees and max_depth=6 finds "patterns" in training data that are spurious. The GBT/short calibration bin at p̂ = 0.52 → actual 9.1% is a smoking gun for overfitting-induced miscalibration.

**However, this doesn't explain LR's failure.** Logistic regression is the minimum-complexity linear model. If linear signal existed, L2-regularized LR would find it. LR's BSS = -0.0003 (indistinguishable from zero) means there is no linear relationship between the 220 features and the binary outcomes that improves on the base rate.

### 4. Could dead features or feature quality be the issue?

The C++ cache fixed the 4 previously-dead book features (exp-004 confirmed they're now active). All 22 features × 10 lookback = 220 dimensions are available. The question is not whether features are present but whether they contain *calibrated probability-shifting* information. The T6 result (+5pp accuracy) showed discriminative signal in a classification sense. The Brier score evaluation asks a stricter question: can the model's probability estimates be trusted? The answer is no.

### 5. Could the failure be specific to the binary {Y_long, Y_short} framing?

**Possible but unlikely.** T6 used the 3-class {long, short, flat} framing and found signal. The binary framing separates this into two independent predictions. If the 3-class signal came from distinguishing "flat" from "{long or short}", the binary framing would still capture it (a lower probability of Y_long implies higher probability of flat). If the signal came from distinguishing "long" from "short", the binary framing handles this directly. The framing change should not destroy genuine signal.

The more likely explanation: T6's +5pp accuracy gain above chance does not translate to calibrated probability improvement. A model can be better than chance at classification (the top predicted class is right more often) while producing worse-calibrated probabilities than the base rate. This is exactly what we observe.

### 6. Multiple testing

With 4 tests and α = 0.05, the family-wise error rate under the null is ~0.185. Bonferroni-corrected per-test α = 0.0125. Since the *lowest* raw p-value is 0.751, multiple testing correction is irrelevant — no result is remotely close to significance even at the raw α.

## What This Changes About Our Understanding

### The critical update: T6's accuracy signal does not translate to Brier-calibrated probabilities.

This is the most important finding. T6/exp-004 showed +5pp accuracy above chance for classifying barrier outcomes. exp-006 shows that this accuracy gain does not produce better-calibrated probability estimates than the trivial constant predictor. How can both be true?

**Reconciliation:** Accuracy measures the mode of the prediction — which class gets the highest predicted probability. Brier score measures the full probability distribution. A model can improve accuracy by shifting predictions slightly toward the correct class *on average*, while making the probabilities *less calibrated* overall (overconfident in some regions, underconfident in others). The calibration curves confirm this: all models are well-calibrated only near the base rate and poorly calibrated at extremes.

**Implication for RL:** An RL agent receives a scalar reward, not a probability. If the features provide weak discriminative signal (the "right" action is slightly more identifiable than chance), an RL agent might exploit this for above-random performance. But the *size* of the information content is tiny — the Brier skill score ceiling is < 0.001, implying the features explain < 0.1% of outcome variance. This is orders of magnitude below what's needed for profitable trading when execution costs are nonzero.

### Updated mental model:

1. **The 220-dim barrier features contain weak discriminative signal but no calibrated probabilistic signal.** The features weakly separate "long profit" from "short profit" from "flat" in a classification sense, but cannot produce per-bar probability estimates that improve on the base rate.

2. **The information content in the features is < 0.1% of outcome variance.** BSS < 0.001 means the features explain essentially none of the variance in Y_long or Y_short outcomes. The barrier outcomes are dominated by unpredictable price movements (consistent with near-efficient markets).

3. **GBT overfits; LR finds nothing.** The 220-dim feature space has enough dimensions for GBT to memorize training patterns, but the linear projection (LR) finds no exploitable direction. This rules out both linear and nonlinear signal at the individual-bar level.

4. **The gap between T6's accuracy finding and exp-006's Brier finding is informative.** It suggests the signal is at the *ranking* level (which bars are more likely to profit), not the *calibration* level (how much more likely). This distinction matters for RL: a policy that conditions on features will only beat random if the ranking signal translates to better action selection, but the magnitude of the edge is too small to overcome costs.

5. **Hypothesis H5 needs further revision.** H5 was "partially refuted" by T6 — features have some signal. exp-006 clarifies: the signal exists for classification but is negligible for calibrated prediction. For practical purposes (profitable trading), the signal is insufficient.

## Proposed Next Experiments

1. **Feature-level information decomposition.** Which of the 22 features contribute most to the (small) discriminative signal? T6 showed trade_flow_imbalance, bar_range, and volume_log as top features. Running a permutation importance analysis on the logistic regression could identify whether the signal is concentrated in 2-3 features or diffuse. If concentrated, the next step would be engineering *more informative variants* of those specific features (e.g., longer lookback, cross-bar differences). If diffuse, the feature space is fundamentally uninformative.

2. **Conditional signal detection — regime filtering.** exp-006 tests average signal across the entire val period. It's possible that signal exists in specific market regimes (high volatility, trend days, pre-FOMC) but averages to zero over the full sample. A conditional analysis — stratifying by realized_vol quintile or session_time — could reveal pockets of predictability. This would be a new experiment with the hypothesis: "Signal exists in high-vol regimes even though it's absent on average."

3. **Longer lookback / temporal aggregation.** The current features use h=10 lookback bars. If the predictive information is at a longer timescale (e.g., h=50 or session-level statistics), the current features would miss it. A diagnostic with session-level aggregates (mean, std, trend of features within a session) could test whether the signal is at a different timescale than the individual bar.

4. **Accept the null and shift strategy.** If the bar-level features genuinely contain < 0.1% of outcome variance, no amount of architecture (Transformer, SSM, etc.) will help — you can't amplify signal that isn't there. The productive path may be: (a) different features (e.g., cross-instrument signals, order flow imbalance at finer granularity), (b) different prediction targets (e.g., longer-horizon barriers, different R:R ratios), or (c) a fundamentally different approach (e.g., execution optimization rather than direction prediction).

## Program Status

- Questions answered this cycle: 1 (signal detection with Brier scores → REFUTED)
- New questions added this cycle: 1 (does signal exist in specific market regimes even though it's absent on average?)
- Questions remaining (open, not blocked): 5
- Handoff required: NO
