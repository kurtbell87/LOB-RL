# Empirical Research Plan: Asymmetric First-Passage Trading

**Document Type:** Research Plan  
**Prerequisite:** Successful LEAN formalization of proof requirements (v2)  
**Audience:** Research team  
**Date:** 2026-02-11

---

## Overview

The formalization establishes three categories of results that directly constrain the empirical work:

1. **Well-defined optimization targets** — $P_{\text{long}}^{(n)} = \mathbb{E}[Y_n^{\text{long}} \mid \mathcal{G}_n]$ exists in $L^2$ and is unique (T1).
2. **Testable invariants** — Under the martingale null, $\mathbb{E}[Y_n^{\text{long}}] = 1/3$ (T2). Prediction sequences across prefix lengths form a martingale (E2).
3. **Information-theoretic bounds** — The variance decomposition (T3) and approximation hierarchy (T4) provide ceilings and stopping criteria.

This plan is structured as a sequence of phases with explicit gate criteria. Each phase answers a specific question, and the answer determines whether to proceed, iterate, or terminate. The most expensive work (sequence model training) is deliberately last.

---

## Phase 0: Data Pipeline and Label Generation

### Objective

Produce the labeled dataset: for each bar $n$, compute features $X_n \in \mathbb{R}^d$ and binary labels $Y_n^{\text{long}}, Y_n^{\text{short}} \in \{0, 1\}$.

### Inputs

- Tick-level market-by-order (MBO) data with timestamps, prices, volumes, and order book snapshots.
- Choice of instrument(s) and date range.
- Bar size parameter $B$ (start with $B = 500$ ticks as a baseline; this will be swept later).
- Risk unit $R$ (start with $R$ calibrated to roughly 2x the median single-bar range; this will also be swept).

### Tasks

**0.1 — Bar construction.** Implement deterministic tick bars: aggregate every $B$ ticks into a bar. Record $T_n = nB$. For each bar, compute the minimal feature set:

- OHLCV (open, high, low, close, volume)
- VWAP
- Trade count
- Buy/sell volume imbalance (using Lee-Ready or similar tick classification)
- Order book imbalance at best bid/ask at bar open and close

This is the baseline $X_n$. Additional features come in Phase 3.

**0.2 — First-passage labeling.** For each bar boundary $T_n$, run the first-passage race forward in tick time:

- Initialize at $S_{T_n}$.
- Scan forward tick-by-tick. Record the first time $S_t - S_{T_n} \geq 2R$ (long reward) or $S_{T_n} - S_t \geq R$ (long risk).
- Set $Y_n^{\text{long}} = 1$ if reward hit first, $0$ if risk hit first.
- Handle the timeout case: if neither barrier is hit within $T_{\max}$ ticks, label as $0$ (conservative) and flag the sample. Track the timeout rate. If it exceeds 5%, $R$ is too large relative to the volatility at this bar scale, or $T_{\max}$ is too small.
- Repeat symmetrically for short.

**0.3 — Lattice verification.** Confirm the tick size $\delta$ of the instrument. Verify that $R / \delta \in \mathbb{N}$. If not, round $R$ to the nearest valid value. This is required for T2 to hold exactly.

**0.4 — Train/validation/test split.** Split temporally, not randomly. Use the first 60% for training, next 20% for validation, final 20% for held-out test. No shuffling. Time series data requires temporal splits to prevent leakage from autocorrelation.

### Gate Criteria

- Pipeline produces labels without errors.
- Timeout rate $< 5\%$.
- Label counts are reasonable (neither $Y^{\text{long}}$ nor $Y^{\text{short}}$ is degenerate — both 0 and 1 occur with meaningful frequency).

### Deliverables

- Labeled dataset with columns: bar index $n$, features $X_n$, labels $Y_n^{\text{long}}$, $Y_n^{\text{short}}$, race termination time (ticks after $T_n$), timeout flag.
- Summary statistics: label distribution, timeout rate, mean race duration, distribution of race durations.

---

## Phase 1: Null Hypothesis Calibration

### Objective

Verify that the data is consistent with the theoretical framework before any modeling. This is a sanity check on the pipeline and a first measurement of market efficiency at this bar scale.

### Tasks

**1.1 — Marginal label frequency.** Compute $\bar{Y}^{\text{long}} = \frac{1}{N}\sum_{n=1}^N Y_n^{\text{long}}$ and $\bar{Y}^{\text{short}} = \frac{1}{N}\sum_{n=1}^N Y_n^{\text{short}}$.

T2 predicts these are approximately $1/3$ under the martingale null. Compute the standard error: $\text{SE} = \sqrt{\frac{\bar{Y}(1-\bar{Y})}{N_{\text{eff}}}}$, where $N_{\text{eff}}$ accounts for autocorrelation in the labels (use Newey-West or block bootstrap to estimate effective sample size).

**Interpretation matrix:**

| Observation | Interpretation | Action |
|------------|---------------|--------|
| $\bar{Y} \approx 1/3$ within 2 SE | Consistent with martingale null at this bar scale. Predictability, if any, is in the conditional structure. | Proceed to Phase 2. |
| $\bar{Y}$ significantly $> 1/3$ | Positive drift in $S$ relative to $R$. Either the instrument trends upward over the sample period (likely for equities), or there's a labeling bug. | Check for bugs. If clean, the deviation estimates aggregate drift magnitude. Adjust the null benchmark from $1/3$ to $\bar{Y}$. |
| $\bar{Y}$ significantly $< 1/3$ | Negative drift or bug. | Same as above. |
| $\bar{Y}$ wildly off (e.g., 0.15 or 0.55) | Almost certainly a labeling or data bug. | Stop. Debug pipeline. |

**1.2 — Non-complementarity check.** Verify L5 empirically: $\bar{Y}^{\text{long}} + \bar{Y}^{\text{short}} \neq 1$. Under the martingale null, both should be $\approx 1/3$, summing to $\approx 2/3$. If they sum to $\approx 1$, there's a labeling bug (likely computing the short label as $1 - Y^{\text{long}}$).

**1.3 — Temporal stability.** Plot $\bar{Y}^{\text{long}}$ computed over rolling windows (e.g., weekly or monthly). If the label frequency is non-stationary (drifts significantly over time), the unconditional target is moving, which complicates modeling. This is expected for equities over long horizons but should be mild over intraday or weekly scales for liquid instruments.

**1.4 — Joint label distribution.** Compute the empirical joint distribution of $(Y_n^{\text{long}}, Y_n^{\text{short}})$. Record the frequencies of all four outcomes: (1,1), (1,0), (0,1), (0,0). This is the empirical input to S1 and informs whether the output layer needs joint constraints.

### Gate Criteria

- $\bar{Y}^{\text{long}}$ and $\bar{Y}^{\text{short}}$ are each within a reasonable range of $1/3$ (or a justifiable drift-adjusted value).
- $\bar{Y}^{\text{long}} + \bar{Y}^{\text{short}} \approx 2/3$ (not $\approx 1$).
- No catastrophic non-stationarity.

### Deliverables

- Null calibration report: estimated $\bar{Y}$, confidence intervals, drift estimate.
- Joint label distribution table.
- Rolling label frequency plot.

---

## Phase 2: Signal Detection

### Objective

Determine whether bar-level features carry *any* predictive signal for the first-passage outcome. This is the critical gate. If simple models can't beat the null Brier score, the conditional probabilities are approximately constant, and no amount of model complexity will help.

### Background

Under the martingale null with constant $P_{\text{long}}^{(n)} = p$ for all $n$, the irreducible Brier score is $p(1-p)$. At $p = 1/3$, this is $2/9 \approx 0.2222$. Any model that achieves Brier score below this value is capturing conditional structure.

More precisely, for a model $\hat{f}(X_n)$:

$$\text{Brier} = \mathbb{E}[(Y_n - \hat{f}(X_n))^2] = \text{Var}(P) + \mathbb{E}[P(1-P)] - (\text{Var}(P) - \text{Var}(P - \hat{f}))$$

Wait — let's be concrete. The Brier score decomposes as:

$$\text{Brier}(\hat{f}) = \underbrace{\mathbb{E}[(P_n - \hat{f}(X_n))^2]}_{\text{model gap}} + \underbrace{\mathbb{E}[P_n(1-P_n)]}_{\text{irreducible noise}}$$

A perfect model ($\hat{f} = P$) achieves $\text{Brier} = \mathbb{E}[P(1-P)]$. A constant model ($\hat{f} = \bar{Y}$) achieves $\text{Brier} = \text{Var}(Y) = \bar{Y}(1-\bar{Y})$. The gap $\text{Var}(Y) - \text{Brier}(\hat{f})$ is the variance explained by the model, which is bounded above by $\text{Var}(P)$.

### Tasks

**2.1 — Constant baseline.** Compute the Brier score of the constant predictor $\hat{f} = \bar{Y}^{\text{long}}$. This is $\bar{Y}(1-\bar{Y})$. This is the score to beat.

**2.2 — Logistic regression.** Fit logistic regression on $X_n$ (the baseline features from Phase 0) to predict $Y_n^{\text{long}}$. Use 5-fold temporal cross-validation (time-aware splits within the training set). Report cross-validated Brier score, log-loss, and calibration curve.

**2.3 — Gradient-boosted trees (GBT).** Fit XGBoost or LightGBM on the same features. GBTs are strong baselines for tabular data and will capture nonlinear relationships that logistic regression misses. Same cross-validation protocol. Report same metrics.

**2.4 — Signal significance test.** Compute $\Delta = \text{Brier}_{\text{constant}} - \text{Brier}_{\text{model}}$ for both models. This is the variance explained. Test whether $\Delta > 0$ using a paired bootstrap test (resample bars with replacement, preserving temporal blocks, recompute $\Delta$ for each resample, check if the 95% CI excludes zero).

**2.5 — Repeat for short.** Run 2.1-2.4 for $Y_n^{\text{short}}$.

### Interpretation

| $\Delta$ | Meaning | Action |
|----------|---------|--------|
| Not significantly $> 0$ | No detectable signal at this bar scale and feature set. | Go to Phase 2b (parameter sweep) before giving up. |
| Small but significant (e.g., $\Delta \approx 0.001$) | Signal exists but is weak. | Proceed cautiously. Compute rough profitability bound (see below). If unprofitable after costs, consider different $B$ or $R$. |
| Moderate ($\Delta \approx 0.01$+) | Meaningful signal. | Proceed to Phase 3. |

**Rough profitability bound.** If the model predicts $\hat{p} > 1/3$ (adjusted for drift), a long trade has positive expected value: $\text{EV} = 2R \cdot \hat{p} - R \cdot (1-\hat{p}) = R(3\hat{p} - 1)$. For this to overcome round-trip transaction costs $C$, you need $3\hat{p} - 1 > C/R$. At $R = 10$ ticks and $C = 2$ ticks (spread + fees), you need $\hat{p} > 0.4$. If the model's predictions never exceed 0.4, signal exists but isn't tradeable. This is a rough filter — a proper Kelly analysis comes later, but it prevents wasting time on signal that can't pay for itself.

### Phase 2b: Parameter Sweep (Conditional on No Signal)

If Phase 2 finds no signal, the issue may be the bar scale or risk unit, not the features. Sweep:

- $B \in \{200, 500, 1000, 2000\}$
- $R$ calibrated to $\{1\times, 2\times, 3\times\}$ the median bar range for each $B$

For each $(B, R)$ pair, regenerate labels (Phase 0) and repeat Phase 2. If no $(B, R)$ combination shows signal, the conclusion is that single-bar MBO features at tick resolution do not predict first-passage outcomes for this instrument. This is a legitimate negative result. Document it and reconsider the information source (e.g., cross-asset features, longer-horizon conditioning, different instruments).

### Gate Criteria

- At least one $(B, R)$ configuration shows statistically significant $\Delta > 0$.
- The implied signal strength is plausibly tradeable after costs.

### Deliverables

- Signal detection report: Brier scores, $\Delta$ with confidence intervals, calibration curves.
- Parameter sweep results (if applicable).
- Go/no-go recommendation with justification.

---

## Phase 3: Feature Engineering Guided by Information Ceiling

### Objective

Determine which features carry signal and estimate how much of the theoretical information ceiling $\text{Var}(P_{\text{long}}^{(n)})$ the feature set captures. This phase uses T4 directly: the gap between $\text{Var}(\mathbb{E}[Y \mid \sigma(X_n)])$ and $\text{Var}(P)$ is the information discarded by feature engineering.

### Tasks

**3.1 — Feature group taxonomy.** Organize candidate features into groups by information source:

| Group | Features | Information Content |
|-------|----------|-------------------|
| G1: Price summary | OHLCV, bar range, returns | Bar-level price dynamics |
| G2: Volume profile | VWAP, volume distribution within bar, trade count | Intra-bar activity |
| G3: Order flow | Buy/sell imbalance, net order flow, trade arrival rate | Directional pressure |
| G4: Book state | Bid/ask depth at multiple levels, book imbalance, spread | Liquidity and supply/demand |
| G5: Book dynamics | Change in depth, cancellation rate, queue position changes | Order book evolution |
| G6: Cross-bar | Rolling statistics over prior $k$ bars (momentum, vol, flow persistence) | Temporal context at single-bar level |

**3.2 — Incremental feature addition.** Using GBT (the strongest single-bar model from Phase 2), train models with progressively richer feature sets:

- $X_n^{(1)} = G1$
- $X_n^{(2)} = G1 \cup G2$
- $X_n^{(3)} = G1 \cup G2 \cup G3$
- ... and so on.

For each, compute cross-validated Brier score and the variance explained $\Delta_k = \text{Brier}_{\text{constant}} - \text{Brier}_k$.

**3.3 — Marginal contribution plot.** Plot $\Delta_k$ vs. feature group. The curve should be concave (diminishing returns). When the marginal improvement from adding a new group is within noise ($< 1$ SE of the bootstrap), you've approximately saturated the information in your features.

**3.4 — Information ceiling bounds.** From T3:

$$\text{Var}(P) = \text{Var}(Y) - \mathbb{E}[P(1-P)]$$

You can't observe $P$ directly, but using the best model $\hat{f}$:

- **Lower bound on** $\text{Var}(P)$: $\text{Var}(\hat{f}(X_n)) \leq \text{Var}(P)$ (since $\hat{f}$ is a coarser function of the information).
- **Upper bound on** $\text{Var}(P)$: $\text{Var}(P) \leq \text{Var}(Y) - \mathbb{E}[\hat{f}(1-\hat{f})]$. This follows because the true Bayes error $\mathbb{E}[P(1-P)]$ is a lower bound on any model's residual, so $\mathbb{E}[\hat{f}(1-\hat{f})] \geq \mathbb{E}[P(1-P)]$ when $\hat{f}$ is well-calibrated. (Caution: this bound is tight only when the model is well-calibrated. If the model is miscalibrated, recalibrate it first — e.g., Platt scaling — before computing this bound.)

Report: lower bound, upper bound, and the ratio $\text{Var}(\hat{f}) / \text{upper bound}$. This ratio estimates what fraction of the available signal the model captures.

**3.5 — Feature importance for downstream architecture design.** From the GBT, extract SHAP values or permutation importances. Identify which features drive predictions. This informs the sequence model: if cross-bar features (G6) are important, the sequence model has room to improve over single-bar models. If only contemporaneous features (G1-G5) matter, the temporal context may add little.

### Gate Criteria

- Feature contribution curve has clearly flattened.
- Information ceiling bounds are computed and the ratio is reported.
- If the ratio $\text{Var}(\hat{f}) / \text{upper bound} > 0.8$: the single-bar model is already close to optimal. Sequence modeling may yield marginal improvement at best. Proceed to Phase 4 only if the absolute signal level justifies the engineering cost.
- If the ratio is $< 0.5$: substantial room for improvement. Either the feature set is incomplete, or temporal dependencies (which single-bar models can't capture) carry the remaining signal. Sequence modeling is justified.

### Deliverables

- Feature contribution curve with error bars.
- Information ceiling bounds.
- Feature importance ranking.
- Recommendation: proceed to sequence model, iterate on features, or conclude that signal is saturated.

---

## Phase 4: Sequence Model Training

### Objective

Train a sequence model (transformer or SSM) on $(X_1, \ldots, X_n)$ to approximate $P_{\text{long}}^{(n)}$. Use the formalization's diagnostics (T2, E2) as calibration checks throughout.

### Tasks

**4.1 — Architecture selection.** Start with one of:

- **Transformer** (encoder-only, causal attention): standard choice. Context window of 64-256 bars.
- **State space model** (Mamba or S4 variant): better scaling for long sequences if temporal dependencies extend far.

Do not sweep architectures initially. Pick one, get it working, validate with the diagnostics, then iterate.

**4.2 — Training setup.**

- **Loss function:** Binary cross-entropy (equivalent to minimizing Brier score up to a monotone transformation). Not MSE on raw labels — BCE is the proper scoring rule for probability estimation.
- **Output layer:** Independent sigmoids for $p_l$ and $p_s$ (justified unless S1 reveals binding joint constraints).
- **Context length:** Start with 64 bars. Increase if Phase 3 showed significant cross-bar signal.
- **Regularization:** Standard (dropout, weight decay). No special tricks yet.

**4.3 — Calibration check (T2).** After training, compute $\frac{1}{N}\sum \hat{f}(X_n)$ on the validation set. This should be close to $\bar{Y}^{\text{long}}$ (the empirical label frequency, which is close to $1/3$ under the null). If the model's mean prediction deviates significantly from $\bar{Y}$:

- If higher: model is systematically overconfident about long wins.
- If lower: model is systematically underconfident.
- Either way: check for label leakage, feature leakage, or training issues before proceeding.

**4.4 — Calibration curve.** Bin predictions into deciles. For each bin, plot mean predicted probability vs. observed frequency. A well-calibrated model falls on the diagonal. If miscalibrated, apply Platt scaling or isotonic regression on the validation set and re-evaluate.

**4.5 — Prefix martingale diagnostic (E2).** For a held-out set of bars, evaluate the model at multiple prefix lengths. Fix a target bar $n$. Feed the model $(X_1, \ldots, X_m)$ for $m = 1, 2, \ldots, n$ and record $\hat{P}_m = \hat{f}(X_1, \ldots, X_m)$.

The sequence $\hat{P}_1, \hat{P}_2, \ldots, \hat{P}_n$ should behave like a martingale:

- The increments $\hat{P}_{m+1} - \hat{P}_m$ should be approximately mean-zero.
- The increments should be uncorrelated with features $X_1, \ldots, X_m$ (i.e., the model doesn't systematically revise upward when certain features are present, unless new information justifies it).
- The variance of increments should be non-negative and roughly decreasing (the model "settles" as it sees more context).

**Diagnosing violations:**

| Violation | Likely Cause |
|-----------|-------------|
| Predictions systematically increase with context length | Model is leaking future information through positional encoding or attention patterns |
| Predictions oscillate wildly | Model is not learning stable representations; regularization or context length issue |
| Variance of increments increases with $m$ | Model is amplifying noise in later context; possible overfitting to sequence position |
| Increments correlate with lagged features | Model is learning spurious temporal patterns |

**4.6 — Comparison to single-bar baseline.** Compare the sequence model's Brier score to the GBT from Phase 3. The sequence model should do at least as well (it has strictly more information). If it does worse, the architecture or training is the bottleneck, not the signal.

Compute:

- $\Delta_{\text{seq}} = \text{Brier}_{\text{GBT}} - \text{Brier}_{\text{seq}}$

This is the marginal value of temporal context. If $\Delta_{\text{seq}} \approx 0$, single-bar features capture essentially all the signal, and the added complexity of the sequence model is not justified.

### Gate Criteria

- Model passes T2 calibration check (mean prediction $\approx \bar{Y}$).
- Calibration curve is approximately diagonal (or post-calibration is).
- Prefix martingale diagnostic shows no systematic violations.
- $\text{Brier}_{\text{seq}} \leq \text{Brier}_{\text{GBT}}$ (sequence model is at least as good as single-bar baseline).

### Deliverables

- Trained model with hyperparameters documented.
- Calibration report (T2 check, calibration curve, prefix martingale plots).
- Brier score comparison: constant baseline → GBT → sequence model.
- Assessment of marginal value of temporal context.

---

## Phase 5: Information Ceiling Estimation and Stopping Criterion

### Objective

Determine how close the best model is to the theoretical ceiling and decide whether further model development is justified.

### Tasks

**5.1 — Tighten ceiling bounds.** Using the best model $\hat{f}$ from Phase 4 (post-calibration):

- **Lower bound:** $\text{Var}(\hat{f}(X_n))$
- **Upper bound:** $\text{Var}(Y) - \mathbb{E}[\hat{f}(1 - \hat{f})]$

If the model is well-calibrated, these bounds should be relatively tight.

**5.2 — Ceiling gap analysis.**

$$\text{Gap} = \text{Upper bound} - \text{Var}(\hat{f})$$

This gap decomposes into two sources:

1. **Feature gap:** $\text{Var}(P) - \text{Var}(\mathbb{E}[Y \mid \sigma(X)])$ — information in $\mathcal{G}_n$ not captured by $X_n$.
2. **Approximation gap:** $\text{Var}(\mathbb{E}[Y \mid \sigma(X)]) - \text{Var}(\hat{f})$ — information in $X_n$ not captured by the model.

You can't decompose these exactly without knowing $P$, but you can probe:

- If adding more features (Phase 3 iteration) doesn't improve the model, the feature gap is likely small and the approximation gap dominates → try a larger/different model.
- If a much larger model doesn't improve over the current one on the same features, the approximation gap is likely small and the feature gap dominates → invest in feature engineering or accept the ceiling.

**5.3 — Stopping criterion.** Model development should stop when:

- The gap is within estimation uncertainty (bootstrap CI of the gap includes zero), OR
- The marginal improvement from the last architecture/feature iteration is below the profitability threshold ($\Delta < C / (3R)$ roughly), OR
- The absolute signal level, even at the ceiling, is not tradeable after costs.

### Deliverables

- Final information ceiling report with bounds and gap decomposition.
- Recommendation: ship to production, iterate further, or terminate the research line.

---

## Cross-Cutting Concerns

### Leakage Prevention Protocol

Every model at every phase must pass the following checks before results are trusted:

1. **Feature timestamp audit.** For every feature in $X_n$, verify that it uses only data from $t \leq T_n$. Automated: compute the maximum timestamp of any data point used in each feature and assert $\leq T_n$.
2. **Label independence.** Verify that no feature is mechanically correlated with the label by construction (e.g., a feature that encodes price movement in the interval used for labeling).
3. **Temporal split integrity.** No data from the validation/test period is used in training, including for normalization statistics (compute mean/std on training data only).

### Reproducibility

- Fix random seeds for all stochastic operations.
- Log all hyperparameters, data versions, and code commits.
- Results should be reproducible from a single config file + data snapshot.

### Instrument and Market Regime

All results are conditional on the instrument and time period. Before drawing general conclusions:

- Run at least two instruments (e.g., one equity index future, one FX pair) to check if signal structure is instrument-specific.
- Segment results by volatility regime (e.g., VIX terciles) to check for regime dependence.

---

## Summary Decision Tree

```
Phase 0: Build pipeline, generate labels
    │
Phase 1: Check ȳ ≈ 1/3?
    │   NO (wildly off) → Debug pipeline
    │   YES (or explainable drift) ↓
    │
Phase 2: Simple models beat constant baseline?
    │   NO → Phase 2b: Sweep (B, R). Still no? → STOP. Negative result.
    │   YES ↓
    │
Phase 3: Feature engineering. Curve flattened?
    │   YES, and ceiling ratio > 0.8 → Single-bar model may suffice. Phase 4 optional.
    │   YES, and ceiling ratio < 0.5 → Proceed to Phase 4.
    │   NO → Add more features. Iterate.
    │
Phase 4: Sequence model. Passes diagnostics?
    │   NO → Debug (leakage, calibration, architecture)
    │   YES, Δ_seq ≈ 0 → Temporal context doesn't help. Use single-bar model.
    │   YES, Δ_seq > 0 → Sequence model adds value.
    │
Phase 5: Close to ceiling?
    │   YES → Ship or STOP (if unprofitable after costs).
    │   NO, feature gap dominates → Return to Phase 3.
    │   NO, approximation gap dominates → Iterate Phase 4 architecture.
```
