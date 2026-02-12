# Experiment: Book-State vs Trade-Flow Feature Group Ablation

## Hypothesis

Trade-flow features (Group B) produce higher mean AUC than book-state features (Group A) by at least Δ_AUC > 0.005 across quarterly expanding-window folds, for logistic regression on both Y_long and Y_short barrier labels at B=500.

**Direction:** AUC(Group B) > AUC(Group A).
**Magnitude:** Mean Δ_AUC(B−A) > 0.005 with bootstrap 95% CI excluding 0.

**Rationale:** T6's RF importance ranking placed trade-derived features (trade_flow_imbalance 0.128, bar_range 0.122, volume_log 0.121) in the top 3, while 4/13 book features were dead. Now that the C++ cache has activated all book features, this experiment directly measures whether the T6 ranking holds — or whether the newly-activated book features carry comparable or superior signal. The literature (Hasbrouck 1991, Cont-Stoikov-Talreja 2010) suggests trade flow dominates price discovery, but MES's book may have unique structure as a derivative of ES.

## Independent Variables

**Feature group** (3 levels):

| Arm | Base columns | Description | Dims (×10 lookback) |
|-----|-------------|-------------|---------------------|
| Group A (Book) | 1, 2, 10, 11, 13, 14, 15, 16, 17, 20 | Book-state features + temporal controls | 120 |
| Group B (Trade) | 0, 3, 4, 5, 6, 7, 8, 18, 19, 21 | Trade-flow features + temporal controls | 120 |
| Group A+B (Full) | 0–21 | All 22 features (baseline) | 220 |

Temporal controls (cols 9, 12: session_time, session_age) are included in ALL arms. They are not informative for the book-vs-flow question but prevent confounding from time-of-day or session-start effects.

**Column-to-feature mapping (verified against `feature_pipeline.py:35`):**

Group A (Book State — snapshot + dynamics from LOB reconstruction):
- Col 1: BBO imbalance [0, 1]
- Col 2: Depth imbalance [0, 1]
- Col 10: Cancel rate asymmetry [-1, +1]
- Col 11: Mean spread (ticks)
- Col 13: Order Flow Imbalance (OFI) [-1, +1]
- Col 14: Multi-level depth ratio [0, 1]
- Col 15: Weighted mid displacement (ticks)
- Col 16: Spread dynamics std (ticks)
- Col 17: VAMP displacement (ticks)
- Col 20: Cancel-to-trade ratio

Group B (Trade Flow — price + volume + execution from trade tape):
- Col 0: Trade flow imbalance [-1, +1]
- Col 3: Bar range (ticks)
- Col 4: Bar body (ticks)
- Col 5: Body/range ratio [-1, +1]
- Col 6: VWAP displacement [-1, +1]
- Col 7: Volume log
- Col 8: Trailing realized volatility
- Col 18: Aggressor imbalance [-1, +1]
- Col 19: Trade arrival rate
- Col 21: Price impact per trade

**Label** (2 levels): Y_long, Y_short. Both tested independently per exp-006 convention.

**Lookback column mapping:** With h=10, the 220-dim vector has cols [i×22..(i+1)×22-1] for lookback step i (i=0 is oldest, i=9 is newest). To select Group A across all 10 steps:
```python
group_a_base = [1, 2, 10, 11, 13, 14, 15, 16, 17, 20]
temporal = [9, 12]
group_a_cols = sorted([base + step * 22 for step in range(10) for base in group_a_base + temporal])
# 120 dims
```
Similarly for Group B with base cols [0, 3, 4, 5, 6, 7, 8, 18, 19, 21] + temporal.

## Controls

- **Data:** `cache/barrier/` — 248 sessions, B=500, 454K usable bars, 220-dim features. C++ backend. All 22 features active. Identical to exp-006.
- **Model:** Logistic regression (L2, C=1.0, solver=lbfgs, max_iter=1000). Consistent with exp-006. LR is the established least-bad model. GBT excluded from primary analysis per exp-006's finding that it is strictly dominated by LR on BSS. GBT included as optional secondary (see below).
- **Label handling:** All bars included (no timeout exclusion). Timeout rate is ~0.3% at B=500 (exp-005/exp-009). Timeouts are labeled as Y=0 for both Y_long and Y_short. Consistent with exp-006.
- **Split:** Quarterly expanding-window CV (3 folds). This differs from exp-006's 60/20/20 temporal split but provides per-regime resolution:
  - Fold 1: Q1 (Jan–Mar) train → Q2 (Apr–Jun) test
  - Fold 2: Q1–Q2 (Jan–Jun) train → Q3 (Jul–Sep) test
  - Fold 3: Q1–Q3 (Jan–Sep) train → Q4 (Oct–Dec) test
  - Session-to-quarter mapping via dates extracted by `load_binary_labels()`.
- **Baseline:** Constant predictor ȳ = mean(y_train) per fold. Same as exp-006.
- **Bootstrap:** Block bootstrap for paired Δ_AUC, block_size=50, n_boot=1000. Block size accounts for label autocorrelation (mean race duration ~12 bars at B=500).
- **Seed:** 42.
- **Hardware:** Local macOS, CPU only. No GPU needed.
- **Software:** Python via `uv`. sklearn (LR), numpy, sklearn.metrics (roc_auc_score).

**Why these controls are necessary:**
- LR-only eliminates GBT overfitting confound (established exp-006, exp-007).
- Quarterly folds give per-regime resolution (high-vol H1 vs low-vol H2) without needing separate regime labeling.
- Block bootstrap accounts for temporal dependence in barrier labels.
- Including all bars (no timeout exclusion) matches exp-006 exactly and avoids selection bias.
- Temporal controls in all arms prevent time-of-day from confounding the group comparison.

## Metrics (ALL must be reported)

### Primary

1. **AUC** (ROC-AUC via `sklearn.metrics.roc_auc_score`) for each arm × label × fold. AUC measures discrimination — can the features rank bars by likelihood of barrier hit? This is the metric where T6 found signal (+5pp accuracy ≈ AUC > 0.5).
2. **Paired Δ_AUC = AUC(B) − AUC(A)** per label, averaged across folds, with block bootstrap 95% CI on the mean difference. This is the primary test statistic.

### Secondary

- **BSS** (Brier Skill Score) for each arm × label × fold. Included for continuity with exp-006 through exp-009, but expected to be negative for all arms (established pattern).
- **Paired Δ_BSS = BSS(B) − BSS(A)** per label.
- **AUC(A+B) − max(AUC(A), AUC(B))** — tests for feature interaction. If positive, the groups carry complementary information.
- **Per-fold AUC** — reveals regime effects (Fold 1/Q2 = high-vol, Fold 3/Q4 = low-vol).
- **N_train, N_test, ȳ_train, ȳ_test** per fold — verifies label balance and sample sizes.
- **(Optional) GBT AUC** for each arm × label × fold — only if wall clock permits after primary analysis. If included, report GBT Δ_AUC alongside LR Δ_AUC for consistency check.

### Sanity Checks

- **AUC > 0.5 for Group A+B (full)** on at least 2/3 folds × 2 labels. If the full feature set can't beat chance, the experiment is uninformative (pipeline or data issue).
- **ȳ_train and ȳ_test ∈ [0.20, 0.46]** for all folds and labels. Verifies null calibration holds across quarters.
- **N_test ≥ 50,000** for each fold. Ensures adequate sample size for AUC estimation.
- **LR converged** (no sklearn ConvergenceWarning) for all fits.

## Baselines

**Within-experiment baselines:**
- Group A+B (full 220-dim) serves as the upper-bound baseline. If neither group alone matches Group A+B, the signal requires feature interactions.
- AUC = 0.5 (random) serves as the lower bound. Any arm with AUC ≤ 0.5 carries no discriminative signal.

**Cross-experiment baselines:**
- exp-006: LR on full 220-dim, chronological 60/20/20: BSS_long = -0.0007, BSS_short = -0.0003. All negative.
- T6: RF on 130-dim (90-dim effective due to dead features), shuffle 80/20: 40.5% balanced accuracy vs 34.5% baseline.

**Reproduction:** The Group A+B arm with Fold 3 (Q1–Q3 train → Q4 test) should produce similar BSS to exp-006 on the overlapping test period. Exact match not expected (different train/test split), but same sign and order of magnitude.

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Group B dominates Group A:** Mean Δ_AUC(B−A) > 0.005 across 3 folds for LR on Y_long AND Y_short. Positive in ≥ 2/3 folds for each label.
- [ ] **C2 — Statistical significance:** Bootstrap 95% CI for mean Δ_AUC(B−A) excludes 0 for at least one of {Y_long, Y_short}.
- [ ] **C3 — Sanity:** AUC(A+B) > 0.5 on ≥ 4/6 (fold, label) cells. All ȳ values in [0.20, 0.46]. All N_test ≥ 50K.
- [ ] **C4 — No regression:** No sanity check failure invalidates the comparison.

**Verdict mapping:**
- C1 + C2 + C3 → **CONFIRMED** — Trade-flow features carry more discriminative signal than book-state features. Next step: focus feature engineering on trade-flow refinements.
- C1 fails (Δ_AUC ≤ 0.005 or wrong direction) + C3 → **REFUTED** — No meaningful difference between groups, or book features are comparable/superior. Both groups carry similar (weak) signal.
- C2 fails (CI includes 0) + C3 → **INCONCLUSIVE** — Trend in expected direction but insufficient statistical power. May need larger sample or different split.
- C3 fails → **INVALID** — Pipeline or data issue. Results cannot be interpreted.

## Minimum Viable Experiment

1. Load data: `load_binary_labels('cache/barrier/', lookback=10)`. Verify N_samples ≈ 454K, N_sessions = 248, dates are extractable.
2. Map sessions to quarters using extracted dates. Verify each quarter has ≥ 40 sessions.
3. Build Fold 1 (Q1 train → Q2 test). Verify N_train > 50K and N_test > 50K.
4. Select Group B columns from X_train and X_test.
5. Fit LR on Group B columns for Y_long. Compute AUC on test. Verify AUC is a finite number and > 0.45 (not catastrophically broken).
6. Repeat step 5 for Group A columns. Compute Δ_AUC(B−A). Verify Δ is a finite number.

If any step fails, diagnose before running the full protocol. The MVE takes < 2 minutes and validates:
- Session-to-quarter mapping works
- Feature column selection is correct
- LR converges on reduced feature sets
- AUC computation returns valid numbers

## Full Protocol

### Phase 1: Data loading and fold construction (~1 min)

1. Load data with `load_binary_labels('cache/barrier/', lookback=10)`.
2. Extract dates from the data dict. Map each session to a quarter (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec) using the date string.
3. Build the 3 expanding-window folds:
   - Fold 1: train = sessions in Q1, test = sessions in Q2
   - Fold 2: train = sessions in Q1+Q2, test = sessions in Q3
   - Fold 3: train = sessions in Q1+Q2+Q3, test = sessions in Q4
4. Convert session indices to row indices using session_boundaries.
5. Record N_train, N_test, ȳ_train_long, ȳ_train_short, ȳ_test_long, ȳ_test_short for each fold.
6. Verify sanity checks: all ȳ ∈ [0.20, 0.46], all N_test ≥ 50K.

### Phase 2: Run MVE (~2 min)

Execute the MVE per the section above. If it fails, stop and diagnose.

### Phase 3: Primary analysis — LR on 3 arms × 2 labels × 3 folds (~5 min)

For each fold ∈ {1, 2, 3}:
  For each label ∈ {Y_long, Y_short}:
    For each arm ∈ {Group A, Group B, Group A+B}:
      1. Select feature columns for this arm from X_train and X_test.
      2. Fit LR(C=1.0, max_iter=1000) on (X_train_arm, y_train).
      3. Predict probabilities on X_test_arm: p̂ = lr.predict_proba(X_test)[:, 1].
      4. Compute AUC = roc_auc_score(y_test, p̂).
      5. Compute BSS = 1 − brier_score(y_test, p̂) / brier_score(y_test, ȳ_train).
      6. Record: {arm, label, fold, auc, bss, brier, n_train, n_test, y_rate_train, y_rate_test}.

Total: 3 × 2 × 3 = 18 LR fits. Estimated ~15s each = ~5 min.

### Phase 4: Paired comparison — bootstrap Δ_AUC (~3 min)

For each label ∈ {Y_long, Y_short}:
  1. Collect per-fold Δ_AUC(B−A) = [AUC_B_fold1 − AUC_A_fold1, ...].
  2. Compute mean Δ_AUC across 3 folds.
  3. Bootstrap the mean Δ_AUC: resample the per-fold AUC pairs (A, B) with block bootstrap (block_size=50, n_boot=1000) within each fold, recompute Δ_AUC per fold, average across folds. Record 95% CI.
  4. Repeat for Δ_BSS.
  5. Compute AUC(A+B) − max(AUC(A), AUC(B)) per fold for interaction analysis.

### Phase 5: Optional GBT secondary (~5 min, skip if wall clock > 12 min)

Repeat Phase 3 with GBT (LightGBM: n_estimators=200, max_depth=6, learning_rate=0.05, min_child_samples=100, seed=42) instead of LR. Report Δ_AUC(B−A) for GBT alongside LR.

### Phase 6: Assemble and save (~1 min)

Write all metrics to `results/exp-011-feature-group-ablation/metrics.json` per the output schema below.

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 15 minutes
- Max training runs: 0 (supervised fitting only, no RL)
- Max seeds per configuration: 1
- Max model fits: 18 (primary) + 18 (optional GBT) = 36

**Estimated runtime breakdown:**
- Phase 1 (load + fold construction): ~1 min
- Phase 2 (MVE): ~2 min (subset of Phase 3, counted once)
- Phase 3 (18 LR fits): ~5 min (454K samples × 120-220 dims × LR ≈ 15s/fit)
- Phase 4 (bootstrap): ~3 min
- Phase 5 (optional GBT): ~5 min (skip if > 12 min wall clock)
- Phase 6 (save): ~1 min
- **Total: ~12 min** (without GBT: ~10 min)

## Compute Target

**Compute:** `local`

No GPU needed. All models are CPU-only sklearn. Memory peak: ~2 GB for 454K × 220 float64 matrix. Well within local machine capacity.

## Abort Criteria

- **Wall clock > 30 min:** Abort. Something is wrong. 30 min is ~2.5× the 12 min estimate.
- **LR fails to converge** (ConvergenceWarning) on any arm: Increase max_iter to 5000 and retry once. If still fails, abort that arm but continue with others.
- **AUC = NaN or Inf:** Abort that cell, continue with others.
- **All AUC(A+B) ≤ 0.50:** Abort entirely — no discriminative signal in the full feature set means the ablation is meaningless. This would contradict T6 and requires investigation.

## Confounds to Watch For

1. **Dimensionality imbalance.** Group A and Group B each have 10 base features (120 dims with lookback), while Group A+B has 22 (220 dims). Higher dimensionality gives LR more degrees of freedom and could inflate AUC for A+B independently of feature quality. The key comparison is A vs B (same dimensionality), not A vs A+B.

2. **Collinearity within groups.** Some Group B features are algebraically related (bar_body, bar_range, body_range_ratio). High collinearity could inflate variance of LR coefficients and make AUC estimates noisier. L2 regularization (C=1.0) mitigates this, but watch for AUC variance across folds.

3. **Dead feature residuals.** Although the C++ cache activated book features, some may still have low variance or near-constant values in certain quarters. Check feature variance per group per fold before interpreting low AUC as "no signal."

4. **Quarterly fold size imbalance.** Q1 has ~62 sessions (~113K bars), while Q4 has ~62 sessions. Training set grows across folds (Q1→Q1-Q3). Fold 1 has the smallest training set (~113K) and Fold 3 the largest (~340K). Per-fold AUC variation may reflect sample size, not regime effects.

5. **Feature group assignment ambiguity.** Some features straddle categories:
   - OFI (col 13): classified as Book State because it depends on order additions/cancellations at BBO, but it incorporates trade-induced book changes.
   - Aggressor imbalance (col 18): classified as Trade Flow, but "aggressor" is defined relative to the book.
   - Cancel-to-trade ratio (col 20): classified as Book State (cancel counting requires book reconstruction), but the denominator is trade count.

   These boundary features could shift the result. If Δ_AUC is close to 0, re-run with ambiguous features swapped to test sensitivity.

6. **Multiple testing.** 2 labels × the primary comparison = 2 tests. Bonferroni correction: p < 0.025 per test. With only 3 folds contributing to each mean Δ_AUC, the bootstrap CI width will be dominated by fold-level variance. The Δ_AUC > 0.005 magnitude threshold provides additional protection beyond the significance test.

## Output Schema

All metrics written to `results/exp-011-feature-group-ablation/metrics.json`:

```json
{
  "experiment": "exp-011-feature-group-ablation",
  "timestamp": "ISO-8601",
  "tier": "Quick",
  "arms": {
    "group_a_book": {
      "feature_cols_base": [1, 2, 10, 11, 13, 14, 15, 16, 17, 20],
      "temporal_cols": [9, 12],
      "n_dims": 120
    },
    "group_b_trade": {
      "feature_cols_base": [0, 3, 4, 5, 6, 7, 8, 18, 19, 21],
      "temporal_cols": [9, 12],
      "n_dims": 120
    },
    "group_ab_full": {
      "feature_cols_base": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
      "temporal_cols": [],
      "n_dims": 220
    }
  },
  "folds": {
    "fold_1": {"train": "Q1", "test": "Q2", "n_train_sessions": "int", "n_test_sessions": "int"},
    "fold_2": {"train": "Q1-Q2", "test": "Q3", "n_train_sessions": "int", "n_test_sessions": "int"},
    "fold_3": {"train": "Q1-Q3", "test": "Q4", "n_train_sessions": "int", "n_test_sessions": "int"}
  },
  "results": {
    "<arm>/<model>/<label>/fold_<N>": {
      "auc": "float",
      "bss": "float",
      "brier": "float",
      "brier_constant": "float",
      "n_train": "int",
      "n_test": "int",
      "y_rate_train": "float",
      "y_rate_test": "float"
    }
  },
  "paired_comparisons": {
    "lr/<label>": {
      "mean_delta_auc_b_minus_a": "float",
      "mean_delta_bss_b_minus_a": "float",
      "per_fold_delta_auc": ["float", "float", "float"],
      "per_fold_delta_bss": ["float", "float", "float"],
      "bootstrap_ci_delta_auc_95": ["float", "float"],
      "bootstrap_ci_delta_bss_95": ["float", "float"]
    }
  },
  "interaction": {
    "<label>/fold_<N>": {
      "auc_ab": "float",
      "max_auc_a_b": "float",
      "interaction_delta": "float (auc_ab - max(auc_a, auc_b))"
    }
  },
  "sanity_checks": {
    "all_ybar_in_range": "bool",
    "all_n_test_gte_50k": "bool",
    "full_auc_gt_05_count": "int (out of 6)",
    "lr_converged_all": "bool"
  },
  "success_criteria": {
    "C1_group_b_dominates": "bool",
    "C1_detail": "per-label pass/fail",
    "C2_significant": "bool",
    "C3_sanity": "bool",
    "C4_no_regression": "bool",
    "verdict": "CONFIRMED|REFUTED|INCONCLUSIVE|INVALID"
  },
  "resource_usage": {
    "wall_clock_seconds": "float",
    "n_lr_fits": "int",
    "n_gbt_fits": "int (0 if skipped)"
  }
}
```
