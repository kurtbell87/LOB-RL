# Experiment: Phase 2 Signal Detection — Can LR or GBT Beat the Constant Brier Score on Y_long / Y_short?

## Hypothesis

Logistic regression (L2, C=1.0) or gradient-boosted trees (LightGBM, max_depth=6, 200 trees) will achieve a lower Brier score than the constant predictor (ȳ) on at least one of {Y_long, Y_short}, evaluated on a held-out chronological validation set (sessions 149–198 of 248).

**Direction:** Model Brier < constant Brier (i.e., BSS > 0).
**Magnitude:** BSS ≥ 0.005 (model explains ≥ 0.5% of predictable variance). This is deliberately low — even a small BSS with n≈91K is meaningful if statistically significant. The survey estimates BSS ≈ 0.005–0.02 based on T6's +5pp accuracy gain.

## Independent Variables

**Model family** (2 levels):
1. Logistic regression: `fit_logistic(X_train, y_train)` — L2, C=1.0, solver=lbfgs, max_iter=1000
2. Gradient-boosted trees: `fit_gbt(X_train, y_train, seed=42)` — LightGBM, max_depth=6, n_estimators=200, lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_samples=50

**Target label** (2 levels):
1. Y_long ∈ {0, 1} — did price hit +2R before -R?
2. Y_short ∈ {0, 1} — did price hit -2R before +R?

Total configurations: 4 (2 models × 2 labels). Each is an independent signal detection test.

## Controls

- **Data:** Barrier cache `cache/barrier/`, 248 sessions, 454K usable samples, 220-dim features (22 features × 10 lookback). C++ backend build from 2026-02-10. No cache rebuild.
- **Split:** Chronological temporal split via `temporal_split(248)` — 60/20/20 = 149/49/50 sessions for train/val/test. Train: sessions 0–148 (~272K rows). Val: sessions 149–197 (~91K rows). Test: sessions 198–247 (~91K rows). Only val is used for primary evaluation. Test is held out for future use.
- **Baseline:** Constant predictor ȳ = mean(y_train). From exp-005: ȳ_long ≈ 0.320, ȳ_short ≈ 0.322. Constant Brier = ȳ(1-ȳ) ≈ 0.2176 (long), 0.2183 (short). These will be recomputed from the training set for exactness.
- **Normalization:** Features are z-score normalized and clipped ±5 during precompute. No additional normalization needed.
- **Software:** Python via `uv`. sklearn 1.8.0 (LR), LightGBM 4.6.0 (GBT), numpy.
- **Hardware:** Local (macOS). No GPU.
- **Seed:** 42 (for GBT and bootstrap). Single seed — this is a gate check, not a robustness study.
- **Bootstrap:** Block bootstrap with block_size=50, n_boot=1000 (as implemented in `signal_detection_report()`). Accounts for temporal autocorrelation within sessions.

**Why these controls are necessary:**
- Chronological split prevents future information leakage (labels from overlapping races could leak across a shuffle boundary).
- Single fixed hyperparameters per model prevent overfitting to the validation set through implicit tuning.
- Block bootstrap accounts for temporal autocorrelation (mean race duration ~12 bars means adjacent labels overlap).

## Metrics (ALL must be reported)

### Primary

1. **Brier skill score (BSS)** for each (model, label) pair: BSS = 1 - BS_model / BS_constant. Tests the hypothesis directly.
2. **Bootstrap p-value** for Brier delta (constant - model) > 0. Statistical significance at α = 0.05.

### Secondary

- **Raw Brier scores**: BS_constant, BS_logistic, BS_gbt for each label.
- **Bootstrap 95% CI** for Brier delta.
- **Max predicted probability** for each model × label (profitability threshold: p > 0.40 needed for C=2, R=10).
- **5-fold expanding-window CV Brier scores** for each model × label (within-training stability check).
- **Calibration curves** (10-bin) for each model × label.

### Sanity Checks

- **N_samples** loaded ≈ 454K (expected from exp-005).
- **N_sessions** loaded = 248.
- **ȳ_train_long** ∈ [0.28, 0.38] (consistent with exp-005 global ȳ = 0.320).
- **ȳ_train_short** ∈ [0.28, 0.38] (consistent with exp-005 global ȳ = 0.322).
- **GBT Brier ≤ constant Brier + 0.01** — if GBT is *worse* than constant by more than 0.01, it is catastrophically overfitting and the experiment is invalid.
- **LR converged** (max_iter=1000 sufficient — sklearn will warn if not).

## Baselines

**Constant predictor:** ȳ = mean(y_train) applied uniformly to all validation samples. Brier score = ȳ(1-ȳ).

From exp-005 (global, not train-only):
- ȳ_long = 0.320, BS_constant_long = 0.320 × 0.680 = 0.2176
- ȳ_short = 0.322, BS_constant_short = 0.322 × 0.678 = 0.2183

The experiment will recompute ȳ from the training set only (first 149 sessions), which may differ slightly from the global mean due to the temporal split. This is the correct baseline: the constant predictor uses only training-set information.

**No reproduction step needed.** The constant baseline is computed analytically from training labels and is deterministic.

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Signal detected:** At least one of the 4 (model, label) pairs has bootstrap p < 0.05 AND Brier delta CI lower bound > 0 (i.e., model significantly beats constant).
- [ ] **C2 — Meaningful magnitude:** The best-performing (model, label) pair achieves BSS ≥ 0.005.
- [ ] **C3 — CV consistency:** The best (model, label) pair's mean CV Brier score is below the constant Brier (computed on each fold's training set) in ≥ 3 of 5 folds.
- [ ] **No sanity check failure:** All sanity checks pass within stated tolerances.

**Verdict mapping:**
- C1 + C2 + C3 pass → **CONFIRMED** — calibrated probabilistic signal exists. Proceed to Phase 3 (feature engineering / architecture).
- C1 passes but C2 fails (BSS < 0.005) → **INCONCLUSIVE** — statistically significant but practically negligible. Signal exists but may not be tradeable.
- C1 fails (no model beats constant at p < 0.05) → **REFUTED** — the 220-dim barrier features do not contain calibrated probabilistic signal for Y_long or Y_short at the individual-bar level, despite the discriminative signal found in T6's 3-class framing.
- Sanity check failure → **INVALID** — investigate before interpreting.

## Minimum Viable Experiment

1. Load cache with `load_binary_labels('cache/barrier/', lookback=10)`. Verify N_samples ≈ 454K, N_sessions = 248.
2. Run temporal split. Verify train has ~149 sessions, val has ~49 sessions.
3. Fit logistic regression on training Y_long only. Predict on val. Compute BS_logistic and BS_constant. Print both.
4. If BS_logistic < BS_constant → MVE passes (signal direction is correct). Proceed to full report.
5. If BS_logistic ≥ BS_constant → MVE still proceeds to full report (GBT may succeed where LR fails, or Y_short may differ from Y_long).

The MVE confirms infrastructure works (data loads, model fits, predictions are in [0,1], Brier scores are computable). It takes < 1 minute.

## Full Protocol

1. **Load data:** `load_binary_labels('cache/barrier/', lookback=10)`. Record N_samples, N_sessions. Check sanity.

2. **Run MVE:** Fit LR on train Y_long → predict val → compare BS_logistic vs BS_constant. Log result. Continue regardless of outcome.

3. **Run full signal detection:** Call `signal_detection_report(X, Y_long, Y_short, session_boundaries, seed=42)`. This single function call executes:
   - Temporal split (60/20/20 by session)
   - For each label (Y_long, Y_short):
     - Constant baseline Brier on val
     - LR: fit on train, predict_proba on val, Brier score, BSS, calibration curve, bootstrap test
     - GBT: fit on train, predict_proba on val, Brier score, BSS, calibration curve, bootstrap test
     - 5-fold expanding-window CV for both LR and GBT
   - signal_found flag (any delta CI excludes 0)

4. **Collect all metrics:** Assemble the `signal_detection_report()` return dict into a JSON-serializable structure. Include:
   - All Brier scores (constant, LR, GBT × long, short)
   - All BSS values
   - All bootstrap deltas with CI and p-values
   - Max predicted probabilities
   - CV Brier arrays
   - Calibration curves (as arrays)
   - signal_found flag
   - N_samples, N_sessions
   - ȳ_train_long, ȳ_train_short
   - Success criteria C1, C2, C3 evaluations

5. **Write metrics:** Save to `results/exp-006-signal-detection/metrics.json`.

6. **No test set evaluation.** The test set (sessions 198–247) is held out. Only val is used for primary metrics.

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 15 minutes
- Max training runs: 0 (supervised fitting only, no RL)
- Max seeds per configuration: 1

**Estimated runtime breakdown:**
- Data loading: ~10s (248 .npz files, ~186 MB)
- LR fit (272K × 220): ~30s × 2 labels = 1 min
- GBT fit (272K × 220): ~2 min × 2 labels = 4 min
- 5-fold CV (LR): ~2.5 min × 2 labels = 5 min
- 5-fold CV (GBT): ~10 min × 2 labels = 20 min (this is the bottleneck — 5 GBT fits per label)
- Bootstrap (4 tests × 1000 iterations): < 1 min
- **Total: ~15 min** (dominated by GBT CV folds)

Note: The 5-fold CV fits GBT 10 times total (5 folds × 2 labels). Each fold trains on an expanding window (smallest fold ~55K rows, largest ~245K rows). LightGBM histogram-based fitting is efficient, but 10 GBT fits on 220-dim features may approach 15 min. If wall time exceeds 15 min, this is still acceptable — the CV is secondary evidence and the primary result (single-split Brier on val) completes in < 5 min.

## Compute Target

**Compute:** `local`

No GPU needed. All models are CPU-based (sklearn LR, LightGBM). Memory peak ~2-3 GB (feature matrix + model copies during CV). Fits comfortably on local machine.

## Abort Criteria

- `load_binary_labels()` raises an exception → abort, investigate cache.
- N_sessions < 200 → abort, cache incomplete.
- N_samples < 400,000 → abort, cache incomplete.
- Any NaN or Inf in Brier scores → abort, numerical issue.
- LR `predict_proba` returns values outside [0, 1] → abort, model issue.
- Wall clock > 30 minutes → abort (3× the expected 10 min for primary results; CV may be slow but primary should complete in < 5 min).

**No per-model time abort.** Individual model fits should complete in < 5 min each. If any single fit hangs for > 10 min, something is wrong (e.g., LR not converging).

## Confounds to Watch For

1. **Train/val distributional shift.** The temporal split puts January–August 2022 in training and September–November in validation. The 2022 bear market had distinct phases (crash in Q1, recovery attempt in Q2-Q3, renewed decline in Q4). If the val period has different volatility or microstructure, Brier scores may be artificially inflated or deflated. The 5-fold expanding-window CV partially controls for this by evaluating on different time periods.

2. **Label autocorrelation inflating significance.** Adjacent bars' labels overlap in time (mean race duration ~12 bars from exp-005). The block bootstrap (block_size=50) accounts for this, but if the true autocorrelation length is > 50 bars, p-values may be anti-conservative. The READ agent should check if reducing block_size to 25 or increasing to 100 changes the significance conclusion.

3. **GBT overfitting masquerading as signal.** LightGBM with 200 trees and max_depth=6 is expressive enough to partially memorize training patterns. If GBT beats constant but LR does not, the signal may be overfitting to nonlinear artifacts rather than genuine predictability. **Diagnostic:** Compare GBT's train Brier vs val Brier. A large gap (> 0.05) signals overfitting.

4. **Dead features.** 4 of 22 features (bbo_imbal, depth_imbal, cancel_asym, mean_spread) were dead in the Python cache but are active in the C++ cache. If these features dominate the GBT but not LR, it may indicate the signal is concentrated in book-derived features. This is informative, not a confound, but the READ agent should note it.

5. **Constant predictor is very strong.** At ȳ ≈ 0.32, the constant Brier is ȳ(1-ȳ) ≈ 0.218. This is already close to the minimum possible Brier (0.0 for a perfect predictor). A BSS of 0.005 means the model reduces Brier by only ~0.001 absolute. Even though this may be statistically significant with n=91K, it represents a tiny practical improvement. The READ agent should contextualize BSS in terms of what probability shift it implies (roughly: if the model moves predictions from 0.32 to [0.30, 0.34] based on features, that's a BSS of ~0.002).

6. **Multiple testing.** We test 4 (model, label) pairs. At α = 0.05 per test, the family-wise error rate under the null is ~0.185. C1 requires only one significant result, so a Bonferroni correction would set the per-test threshold at 0.05/4 = 0.0125. The READ agent should report both the raw and Bonferroni-corrected conclusions. However, C2 (BSS ≥ 0.005) serves as an independent magnitude filter that partially controls for false positives from multiple testing.
