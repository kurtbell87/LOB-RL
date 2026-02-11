# Survey: Can logistic regression or GBT beat the constant Brier score baseline on Y_long or Y_short using 220-dim barrier features?

## Prior Internal Experiments

### Directly relevant
- **exp-005-null-calibration (CONFIRMED):** ȳ_long = 0.320, ȳ_short = 0.322. Constant-prediction Brier baseline: BS_long = 0.320 × 0.680 = **0.2176**, BS_short = 0.322 × 0.678 = **0.2183**. Labels are independent (P(1,1) = 0), well-calibrated against Gambler's Ruin null. Phase 1 gate passed. This establishes the exact scores to beat.

### Related (different framing)
- **T6 supervised diagnostic (CONFIRMED, weak signal):** 3-class {long, short, flat} framing. RF 40.5% vs 34.5% baseline (+5.9pp). MLP 39.3% (+4.8pp). Signal consistent across shuffle and chrono splits. But: 3-class balanced accuracy ≠ Brier score. A model can improve classification accuracy without improving probabilistic calibration. T6 strongly suggests signal exists but doesn't directly predict Phase 2 outcome.
- **exp-004 (INCONCLUSIVE, quick tier):** 22-feature RF 49.6% vs 9-feature RF 47.5% on 50K subsample, 3-class. Both beat 38.1% majority baseline by ~11pp. Aborted before full run. Again 3-class, not Brier.
- **exp-003 (REFUTED):** 20-dim bar-level features contain NO signal for next-bar price direction. Irrelevant to barrier features but establishes that bar-level aggregates alone are insufficient.

### Key gap
**No prior experiment has evaluated Brier scores on binary Y_long / Y_short.** All prior diagnostics used 3-class balanced accuracy. The Phase 2 signal detection is the first Brier-based test. This is a genuine novelty — the shift from "can we classify better than majority?" to "can we calibrate probabilities better than a constant?" is a strictly harder test.

## Current Infrastructure

### Complete and ready to use
1. **`first_passage_analysis.py`** — Contains the entire Phase 2 pipeline:
   - `load_binary_labels(cache_dir, lookback=10)` → X (N, 220), Y_long, Y_short, timeouts, tau, session_boundaries
   - `signal_detection_report(X, Y_long, Y_short, session_boundaries, seed=42)` → full Brier analysis
   - `fit_logistic(X_train, y_train)` — L2-regularized LR (C=1.0, solver=lbfgs)
   - `fit_gbt(X_train, y_train, seed)` — LightGBM (max_depth=6, n_estimators=200, lr=0.05) with sklearn fallback
   - `brier_score()`, `constant_brier()`, `brier_skill_score()`
   - `calibration_curve(y_true, y_pred, n_bins=10)`
   - `paired_bootstrap_brier(y_true, pred_model, pred_baseline, n_boot=10000, block_size=50)` → {delta, ci_lower, ci_upper, p_value}
   - `temporal_split(n_sessions)` → 60/20/20 train/val/test by session
   - `temporal_cv_folds(n_sessions, n_folds=5)` → expanding-window CV

2. **Barrier cache** — 248 sessions, 454K usable samples, 220-dim features (22 features × 10 lookback). Fresh C++ backend build. No rebuild needed.

3. **Dependencies** — LightGBM 4.6.0, scikit-learn 1.8.0, numpy all installed and confirmed working.

4. **Test coverage** — `test_first_passage_analysis.py` has ~50 tests covering `signal_detection_report`, including planted-signal and pure-noise edge cases. The function is battle-tested on synthetic data.

### Not yet run on real data
- `signal_detection_report()` has never been executed on the actual 454K-sample barrier cache. It was tested only on small synthetic datasets in unit tests.

## Known Failure Modes

1. **Process crashes on large datasets (exp-004 precedent).** exp-004 was killed twice by SIGPIPE when processing all 454K samples. `signal_detection_report()` calls `fit_gbt()` which fits 200 trees on ~272K training rows × 220 features — significantly heavier than the RF in exp-004. **Mitigation:** Run locally (no parent process to SIGPIPE). LightGBM is memory-efficient (histogram-based). The 454K × 220 feature matrix is ~380 MB float64 — should fit comfortably in memory.

2. **Massive overfitting gap.** T6 saw 90% train → 39% test accuracy. The LR and GBT may similarly overfit, but Brier score penalizes overconfident predictions directly (unlike accuracy). A model that memorizes training data will produce extreme predicted probabilities near 0 or 1 on test data, yielding *worse* Brier than the constant. **Watch for:** GBT Brier > constant Brier, which would indicate harmful overfitting.

3. **Logistic regression on 220-dim with 272K samples may barely beat constant.** LR is linear and the 220-dim space is high-dimensional. With n/p ≈ 1236, regularization is critical. The default C=1.0 may be suboptimal. **Not a blocker** — this is an experiment, not a hyperparameter search.

4. **Temporal autocorrelation in labels.** Labels from adjacent bars within a session overlap in time (mean race duration ~12 bars). The block bootstrap (block_size=50) partially addresses this, but the effective sample size may be smaller than N. Bootstrap p-values should be interpreted cautiously.

5. **GBT internal cross-validation not used.** The `fit_gbt` function uses fixed hyperparameters, not tuned via internal CV. This is appropriate for a signal detection experiment (we're asking "is there any signal?" not "what's the best model?") but means the GBT may be under- or over-regularized.

## Key Codebase Entry Points

| File | Function | Role |
|------|----------|------|
| `python/lob_rl/barrier/first_passage_analysis.py:397` | `signal_detection_report()` | **Primary entry point.** Runs full Phase 2 analysis. |
| `python/lob_rl/barrier/first_passage_analysis.py:294` | `load_binary_labels()` | Loads cache into X, Y_long, Y_short with session boundaries. |
| `python/lob_rl/barrier/first_passage_analysis.py:251` | `fit_logistic()` | L2-regularized logistic regression. |
| `python/lob_rl/barrier/first_passage_analysis.py:260` | `fit_gbt()` | LightGBM (preferred) or sklearn GBT. |
| `python/lob_rl/barrier/first_passage_analysis.py:120` | `paired_bootstrap_brier()` | Block bootstrap for Brier score delta significance test. |
| `python/lob_rl/barrier/first_passage_analysis.py:17` | `brier_score()` | Core metric computation. |
| `python/lob_rl/barrier/first_passage_analysis.py:47` | `temporal_split()` | 60/20/20 chronological split by session. |
| `python/tests/barrier/test_first_passage_analysis.py` | Test class | ~50 tests covering all Phase 2 functions. |
| `cache/barrier/*.npz` | Data files | 248 sessions, 454K rows, 220-dim features. |

## Architectural Priors

This is a **tabular binary classification** problem:
- Input: 220-dim feature vector (22 features × 10 lookback steps, z-score normalized, clipped ±5)
- Output: probability p ∈ [0,1] for binary Y ∈ {0,1}
- Samples: ~454K total, ~272K train (60%), ~91K val (20%), ~91K test (20%)
- Features: mix of continuous (price, volume) and bounded (imbalance, ratios)

**Why logistic regression and GBT are appropriate first models:**
- LR tests whether any linear combination of the 220 features predicts Y.
- GBT tests whether nonlinear interactions improve over linear — GBTs are the strongest general-purpose tabular learner.
- Both produce calibrated probabilities (LR natively, GBT approximately via sigmoid of log-odds).
- MLP is deliberately excluded at this stage: if LR and GBT both fail, adding neural network complexity won't help.

**Brier score as the metric:**
- Proper scoring rule — rewards calibration, not just discrimination.
- Constant predictor baseline is well-defined: ȳ(1-ȳ) ≈ 0.218.
- A model can have high accuracy but poor Brier score (if overconfident), or low accuracy but good Brier score (if well-calibrated).
- The Brier skill score (BSS = 1 - BS_model/BS_constant) gives the fraction of predictable variance explained.

**Expected signal magnitude (from T6 analogy):**
- T6 showed +5pp accuracy gain in 3-class. Converting this to Brier improvement is not straightforward, but a rough estimate: if the model shifts predicted probabilities from 0.32 to [0.28, 0.36] based on features, the Brier improvement over constant is approximately Var(p̂) ≈ 0.04² × proportion ≈ 0.0016. This would give BSS ≈ 0.0016/0.218 ≈ 0.007, or about **0.7% of predictable variance explained**.
- This is a very small signal. The bootstrap test with 454K samples may detect it, but the practical significance is questionable.

## External Context

**Signal detection in financial microstructure is well-studied:**
- Cross-sectional predictability in LOB data is typically weak at the individual-bar level. Strong signals exist only at very short horizons (milliseconds) or very long horizons (days), not at the ~5-minute bar scale we're using.
- Brier score improvements of 0.001–0.005 over the constant baseline are typical for well-engineered financial probability models. BSS > 0.01 would be remarkably strong.
- GBT (LightGBM/XGBoost) consistently outperforms LR and neural networks on tabular financial data with limited sample sizes. This is the standard finding across academic and industry work.
- Temporal CV with expanding windows is the correct protocol for financial time series. The code implements this correctly.

**Profitability threshold:** The research plan notes that for the 2:1 reward:risk ratio with C=2 ticks transaction cost, profitability requires predicted p > 0.40. If the model's max prediction never exceeds 0.40, the signal exists but isn't tradeable. The `signal_detection_report()` tracks `max_pred_logistic` and `max_pred_gbt` for this check.

## Constraints and Considerations

1. **Compute:** Local only. LightGBM on 272K × 220 should fit in < 2 minutes. LR in < 30 seconds. Total wall time for `signal_detection_report()` including 5-fold CV: estimated 15–30 minutes.

2. **Memory:** Feature matrix X is 454K × 220 × 8 bytes ≈ 800 MB float64. Plus model copies during CV. Total memory peak ~2–3 GB. Should be fine on local machine.

3. **Bootstrap cost:** The `signal_detection_report()` uses n_boot=1000 (not the 10000 from the standalone function default). Each bootstrap iteration is fast (array indexing + mean), so 1000 bootstraps × 4 (model, label) combinations ≈ 4000 bootstraps total. < 1 minute.

4. **No hyperparameter tuning.** LR uses C=1.0. GBT uses fixed hyperparameters from the research plan. This is deliberate — Phase 2 is a signal detection gate, not a model optimization phase.

5. **Single seed.** The function takes seed=42. For a gate experiment, one seed is sufficient. If the result is borderline, re-running with multiple seeds is a natural follow-up.

## Recommendation

**The FRAME agent should design a straightforward Phase 2 signal detection experiment:**

1. **Just call `signal_detection_report()`.** The infrastructure is complete and tested. The experiment is a single function call on the full dataset.

2. **Key outputs to pre-commit:**
   - `brier_constant_long` / `brier_constant_short` (≈ 0.218)
   - `brier_logistic_long` / `brier_gbt_long` (and short variants)
   - `bss_logistic_long` / `bss_gbt_long` (Brier skill scores)
   - `delta_logistic_long` / `delta_gbt_long` (bootstrap delta with CI and p-value)
   - `signal_found` (True if any delta CI excludes 0)

3. **Success criterion:** At least one of {logistic, GBT} × {Y_long, Y_short} achieves Brier score significantly below the constant baseline (bootstrap CI for delta excludes 0, p < 0.05).

4. **Most likely outcome:** Based on T6's +5pp accuracy gain, signal likely exists but is weak. GBT will likely beat constant by a small margin (BSS ≈ 0.005–0.02). LR may or may not beat constant — if it does, signal is partially linear. If only GBT beats constant, signal is nonlinear.

5. **Failure scenario:** If no model beats constant, the features have discriminative power (T6 showed this) but not calibrated probabilistic power at the binary level. This would be surprising but informative — it would suggest the signal is in the joint (Y_long, Y_short) structure, not the marginals.

6. **What NOT to do:** Do not add new features, tune hyperparameters, or try other model families in this experiment. Phase 2 is a gate, not an optimization. Keep it simple.
