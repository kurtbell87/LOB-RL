# Experiment: 22-Feature Barrier Supervised Diagnostic — Feature Ablation

## Hypothesis

The expanded 22-feature barrier observation set (220-dim with h=10 lookback) will produce RF classification accuracy **>42.5%** on the bidirectional 3-class task (long_profitable / short_profitable / flat), representing a **>2pp improvement over the 9-active-feature baseline RF accuracy of 40.5%**. The improvement will be driven primarily by the 9 newly activated book + microstructure features (OFI, depth ratio, weighted mid displacement, spread std, VAMP displacement, aggressor imbalance, trade arrival rate, cancel-to-trade ratio, price impact per trade) and 3 previously dead features now activated by the LOB Reconstructor (bbo_imbal, depth_imbal, cancel_asym).

**Null hypothesis:** The 22-feature RF accuracy is within ±2pp of the 9-feature baseline (40.5%), indicating the new features carry no incremental signal for barrier-hit direction prediction. The improvement, if any, is within seed-to-seed variance.

**Why 2pp?** The baseline is 40.5% on a 34.5% majority-class baseline (+6pp lift). A 2pp improvement (to 42.5%) would represent a 33% increase in signal above chance (from +6pp to +8pp). This is the minimum threshold to justify the added complexity of 22 features in the RL observation space. Smaller improvements would be consumed by the curse of dimensionality in RL training.

## Independent Variables

| Variable | Values | Notes |
|----------|--------|-------|
| Feature set | **A: All 22 features (220-dim)**, **B: Original 9 active features (90-dim)**, **C: New 13 features only (130-dim)** | A is the full set. B reconstructs the T6 baseline on the new cache. C isolates the new features' contribution. |
| Model | **RF (100 trees)**, **MLP (2×256 ReLU)** | RF is primary (stable, importance for free). MLP is secondary (RL policy proxy). |
| Split method | **Shuffle-split (80/20)**, **Chronological (first 80% / last 20%)** | Both reported. T6 used both; we maintain comparability. |

**Feature set column mapping (in the 22-feature cache):**

| Set | Columns (0-indexed) | Dim (×10 lookback) | Description |
|-----|---------------------|---------------------|-------------|
| B (original 9 active) | 3,4,5,6,7,8,9,12 + col 0 if alive | 90 | bar_range, bar_body, body_range_ratio, vwap_displace, volume_log, realized_vol, session_time, session_age, trade_flow_imbal. Note: col 0 (trade_flow_imbal) is dead in the new cache per survey. Use cols 3-9,12 = 8 features × 10 = 80-dim. See confound #5. |
| C (new 13) | 1,2,10,11,13,14,15,16,17,18,19,20,21 | 130 | bbo_imbal, depth_imbal, cancel_asym, mean_spread, OFI, depth_ratio, weighted_mid_disp, spread_std, VAMP_disp, aggressor_imbal, trade_arrival, cancel_to_trade, price_impact |
| A (all 22) | 0-21 | 220 | Everything |

**IMPORTANT — Dead features in the new cache:** The survey identifies **2/22 dead features**: col 0 (trade_flow_imbal) and col 11 (mean_spread), both with std=0. This means:
- Set A (all 22): includes 2 dead features (220-dim, 200 effective)
- Set B (original 9 active): trade_flow_imbal (col 0) was the **top feature** in T6 (importance 0.128) but is now dead in the new cache. This is a critical change — set B is now 8 active features, not 9. Use only cols 3-9, 12 = 80-dim.
- Set C (new 13): includes 1 dead feature (mean_spread, col 11). 12 effective features = 120 effective dim.

**Total configurations:** 3 feature sets × 2 models × 2 splits × 5 seeds = 60 classifier runs.

## Controls

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Barrier parameters | bar_size=500, lookback h=10, stop=10 ticks, profit=20 ticks | Matches the existing barrier cache. Same as T6. |
| RF hyperparameters | n_estimators=100, max_features='sqrt', random_state=seed | Matches T6. Default sklearn RF. No tuning — this is signal detection, not model optimization. |
| MLP architecture | 2×256 ReLU, Adam lr=1e-3, 100 epochs, batch_size=512 | Matches T6. RL policy proxy. |
| Label framing | Bidirectional 3-class: {long_profitable, short_profitable, flat} | Matches T6 v2. Balanced ~33/33/35%. |
| Dataset | `cache/barrier/*.npz` — 248 sessions, ~454K usable bars | Same data as T6 but re-precomputed with C++ backend + 22 features. |
| Normalization | Features are pre-normalized (z-score with trailing window, clipped [-5,+5]). No additional normalization. | Matches T6. The barrier cache applies normalization at precompute time. |
| Random seeds | 42, 43, 44, 45, 46 | 5 seeds (up from T6's single run) to assess variance. Survey recommends 5+. |
| Software | scikit-learn (RF), PyTorch (MLP), numpy | Same as T6. |
| Hardware | Local (Apple Silicon CPU) | No GPU needed. |

## Metrics (ALL must be reported)

### Primary

1. **RF test accuracy (balanced) on all-22-features (set A) vs original-9 (set B)** — The direct comparison that tests the hypothesis. Balanced accuracy corrects for the ~33/33/35% class distribution.
2. **RF accuracy delta: set A minus set B** — The incremental signal from the new features. Must be >2pp to confirm hypothesis.

### Secondary

| Metric | Purpose |
|--------|---------|
| MLP test accuracy (balanced) per feature set | RL policy proxy — can the MLP architecture exploit the new features? |
| Majority-class baseline (per split) | Reference for signal above chance. |
| Train accuracy per configuration | Overfitting diagnostic. T6 showed 90% train → 39% test for MLP. |
| Per-feature-group permutation importance (RF, set A) | Which feature *groups* drive the improvement? Aggregate importance by: original trade features (cols 3-9,12), book features (cols 1,2,10,13,14), microstructure features (cols 15-21). |
| Individual feature permutation importance (RF, set A) | Rank all 22 features. Compare against T6's ranking (where trade_flow_imbalance was #1). |
| Chrono vs shuffle accuracy gap | If shuffle >> chrono, signal may be distributional similarity, not genuine. |
| Set C (new-features-only) accuracy | Can the new features alone match or beat the original 9? |

### Sanity Checks

| Check | Expected | If Violated |
|-------|----------|-------------|
| Set B accuracy within ±3pp of T6 baseline (40.5% RF shuffle) | Confirms the new cache produces comparable results on the same features | If >3pp off, the cache rebuild changed something (normalization, bar boundaries, label distribution). Investigation required before interpreting set A results. |
| Label distribution within ±3pp of T6 (33/33/35%) | Same barrier parameters → same distribution | If distribution shifted, the precompute pipeline changed barrier hit rates. |
| Dead features (cols 0, 11) have importance ≈ 0.000 | Dead features carry no information | If importance > 0.01, the feature is not actually dead — re-check std. |
| MLP overfit-256 accuracy > 95% on 256-sample subset | Model capacity sufficient | If < 95%, implementation bug. |
| 5-seed std < 2pp for RF accuracy | RF should be stable across seeds | If std > 2pp, the signal is too unstable for a 2pp threshold to be meaningful. |

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| RF 9-active-feature shuffle accuracy | T6 v2 result (2026-02-10) | 40.5% |
| RF 9-active-feature chrono accuracy | T6 v2 result | 39.6% |
| MLP 9-active-feature shuffle accuracy | T6 v2 result | 39.3% |
| MLP 9-active-feature chrono accuracy | T6 v2 result | 39.1% |
| Majority-class baseline | T6 v2 result | 34.5% (shuffle), 35.0% (chrono) |
| Set B (reconstructed baseline on new cache) | This experiment | To be measured — the primary sanity check |

**NOTE:** The T6 baseline was computed on the **old 130-dim cache** with 9 active features. The new 220-dim cache was re-precomputed with the C++ backend. Set B on the new cache may differ slightly from T6 due to implementation differences between the Python and C++ precompute pipelines. This is why the set B sanity check is critical.

**ADDITIONAL NOTE:** T6's #1 feature (trade_flow_imbal, col 0) is **dead** in the new cache. Set B in this experiment has only 8 active features (cols 3-9, 12), not 9. If set B accuracy is significantly lower than T6's 40.5%, the loss of trade_flow_imbal is the likely explanation. This would make the set A vs set B comparison MORE favorable to the hypothesis (lower baseline to beat). The READ agent must flag this and compare set A vs T6 directly as well.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1:** Mean RF balanced accuracy on set A (all 22 features) exceeds mean RF balanced accuracy on set B (original features) by >2pp across 5 seeds on shuffle-split. CI of the paired difference excludes zero.
- [ ] **SC-2:** Mean RF balanced accuracy on set A exceeds 42.5% on shuffle-split (direct threshold test).
- [ ] **SC-3:** The improvement in SC-1 is also observed on chronological split (set A > set B by >1pp). Direction must be consistent across both splits — if chrono shows regression, the shuffle-split improvement is likely spurious.
- [ ] **SC-4:** All sanity checks pass (set B reproduces T6 within ±3pp, label distribution stable, dead features have zero importance, MLP overfit test passes, 5-seed std < 2pp).
- [ ] **SC-5:** At least 2 of the 13 new features (cols 1,2,10,13-21) appear in the top-5 feature importance ranking for the set A RF model. This confirms the improvement is driven by the new features, not random interactions.

**Interpretation guide:**

- SC-1 + SC-2 + SC-3 + SC-5 pass → **CONFIRMED**: The 22-feature set carries meaningfully more signal than the original 9. Proceed to RL training with 22 features. Architecture comparison (exp-005+) should use the 22-feature observation.
- SC-1 passes but SC-3 fails → **INCONCLUSIVE**: New features add signal in-distribution (shuffle) but not out-of-distribution (chrono). The new features may capture distributional properties rather than predictive signal.
- SC-1 fails (delta < 2pp or CI includes zero) → **REFUTED**: The 13 new features do not carry incremental signal for barrier-hit prediction. The original 9 features are sufficient. Proceed to RL training with original feature set. Do not invest in additional feature engineering.
- SC-4 fails (set B ≠ T6) → **INCONCLUSIVE**: Cannot compare against T6 baseline. The cache rebuild changed something fundamental. Must investigate before drawing conclusions.

## Minimum Viable Experiment

Before the full protocol, run a single quick validation:

1. Load 10 sessions from `cache/barrier/`.
2. Extract 220-dim features and barrier labels.
3. Verify feature dimensions: `X.shape[1] == 220`, `y ∈ {0, 1, 2}`.
4. Verify label distribution: each class has >20% of samples.
5. Verify dead features: std of cols 0*10:(0*10+10) ≈ 0 and cols 11*10:(11*10+10) ≈ 0.
6. Train RF (100 trees) on 8 sessions, test on 2.
7. Verify: accuracy > 30% and < 60% (reasonable range). Confusion matrix shows all 3 classes predicted.
8. Verify: permutation importance of dead features ≈ 0.
9. Train on set B (cols 3-9, 12 × 10 lookback = 80-dim subset). Verify accuracy is comparable to T6 range (35-45%).

**MVE success gate:** Feature loading, dimension checks, dead feature verification, and classification all complete without error. Results are in reasonable ranges. Only proceed to full protocol if MVE passes.

## Full Protocol

### Phase 0: Infrastructure Validation (MVE)

1. Update `scripts/run_barrier_diagnostic_v2.py`:
   - Change assertion from `X.shape[1] == 130` to `X.shape[1] == 220`.
   - Update `feature_names_base` from 13 entries to 22 entries.
   - Add `--feature-subset` argument to support set A/B/C column selection.
   - Add `--seeds` argument to support multi-seed runs.
   These are trivial script changes (<20 lines), not production code.

2. Run MVE per the criteria above.

### Phase 1: Reconstructed Baseline (Set B — Critical Sanity Check)

3. Run set B (original 8 active features, cols 3-9,12, 80-dim) with RF and MLP:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python \
     ../scripts/run_barrier_diagnostic_v2.py \
     --cache-dir ../cache/barrier/ \
     --feature-subset original \
     --seeds 42,43,44,45,46
   ```
4. Compare against T6 baseline (RF 40.5% shuffle). If delta > 3pp, investigate before proceeding.

### Phase 2: Full Feature Set (Set A — Primary Test)

5. Run set A (all 22 features, 220-dim) with RF and MLP:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python \
     ../scripts/run_barrier_diagnostic_v2.py \
     --cache-dir ../cache/barrier/ \
     --feature-subset all \
     --seeds 42,43,44,45,46
   ```

### Phase 3: New Features Only (Set C — Isolation)

6. Run set C (new 13 features, cols 1,2,10,11,13-21, 130-dim) with RF and MLP:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python \
     ../scripts/run_barrier_diagnostic_v2.py \
     --feature-subset new \
     --seeds 42,43,44,45,46
   ```

### Phase 4: Statistical Analysis

7. For each feature set × model × split:
   - Compute mean and std of balanced accuracy across 5 seeds.
   - For the paired comparison (set A vs set B): compute per-seed delta, mean delta, std of delta. Compute 95% CI: mean ± t(0.025, df=4) × std/√5, where t(0.025, 4) ≈ 2.776.
   - Report whether CI excludes zero.

8. Compute and report permutation importance for RF on set A (averaged across 5 seeds). Rank all 22 features. Group by: original trade (cols 3-9,12), book (cols 1,2,10,13,14), microstructure (cols 15-21), dead (cols 0,11).

### Phase 5: Collect Results

9. Write all results to `results/exp-004-22-feature-supervised-diagnostic/metrics.json`:
   ```json
   {
     "set_a": {
       "rf_shuffle": {"mean_balanced_acc": ..., "std": ..., "seeds": {...}},
       "rf_chrono": {"mean_balanced_acc": ..., "std": ..., "seeds": {...}},
       "mlp_shuffle": {...},
       "mlp_chrono": {...}
     },
     "set_b": { ... },
     "set_c": { ... },
     "paired_delta_a_minus_b": {
       "rf_shuffle": {"mean": ..., "std": ..., "ci_95": [...], "ci_excludes_zero": bool},
       "rf_chrono": {...}
     },
     "feature_importance": {
       "individual": {"feature_name": mean_importance, ...},
       "grouped": {"original_trade": ..., "book": ..., "microstructure": ..., "dead": ...}
     },
     "sanity_checks": {
       "set_b_vs_t6_delta": ...,
       "label_distribution": {...},
       "dead_feature_importance": {...},
       "mlp_overfit_test": ...,
       "seed_std": ...
     },
     "success_criteria": {
       "SC-1": bool, "SC-2": bool, "SC-3": bool, "SC-4": bool, "SC-5": bool
     }
   }
   ```

10. Evaluate all success criteria. Record pass/fail.

## Resource Budget

- Max GPU-hours: **0** (all CPU)
- Max wall-clock time: **3 hours**
- Max training runs: **60** (3 feature sets × 2 models × 2 splits × 5 seeds)
- Max seeds per configuration: **5** (42-46)

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Data loading (248 sessions + label compute) | ~4 min | Python label computation on cached bars |
| MVE (5k samples, RF + overfit check) | ~2 min | Quick validation |
| Set A (22-feature, 2 models × 2 splits × 5 seeds) | ~40 min | RF on 454K × 220 takes ~200s/fit |
| Set B (9-feature, 2 models × 2 splits × 5 seeds) | ~25 min | RF on 454K × 90 takes ~120s/fit |
| Set C (13-feature, 2 models × 2 splits × 5 seeds) | ~30 min | RF on 454K × 130 takes ~150s/fit |
| Permutation importance (set A, RF, 5 seeds, 20K subsample) | ~15 min | 22 base features × 5 repeats × 5 seeds |
| Statistical analysis + write results | ~1 min | Computation only |
| **Total** | **~120 min** | Based on actual 454K sample dataset. |

## Compute Target

**Compute:** `local`

This is a supervised classification experiment on pre-cached numpy arrays. Dataset is 454K samples × 220 dims. RF (100 trees) takes ~200s per fit at this scale. No GPU needed. Total wall time ~2 hours.

## Abort Criteria

| Condition | Action |
|-----------|--------|
| Cache loading fails or produces unexpected dimensions | Abort. Verify `cache/barrier/` files are 220-dim. |
| Set B accuracy < 33% (below majority baseline) | Abort. Something is fundamentally broken — the features have no signal at all on the known-working feature subset. |
| Label distribution has any class < 15% | Abort. Barrier parameters may have changed. Verify precompute settings. |
| Any single RF fit takes > 600 seconds | Abort that run. Investigate — RF on 454K × 220 typically takes ~200s. |
| Total wall time exceeds 3 hours | Complete current phase and report partial results. |
| MLP overfit test (256 samples) fails to reach 90% | Abort MLP runs. Report RF results only. |

## Confounds to Watch For

1. **trade_flow_imbal is dead in the new cache.** This was the #1 feature in T6 (importance 0.128). Its absence from set B means set B is weaker than the T6 baseline. The set A vs set B comparison may overstate the improvement because set B lost a strong feature, not because set A gained strong features. **Mitigation:** Compare set A accuracy directly against the T6 recorded baseline (40.5%), not just against set B. If set A > 40.5% AND set A > set B, both the direct and relative improvements are real.

2. **Python vs C++ precompute differences.** The old cache was built with the Python pipeline; the new cache uses the C++ backend. Subtle differences in bar boundary alignment, normalization window, or rounding could change feature values. **Mitigation:** The set B sanity check (±3pp of T6) catches this. If set B is far from T6, the caches are not comparable.

3. **Feature dimensionality curse.** Going from 80-dim to 220-dim increases the risk that RF overfits to noise dimensions. RF is naturally regularized (sqrt feature sampling), but with 20 features × 10 lookback, some lookback positions of weak features may create spurious splits. **Mitigation:** Report train/test gap for each feature set. If set A has a much larger train/test gap than set B, the improvement is likely overfitting.

4. **Dead feature dilution.** Two dead features (cols 0, 11) contribute 20 constant-zero dimensions. For RF, these are harmless (never selected for splits). For MLP, they waste capacity and could impair optimization. **Mitigation:** Report dead feature importance (should be ≈0). If MLP accuracy is lower on set A than set B, dead feature dilution may explain it.

5. **Baseline reconstruction fidelity.** Set B attempts to reconstruct the T6 baseline by selecting cols 3-9, 12 from the new cache. But the new cache's feature values may differ subtly from the old cache's values for the same columns (different precompute pipeline). The reconstruction may not be exact. **Mitigation:** This is flagged as a known limitation. The comparison against the *recorded* T6 number (40.5%) provides a second reference point. Both comparisons (set A vs set B, set A vs T6 recorded) should agree in direction.

6. **Mean_spread (col 11) was dead in both caches.** It was dead in T6 (importance 0.000) and is dead in the new cache. It appears in set C but carries no information. This reduces set C's effective dimensionality from 130 to 120. **Mitigation:** Exclude dead features from the count of "new features with signal" when evaluating SC-5.
