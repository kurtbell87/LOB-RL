# Experiment: Observation Space Signal Audit — Supervised Classification

## Hypothesis

A supervised MLP classifier trained on the 20-dim market features (21-dim bar-level obs minus the agent position feature) will predict next-bar price direction with **test accuracy > 55%** on at least one bar-size configuration, demonstrating that the observation space contains exploitable predictive signal that RL is failing to exploit.

Specifically:
1. **Bar-level (20-dim) classifier at bar_size=1000 will achieve test accuracy > 55% on the 3-class no-exec-cost oracle** (short/flat/long, where flat = zero mid_delta).
2. **If bar_size=1000 fails, bar_size=200 or bar_size=500 will achieve test accuracy > 55%**, indicating signal exists at higher frequency but is destroyed by aggregation.

**Null hypothesis:** Test accuracy at all bar sizes is within ±2% of the majority-class baseline (~50%), indicating the 20-dim features lack predictive power for next-bar direction. No RL algorithm can succeed on features that a supervised classifier cannot exploit.

**Why 55%?** In a 3-class problem where the majority class is ~40-50%, 55% represents a 5-15 percentage point lift over baseline. This is a meaningful but modest threshold. At 55% accuracy, a naive trading strategy (go long when predicted "long," go short when predicted "short," flat otherwise) would have positive expected gross PnL, assuming equal-magnitude moves. A lower bar (e.g., 52%) would be too close to noise on our sample size.

## Independent Variables

| Variable | Values | Notes |
|----------|--------|-------|
| `bar_size` | **200, 500, 1000** | Tests whether signal exists at higher frequencies and is destroyed by aggregation. 1000 is the RL training config. 200 is ~5x finer (more bars per day, shorter prediction horizon). |
| Feature set | **20-dim bar-level** (primary), **53-dim tick-level** (secondary upper bound) | Bar-level matches the RL agent's input. Tick-level establishes maximum achievable accuracy from the raw features before aggregation. |
| Oracle definition | **3-class no-exec-cost** (primary), **3-class with-exec-cost** (secondary) | No-exec-cost: short if mid_delta < 0, flat if = 0, long if > 0. With-exec-cost: short if -mid_delta > half_spread, flat if |mid_delta| ≤ half_spread, long if mid_delta > half_spread. |

**Total configurations:** 3 bar sizes × 2 feature sets × 2 oracle definitions = 12 classifiers. Plus 1 tick-level baseline (53-dim, no bar aggregation) × 2 oracle definitions = 2. Total: 14 classifier configurations.

Note: The tick-level (53-dim) configurations only apply at the original tick resolution — they are NOT aggregated into bars. They serve as an upper bound on the information content of the raw features.

## Controls

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MLP architecture | 2×256 ReLU | Matches the RL policy network. Ensures the comparison is "same model capacity, better training signal (supervised vs RL)." |
| Additional architectures | 2×64 Tanh (SB3 default), 2×64 ReLU, 2×512 ReLU | Tests whether model capacity matters. If 2×512 >> 2×256, the RL policy may be capacity-limited. |
| Training epochs | 100 | Sufficient for convergence on this dataset size. Existing script uses 100. |
| Batch size | 512 | Standard for tabular classification. |
| Optimizer | Adam, lr=1e-3 | Standard. Matches existing script. |
| Normalization | Z-score (fit on train, apply to test) | Standard supervised learning. Differs from VecNormalize (running mean/var) used in RL — this is intentional to test feature quality independent of normalization method. |
| Train/test split | First 199 days train, last 50 days test (chronological) | Matches the RL 199-day config. Chronological split is stricter than shuffle-split for testing temporal generalization. |
| Random seeds | 42, 43, 44 | 3 seeds for model initialization variance. Data split is fixed (chronological). |
| Position feature (dim 20) | **Excluded** | Agent state, not market signal. The supervised test asks "can market features predict price direction?" not "can the model learn given a position." |
| Software | PyTorch (same version as RL experiments), numpy, scikit-learn for metrics | |
| Hardware | Local (Apple Silicon CPU) | No GPU needed for supervised MLP on this dataset size. |

## Metrics (ALL must be reported)

### Primary

1. **Test accuracy** — Fraction of correctly classified test samples. The headline metric for each configuration.
2. **Test accuracy delta over majority-class baseline** — Test accuracy minus the majority-class test frequency. This is the signal-above-noise measure.

### Secondary

| Metric | Purpose |
|--------|---------|
| Majority-class baseline (train and test) | Reference. If baseline is 48%, 55% accuracy = 7-point lift. |
| Train accuracy | Overfitting diagnostic. If train >> test, features overfit. If train ≈ test ≈ baseline, no signal. |
| Per-class precision, recall, F1 | Identifies whether signal is concentrated in one class (e.g., model only predicts "flat" but does so well). |
| Overfit-64 accuracy | Can the 2×256 model memorize a 64-sample batch to 100%? Verifies model capacity is sufficient. |
| Label distribution (train and test, per oracle definition) | Class balance. Heavy imbalance (e.g., 80% "flat" with exec cost) makes accuracy misleading. Report balanced accuracy alongside. |
| Balanced accuracy (macro-averaged recall) | Corrects for class imbalance. More informative than raw accuracy when classes are unbalanced. |
| Per-feature permutation importance (test set) | Identifies which features (if any) carry signal. Reports the mean accuracy drop when each feature is permuted. Top-5 most important features should be highlighted. |
| Cross-entropy loss (test set) | Calibration — is the model confidently wrong or uncertain? Low loss + low accuracy = overconfident mistakes. |
| Confusion matrix | Full 3×3 matrix for each configuration. Reveals systematic biases (e.g., model never predicts "short"). |

### Sanity Checks

| Check | Expected | If Violated |
|-------|----------|-------------|
| Overfit-64 accuracy > 90% for 2×256 ReLU | Model has sufficient capacity to memorize | If < 90%, model is too small — add larger architecture |
| Tick-level test accuracy ≥ bar-level test accuracy at bar_size=1000 | Aggregation cannot create signal | If bar-level > tick-level, something is wrong with the tick pipeline |
| Bar_size=200 test accuracy ≥ bar_size=1000 test accuracy | Finer granularity preserves more signal | If 1000 > 200, the aggregation is adding useful smoothing — not destroying signal |
| "Flat" class frequency with exec cost > 50% | Most bar moves are smaller than half-spread | If flat < 50%, the spread or bar_size assumptions are wrong |
| All 3 classes appear in test predictions | Model is not trivially degenerate | If model predicts only one class, accuracy = baseline and model learned nothing |

## Baselines

| Baseline | Description | Expected Accuracy |
|----------|-------------|-------------------|
| Majority-class classifier | Always predicts the most common class | ~40-50% (depends on oracle definition and bar size) |
| Random classifier | Predicts uniformly at random among 3 classes | 33.3% |
| Existing `supervised_diagnostic.py` on tick-level | 53-dim tick features, step_interval=10 | Unknown — has never been run. This experiment runs it. |

No published external baselines exist for this exact feature set and instrument. The majority-class baseline is the primary reference.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1:** At least one bar-level (20-dim) configuration achieves mean test accuracy > 55% across 3 seeds (mean of seeds 42, 43, 44) on the 3-class no-exec-cost oracle. This would demonstrate exploitable signal exists.
- [ ] **SC-2:** The best bar-level configuration's test accuracy minus majority-class baseline > 5 percentage points (mean across seeds). This ensures the lift is meaningful, not an artifact of class distribution.
- [ ] **SC-3:** The test accuracy delta over baseline is statistically significant: the 95% confidence interval of the 3-seed mean delta excludes zero. With 3 seeds, this requires the delta to be at least ~2.5× the seed standard deviation.
- [ ] **SC-4:** No sanity check failures (overfit capacity, monotonicity across granularities, class coverage).
- [ ] **SC-5:** Train accuracy is within 10 percentage points of test accuracy for the best configuration (no gross overfitting in the supervised setting). If the supervised model overfits, the signal may be spurious.

**Interpretation guide:**

- SC-1 + SC-2 + SC-3 pass → **CONFIRMED**: Predictive signal exists in the observation space. RL is failing to exploit it. Investigate RL training dynamics (reward shaping, exploration, credit assignment). Feature engineering may not be the bottleneck.
- SC-1 passes but SC-3 fails (not statistically significant) → **INCONCLUSIVE**: There may be weak signal, but 3 seeds cannot confirm it. Run with more seeds (10+) or larger test set.
- SC-1 fails at bar_size=1000 but passes at bar_size=200 or 500 → **PARTIALLY CONFIRMED**: Signal exists at higher frequency but is destroyed by bar_size=1000 aggregation. The RL agent should train at the finer bar size (requires infrastructure check — more bars per episode = longer training).
- SC-1 fails at all bar sizes AND tick-level also fails → **REFUTED**: The feature set lacks predictive power for next-step price direction. Future work must focus on richer features (order flow, trade direction, higher-frequency microstructure signals) before any RL algorithm can succeed. This is the most important possible outcome — it terminates an entire class of RL experiments.
- SC-1 fails at all bar sizes but tick-level shows signal → **PARTIALLY CONFIRMED (feature engineering)**: Raw tick-level features contain signal that bar aggregation destroys. The aggregation function is the bottleneck, not the underlying data. Redesign bar aggregation to preserve predictive tick-level patterns.

## Minimum Viable Experiment

Before the full protocol, run a single quick validation:

1. **Load 10 days of cached data at bar_size=1000.**
2. **Extract 20-dim bar-level features and 3-class no-exec-cost labels.**
3. **Train a 2×256 ReLU MLP for 50 epochs on the first 8 days, test on the last 2.**
4. **Verify:**
   - Features load without error, shape is (N, 20).
   - Labels are distributed across all 3 classes.
   - Overfit-64 accuracy > 90% (model capacity check).
   - Training completes in < 2 minutes.
   - Train and test accuracy are reported and reasonable (between 30% and 99%).
5. **If MVE fails:** The supervised diagnostic pipeline needs debugging before running the full protocol. Most likely: bar-level feature extraction needs implementation (the existing `supervised_diagnostic.py` is tick-level only).

**MVE success gate:** Feature extraction, label generation, training, and evaluation complete without error. Overfit-64 > 90%. Results are numerically reasonable. Only proceed to full protocol if MVE passes.

**Implementation note:** The existing `supervised_diagnostic.py` operates at tick-level (53-dim) only. The RUN agent will need to implement bar-level feature extraction using `BarLevelEnv`'s `aggregate_bars()` and `_precompute_temporal()` logic, or extend the existing script. This is a diagnostic script, not a production code change — it should live in `scripts/` alongside the existing diagnostic.

## Full Protocol

### Phase 0: Infrastructure Validation (MVE)

1. Verify the existing `supervised_diagnostic.py` runs on tick-level data without errors:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python \
     ../scripts/supervised_diagnostic.py --cache-dir ../cache/mes/ \
     --train-days 10 --epochs 50
   ```
2. Implement or extend the diagnostic script to support bar-level feature extraction at multiple bar sizes. The script must:
   - Load `.npz` cache files
   - Call `aggregate_bars()` with configurable `bar_size`
   - Apply `_precompute_temporal()` logic for cross-bar features
   - Exclude position feature (use 20-dim market features only)
   - Generate oracle labels from `bar_mid_close[t+1] - bar_mid_close[t]` and `bar_spread_close[t]`
   - Support both with-exec-cost and no-exec-cost oracle definitions
   - Report majority baseline, train/test accuracy, balanced accuracy, per-class metrics
   - Report per-feature permutation importance on test set
   - Support multiple random seeds for model initialization
3. Run MVE (10 days, bar_size=1000, 50 epochs). Verify per MVE criteria above.

### Phase 1: Tick-Level Baseline (Upper Bound)

4. Run the existing `supervised_diagnostic.py` at tick-level on full dataset:
   ```bash
   cd build-release && PYTHONPATH=.:../python uv run python \
     ../scripts/supervised_diagnostic.py --cache-dir ../cache/mes/ \
     --train-days 199 --epochs 100
   ```
   This establishes the upper bound: how much signal exists in the raw 53-dim tick features before any bar aggregation.

### Phase 2: Bar-Level Classification (Primary Test)

5. For each bar_size in {200, 500, 1000}:
   - For each seed in {42, 43, 44}:
     - For each oracle in {no-exec-cost, with-exec-cost}:
       - Train 4 MLP architectures (2×64 Tanh, 2×64 ReLU, 2×256 ReLU, 2×512 ReLU)
       - Record: train accuracy, test accuracy, balanced accuracy, per-class precision/recall/F1, confusion matrix, cross-entropy loss
     - Report per-feature permutation importance for the 2×256 ReLU model on the no-exec-cost oracle (1 permutation importance run per bar_size per seed — the most relevant configuration)

6. For each configuration, compute the mean and std of test accuracy across 3 seeds.

### Phase 3: Statistical Analysis

7. For each bar_size and oracle definition:
   - Compute mean test accuracy across 3 seeds.
   - Compute the delta: mean test accuracy minus majority-class baseline.
   - Compute 95% CI for the delta: mean ± t(0.025, df=2) × std / sqrt(3), where t(0.025, 2) ≈ 4.303.
   - Report whether the CI excludes zero.

8. Identify the best-performing configuration (highest mean test accuracy delta over baseline).

9. Report per-feature permutation importance (averaged across 3 seeds) for the best configuration. Identify top-5 features.

### Phase 4: Collect Results

10. Write all results to `results/exp-003-does-the-observation-space-contain-any-p/metrics.json` in the standard format:
    ```json
    {
      "tick_level": { ... },
      "bar_level": {
        "bar_size_200": { "no_exec_cost": { "seed_42": { ... }, ... }, ... },
        "bar_size_500": { ... },
        "bar_size_1000": { ... }
      },
      "best_config": { ... },
      "permutation_importance": { ... },
      "success_criteria": { "SC-1": bool, "SC-2": bool, ... }
    }
    ```

11. Evaluate all success criteria. Record pass/fail.

### Commands

All commands run locally. No GPU or AWS needed.

**MVE (tick-level, 10 days):**
```bash
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/supervised_diagnostic.py --cache-dir ../cache/mes/ \
  --train-days 10 --epochs 50
```

**Full tick-level baseline:**
```bash
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/supervised_diagnostic.py --cache-dir ../cache/mes/ \
  --train-days 199 --epochs 100
```

**Bar-level diagnostic (the RUN agent must implement or extend the script):**
```bash
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/supervised_diagnostic.py --cache-dir ../cache/mes/ \
  --train-days 199 --epochs 100 --bar-size 1000 --seeds 42,43,44
```

Repeat for `--bar-size 200` and `--bar-size 500`.

## Resource Budget

- Max GPU-hours: **0** (all CPU)
- Max CPU-hours: **2**
- Max wall-clock time: **2 hours**
- Max training runs: **168** (14 configurations × 4 architectures × 3 seeds)
- Max seeds per configuration: **3** (42, 43, 44)

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| MVE (10 days, 1 config) | ~2 min | Quick validation |
| Tick-level baseline (199 days, 4 architectures, 2 oracles) | ~10 min | Existing script, no modification needed |
| Bar-level bar_size=1000 (4 arch × 2 oracle × 3 seeds) | ~15 min | 24 training runs |
| Bar-level bar_size=500 (same) | ~20 min | More samples per day, slightly longer |
| Bar-level bar_size=200 (same) | ~30 min | Most samples, longest per run |
| Permutation importance (9 runs) | ~15 min | 3 bar_sizes × 3 seeds |
| Total | **~90 min** | Conservative estimate. Actual likely < 60 min. |

This is well within the 8 GPU-hour budget (uses 0 GPU-hours) and well within the 10-run budget (training "runs" here are lightweight supervised fits, not RL training episodes).

## Compute Target

**Compute:** `local`

This is a supervised classification experiment using tabular MLP models on cached numpy data. No GPU needed — all runs complete in minutes on CPU. Total dataset is ~18GB of cached .npz files, but only the extracted features (~100MB after aggregation) are loaded into memory.

## Abort Criteria

| Condition | Action |
|-----------|--------|
| Feature extraction crashes or produces NaN | Abort. Debug the feature extraction pipeline. Likely a divide-by-zero in bar aggregation or temporal computation. |
| Overfit-64 accuracy < 50% for 2×256 ReLU | Abort. Model capacity is fundamentally broken — check the MLP implementation. |
| Any single run takes > 10 minutes | Abort that run. Investigate — likely a data loading or memory issue. Reduce dataset size. |
| Label distribution is 100% one class | Abort for that oracle/bar_size combination. The oracle definition is degenerate at this granularity. |
| Total wall time exceeds 2 hours | Complete the current phase and report partial results. Prioritize bar_size=1000 (the RL training config) over 200 and 500. |

## Confounds to Watch For

1. **Z-score normalization vs VecNormalize.** The supervised classifier uses per-feature z-scoring (mean/std from training data). The RL agent sees VecNormalize'd observations (running mean/variance updated incrementally). If the supervised classifier finds signal but RL doesn't, the normalization difference is one possible explanation. **Mitigation:** Report both z-score and raw (unnormalized) test accuracy. If signal disappears without normalization, it may be normalization-dependent.

2. **Chronological vs shuffle split.** The RL experiments use shuffle-split; this experiment uses chronological split (first 199 days train, last 50 test). If signal exists in shuffle-split but not chronological, the "signal" is cross-day distributional similarity, not temporal prediction. **Mitigation:** Report results with both split methods. Primary analysis uses chronological (stricter). If chronological fails but shuffle-split shows signal, flag as a confound.

3. **Oracle label noise at bar level.** The oracle is `sign(mid_close[t+1] - mid_close[t])`. At bar_size=1000, the mid_close is a single price snapshot at bar boundary. If two adjacent bars have very similar closing mids, the oracle labels flip randomly based on sub-tick price jitter. This creates label noise that suppresses test accuracy. **Mitigation:** Report the fraction of "small moves" (|mid_delta| < 1 tick = $1.25). If > 30% of labels are effectively noise, consider a 2-class oracle (remove the "flat" class or define flat as |delta| < 1 tick).

4. **Temporal autocorrelation in train/test.** With chronological split, the last training day and first test day are adjacent calendar days. Features may be autocorrelated (e.g., spread levels tend to persist across days). This can inflate test accuracy if the model learns the level rather than the predictive relationship. **Mitigation:** Report test accuracy on the first 10 test days vs last 10 test days. If accuracy decays with distance from training data, temporal leakage is present.

5. **Information leakage via temporal features.** The 7 temporal features (return_lag1, return_lag3, etc.) are backward-looking. They cannot directly predict the future. However, if there is strong short-term mean-reversion or momentum, these features are legitimately informative. This is not a confound — it's the kind of signal we're looking for. The only risk is if the temporal feature computation accidentally includes the current bar's return in its own lag (which would be leakage). **Mitigation:** Verify that `return_lag1` at bar t equals `bar_return` at bar t-1, NOT bar t.

6. **Tick-level vs bar-level comparisons are not apples-to-apples.** Tick-level has ~350,000 samples per day; bar_size=1000 has ~350 samples per day. The tick-level model trains on ~1,000× more samples. Higher test accuracy at tick-level could reflect larger training set, not richer features. **Mitigation:** The comparison is directional (upper bound), not quantitative. We do not claim tick-level is "N% better" — we note whether signal exists at any level.

7. **Feature set is not the RL agent's actual input.** The RL agent sees VecNormalize'd 21-dim observations (including position). The supervised classifier sees z-scored 20-dim observations (excluding position). If the supervised test shows signal but RL doesn't, the difference could be VecNormalize, the position feature, or the RL training dynamics. The supervised test isolates the question of feature quality from RL-specific confounds. **Mitigation:** This is by design. The goal is to test feature quality, not RL-specific issues.
