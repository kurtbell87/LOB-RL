# Survey: Do the 22 barrier features (including 9 new book + microstructure features) significantly improve supervised classification signal vs the old 9-active-feature baseline (RF 40.5%)?

## Prior Internal Experiments

### T6 Supervised Diagnostic v2 (2026-02-10) — The Baseline

The most directly relevant experiment. Ran on the **old 130-dim cache** (13 features x 10 lookback), of which **4/13 features were dead** (bbo_imbal, depth_imbal, cancel_asym, mean_spread — all constant zero because `precompute_barrier_cache.py` didn't pass `mbo_data`). Effectively 9 active features x 10 lookback = 90 effective dimensions.

**Bidirectional framing** (3-class: long_profitable, short_profitable, flat):
- Distribution: 33/33/35% — nearly balanced.
- **RF 40.5%** / **MLP 39.3%** on shuffle-split vs 34.5% baseline (+6pp / +5pp).
- Chrono split: RF 39.6% / MLP 39.1% vs 35.0% baseline.
- Overfit test: MLP 100% on 256 samples (capacity confirmed).
- Train accuracy: MLP ~90% train → 39% test (massive overfitting).

**RF feature importance** (aggregate across lookback, 9 active features):
1. trade_flow_imbalance (0.128)
2. bar_range (0.122)
3. volume_log (0.121)
4. vwap_displacement (0.121)
5. body_range_ratio (0.117)
6. bar_body (0.115)
7. realized_vol (0.114)
8. session_time (0.089)
9. session_age (0.074)

All 9 active features contribute meaningfully (importance 0.07-0.13). No single feature dominates. Dead features: exactly 0.000.

### exp-003 — Next-Bar Direction Classification (earlier, different framing)

Used the **old 20-dim bar-level** BarLevelEnv features (not barrier features). Predicted next-bar price direction (short/flat/long). Results at bar_size=1000:
- 2x256 MLP: test 48.3% vs 48.8% baseline (mean delta **-0.5pp**, CI includes zero).
- **No signal** in the old 20-dim features for next-bar direction prediction.

This is a fundamentally different problem (next-bar direction vs barrier outcome), but establishes that the older feature set had **zero signal** for even a simpler prediction task.

## Current Infrastructure

### Barrier Cache (FRESH, 220-dim)

- **Location:** `cache/barrier/` — 248 `.npz` files.
- **Built with:** C++ backend (`lob_rl_core.barrier_precompute()`) including LOB Reconstructor.
- **Feature dim:** 220 = 22 features x 10 lookback.
- **Dead features in new cache:** 2/22 — `trade_flow_imbal` (col 0) and `mean_spread` (col 11). Both have std=0 across all sessions. The other 20 features have healthy variance.
- **Previously dead, now active:** `bbo_imbal` (col 1), `depth_imbal` (col 2), `cancel_asym` (col 10) — activated by LOB Reconstructor (PR #34). Plus 9 entirely new features (cols 13-21).
- **Total bars:** 461K total, 454K usable.
- **Cache size:** 186 MB.

### New Features (cols 13-21, Phase 1+2 Microstructure)

Phase 1 (PR #35): OFI, depth ratio, weighted mid displacement, spread std.
Phase 2 (PR #36): VAMP displacement, aggressor imbalance, trade arrival rate, cancel-to-trade ratio, price impact per trade.

### Supervised Diagnostic Script

- **v2 script:** `scripts/run_barrier_diagnostic_v2.py` — trains MLP + RF on barrier features with bidirectional framing.
- **INCOMPATIBILITY:** Line 183 has `assert X.shape[1] == 130`. The new cache produces 220-dim features. **The script must be updated** (trivially: change the assertion, update feature_names list from 13 to 22 entries).
- **Feature names list:** Hardcoded to 13 names at line 302-307. Needs 22 names.

### Diagnostic Module

- `python/lob_rl/barrier/supervised_diagnostic.py` — contains `BarrierMLP`, `evaluate_classifier()`, `train_random_forest()`, `run_diagnostic()`. These functions are dim-agnostic (input_dim parameter).
- `precompute_barrier_cache.py::load_session_from_cache()` — loads `.npz` and returns dict with bars + features. Dim-agnostic.

## Known Failure Modes

1. **Script assertion failure:** The v2 diagnostic script asserts `X.shape[1] == 130`. Must be updated to 220.
2. **Feature name mismatch:** The RF importance printout uses a 13-element feature name list. Must be expanded to 22.
3. **Dead feature dilution:** 2/22 features are dead (constant zero after z-score normalization = NaN or zero). The v2 script already handles NaN/Inf replacement (line 196-200). However, dead features consume model capacity and add noise to RF splits. Consider whether to exclude them.
4. **Overfitting risk:** MLP went from 90% train → 39% test on 130-dim. On 220-dim, overfitting will likely be **worse** (more parameters, more capacity to memorize). The RF is more resistant (40.5% on old).
5. **Feature importance instability:** The T6 permutation importance was only computed once per RF. exp-003 showed high variance in permutation importance across seeds. Multiple seeds are essential.
6. **Normalization already applied:** The barrier cache features are **pre-normalized** (z-score with trailing window). The diagnostic script does NOT re-normalize. This is correct — but worth noting that the features are already clipped to [-5, +5].

## Key Codebase Entry Points

| File | Role |
|------|------|
| `scripts/run_barrier_diagnostic_v2.py` | Main diagnostic script. Needs 130→220 fix. |
| `scripts/precompute_barrier_cache.py` | Cache builder + `load_session_from_cache()` |
| `python/lob_rl/barrier/feature_pipeline.py` | Feature extraction: `compute_bar_features()` (22 cols) |
| `python/lob_rl/barrier/supervised_diagnostic.py` | `BarrierMLP`, `evaluate_classifier()` |
| `python/lob_rl/barrier/__init__.py` | `N_FEATURES = 22` constant |
| `python/lob_rl/barrier/lob_reconstructor.py` | `OrderBook` class for book features |
| `cache/barrier/*.npz` | Pre-built 220-dim features |
| `cache/t6_diagnostic_v2_results.json` | Old 130-dim results (baseline comparison) |

## Architectural Priors

This is a **tabular supervised classification** task. No temporal architecture (LSTM, Transformer) is needed here — the lookback window is already flattened into the feature vector.

**MLP + Random Forest are the correct tools.** RF is particularly suitable for:
- Feature importance analysis (built-in via Gini importance or permutation)
- Robustness to dead/irrelevant features (RF ignores them naturally)
- No normalization sensitivity (RF works on raw features)

MLP is useful as the **RL policy architecture proxy** — if MLP can't learn the signal, the RL policy (same architecture) won't either.

The question is not "which model is best" but **"do the additional features improve signal?"** — a feature ablation / comparison study.

## External Context

This is a well-studied problem class: **feature engineering evaluation for financial prediction.**

Key insights from microstructure literature:
- **OFI (Order Flow Imbalance)** is one of the strongest short-horizon predictors of price movement (Cont, Kukanov, Stoikov 2014). It captures the net buying/selling pressure from order book changes.
- **VAMP (Volume-Adjusted Mid Price)** provides a more robust mid-price estimate than simple mid. Displacement from last trade is informative.
- **Depth ratio** captures asymmetric liquidity — predictive of short-term price pressure direction.
- **Aggressor imbalance** (buyer vs seller initiated trades) is a direct measure of informed trading flow.
- **Cancel-to-trade ratio** and **price impact** capture HFT activity patterns.

These features encode order book microstructure dynamics that the original 9 trade-derived features **could not capture**. The original features (bar_range, bar_body, volume, VWAP, etc.) are all derived from **trade prices and sizes**. The new features capture **order book state** (depth, spread, OFI) and **trade quality** (aggressor side, cancellation patterns).

**Expected outcome:** The book-derived features should add meaningful signal because:
1. They capture supply/demand imbalance (OFI, depth ratio) which is directly predictive.
2. They capture market maker behavior (cancellation patterns, spread dynamics).
3. They provide information about the state of liquidity, which affects barrier hit probabilities.

**Magnitude expectation:** In microstructure literature, OFI alone typically adds 1-3pp to predictive accuracy on similar classification tasks. The combined effect of 9 new features could plausibly add 2-5pp, bringing accuracy from ~40% to ~42-45%. However, at bar_size=500 (1000 MBO events), much of the intra-bar microstructure information is already summarized in the bar-level features, which may limit the incremental value.

## Constraints and Considerations

1. **Compute:** This is CPU-only. MLP + RF training on 157K samples x 220 dims should complete in < 10 minutes. No resource constraints.
2. **Script modification needed:** The v2 script requires a trivial fix (assertion + feature names). This can be done inline by the RUN agent as a diagnostic script change (not a code change requiring TDD).
3. **Apples-to-apples comparison:** The old T6 ran on a different cache (130-dim). For a fair comparison, both the old 130-dim and new 220-dim results should be compared. The old results are saved in `cache/t6_diagnostic_v2_results.json`. However, the cache itself has been rebuilt — there is no 130-dim cache available. The comparison must be against the **recorded** old results.
4. **Ablation design:** To isolate the contribution of new features, ideally run:
   - (a) All 22 features (220-dim) — the new full set.
   - (b) Original 9 active features only (cols 3-9, 12 in the 22-feature layout → 90-dim) — reconstructed baseline.
   - (c) New 13 features only (cols 1-2, 10-11, 13-21 → 130-dim) — book + microstructure features alone.
   This requires subsetting feature columns, which is straightforward in numpy.
5. **Dead features:** 2/22 are dead. Should be excluded from the "new features" condition but kept in the "all features" condition (to match what the RL agent would see).
6. **Statistical comparison:** Need to determine if the improvement is statistically significant. With 3 seeds + both split methods, a paired test (old vs new accuracy) is possible but underpowered. Consider using more seeds (e.g., 10) or bootstrap confidence intervals.

## Recommendation

The FRAME agent should design an experiment with these elements:

1. **Primary test:** Run the v2 bidirectional diagnostic on the new 220-dim cache (all 22 features). Compare RF and MLP accuracy vs the recorded baseline (RF 40.5%, MLP 39.3%).

2. **Ablation:** Subset features to isolate contribution:
   - Full 22 features (220-dim)
   - Original 9 active features only (90-dim) — reconstruct the old baseline on the new cache
   - New features only — book + microstructure (the newly activated + entirely new)

3. **Key question to answer:** Is the improvement from new features larger than seed-to-seed variance? Use 5+ seeds and bootstrap CIs.

4. **Script fix:** Update the v2 script's assertion and feature names. This is a 5-line change, not a TDD task.

5. **Success criterion:** RF accuracy on all-22-features > RF accuracy on 9-original-features by > 2pp, with CI excluding zero. This would confirm the new features carry incremental signal.

6. **Focus on RF first.** RF is more stable, less prone to overfitting, and provides feature importance for free. MLP results are secondary (as the RL policy proxy).

7. **Do NOT invest time in hyperparameter tuning.** This is a signal detection test, not a model optimization task. Default RF (100 trees) and MLP (256x2 ReLU) are appropriate.
