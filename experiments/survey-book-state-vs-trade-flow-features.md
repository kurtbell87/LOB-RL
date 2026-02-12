# Survey: Book-State vs Trade-Flow Feature Predictive Power for Barrier Outcomes

## Research Question

Do book-state features (imbalance, depth, spread, queue) and trade-flow features (signed volume, aggressor sequences, arrival rate, trade size) have different predictive power for barrier outcomes on /MES? Compare AUC/BSS per group via logistic regression and GBT with temporal cross-validation. Also test per-regime (high-vol H1 2022 vs consolidation H2 2022).

---

## Prior Internal Experiments

### Directly relevant

**T6 Supervised Diagnostic (CONFIRMED — weak signal):** RF feature importance on 3-class {long, short, flat} showed a clear ordering:

| Rank | Feature | Importance | Category |
|------|---------|-----------|----------|
| 1 | trade_flow_imbalance (col 0) | 0.128 | Trade flow |
| 2 | bar_range (col 3) | 0.122 | Price summary |
| 3 | volume_log (col 7) | 0.121 | Trade flow |
| 4 | vwap_displacement (col 6) | 0.121 | Price summary |
| 5 | body_range_ratio (col 5) | 0.117 | Price summary |
| 6 | bar_body (col 4) | 0.115 | Price summary |
| 7 | realized_vol (col 8) | 0.114 | Price summary |
| 8 | session_time (col 9) | 0.089 | Temporal |
| 9 | session_age (col 12) | 0.074 | Temporal |
| — | 4 dead book features | 0.000 | Book state |

**Critical caveat:** At T6 time, 4/13 features were dead (bbo_imbal, depth_imbal, cancel_asym, mean_spread). These book features are NOW active in the C++ cache. T6's importance ranking is therefore incomplete — book features were untested.

**exp-004 (INCONCLUSIVE):** 22-feature set (A) scored 49.6% balanced accuracy vs 9-feature original set (B) at 47.5%. The +2.1pp gap suggests newly-activated book features contribute marginally. But this was a quick test (50K subsample, 2 seeds, RF only) and was aborted before full analysis.

**exp-006 (REFUTED):** LR and GBT on full 220-dim features both fail to beat constant Brier. Best BSS = -0.0003 (LR/short). The analysis noted: "If GBT beats constant but LR does not, the signal may be overfitting to nonlinear artifacts." In fact, GBT was worse than LR in all 4 cells — more expressiveness → worse performance. **No feature-group decomposition was performed.**

**exp-007 (REFUTED):** LSTM and Transformer with full-session causal context also fail. Transformer collapses to near-constant predictions (p̂_std = 0.014). Six model families tested; all fail. **No feature-group decomposition was performed.**

**exp-009 (REFUTED at B=2000):** Realistic barrier sweep R ∈ {10,20,30,40} at B=2000. All 8 cells BSS < 0 except R=30/short (BSS +0.0016, p=0.248) and R=40/short (BSS +0.0027, p=0.168) — both non-significant. **No feature-group decomposition was performed.**

### Key gap

**No experiment has ever decomposed signal by feature group.** All prior experiments used the full 220-dim feature vector. The proposed experiment — splitting features into book-state vs trade-flow groups — has never been attempted. This is the most important finding of this survey.

---

## Current Infrastructure

### Feature definitions (feature_pipeline.py)

The 22 features split naturally into the following groups for this experiment:

**Group A — Book State (snapshot + dynamics):** Features derived from order book reconstruction via MBO data.

| Col | Feature | Type | Notes |
|-----|---------|------|-------|
| 1 | BBO imbalance | Snapshot | bid_qty / (bid+ask) at bar close |
| 2 | Depth imbalance (5-level) | Snapshot | total_bid / (bid+ask) depth at bar close |
| 10 | Cancel rate asymmetry | Dynamics | (bid_cancels − ask_cancels) / total |
| 11 | Mean spread | Snapshot | Average spread across bar |
| 13 | Order Flow Imbalance (OFI) | Dynamics | Signed add-volume at BBO |
| 14 | Multi-level depth ratio | Snapshot | top3 / top10 depth |
| 15 | Weighted mid displacement | Dynamics | (wmid_end − wmid_start) / tick |
| 16 | Spread dynamics (std) | Dynamics | Spread volatility within bar |
| 17 | VAMP displacement | Dynamics | 3-level VAMP change mid→end of bar |
| 20 | Cancel-to-trade ratio | Dynamics | log(1 + cancels/trades) |

**Group B — Trade Flow (price + volume + execution):** Features derived from trade tape and bar OHLCV, without book reconstruction.

| Col | Feature | Type | Notes |
|-----|---------|------|-------|
| 0 | Trade flow imbalance | Trade tape | Tick-rule signed volume |
| 3 | Bar range (ticks) | Price | high − low |
| 4 | Bar body (ticks) | Price | close − open |
| 5 | Body/range ratio | Price | (close − open) / range |
| 6 | VWAP displacement | Price | (close − vwap) / range |
| 7 | Volume (log) | Volume | log(volume) |
| 8 | Realized volatility | Price | Trailing 20-bar log-return std |
| 18 | Aggressor imbalance | Trade tape | (buy_agg − sell_agg) / total_agg |
| 19 | Trade arrival rate | Trade tape | log(1 + n_trades) |
| 21 | Price impact per trade | Trade+Price | (close − open) / (n_trades × tick) |

**Ambiguous / Temporal (exclude from both groups or assign separately):**

| Col | Feature | Type |
|-----|---------|------|
| 9 | Normalized session time | Temporal context |
| 12 | Session age | Temporal context |

These two features are not information-bearing for the book-vs-flow question. They should be either excluded or included in both groups as controls.

### Data pipeline

- **Cache:** `cache/barrier/` — 248 sessions, 454K usable bars, 220-dim features (22 × 10 lookback). C++ backend. All 22 features active.
- **Loading:** `load_binary_labels(cache_dir, lookback=10)` returns X (N×220), Y_long, Y_short, session_boundaries.
- **Temporal split:** `temporal_split(248)` → 149 train / 50 val / 49 test sessions.
- **Bootstrap:** `paired_bootstrap_brier()` with block_size=50, n_boot=1000.
- **Signal detection:** `signal_detection_report()` in `first_passage_analysis.py` — fits LR + GBT, computes BSS, bootstrap CI, calibration. This function uses the **full** 220-dim vector; it would need modification or a wrapper to accept feature subsets.

### What exists vs what's needed

| Capability | Status |
|-----------|--------|
| Load full 220-dim features | Ready |
| Select feature columns by group | Needs column masking (trivial — just index the assembled 220-dim matrix by group columns × 10 lookback offsets) |
| Fit LR/GBT per feature group | Ready (sklearn, LightGBM) |
| Temporal CV (expanding window) | Ready |
| Per-regime analysis | **Needs session-level regime labeling** — split sessions by realized vol or calendar half. Not implemented but straightforward. |
| AUC metric | Not currently computed — exp-006 used Brier/BSS. Straightforward to add via `roc_auc_score`. |

### Lookback column mapping

With h=10 lookback, the 220-dim vector is structured as:
- Cols [0..21] = bar at time t-9 (oldest)
- Cols [22..43] = bar at time t-8
- ...
- Cols [198..219] = bar at time t (newest)

To select Group A columns across all 10 lookback steps:
```python
group_a_base = [1, 2, 10, 11, 13, 14, 15, 16, 17, 20]  # 10 features
group_a_cols = [base + step * 22 for step in range(10) for base in group_a_base]  # 100 dims
```

Similarly for Group B:
```python
group_b_base = [0, 3, 4, 5, 6, 7, 8, 18, 19, 21]  # 10 features
group_b_cols = [base + step * 22 for step in range(10) for base in group_b_base]  # 100 dims
```

---

## Known Failure Modes

1. **All BSS values have been negative across 11 experiments.** The feature space explains < 0.1% of outcome variance (exp-006 analysis). A per-group decomposition will likely show both groups with negative BSS. The question is whether one group is *less negative* than the other, which would indicate relative informativeness even if neither group alone beats the constant.

2. **GBT overfits on this data.** exp-006 showed GBT is strictly worse than LR in all 4 cells. exp-007 showed more expressive models are worse. If GBT is included, it should be compared to LR within each group, not trusted on its own.

3. **Dead features were only recently activated.** Book features (cols 1, 2, 10, 11) were dead in the Python cache and only became active with the C++ backend. exp-004 showed the jump from 40.5% → 47.5% for the "same" 9 features was entirely due to newly-active features. The C++ cache is now the source of truth.

4. **Calibration curves are poorly calibrated at the tails.** exp-006 showed all models are well-calibrated only near ȳ ≈ 0.32 and miscalibrated at extremes. Feature subsets with fewer dimensions may produce even worse tail calibration.

5. **Block bootstrap with block_size=50 is critical.** Mean race duration is ~12 bars at B=500. Adjacent labels overlap temporally. Standard bootstrap would underestimate variance.

6. **Small feature groups may have unstable LR coefficients.** With only 10 base features × 10 lookback = 100 dims per group, LR on 272K training samples is well-conditioned. But GBT with 100 dims may overfit more aggressively than with 220 dims (less regularization from irrelevant features competing for splits).

---

## Key Codebase Entry Points

| File | Role |
|------|------|
| `python/lob_rl/barrier/feature_pipeline.py` | Feature computation (22 cols), normalization, lookback assembly |
| `python/lob_rl/barrier/feature_pipeline.py:32` | `compute_bar_features()` — column layout docstring |
| `python/lob_rl/barrier/feature_pipeline.py:146` | `_compute_book_features()` — 13 book-derived columns |
| `python/lob_rl/barrier/__init__.py` | `N_FEATURES = 22`, `TICK_SIZE = 0.25` |
| `python/lob_rl/barrier/first_passage_analysis.py` | `load_binary_labels()`, `signal_detection_report()`, `temporal_split()`, `paired_bootstrap_brier()` |
| `scripts/run_exp006_signal_detection.py` | Reference implementation for signal detection experiment |
| `results/exp-006-signal-detection/metrics.json` | Full BSS results at B=500 (220-dim, full features) |
| `results/exp-009-realistic-barrier-sweep/metrics.json` | BSS results at B=2000, R ∈ {10,20,30,40} |

---

## Architectural Priors

This is a **tabular supervised learning** problem (feature subsets → binary label). The appropriate models are:

- **Logistic regression** — linear baseline, established as the least-bad model in prior experiments (exp-006, exp-007).
- **GBT (LightGBM)** — nonlinear baseline. Known to overfit on this data but included for completeness.
- **No neural networks for per-group analysis.** T6/exp-007 showed more expressiveness → worse performance.

The feature-group question is fundamentally an **ablation study**, not an architecture experiment. The right approach is identical models on different feature subsets, not different models on the same features.

---

## External Context

### Microstructure literature on feature informativeness

The book-state vs trade-flow distinction maps to the classic **information shares** literature in market microstructure:

- **Hasbrouck (1991)** information share decomposition: trade flow (order flow) carries the majority of price discovery for most assets.
- **Cont, Stoikov & Talreja (2010)** showed order flow imbalance (OFI) is a significant predictor of short-term price changes. This is col 13 in our feature set.
- **Bouchaud et al. (2009)** and others: the limit order book's predictive content is concentrated in the bid-ask imbalance at the BBO, with diminishing contribution from deeper levels.
- **Cartea, Jaimungal & Penalva (2015)**: trade arrival rate and aggressor imbalance are informative for direction over very short horizons (seconds to minutes), but the signal decays rapidly.

**Key insight from literature:** Trade flow features tend to predict **very short-term** price changes (next few seconds to minutes), while book-state features predict **ultra-short-term** dynamics (next tick to a few seconds). At the bar level (B=500 or B=1000 MBO events ≈ minutes), both signals may be significantly attenuated. The barrier race (lasting ~12 bars at B=500 ≈ many minutes to hours) may be too long a horizon for either feature class to predict.

### Regime conditioning

- High-vol regimes (H1 2022: Jan–Jun, S&P drawdown ~20%) should have wider bar ranges and stronger directional signals.
- Consolidation (H2 2022: Jul–Dec) should have narrower ranges and weaker signals.
- The 2022 calendar splits naturally: H1 = sessions 0–~124 (mostly in training set, some in val), H2 = sessions ~125–247 (val + test).
- **Complication:** With a 60/20/20 temporal split, H1 is entirely in training (sessions 0–148), while H2 spans val (149–197) and test (198–247). A per-regime analysis needs to either (a) use the val set and stratify by realized vol quintile, or (b) use a different CV scheme (e.g., Q1-Q2 train → Q3 test, Q3-Q4 train → Q1 test as rolling windows).

---

## Constraints and Considerations

1. **Compute:** Local CPU only. LR is fast (~30s per fit). GBT is ~2 min per fit. With 2 groups × 2 labels × 2 models × (1 primary + 5 CV folds) = 48 fits total, plus the full-feature baseline → ~60 min wall clock. Adding per-regime splits (2 regimes) would roughly double this → ~2 hours. Well within budget.

2. **Statistical power concern.** With 100-dim feature subsets on 90K val samples, BSS differences between groups may be very small (< 0.001). Block bootstrap CIs will be ~0.001 wide. Detecting a group difference of 0.001 BSS requires careful paired comparison (e.g., bootstrap the *difference* in BSS between groups, not each BSS independently).

3. **AUC vs BSS.** The research question asks for AUC. Prior experiments used BSS exclusively. Both should be reported — AUC measures discrimination (ranking), BSS measures calibration. T6 showed features have discriminative signal (accuracy +5pp ≈ AUC > 0.5) but no calibrated signal (BSS < 0). AUC may show which group discriminates better even if neither produces positive BSS.

4. **Existing B=500 cache is the primary target.** The question mentions "temporal cross-validation (Q1-Q2 train, Q3 test, etc.)" which differs from the existing 60/20/20 split. A quarterly rolling CV would require: {Q1-Q2 train → Q3 val, Q2-Q3 train → Q4 val, etc.}. This is ~62 sessions per quarter. The cache supports this; `load_binary_labels` returns session boundaries that can be indexed by date.

5. **Interaction between groups.** The full 220-dim model (both groups) may outperform the sum-of-parts (each group alone), indicating feature interactions. Include the full model as a third arm: Group A, Group B, Group A+B. Compare A+B vs max(A, B) to measure interaction.

---

## Recommendation

The FRAME agent should focus on:

1. **Feature-group ablation as the primary experiment.** This is the highest-value next step because:
   - It has never been done (identified gap in prior work)
   - The infrastructure exists (feature indexing + LR/GBT fitting)
   - It directly answers whether book features or trade features drive the weak signal found in T6
   - It informs whether to invest in better book features vs better trade features

2. **Three arms:** Group A (book, 10 features × 10 = 100 dim), Group B (trade, 10 features × 10 = 100 dim), Group A+B (all 20 + 2 temporal = 220 dim). The temporal features (cols 9, 12) can be included in all three arms as controls (they don't belong to either group conceptually).

3. **Both AUC and BSS metrics.** AUC will show discrimination power per group (likely to show differences even when BSS is negative). BSS will show calibration power (likely negative for both but the gap is informative).

4. **Per-regime stratification as a secondary analysis.** Split the validation set by realized-vol median (or use quarterly rolling CV). This tests whether signal concentrates in volatile periods. Keep it simple — don't sweep many regime definitions.

5. **LR as the primary model, GBT as secondary.** LR is the established least-bad model. Include GBT to check for nonlinear signal within each group, but interpret cautiously given known overfitting.

6. **Pre-commit to a paired comparison.** The key result is not "Group A BSS" or "Group B BSS" individually, but `BSS(A) - BSS(B)` with bootstrap CI. This paired comparison has more statistical power than comparing each to the constant baseline independently.

7. **Expected outcome:** Based on T6's feature importance (trade_flow_imbalance, bar_range, volume_log dominating), Group B (trade flow) is likely to show stronger signal than Group A (book state). But with all book features now active (unlike T6), Group A may surprise. Either result is informative for the next iteration of feature engineering.
