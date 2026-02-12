# Experiment: Bar-Size Sweep — Does Any (B, R) Configuration Produce Positive BSS?

## Hypothesis

At least one combination of bar size B ∈ {200, 500, 1000, 2000} and risk calibration R ∈ {1x, 2x, 3x} median bar range produces a positive Brier Skill Score (BSS > 0) for logistic regression on Y_long or Y_short, evaluated on a held-out chronological validation set.

**Direction:** BSS > 0 (model beats constant predictor) for at least one of the up-to-24 (B, R, label) cells.
**Magnitude:** BSS ≥ 0.005 (model explains ≥ 0.5% of outcome variance beyond constant).

**Rationale:** All prior signal detection (exp-006, exp-007) tested a single configuration: B=500, a=20, b=10 (R=10 ticks ≈ 1.67x median bar range). Six model families failed. But bar size controls the resolution/noise trade-off — smaller B has more samples but noisier features; larger B aggregates more information per bar but fewer samples. The risk unit R controls barrier width relative to per-bar volatility, affecting race duration and label quality. It is possible that signal exists at a different (B, R) scale that is absent at B=500/R=10.

## Independent Variables

**Bar size B** (4 levels): {200, 500, 1000, 2000} MBO events per bar.

**Risk calibration multiplier** (3 levels): R = {1x, 2x, 3x} × median_bar_range(B), where median_bar_range(B) is measured empirically in ticks from a sample of sessions at each B.

The barrier parameters derive from R:
- `--b` (stop loss) = R (in ticks)
- `--a` (profit target) = 2R (in ticks), preserving the 2:1 reward:risk asymmetry
- `--t-max` = max(40, 4 × mean_race_duration_estimate), capped at 100

**Label** (2 levels): Y_long, Y_short (independent binary targets).

Total grid: 4 × 3 × 2 = 24 (B, R, label) cells. One cell (B=500, R≈2x ≈ 10 ticks) overlaps with exp-006; it will be re-run for consistency but is not expected to differ.

## Controls

- **Data:** 312 `.mbo.dbn.zst` files in `data/mes/` (249 trading days from 2022 MES, filtered by roll calendar).
- **Precompute:** C++ backend `lob_rl_core.barrier_precompute()` via `scripts/precompute_barrier_cache.py`. `--lookback 10`, `--workers 8`. Each (B, R) pair gets a separate cache directory: `cache/barrier-b{B}-r{R}/`.
- **Feature space:** 22 features × 10 lookback = 220 dimensions (same feature set at every scale).
- **Model:** Logistic regression only (L2, C=1.0, solver=lbfgs, max_iter=1000). GBT is excluded — exp-006 and exp-007 demonstrated it is strictly dominated by LR on BSS (more expressive = more noise fitting).
- **Split:** Chronological temporal 60/20/20 by session (≈149/50/49 sessions). Same `temporal_split()` as exp-006.
- **Baseline:** Constant predictor ȳ = mean(y_train) per (B, R, label) cell. Brier = ȳ(1-ȳ).
- **Bootstrap:** Block bootstrap, block_size=50, n_boot=1000 (same as exp-006).
- **Seed:** 42.
- **Hardware:** Local (macOS, CPU only). No GPU needed.
- **Software:** Python via `uv`. sklearn (LR), numpy.

**Why these controls are necessary:**
- Separate cache directories prevent file reuse across (B, R) configs (known skip bug).
- LR-only eliminates the GBT overfitting confound that dominated exp-006.
- Chronological split prevents label leakage from overlapping barrier races.
- Block bootstrap accounts for label autocorrelation (race duration varies by B and R).

## Metrics (ALL must be reported)

### Primary

1. **BSS** for each (B, R, label) cell: BSS = 1 − BS_model / BS_constant. Tests the hypothesis directly.
2. **Bootstrap p-value** for Brier delta (constant − model) > 0 at α = 0.05.

### Secondary

- **Raw Brier scores:** BS_constant and BS_logistic for each cell.
- **Bootstrap 95% CI** for Brier delta.
- **N_samples** and **N_sessions** at each (B, R) — tracks how sample size varies with bar size.
- **ȳ_long** and **ȳ_short** at each (B, R) — verifies null calibration holds across scales.
- **Mean race duration** (mean_tau) at each (B, R) — characterizes barrier dynamics.
- **Timeout rate** (pre-bias) at each (B, R) — if > 5%, barriers are too wide.
- **Median bar range** (in ticks) at each B — documents the R calibration input.
- **Best BSS across all cells** (with Bonferroni-corrected p-value for 24 tests).

### Sanity Checks

- **Null calibration gate** at each (B, R): ȳ_long ∈ [0.20, 0.46] and ȳ_short ∈ [0.20, 0.46] (wider than exp-005's [0.28, 0.38] to accommodate extreme R values that may shift the null). If any (B, R) fails this gate, its signal detection results are unreliable.
- **Timeout rate** < 10% at each (B, R). If > 10%, the barriers are too wide and many labels are biased timeouts rather than genuine first-passage outcomes.
- **N_sessions** ≥ 200 at each B. Fewer sessions means the precompute is failing on too many days.
- **LR converged** (no sklearn convergence warnings) for every cell.

## Baselines

**Constant predictor:** ȳ = mean(y_train) for each (B, R, label) cell. This is the only baseline.

**Cross-reference:** The B=500, R≈2x cell should approximately reproduce exp-006's LR results:
- exp-006 LR/long BSS = -0.0007, LR/short BSS = -0.0003
- If the re-run differs by > |0.002| in BSS, investigate (cache difference, split difference, etc.).

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Signal detected:** At least one of the 24 (B, R, label) cells has BSS > 0 AND Bonferroni-corrected p < 0.05 (i.e., raw p < 0.05/24 = 0.00208).
- [ ] **C2 — Meaningful magnitude:** The best-performing cell achieves BSS ≥ 0.005.
- [ ] **C3 — Null calibration holds:** At least 10 of the 12 (B, R) configurations pass the null calibration gate (ȳ ∈ [0.20, 0.46], timeout < 10%).
- [ ] No sanity check failure invalidates the best-performing cell.

**Verdict mapping:**
- C1 + C2 + C3 pass → **CONFIRMED** — calibrated signal exists at a specific (B, R) scale. Proceed to Phase 3 with that configuration.
- C1 passes but C2 fails → **INCONCLUSIVE** — statistically significant but negligible. Explore further (longer lookback, more features) at the best scale.
- C1 fails (no cell beats constant at corrected p < 0.05) → **REFUTED** — the 22-feature barrier observation space does not contain calibrated probabilistic signal for first-passage prediction at any tested (B, R) scale. This closes Phase 2b and motivates a pivot to different features or targets.
- C3 fails (most configurations have broken null calibration) → **INVALID** — the R calibration is off; redesign the sweep.

## Minimum Viable Experiment

1. **Measure median bar range at B=200:** Precompute a mini-cache for B=200 using 10 sessions (first 10 trading days). Extract bar_high − bar_low from the OHLCV output. Compute median in ticks (divide by TICK_SIZE=0.25). This gives the R calibration anchor for B=200.
2. **Precompute one full (B, R) configuration** (B=200, R=1x) to `cache/barrier-b200-r{R}/`. Verify 248 sessions produce output.
3. **Load with `load_binary_labels()`**. Verify N_samples > 0, N_sessions ≥ 240.
4. **Run null calibration.** Verify gate passes (ȳ ∈ [0.20, 0.46]).
5. **Fit LR on Y_long, predict on val, compute BSS.** Verify BSS is a finite number.

If any step fails, diagnose before proceeding to the full sweep. The MVE takes ~15 min (one precompute + one signal detection) and validates the entire pipeline at a new bar size.

## Full Protocol

### Phase 0: Measure median bar ranges (≈20 min)

For each B ∈ {200, 500, 1000, 2000}:

1. **B=500 (existing cache):** Load 10 `.npz` files from `cache/barrier/`. Compute median((bar_high − bar_low) / TICK_SIZE) across all bars. Expected: ~6 ticks (from prior analysis).

2. **B ∈ {200, 1000, 2000} (need fresh precompute):** Run `precompute_barrier_cache.py` with `--bar-size B --a 20 --b 10 --t-max 40 --workers 8` on a subset of 10 sessions to a temporary directory (e.g., `cache/tmp-b{B}/`). Extract median bar range from the OHLCV output. Delete temp directory afterward.

3. **Compute R grid:** For each B, compute R_1x = round(1.0 × median_range), R_2x = round(2.0 × median_range), R_3x = round(3.0 × median_range), all in ticks (integers). If any R < 2 ticks, set R = 2 (minimum viable barrier width). Record the full (B, R_1x, R_2x, R_3x) grid.

4. **Set t_max for each R:** t_max = max(40, round(4 × R)). Larger R means longer expected race duration; t_max must accommodate this. Cap at 100.

### Phase 1: Precompute barrier caches (≈2 hours)

For each of the 12 (B, R) configurations:

```bash
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/precompute_barrier_cache.py \
  --data-dir ../data/mes/ \
  --output-dir ../cache/barrier-b${B}-r${R}/ \
  --roll-calendar ../data/mes/roll_calendar.json \
  --bar-size ${B} \
  --a $((2*R)) \
  --b ${R} \
  --t-max ${T_MAX} \
  --lookback 10 \
  --workers 8
```

**Run configurations sequentially** (each takes ~10 min; parallel precompute would contend on disk I/O). Total: ~2 hours.

After each precompute, verify:
- Output directory has ≥ 240 `.npz` files.
- Spot-check one file: has `features`, `bar_features`, `label_values`, `short_label_values`, `bar_high`, `bar_low` keys.

**Special case: B=500, R=10** — if the existing `cache/barrier/` matches (a=20, b=10, t_max=40), symlink or copy instead of re-precomputing. But only if the parameters match exactly. If the R_2x calibration for B=500 differs from R=10, precompute fresh.

### Phase 2: Run MVE (≈15 min)

Execute the MVE on the first completed cache (smallest B) per the Minimum Viable Experiment section. If the MVE fails, stop and diagnose.

### Phase 3: Null calibration + signal detection (≈30 min)

For each of the 12 (B, R) configurations:

1. Load data: `load_binary_labels(cache_dir, lookback=10)`. Record N_samples, N_sessions.

2. Run null calibration: `null_calibration_report(Y_long, Y_short, tau_long, tau_short, timeout_long, timeout_short, session_boundaries)`. Record ȳ_long, ȳ_short, timeout rates, mean_tau, gate_passed.

3. If gate fails (ȳ outside [0.20, 0.46] or timeout > 10%): log the failure, skip signal detection for this config, mark as INVALID.

4. If gate passes: Run signal detection. Because `signal_detection_report()` includes GBT (which we don't need), either:
   - Call `signal_detection_report()` and extract only LR metrics (ignore GBT). Runtime ~74 sec.
   - Or call `fit_logistic()` + bootstrap manually for LR only. Runtime ~30 sec.

   The RUN agent should use whichever approach is simpler. The key outputs per (B, R, label) cell are: BSS, bootstrap delta CI, p-value, raw Brier scores.

5. Write all metrics for this configuration to a per-config dict.

Estimated runtime: 12 configs × ~2 min each = ~24 min.

### Phase 4: Aggregate and save

1. Assemble all per-config results into a single JSON structure:
   ```json
   {
     "experiment": "exp-008-bar-size-sweep",
     "median_bar_ranges": {"200": N, "500": N, "1000": N, "2000": N},
     "r_grid": {"200": [R1, R2, R3], "500": [...], ...},
     "configs": [
       {
         "bar_size": 200, "R_ticks": N, "a": 2N, "b": N, "t_max": M,
         "n_samples": ..., "n_sessions": ...,
         "ybar_long": ..., "ybar_short": ...,
         "timeout_rate_long": ..., "timeout_rate_short": ...,
         "mean_tau_long": ..., "mean_tau_short": ...,
         "gate_passed": true/false,
         "bss_logistic_long": ..., "bss_logistic_short": ...,
         "brier_constant_long": ..., "brier_logistic_long": ...,
         "delta_logistic_long": {"delta": ..., "ci_lower": ..., "ci_upper": ..., "p_value": ...},
         ...
       },
       ...
     ],
     "best_cell": {"bar_size": ..., "R_ticks": ..., "label": ..., "bss": ..., "p_value": ..., "bonferroni_p": ...},
     "success_criteria": {"C1": true/false, "C2": true/false, "C3": true/false},
     "exp006_consistency": {"bss_long": ..., "bss_short": ..., "delta_vs_exp006": ...}
   }
   ```

2. Save to `results/exp-008-bar-size-sweep/metrics.json`.

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0
- Max wall-clock time: 3 hours (2h precompute + 0.5h analysis + 0.5h buffer)
- Max training runs: 0 (supervised fitting only, no RL)
- Max seeds per configuration: 1
- Max precompute jobs: 12 (sequential, ~10 min each)
- Max signal detection runs: 12 (one per configuration)
- Max disk: ~4 GB (12 caches × ~150-350 MB each)

**Estimated runtime breakdown:**
- Phase 0 (measure bar ranges): ~20 min (3 temp precomputes × 5 min + B=500 from cache)
- Phase 1 (precompute 12 caches): ~120 min (12 × 10 min sequential)
- Phase 2 (MVE): ~15 min
- Phase 3 (null calibration + signal detection): ~30 min
- Phase 4 (aggregation): ~1 min
- **Total: ~3 hours** (dominated by precompute)

## Compute Target

**Compute:** `local`

No GPU needed. The precompute uses the C++ backend (CPU-intensive, I/O-intensive). Signal detection is CPU-only sklearn. Memory peak: ~3 GB per precompute worker × 8 workers = ~24 GB. The local machine should handle this.

If precompute per-config time exceeds 20 min (suggesting the C++ backend is slower than expected at large B), reduce workers to 4.

## Abort Criteria

- **Precompute failure:** If any single B fails to produce ≥ 200 sessions, abort that B (skip all R calibrations for it) but continue with remaining B values. If ≥ 2 of 4 bar sizes fail, abort the entire experiment.
- **Median bar range degenerate:** If median bar range < 1 tick at any B, that B is too fine-grained. Skip it.
- **Wall clock > 5 hours:** Abort. Something is wrong with precompute performance. The 5h abort is 1.67× the 3h budget — generous enough to avoid kill-restart cycles.
- **All null calibration gates fail:** If 0 of 12 configs pass the gate, abort. The R calibration approach is fundamentally broken.
- **NaN or Inf in any Brier score:** Abort that config, continue with others.

## Confounds to Watch For

1. **Multiple testing inflation.** 24 (B, R, label) cells means the family-wise error rate under the null at α = 0.05 per test is ~1 − (0.95)^24 ≈ 0.71. The Bonferroni correction (p < 0.05/24 = 0.00208) is conservative but necessary. If the best raw p is < 0.05 but > 0.00208, this is borderline — the READ agent should report both raw and corrected results and note the ambiguity.

2. **Sample size confound.** B=200 will have ~4-5× more samples than B=2000 (~1.1M vs ~110K). Larger sample sizes give more statistical power to detect tiny effects. A BSS of 0.0005 might be "significant" at B=200 with 1.1M samples but practically useless. C2 (BSS ≥ 0.005) guards against this.

3. **R calibration circularity.** Measuring median bar range from the data and then using it to set R introduces a mild form of data snooping — the barrier widths are adapted to the data. In practice this is standard (adaptive discretization), but the READ agent should note it. Using fixed R values across all B would be an alternative but would compare apples to oranges (R=10 at B=200 means something very different than R=10 at B=2000).

4. **Feature quality variation across B.** The 22 features are computed identically at each bar size, but their information content may change. At small B (200 events), microstructure features (OFI, depth ratio, etc.) may be noisier. At large B (2000 events), they aggregate more information. This is part of what the sweep is testing — it's a feature, not a bug — but the READ agent should consider whether any positive result at large B is driven by better features vs. better label dynamics.

5. **Timeout rate as a hidden confounder.** If R is too large relative to per-bar volatility, many races timeout. The `flatten_and_bias_labels` step assigns timeouts to the nearer barrier, which dilutes the label quality. High timeout rates (> 5%) mean the labels are partially synthetic. The null calibration gate catches extreme cases, but rates of 3-5% may still degrade BSS without triggering the gate.

6. **Train/val distributional shift (same as exp-006).** The chronological split puts Jan–Aug in training and Sep–Nov in validation. The 2022 bear market had distinct phases. This is an unavoidable consequence of chronological splitting. The CV within `signal_detection_report()` partially addresses this, but the primary result is on the single temporal split.

7. **B=500/R≈10 consistency check.** If the re-run at B=500 with the closest R to 10 ticks produces substantially different BSS than exp-006 (|ΔBSS| > 0.002), something changed between experiments (cache version, feature normalization, split boundaries). This would invalidate cross-experiment comparisons.
