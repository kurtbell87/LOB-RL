# Experiment: Realistic Barrier Sweep — Does B=2000 with Fixed Tick-Based Barriers R ∈ {10, 20, 30, 40} Produce Positive BSS?

## Hypothesis

At least one of the 8 (R, label) cells at B=2000 with fixed barrier sizes R ∈ {10, 20, 30, 40} ticks produces BSS ≥ 0.005 with Bonferroni-corrected p < 0.05 for logistic regression on Y_long or Y_short, evaluated on a chronological validation set.

**Direction:** BSS > 0 (model beats constant predictor).
**Magnitude:** BSS ≥ 0.005 (same threshold as exp-008).

**Rationale:** exp-008 tested B=2000 with R calibrated as multiples of median bar range (R ∈ {12, 24, 36} ticks). All 6 cells were negative. This experiment tests fixed tick-based barrier sizes anchored to realistic MES intraday trading parameters: R=20 (b=20 ticks = 5 points stop, a=40 ticks = 10 points target) matches actual discretionary trading risk parameters. R=10 and R=40 bound the range; R=30 fills the gap. Combined with exp-008's 3 values, this gives a 7-point R sweep at B=2000: {10, 12, 20, 24, 30, 36, 40}.

**Prior expectation:** Low probability of success at R=10 (close to exp-008's R=12 which was negative). R=20, 30, 40 are genuinely novel and test whether wider, more realistic barriers — where price has room to develop directional moves — unlock signal that sub-point barriers cannot detect.

## Independent Variables

**Barrier width R** (4 levels): R ∈ {10, 20, 30, 40} ticks (fixed, not calibrated to median bar range).

Derived barrier parameters:
- R=10: `--a 20 --b 10 --t-max 40` (a=2R, b=R, t_max=max(40, 4×10)=40)
- R=20: `--a 40 --b 20 --t-max 80` (a=2R, b=R, t_max=4×20=80)
- R=30: `--a 60 --b 30 --t-max 100` (a=2R, b=R, t_max=min(4×30, 100)=100, capped)
- R=40: `--a 80 --b 40 --t-max 100` (a=2R, b=R, t_max=min(4×40, 100)=100, capped)

**Label** (2 levels): Y_long, Y_short.

**Bar size B is FIXED at 2000.** It is NOT an independent variable.

Total grid: 4 R × 2 labels = 8 cells. Combined with exp-008's 6 cells at B=2000, this gives a 7-point R sweep: {10, 12, 20, 24, 30, 36, 40}.

## Controls

- **Data:** 312 `.mbo.dbn.zst` files in `data/mes/` (249 trading days from 2022 MES, filtered by roll calendar). Identical to exp-008.
- **Bar size:** B=2000 MBO events per bar. Identical to exp-008.
- **Precompute:** C++ backend `lob_rl_core.barrier_precompute()` via `scripts/precompute_barrier_cache.py`. `--lookback 10`, `--workers 8`. Separate cache directory per R: `cache/barrier-b2000-r{R}/`.
- **Feature space:** 22 features × 10 lookback = 220 dimensions. Same feature extraction as exp-008.
- **Model:** Logistic regression only (L2, C=1.0, solver=lbfgs, max_iter=1000). Consistent with exp-006 and exp-008. GBT excluded per exp-006's finding that it is strictly dominated by LR on BSS.
- **Split:** Chronological temporal 60/20/20 by session (≈149/50/49 sessions). Same `temporal_split()` as exp-006 and exp-008.
- **Baseline:** Constant predictor ȳ = mean(y_train) per cell. Brier = ȳ(1−ȳ).
- **Bootstrap:** Block bootstrap, block_size=50, n_boot=1000. Same as exp-006 and exp-008.
- **Seed:** 42.
- **Hardware:** Local (macOS, CPU only). No GPU needed.
- **Software:** Python via `uv`. sklearn (LR), numpy.

**Why these controls are necessary:**
- Separate cache directories prevent cross-config file reuse (known skip bug from exp-008).
- LR-only eliminates GBT overfitting confound (established in exp-006).
- Chronological split prevents label leakage from overlapping barrier races.
- Block bootstrap accounts for label autocorrelation (race duration varies by R).
- All parameters match exp-008 exactly so results are directly comparable.

## Metrics (ALL must be reported)

### Primary

1. **BSS** for each of the 4 (R, label) cells: BSS = 1 − BS_model / BS_constant.
2. **Bootstrap p-value** for Brier delta (constant − model) > 0, Bonferroni-corrected for 4 tests (raw p < 0.0125).

### Secondary

- **Raw Brier scores:** BS_constant and BS_logistic for each cell.
- **Bootstrap 95% CI** for Brier delta.
- **N_samples** and **N_sessions** at each R.
- **ȳ_long** and **ȳ_short** at each R — verifies null calibration.
- **Mean race duration** (mean_tau) at each R.
- **Timeout rate** (pre-bias) at each R.
- **exp-008 cross-reference:** Table comparing new R={10,40} results against exp-008's R={12,24,36} at B=2000 to verify monotonic BSS degradation pattern holds.

### Sanity Checks

- **Null calibration gate** at each R: ȳ_long ∈ [0.20, 0.46] AND ȳ_short ∈ [0.20, 0.46]. Same gate as exp-008.
- **Timeout rate** < 10% at each R. R=30 and R=40 are highest risk: wider barriers with t_max=100 may truncate some races. Monitor closely.
- **N_sessions** ≥ 200 at each R. Fewer sessions means precompute is failing.
- **LR converged** (no sklearn convergence warnings) for all 8 cells.

## Baselines

**Constant predictor:** ȳ = mean(y_train) for each (R, label) cell. This is the only baseline.

**Cross-reference with exp-008:** The new R=10 and R=40 cells should be interpretable alongside exp-008's B=2000 results:
- exp-008 B=2000/R=12: BSS_long = -0.0032, BSS_short = -0.0028
- exp-008 B=2000/R=24: BSS_long = -0.0059, BSS_short = -0.0019
- exp-008 B=2000/R=36: BSS_long = -0.0114, BSS_short = +0.0021 (not significant, p=0.208)

If R=10 BSS is not better than R=12 BSS (the nearest tested value, 2 ticks away), the hypothesis fails trivially — tighter barriers at B=2000 don't help.

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Signal detected:** At least one of the 8 (R, label) cells has BSS > 0 AND Bonferroni-corrected p < 0.05 (raw p < 0.05/8 = 0.00625).
- [ ] **C2 — Meaningful magnitude:** The best-performing cell achieves BSS ≥ 0.005.
- [ ] **C3 — Null calibration holds:** At least 3 of 4 R configurations pass the null calibration gate (ȳ ∈ [0.20, 0.46], timeout < 10%).
- [ ] No sanity check failure invalidates the best-performing cell.

**Verdict mapping:**
- C1 + C2 + C3 pass → **CONFIRMED** — signal exists at B=2000 with realistic barriers. Warrants investigation at the successful R value with more models.
- C1 passes but C2 fails → **REFUTED** — statistically significant but negligible, same as exp-008's B=200/R=4 result.
- C1 fails → **REFUTED** — no cell beats constant at corrected p < 0.05. Combined with exp-008's 6 cells, all 14 B=2000 cells (R ∈ {10, 12, 20, 24, 30, 36, 40}) fail. B=2000 is definitively closed.
- C3 fails → **INVALID** — label quality is compromised (likely at R=40 due to high timeout rate). Results cannot be interpreted.

**Note on C1+C2 fail verdict:** exp-008 mapped C1-pass/C2-fail to INCONCLUSIVE, but the READ agent upgraded to REFUTED because the magnitude was 2.3× below threshold. This spec pre-commits to REFUTED for this case to avoid ambiguity.

## Minimum Viable Experiment

1. **Precompute R=10 cache** to `cache/barrier-b2000-r10/` (a=20, b=10, t_max=40, lookback=10, workers=8). Verify ≥ 240 `.npz` files produced.
2. **Load with `load_binary_labels()`**. Verify N_samples > 0, N_sessions ≥ 240.
3. **Run null calibration.** Verify ȳ_long and ȳ_short are both in [0.20, 0.46].
4. **Fit LR on Y_long, predict on val, compute BSS.** Verify BSS is a finite number.

If any step fails, diagnose before proceeding to R=40. The MVE takes ~12 min (10 min precompute + 2 min signal detection) and validates the pipeline for a novel R at B=2000.

## Full Protocol

### Phase 1: Precompute barrier caches (~20 min)

For each R ∈ {10, 20, 30, 40}, run sequentially:

```bash
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/precompute_barrier_cache.py \
  --data-dir ../data/mes/ \
  --output-dir ../cache/barrier-b2000-r${R}/ \
  --roll-calendar ../data/mes/roll_calendar.json \
  --bar-size 2000 \
  --a $((2*R)) \
  --b ${R} \
  --t-max ${T_MAX} \
  --lookback 10 \
  --workers 8
```

Where T_MAX=40 for R=10, T_MAX=80 for R=20, T_MAX=100 for R=30 and R=40.

After each precompute, verify:
- Output directory has ≥ 240 `.npz` files.
- Spot-check one file: has `features`, `bar_features`, `label_values`, `short_label_values`, `bar_high`, `bar_low` keys.

### Phase 2: Run MVE (~2 min)

Execute the MVE on the R=10 cache per the Minimum Viable Experiment section. If the MVE fails, stop and diagnose.

### Phase 3: Null calibration + signal detection (~5 min)

For each R ∈ {10, 20, 30, 40}:

1. Load data: `load_binary_labels(cache_dir, lookback=10)`. Record N_samples, N_sessions.
2. Run null calibration: record ȳ_long, ȳ_short, timeout rates, mean_tau, gate_passed.
3. If gate fails: log the failure, skip signal detection, mark as INVALID.
4. If gate passes: Fit LR + bootstrap for Y_long and Y_short. Record BSS, CI, p-value, raw Brier scores.

Estimated runtime: 4 configs × ~2 min each = ~8 min.

### Phase 4: Aggregate and save (~1 min)

1. Assemble results into JSON structure:
   ```json
   {
     "experiment": "exp-009-realistic-barrier-sweep",
     "bar_size": 2000,
     "configs": [
       {
         "R_ticks": 10, "a": 20, "b": 10, "t_max": 40,
         "n_samples": ..., "n_sessions": ...,
         "ybar_long": ..., "ybar_short": ...,
         "timeout_rate_long": ..., "timeout_rate_short": ...,
         "mean_tau_long": ..., "mean_tau_short": ...,
         "gate_passed": true/false,
         "bss_logistic_long": ..., "bss_logistic_short": ...,
         "brier_constant_long": ..., "brier_logistic_long": ...,
         "brier_constant_short": ..., "brier_logistic_short": ...,
         "delta_logistic_long": {"delta": ..., "ci_lower": ..., "ci_upper": ..., "p_value": ...},
         "delta_logistic_short": {"delta": ..., "ci_lower": ..., "ci_upper": ..., "p_value": ...}
       },
       ...
     ],
     "exp008_b2000_cross_reference": {
       "R12": {"bss_long": -0.0032, "bss_short": -0.0028},
       "R24": {"bss_long": -0.0059, "bss_short": -0.0019},
       "R36": {"bss_long": -0.0114, "bss_short": 0.0021}
     },
     "best_cell": {"R_ticks": ..., "label": ..., "bss": ..., "p_value": ..., "bonferroni_p": ...},
     "success_criteria": {"C1": true/false, "C2": true/false, "C3": true/false}
   }
   ```

2. Save to `results/exp-009-realistic-barrier-sweep/metrics.json`.

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 60 minutes (40 min precompute + 10 min analysis + 10 min buffer)
- Max training runs: 0 (supervised fitting only, no RL)
- Max seeds per configuration: 1
- Max precompute jobs: 4 (sequential, ~10 min each)
- Max signal detection runs: 4
- Max disk: ~1.4 GB (4 caches × ~350 MB each)

**Estimated runtime breakdown:**
- Phase 1 (precompute 4 caches): ~40 min (4 × 10 min sequential)
- Phase 2 (MVE): ~2 min (subset of Phase 3, counted once)
- Phase 3 (null calibration + signal detection): ~8 min
- Phase 4 (aggregation): ~1 min
- **Total: ~50 min**

## Compute Target

**Compute:** `local`

No GPU needed. Precompute is CPU-intensive using the C++ backend. Signal detection is CPU-only sklearn. Memory peak: ~3 GB per precompute worker × 8 workers = ~24 GB. Well within local machine capacity.

## Abort Criteria

- **Precompute failure:** If either R config fails to produce ≥ 200 sessions, abort that config but continue with the other.
- **R=40 timeout rate > 20%:** If R=40 has timeout rate > 20%, labels are severely degraded. Mark R=40 as INVALID but still report metrics for transparency.
- **Wall clock > 90 min:** Abort. Something is wrong. 90 min is ~2× the 50 min budget.
- **NaN or Inf in any Brier score:** Abort that config, continue with the other.

## Confounds to Watch For

1. **Sample size at B=2000.** Only ~108K samples (same as exp-008's B=2000 configs). Bootstrap CI widths will be ~0.001–0.002. A BSS of 0.005 is at the edge of detectability. The experiment may be unable to distinguish BSS=0.003 from BSS=0 at conventional significance levels. This is an inherent limitation of B=2000, not a flaw in the design.

2. **R=40 timeout truncation.** With a=80 and t_max=100, races that haven't resolved by bar 100 are truncated and assigned to the nearer barrier. At R=40, expected mean race duration is ~50+ bars (extrapolating from exp-008's R=36 → mean_tau ≈ 34). The t_max=100 provides ~2× mean_tau headroom, which should be adequate, but monitor timeout rates. If timeout rate > 5%, R=40 label quality is questionable.

3. **R=10 proximity to exp-008's R=12.** R=10 is only 2 ticks (17%) away from the already-tested R=12. If BSS at R=10 ≈ BSS at R=12 (≈ -0.003), this confirms smooth interpolation and adds no new information. If BSS at R=10 is meaningfully different (e.g., positive while R=12 is -0.003), that would be surprising and worth investigating — but also suspicious given the monotonic pattern.

4. **Multiple testing across experiments.** Combined with exp-008's 24 cells, this experiment adds 4 more for a total of 28 cells tested at B=2000 (10 cells) and other B values (18 cells). The family-wise Bonferroni across all 28 would require p < 0.0018, stricter than the within-experiment correction (p < 0.0125). The READ agent should report both within-experiment and cross-experiment corrected p-values.

5. **Asymmetric information from prior.** We designed this experiment knowing exp-008's results. The choice of R=10 and R=40 was informed by the gap analysis, not by a pre-registered sweep. This is acceptable for gap-filling but means positive results should be treated with extra skepticism (hindsight-guided boundary probing is a mild form of data dredging).
