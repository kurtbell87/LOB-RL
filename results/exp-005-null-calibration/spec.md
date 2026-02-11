# Experiment: Null Calibration — Is ȳ_long ≈ ȳ_short ≈ 1/3?

## Hypothesis

Under the martingale null with 2:1 reward:risk asymmetric barriers (a=20 ticks profit, b=10 ticks stop), the unconditional label frequencies satisfy:
- ȳ_long = mean(Y_long) ∈ [0.28, 0.38] (centered on 1/3 ≈ 0.333)
- ȳ_short = mean(Y_short) ∈ [0.28, 0.38]
- ȳ_long + ȳ_short ∈ [0.58, 0.72] (centered on 2/3 ≈ 0.667, NOT ≈ 1.0)

**Direction:** Both marginals will be approximately 1/3. The 2022 bear market may push ȳ_long slightly below 1/3 and ȳ_short slightly above 1/3 (negative drift favors short profit), but both should remain within ±5pp of the theoretical value.

**Magnitude:** Departures from 1/3 should be < 5pp (|ȳ - 0.333| < 0.05). Larger departures indicate either a labeling bug or an unexpectedly strong unconditional drift.

## Independent Variables

None. This is a measurement experiment, not a manipulation experiment. The only "variable" is the data itself (248 sessions from 2022 MES).

## Controls

- **Cache version:** C++ barrier cache (`cache/barrier/`), 248 `.npz` files, bar_size=500, a=20, b=10, t_max=40, lookback=10. Precomputed 2026-02-10. No regeneration.
- **Label extraction:** `load_binary_labels()` from `first_passage_analysis.py`. Y_long = (label_values == 1), Y_short = (short_label_values == -1). These conventions are verified correct per the C++ pipeline: `compute_labels(bars, b, a, t_max)` swaps a↔b for short, so `-1` = lower barrier hit = price went down = short profit.
- **Timeout biasing:** The C++ pipeline's `flatten_and_bias_labels()` converts label==0 (timeout) to ±1 based on proximity to the nearer barrier. Post-bias, almost no label==0 entries remain (~0.46%). True pre-bias timeout rates are available from per-session `p_zero` / `short_p_zero` summary stats stored in each `.npz` file.
- **Software:** Python via `uv`, numpy for computation. No model fitting needed.
- **Hardware:** Local (macOS). This is a pure data analysis — loads 248 files and computes means/SEs.

## Metrics (ALL must be reported)

### Primary

1. **ȳ_long** — unconditional mean of Y_long across all usable samples, with session-blocked SE.
2. **ȳ_short** — unconditional mean of Y_short across all usable samples, with session-blocked SE.

### Secondary

- **sum_ȳ** — ȳ_long + ȳ_short. Should be ≈ 2/3. If ≈ 1.0, there is a labeling bug (Y_short = 1 - Y_long instead of independent).
- **Joint distribution** — P(Y_long=i, Y_short=j) for all four (i,j) ∈ {0,1}². P(1,1) should be ≈ 0 (both directions profit simultaneously is near-impossible with asymmetric barriers). The dominant cells should be (1,0), (0,1), and (0,0).
- **Pre-bias timeout rates** — mean of per-session `p_zero` and `short_p_zero` from the `.npz` summary stats. This captures the true timeout rate before `flatten_and_bias_labels()` assigns timeouts to the nearer barrier.
- **Post-bias timeout rates** — fraction of label_values == 0 and short_label_values == 0 in the biased cache. Expected ≈ 0 (confirms bias step ran).
- **Mean race duration** — mean(tau_long) and mean(tau_short) in bars.
- **Per-session rolling ȳ** — ȳ_long and ȳ_short computed per session (248 values each). Report min, max, mean, std.

### Sanity Checks

- **N_samples** — total usable samples. Expected ~454K (from exp-004).
- **N_sessions** — number of sessions loaded. Expected 248.
- **sum_ȳ ≠ 1.0** — If sum_ȳ > 0.90, the labels are likely coupled (short = 1 - long). This would indicate a critical bug.
- **P(1,1) ≈ 0** — Both directions profiting simultaneously should be extremely rare with a=20, b=10 barriers. If P(1,1) > 0.01, investigate: are labels truly independent?
- **Post-bias timeout ≈ 0** — Confirms `flatten_and_bias_labels()` is working as expected.

## Baselines

**Theoretical null (Gambler's Ruin):** For a symmetric random walk on a tick lattice with absorbing barriers at +a and -b:

P(hit +a before -b) = b / (a + b) = 10 / 30 = 1/3

This is the comparison point. No prior experiment baseline is needed because this is the first measurement of binary ȳ_long / ȳ_short.

**Prior three-class evidence (for context, not formal baseline):**
- T6 v2 (Python cache, 157K samples): long 32.6%, short 32.9%, flat 34.6%
- exp-004 (C++ cache, 454K samples): long 30.8%, short 31.2%, flat 38.1%

These three-class distributions suggest ȳ ≈ 1/3, but they are not the same measurement as the binary Y_long / Y_short formulation.

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Long marginal:** ȳ_long ∈ [0.28, 0.38]
- [ ] **C2 — Short marginal:** ȳ_short ∈ [0.28, 0.38]
- [ ] **C3 — Non-complementarity:** ȳ_long + ȳ_short ∈ [0.58, 0.72]
- [ ] **C4 — Pre-bias timeout rate:** Both long and short pre-bias timeout rates (from p_zero / short_p_zero) < 5%
- [ ] **C5 — Temporal stability:** No individual session has ȳ_long or ȳ_short outside [0.10, 0.56] (wide bounds to avoid spurious failure on low-count sessions; the per-session mean across sessions must be within [0.28, 0.38])
- [ ] **C6 — Joint sanity:** P(Y_long=1, Y_short=1) < 0.01

**Verdict mapping:**
- All C1–C6 pass → **CONFIRMED** (null calibration holds, proceed to Phase 2 signal detection)
- C1 or C2 fail with ȳ wildly off (outside [0.15, 0.55]) → **REFUTED** (pipeline bug suspected)
- C1 or C2 fail but ȳ is in [0.15, 0.55] with consistent drift → **INCONCLUSIVE** (drift-adjusted null needed; adjust baseline from 1/3 to empirical ȳ for Phase 2)
- C3 fails (sum ≈ 1.0) → **REFUTED** (labels are coupled, not independent; critical bug)
- C6 fails → **INCONCLUSIVE** (joint structure unexpected; investigate before Phase 2)

## Minimum Viable Experiment

1. Load a single `.npz` file and verify that `load_binary_labels()` produces sensible output: Y_long and Y_short are boolean arrays, timeout arrays exist, session_boundaries are well-formed.
2. Check that Y_long rate ≈ 0.32 and Y_short rate ≈ 0.32 on that one file (matching the per-session p_plus and short_p_minus).
3. Verify that `p_zero` and `short_p_zero` fields exist and are accessible.

If the MVE fails, the infrastructure has a bug and the full run should not proceed.

## Full Protocol

1. **MVE (Step 0):** Load one `.npz` file. Verify Y_long, Y_short, timeout flags, and per-session summary stats. Confirm Y_long rate and Y_short rate are each in [0.20, 0.50]. Check that p_zero and short_p_zero are present in the file. Write MVE pass/fail to results.

2. **Load full dataset:** Call `load_binary_labels('cache/barrier/', lookback=10)`. Record N_samples and N_sessions.

3. **Extract pre-bias timeout rates:** Separately iterate over all 248 `.npz` files and collect `p_zero` and `short_p_zero` from each. Compute the weighted mean (weighted by n_bars) as the true pre-bias timeout rate.

4. **Run null calibration report:** Call `null_calibration_report(Y_long, Y_short, tau_long, tau_short, timeout_long, timeout_short, session_boundaries)`. This computes ȳ_long, ȳ_short, SE, joint distribution, rolling per-session ȳ, and the built-in gate check.

5. **Collect all metrics:** Assemble primary, secondary, and sanity check metrics into a single dict. Include:
   - `ybar_long`, `ybar_short`, `se_long`, `se_short`
   - `sum_ybar`
   - `joint_distribution` (4-cell table)
   - `pre_bias_timeout_long`, `pre_bias_timeout_short`
   - `post_bias_timeout_long`, `post_bias_timeout_short`
   - `mean_tau_long`, `mean_tau_short`
   - `rolling_ybar_long_stats` (min, max, mean, std of per-session values)
   - `rolling_ybar_short_stats`
   - `n_samples`, `n_sessions`
   - `gate_passed` (from null_calibration_report)
   - Individual criterion pass/fail for C1–C6

6. **Write metrics:** Save to `results/exp-005-null-calibration/metrics.json`.

7. **No plotting required.** The READ agent will interpret the numbers directly. Rolling per-session arrays should be saved in the metrics for the READ agent to inspect if needed.

## Resource Budget

**Tier:** Quick

- Max GPU-hours: 0
- Max wall-clock time: 5 minutes
- Max training runs: 0
- Max seeds per configuration: N/A (deterministic computation)

**Estimated runtime:** Loading 248 `.npz` files (~186 MB) and computing means/SEs takes < 30 seconds. The `null_calibration_report()` function iterates over sessions once. Total wall time < 1 minute.

## Compute Target

**Compute:** `local`

No GPU, no model training, no heavy computation. Pure numpy data analysis on cached files.

## Abort Criteria

- `load_binary_labels()` raises an exception → abort, investigate cache integrity.
- N_sessions < 200 → abort, cache is incomplete (expected 248).
- N_samples < 400,000 → abort, cache is incomplete (expected ~454K).
- Any NaN in computed metrics → abort, data corruption.

**Note:** Time-based abort is unnecessary. The entire experiment runs in < 1 minute. If it hasn't finished in 5 minutes, something is fundamentally broken (e.g., disk I/O issue).

## Confounds to Watch For

1. **Timeout biasing distortion.** The C++ `flatten_and_bias_labels()` assigns timeouts to the nearer barrier. This changes the label distribution relative to a "conservative" timeout=0 scheme. The pre-bias `p_zero` / `short_p_zero` summary stats let us quantify this effect. If pre-bias timeout rate is > 5%, the biased labels may not faithfully represent first-passage outcomes.

2. **Small-session instability.** Some sessions may have very few bars (e.g., holiday-shortened trading days), making per-session ȳ noisy. The stability criterion uses wide bounds [0.10, 0.56] for individual sessions to account for this, while the aggregate criterion [0.28, 0.38] applies to the cross-session mean.

3. **2022 bear market drift.** The S&P 500 fell ~20% in 2022, creating a negative price drift. This should push ȳ_long slightly below 1/3 (harder to profit going long) and ȳ_short slightly above 1/3 (easier to profit going short). This is an expected and interpretable departure from the exact null, not a bug. The per-session rolling ȳ values will reveal whether the drift is concentrated in specific months or spread uniformly.

4. **Label convention confusion.** The short label convention (short_label_values == -1 = short profit) is non-obvious because the C++ pipeline swaps a↔b for short direction. The MVE explicitly checks that the derived rates match the per-session summary stats, catching any convention mismatch.

5. **Lookback warmup exclusion.** The first `lookback-1 = 9` bars of each session are excluded (consumed by the feature lookback window). This slightly biases the sample toward bars later in the session. At ~1100 bars/session, this is a < 1% effect.
