# Survey: Under 2:1 reward:risk with bar_size=500, is ybar_long ≈ ybar_short ≈ 1/3?

## Prior Internal Experiments

### Direct evidence (three-class framing, biased labels)

Two experiments have measured label distributions on the existing barrier cache (bar_size=500, a=20 ticks profit, b=10 ticks stop, t_max=40 bars):

| Source | Sessions | Samples | Long % | Short % | Flat % |
|--------|----------|---------|--------|---------|--------|
| T6 v2 (old 130-dim Python cache) | 247 | 157,040 | 32.6% | 32.9% | 34.6% |
| exp-004 (new 220-dim C++ cache) | 248 | 454,164 | 30.8% | 31.2% | 38.1% |

Both show long and short near 1/3, consistent with the martingale null prediction. However, these are **three-class distributions** (long_profitable / short_profitable / flat), not the binary Y_long/Y_short labels required by Phase 1.

### Critical caveat: timeout biasing

The C++ precompute pipeline applies `flatten_and_bias_labels()` which **converts all timeout labels (label==0) to +1 or -1** based on which barrier the price came closer to reaching. This means:

- The cached label distributions **do not distinguish timeouts from actual barrier hits**.
- The "flat" class in T6/exp-004 is defined as "neither long_profitable nor short_profitable" — i.e., the biased label was -1 for long AND +1 for short (both directions stopped out or biased toward stop).
- `load_binary_labels()` checks `label_values == 0` for timeouts, which will **always be empty** since all 0s have been biased away.

**Implication:** The existing cache cannot directly answer the Phase 1 questions about timeout rates. To get true Y_long/Y_short with timeout tracking, either:
1. Use the raw (pre-bias) labels from the C++ `compute_labels()` output before `flatten_and_bias_labels()`, or
2. Recompute labels skipping the bias step, or
3. Accept that the biased labels are the operationally correct ones (timeout → biased toward the nearer barrier is a reasonable label assignment).

### No prior experiment has computed binary ybar_long/ybar_short

No experiment has directly measured ybar_long = mean(Y_long) or ybar_short = mean(Y_short) with the independent binary formulation. The three-class framing conflates long_profitable/short_profitable/flat into mutually exclusive categories, whereas the binary formulation treats Y_long and Y_short as **independent** indicators.

## Current Infrastructure

### Phase 1 implementation exists and is complete

`python/lob_rl/barrier/first_passage_analysis.py` contains:

- **`load_binary_labels(cache_dir, lookback=10)`** — loads all `.npz` files, extracts X (features), Y_long (label_values == 1), Y_short (short_label_values == -1), timeout flags, tau values, session boundaries.
- **`null_calibration_report(...)`** — computes ybar_long, ybar_short, sum_ybar, SE (session-level blocking), joint distribution, rolling per-session ybar, and gate criteria.
- **Gate criteria hardcoded:**
  - ybar ∈ [0.28, 0.38] for both directions
  - sum_ybar ∈ [0.58, 0.72]
  - timeout_rate < 5% for both
  - per-session stability: all rolling values ∈ [0.20, 0.46]

### Barrier cache is fresh and ready

- 248 `.npz` files in `cache/barrier/`
- bar_size=500, a=20, b=10, t_max=40, lookback=10
- 461K total bars, 454K usable (after warmup + lookback)
- 22 features × 10 lookback = 220-dim observations
- Re-precomputed 2026-02-10 via C++ backend in ~570s

### Test suite exists

`python/tests/barrier/test_first_passage_analysis.py` has ~50 tests covering null calibration, Brier scores, temporal splits, bootstrap tests, label loading, and lattice verification.

### Signal detection (Phase 2) also implemented

`signal_detection_report()` in the same module: fits logistic regression + GBT, computes Brier scores, BSS, bootstrap CI, temporal CV. Ready to run after Phase 1 passes.

## Known Failure Modes

1. **Timeout biasing masks true timeout rate.** The cache has zero label==0 entries because `flatten_and_bias_labels()` assigns all timeouts to the nearer barrier. `load_binary_labels()` detects timeouts via `label_values == 0`, which will always yield 0%. The null calibration gate `timeout_rate < 5%` will trivially pass, but the **true** timeout rate is unknown from the cache alone.

2. **Three-class vs binary formulation mismatch.** T6 and exp-004 used {long_profitable, short_profitable, flat} where "flat" requires BOTH directions to fail. The binary Y_long and Y_short are independent — a bar can have Y_long=0 and Y_short=0 (flat), Y_long=1 and Y_short=0, Y_long=0 and Y_short=1, or (theoretically) both=1. The joint distribution has 4 outcomes, not 3. But with timeout biasing, the labels are all +1/-1, so Y_long = (label==1) directly maps from the biased cache.

3. **2022 bear market drift.** S&P 500 fell ~20% in 2022. This negative drift could push ybar_long below 1/3 and ybar_short above 1/3. The plan's Phase 1.3 (temporal stability via rolling windows) is designed to detect this.

4. **exp-004 aborted.** The full 60-run experiment was killed twice by SIGPIPE. Only quick-tier results (2 seeds, 50K subsample) survived.

## Key Codebase Entry Points

| File | Role |
|------|------|
| `python/lob_rl/barrier/first_passage_analysis.py` | Phase 1 & 2 implementation: `load_binary_labels()`, `null_calibration_report()`, `signal_detection_report()` |
| `scripts/precompute_barrier_cache.py` | Cache generation script; default params: bar_size=500, a=20, b=10, t_max=40 |
| `src/barrier/barrier_precompute.cpp` | C++ precompute pipeline with `flatten_and_bias_labels()` |
| `src/barrier/barrier_label.cpp` | C++ label computation: `compute_labels()`, `resolve_tiebreak()` |
| `python/lob_rl/barrier/label_pipeline.py` | Python label computation (fallback): `compute_labels()`, `_label_single_bar()` |
| `python/tests/barrier/test_first_passage_analysis.py` | Test suite (~50 tests) |
| `cache/barrier/*.npz` | Precomputed cache (248 sessions, 220-dim) |
| `experiments/Asymmetric First-Passage Trading.md` | Full 5-phase research plan |

## Architectural Priors

### Theoretical prediction (T2: Gambler's Ruin)

For a symmetric random walk on a tick lattice with absorbing barriers at +a (profit) and -b (stop):

P(hit +a before -b) = b / (a + b)

With a=20, b=10: P(long profit) = 10/30 = **1/3 exactly**.

By symmetry (swapping a and b for the short direction): P(short profit) = **1/3 exactly**.

This is the **martingale null** — if MES intraday prices are approximately a martingale at the 500-tick bar scale, both marginal label frequencies should be ≈ 1/3.

### What departures would mean

- ybar > 1/3: Positive drift (prices tend to move toward the profit barrier). For equities in a trending market, mild departure is expected.
- ybar < 1/3: Negative drift or mean-reversion.
- ybar_long ≈ ybar_short ≈ 1/3: Market is approximately efficient at this bar scale. Any exploitable signal is in the **conditional** structure P(Y|X), not the unconditional marginal.
- ybar_long + ybar_short ≈ 1 (instead of 2/3): Labeling bug — short labels are computed as 1 - Y_long instead of independently.

### Architecture is irrelevant at this phase

Phase 1 is purely a data analysis question — no model training needed. MLP vs RF vs Transformer does not apply.

## External Context

The 1/3 prediction for 2:1 asymmetric barriers under a martingale is a textbook result from the theory of random walks with absorbing barriers (gambler's ruin). For continuous-time Brownian motion, the exact result is P(hit +a before -b) = b/(a+b), which holds for any symmetric diffusion process. For discrete-time random walks on a lattice, the same formula holds when barriers align with lattice points.

MES intraday prices are known to be approximately a martingale with small drift and moderate volatility. At the 500-tick bar scale (roughly 5-10 minutes during RTH), the drift is negligible relative to the noise, so ybar ≈ 1/3 is the expected finding. Departures of ±2-3pp are typical for instruments with mild directional bias over a year-long sample.

## Constraints and Considerations

1. **No new computation needed.** The cache exists, `load_binary_labels()` and `null_calibration_report()` are implemented. This is a pure "run the code" experiment.

2. **Timeout interpretation requires care.** The cache's biased labels mean "timeouts" are already assigned a direction. This is operationally correct (the label reflects which barrier was closer), but the Phase 1 timeout_rate gate will trivially pass (0%). If the true pre-bias timeout rate is needed, it can be estimated from the per-session p_zero values stored in the cache.

3. **Cache parameters are fixed.** The current cache is bar_size=500, a=20, b=10. If Phase 1 shows ybar wildly off, regenerating the cache with different R is ~10 minutes (C++ backend). Phase 2b sweeps B ∈ {200, 500, 1000, 2000} and R calibrations, each requiring a new cache.

4. **Per-session summary stats already in cache.** Each `.npz` file stores `p_plus`, `p_minus`, `p_zero` and their short equivalents. These can provide pre-bias label distributions without needing to re-run the full label pipeline.

5. **Compute budget: ~0.** Loading 248 `.npz` files and computing means/SE takes seconds.

## Recommendation

The FRAME agent should design a lightweight Phase 1 experiment that:

1. **Runs `null_calibration_report()` on the full cache** (454K samples, 248 sessions). This directly answers the question.

2. **Also extracts pre-bias timeout rates** from per-session `p_zero` / `short_p_zero` summary stats stored in the `.npz` files, to verify the true timeout rate is < 5%.

3. **Reports the joint distribution** (Y_long, Y_short) to confirm the 4-outcome structure and that P(1,1) ≈ 0 (as expected when barriers are large enough to prevent both directions from profiting).

4. **Reports rolling per-session ybar** to check temporal stability and detect the 2022 bear market drift.

5. **Pre-commits gate criteria** consistent with the Asymmetric First-Passage Trading plan:
   - ybar_long and ybar_short each within [0.28, 0.38] (≈ 1/3 ± 5pp)
   - sum_ybar within [0.58, 0.72] (≈ 2/3 ± 5pp)
   - Pre-bias timeout rate < 5% for both directions
   - No month shows ybar outside [0.20, 0.46]

The expected outcome is **CONFIRMED** — ybar ≈ 1/3 for both directions — since the exp-004 three-class distribution (30.8%/31.2%/38.1%) already strongly suggests this. The experiment's value is in formalizing the measurement with proper SE, joint distribution, and temporal stability analysis, establishing the null benchmark for Phase 2.
