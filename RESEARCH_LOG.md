# Research Log -- LOB-RL

Cumulative findings from all experiments. Each entry is a concise summary.
Read this file FIRST when starting any new research task. It is the institutional memory of this project.

---

## exp-005-null-calibration — CONFIRMED
**Date:** 2026-02-11
**Hypothesis:** Under the martingale null with 2:1 asymmetric barriers, ȳ_long ≈ ȳ_short ≈ 1/3.
**Key result:** ȳ_long = 0.320, ȳ_short = 0.322 (both within [0.28, 0.38]). sum_ȳ = 0.643 ≈ 2/3. P(1,1) = 0. Labels are independent, well-calibrated, and consistent across 248 sessions.
**Lesson:** The Gambler's Ruin null is a valid first approximation for MES barrier labels. Constant-prediction Brier score baseline is 0.218. Phase 2 signal detection can proceed with empirical ȳ ≈ 0.32.
**Next:** Phase 2 — logistic regression + GBT on 22 features to predict Y_long / Y_short with Brier score evaluation.
**Details:** results/exp-005-null-calibration/analysis.md

---

## exp-004-22-feature-supervised-diagnostic — INCONCLUSIVE
**Date:** 2026-02-11
**Hypothesis:** 22-feature barrier obs (with all book + microstructure features active in C++ cache) improves RF balanced accuracy by >2pp over the original 9-feature set.

**Quick results (2 seeds, 50K subsample, shuffle split, RF only):**

| Feature set | Mean bal_acc | Train acc |
|------------|-------------|-----------|
| A (all 22) | 49.6% | 100% |
| B (original 9) | 47.5% | 100% |
| Majority baseline | 38.1% | — |

Delta: +2.1pp (A > B). Both sets dramatically exceed T6 baseline (40.5%) because 4 previously-dead features (cols 0,1,2,11) are now active in the C++ cache.

**Status:** Aborted. Full 60-run experiment (5 seeds × 2 models × 2 splits × 3 feature sets + permutation importance) crashed twice — Python process killed by SIGPIPE when parent experiment.sh was terminated. Quick-tier metrics saved.

**Lesson:** The C++ cache producing active features where Python had dead zeros explains the jump from 40.5% → 47.5% for the "same" 9 features. The new features add a marginal +2pp. More importantly, the three-class framing {long, short, flat} may be suboptimal — the Asymmetric First-Passage Trading plan uses two independent binary predictions (Y_long, Y_short) with Brier score evaluation instead.

**Details:** results/exp-004-22-feature-supervised-diagnostic/metrics.json

---

## t6-supervised-diagnostic — CONFIRMED (weak signal)
**Date:** 2026-02-10
**Hypothesis:** The 130-dim barrier feature set contains learnable signal about barrier-hit direction that exceeds majority-class baseline.

**Key result (v2 — bidirectional framing):** Proper framing classifies {long_profitable, short_profitable, flat} — a balanced 3-class problem (33/33/35%). Both MLP and RF beat baseline on BOTH splits:

| Model | Shuffle Acc | Shuffle Bal | Chrono Acc | Chrono Bal | Baseline |
|-------|------------|------------|------------|------------|----------|
| MLP [256,256] | 39.3% | 39.4% | 39.1% | 39.2% | 34.5% |
| RF (100 trees) | 40.5% | 40.5% | 39.6% | 39.7% | 34.5% |

Signal is +4-6pp above chance, consistent across both splits. RF marginally outperforms MLP. Massive overfitting: MLP 90% train → 39% test.

**v1 (long-only, misleading):** Single-direction labels gave 67/33/0% imbalanced distribution. MLP 61.6% vs 67.4% baseline — appeared to show no signal. This was wrong: unidirectional framing tests gambler's ruin, not direction.

**Critical insight: τ_{+} and τ_{-} are mutually exclusive.** Long profit (price up 20 before down 10) precludes short profit (short stops at +10). Joint distribution: 32.3% long, 32.6% short, 35.1% neither, 0% both.

**Feature caveat:** 4/13 features dead (BBO imbal, depth imbal, cancel asym, spread = constant zero). `precompute_barrier_cache.py` doesn't pass `mbo_data`. Effective dim = 90, not 130.

**Feature importance (RF):** trade_flow_imbalance (0.128) > bar_range (0.122) > volume_log (0.121) > vwap_displacement (0.121) > body_range_ratio (0.117) > bar_body (0.115) > realized_vol (0.114) > session_time (0.089) > session_age (0.074). Dead features: 0.000.

**Lesson:** Trade-derived features have weak but real directional signal (~5pp above chance). Signal is consistent across time periods. Massive train/test gap (90→39%) suggests regularization matters. 4 dead book features are untested and could add meaningful signal. RL has something to learn.

**Next:** (1) Activate book features, re-run. (2) Proceed to T10-T12 — signal exists. (3) Architecture comparison now unblocked.

**Details:** cache/t6_diagnostic_v2_results.json (v2), cache/t6_diagnostic_results.json (v1), scripts/run_barrier_diagnostic_v2.py

---

## exp-001-does-increasing-training-data-from-20-to — REFUTED
**Date:** 2026-02-09
**Hypothesis:** Increasing training data from 20 to 199 days (10x) reduces overfitting and produces meaningfully less negative OOS returns for the LSTM agent.
**Key result:** LSTM 199d val -59.95 (threshold -16.7), MLP 199d val -75.53 (identical to 20d control -75.82). All OOS deeply negative. 199 days eliminates memorization (expl_var 0.30 vs 0.97) but does not improve OOS. Data quantity is not the primary bottleneck.
**Lesson:** MLP 199d val is identical to MLP 20d val despite 5x lower explained_variance — overfitting is a symptom, not the cause of bad OOS. The 21-dim bar-level obs space likely lacks exploitable signal. Seed sensitivity is extreme (LSTM 199d: -59.95 s42 vs -97.0 s43 on 5 val episodes). Entropy collapsed below -0.60 on all 199d runs. LSTM clip_fraction 40-50% confirms lr=1e-3 is too aggressive for RecurrentPPO.
**Next:** P3 observation signal audit (supervised classifier) is now highest priority. Also: 199d no-exec-cost at 10M steps to check if exp-002's +10.93 val persists.
**Details:** results/exp-001-does-increasing-training-data-from-20-to/analysis.md

---

## exp-002-execution-cost-ablation — REFUTED
**Date:** 2026-02-09
**Hypothesis:** Removing execution cost reveals positive OOS returns, proving the agent learns signal masked by the 1-tick round-trip cost.
**Key result:** Val -4.43 / test -5.03 without exec cost (vs val -39.55 / test -68.68 with). Exec cost accounts for ~35 points of OOS loss but removing it doesn't flip the sign to positive. 2/5 positive val episodes. Cross-eval with exec cost: val -85.22 (worse than baseline), confirming the no-cost policy is impractical.
**Lesson:** Execution cost explains a large fraction (~70%) of OOS loss, but even gross returns are slightly negative on 20 training days. The gap to profitability is ~5 points (gross) not ~50 points (net). Stretch run (199d, no exec cost) hit +10.93 val but was severely undertrained (explained_variance=0.174) — needs 10M+ steps.
**Next:** Run 199d no-exec-cost with sufficient training (10M+ steps) to test if positive val persists at convergence. Also resolve exp-001 (199d WITH exec cost) to isolate data vs cost effects.
**Details:** results/exp-002-execution-cost-ablation/analysis.md

---

## pre-007-gpu-framestack — REFUTED

**Date:** 2026-02-09
**Hypothesis:** VecFrameStack(4) gives temporal context that improves OOS on GPU with 5M steps.
**Key result:** Val -82.3 / test -49.4 — worst OOS of all architectures, much worse than 2M local (-48.4 val).
**Lesson:** Frame-stacking memorizes rather than generalizes. More steps + frame-stack = severe overfitting.
**Next:** Abandon frame-stacking. Focus on LSTM and data quantity.
**Details:** research/experiment_report.md

---

## pre-006-gpu-mlp — REFUTED

**Date:** 2026-02-09
**Hypothesis:** Baseline MLP improves with 5M steps on GPU.
**Key result:** Val -62.9 / test -44.0 — val worse than 2M local (-51.5), test slightly better.
**Lesson:** More steps on 20 training days = more memorization. Model oscillates, doesn't converge.
**Next:** Increase training days before adding more steps.
**Details:** research/experiment_report.md

---

## pre-005-gpu-lstm — REFUTED

**Date:** 2026-02-09
**Hypothesis:** LSTM (RecurrentPPO) generalizes better than feedforward architectures.
**Key result:** Val -36.7 / test -33.4 — best OOS but still deeply negative. 2/5 positive val episodes.
**Lesson:** LSTM is the best direction — least overfit, retains most entropy. Advantage may be regularization, not temporal learning. High KL (0.148) suggests LR may be too high for LSTM.
**Next:** Use LSTM as default architecture. Investigate with more training data.
**Details:** research/experiment_report.md

---

## pre-004-local-framestack — REFUTED

**Date:** 2026-02-09
**Hypothesis:** Frame-stacking(4) gives MLP temporal context to improve OOS.
**Key result:** Val -48.4 / test -50.2 — marginal improvement over baseline MLP on test.
**Lesson:** Frame-stacking provides minimal benefit at 2M steps. Does not solve the generalization problem.
**Next:** Test on GPU with more steps (became pre-007).
**Details:** LAST_TOUCH.md training history

---

## pre-003-local-mlp — REFUTED

**Date:** 2026-02-09
**Hypothesis:** Shuffle-split fixes OOS (chronological split conflates overfitting with regime shift).
**Key result:** Val -51.5 / test -62.5 — still deeply negative with shuffle-split.
**Lesson:** The agent genuinely isn't generalizing. Negative OOS is not explained by regime shift alone.
**Next:** Test other architectures (frame-stack, LSTM) with shuffle-split.
**Details:** LAST_TOUCH.md training history

---

## pre-002-full-dataset-chrono — REFUTED

**Date:** 2026-02-08
**Hypothesis:** Best hyperparameter config (return 139.5 in-sample) generalizes OOS on full dataset.
**Key result:** In-sample 139.5, val -53.8, test -36.6 — massive in-sample/OOS gap.
**Lesson:** Positive in-sample returns are illusory. The agent memorizes training patterns. This was the first clear evidence of overfitting.
**Next:** Try shuffle-split to rule out regime shift explanation.
**Details:** LAST_TOUCH.md training history

---

## pre-001-hyperparam-sweep — CONFIRMED

**Date:** 2026-02-07
**Hypothesis:** Systematic sweep identifies optimal in-sample hyperparameters.
**Key result:** bar_size=1000, ent_coef=0.05, lr=1e-3 → return 139.5, entropy -0.48.
**Lesson:** bar_size=1000 > 500 > 200. ent_coef=0.05 prevents collapse. lr=1e-3 is fastest convergence. These hyperparameters are the baseline for all future experiments.
**Next:** Test OOS generalization with full dataset (became pre-002).
**Details:** LAST_TOUCH.md training history

---
