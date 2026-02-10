# Research Log -- LOB-RL

Cumulative findings from all experiments. Each entry is a concise summary.
Read this file FIRST when starting any new research task. It is the institutional memory of this project.

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
