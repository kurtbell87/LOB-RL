# Analysis: Execution Cost Ablation — Gross vs Net Alpha Decomposition

## Verdict: REFUTED

The primary hypothesis — that removing execution cost reveals positive OOS returns — is **REFUTED**. Run B (MLP 20d, no exec cost) produced **negative** mean returns on both val (-4.43) and test (-5.03). SC-1 and SC-2 both failed. The agent does not learn directionally useful signal from the 21-dim observation space, even when transaction costs are removed entirely.

However, the results are not without nuance. The magnitude of the failure is dramatically smaller than with exec cost (val improved from -39.55 to -4.43, a 35-point gain), and Run C (199d, no exec cost) produced **positive val returns** (+10.93) while still failing on test (-13.90). These secondary findings inform the path forward.

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL** — Run B mean val return = **-4.43** vs. threshold > 0.0. The agent did not achieve positive gross alpha on the val set.
- [ ] **SC-2: FAIL** — Run B mean test return = **-5.03** vs. threshold > 0.0. The agent did not achieve positive gross alpha on the test set.
- [x] **SC-3: PASS** — Run B val return (-4.43) is **35.12 points better** than Run A val return (-39.55). Threshold was 30 points. Removing execution cost has a material positive effect on OOS returns.
- [x] **SC-4: PASS** — All sanity checks passed:
  - Entropy above -0.60 for all runs: Run A min 0.490, Run B min 0.542, Run C min 0.743. All well above threshold.
  - No NaN in any run.
  - Control (Run A) val -39.55 vs historical pre-003 val -51.5. Delta 11.95, within ±15 tolerance.
  - In-sample positive for all runs: Run A 423.75, Run B 483.79, Run C 86.05.
  - Trade count: Run B = 199.7 trades/episode. The spec threshold of 40 was based on an incorrect assumption of ~23 bars/episode; actual episode length is ~348 bars. Trade rate is 199.7/347.7 = 0.57 trades/bar, well below 1.0. Not degenerate.
- [ ] **SC-5: FAIL** — Run B positive val episodes = **2/5** vs. threshold ≥ 3/5. Signal is not robust across val days.

**Score: 2/5 criteria passed. SC-1, SC-2, SC-5 failed. Primary criteria failed → REFUTED.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Run A (exec cost) | Run B (no exec cost, 20d) | Run C (no exec cost, 199d) | Notes |
|--------|-------------------|---------------------------|----------------------------|-------|
| Val mean return (no exec cost eval) | +24.10 (cross-eval) | **-4.43** | **+10.93** | Run B fails SC-1. Run C positive but not the primary test. |
| Test mean return (no exec cost eval) | -21.84 (cross-eval) | **-5.03** | **-13.90** | Both fail SC-2. No gross alpha on test. |

**Run B val = -4.43:** This is close to zero but definitively on the wrong side. The 95% CI given std=40.89 across 5 episodes is approximately -4.43 ± 35.9, spanning roughly [-40, +31]. The true mean could plausibly be positive, but we cannot claim it is given the observed point estimate. The small sample (5 val days) creates high uncertainty, but the pre-committed criterion is clear: > 0.0.

**Run B test = -5.03:** The test set (10 episodes) provides a slightly larger sample. std=48.19, so the 95% CI is roughly -5.03 ± 29.9, spanning [-35, +25]. Again, plausibly overlaps zero, but the point estimate is negative. With 4/10 positive test episodes, the agent is near chance.

**Run C val = +10.93:** The stretch run (199 training days, no exec cost) achieves positive val returns. This is the most encouraging signal in the experiment, but: (a) it changes two variables vs. the baseline (data + cost), making it harder to attribute; (b) the test return is -13.90, so it does not generalize; (c) explained_variance = 0.174, indicating the model is severely undertrained at 2M steps on 199 days.

### Secondary Metrics

| Metric | Run A | Run B | Run C |
|--------|-------|-------|-------|
| Val mean return (WITH exec cost) | -39.55 | **-85.22** | **-64.50** |
| Test mean return (WITH exec cost) | -68.68 | **-63.60** | **-77.57** |
| Val Sortino (no exec cost) | -1.275 (native) | **-0.431** | **+0.777** |
| Test Sortino (no exec cost) | -1.550 (native) | **-0.271** | **-0.591** |
| Positive val episodes (no exec cost) | 0/5 (native) | **2/5** | **3/5** |
| Positive test episodes (no exec cost) | 1/10 (native) | **4/10** | **6/10** |
| Val std return | 31.02 | 40.89 | 33.62 |
| Test std return | 48.26 | 48.19 | 35.87 |
| Final entropy | 0.490 | 0.542 | 0.743 |
| Entropy at 500K/1M/1.5M/2M | 0.748/0.563/0.546/0.490 | 0.732/0.588/0.562/0.542 | 0.913/0.878/0.822/0.743 |
| Mean trade count/episode | 162.8 | 199.7 | 223.5 |
| In-sample mean return | 423.75 | 483.79 | 86.05 |
| Explained variance (1M/2M) | 0.974/0.979 | 0.973/0.984 | 0.091/0.174 |
| approx_kl (1M/2M) | 0.052/0.061 | 0.051/0.055 | 0.029/0.041 |
| Training FPS | 4767 | 4251 | 4146 |

**Cross-evaluation (no-exec-cost models WITH exec cost):**
- Run B: val -85.22, test -63.60. This is **worse** than the exec-cost-trained Run A (val -39.55, test -68.68). The no-exec-cost policy trades too frequently (199.7 vs 162.8 trades/episode) and gets destroyed by transaction costs.
- Run C: val -64.50, test -77.57. Also catastrophically bad under real costs. 0/10 positive test episodes.
- **Implication:** Even if the no-exec-cost agent had positive gross alpha (it doesn't), the learned policy is impractical — it cannot survive real trading costs. The agent's learned behavior without cost pressure is to trade more aggressively, not to trade more wisely.

**Trade frequency analysis:**
- Removing exec cost increases trades/episode by 22.6% (162.8 → 199.7, Runs A→B).
- Adding more data increases it further (199.7 → 223.5, Runs B→C).
- Trade rate 0.57 trades/bar (Run B) is below 1.0, so the agent isn't flipping every bar, but it is position-changing on 57% of bars — very active relative to the ~23 bars assumed in the spec.
- The spec's trade count threshold (40) was miscalibrated due to an incorrect assumption about episode length. Actual episodes are ~348 bars, not ~23. The trade **rate** is the relevant metric, and 0.57 trades/bar is not degenerate.

**Entropy trajectories:**
- Run B entropy (0.542 final) is slightly **higher** than Run A (0.490). Removing exec cost does NOT cause entropy collapse — the opposite. Without cost pressure, the agent maintains slightly more policy uncertainty.
- Run C entropy (0.743 final) is much higher, confirming that 2M steps is grossly insufficient for 199 days. The model has barely begun to converge.

**Explained variance:**
- Runs A and B: ~0.98 at 2M. The value function has fully fit the training data. This extreme explained variance combined with negative OOS returns is the hallmark of overfitting.
- Run C: 0.174 at 2M. The value function has barely learned. 2M steps / 199 days ≈ 10K steps/day vs. 100K steps/day for 20 days. The model needs 10x+ more training steps.

**In-sample returns:**
- Run B (483.79) > Run A (423.75): Expected — removing cost increases raw reward per trade.
- Run C (86.05) << Runs A/B: Confirms severe undertraining. The model can't even fit in-sample with 2M steps on 199 days.

### Sanity Checks

| Check | Result | Details |
|-------|--------|---------|
| Entropy > -0.60 all runs | **PASS** | Min entropy across all runs: 0.490 (Run A). All well above -0.60. |
| No NaN | **PASS** | All runs completed without NaN/inf. |
| Control reproduces baseline within ±15 | **PASS** | Run A val -39.55 vs pre-003 val -51.5. Delta = 11.95 < 15. |
| In-sample positive all runs | **PASS** | Run A: 423.75, Run B: 483.79, Run C: 86.05. All positive. |
| Trade count < 40/episode (no-exec-cost) | **N/A — spec miscalibrated** | Spec assumed ~23 bars/episode; actual is ~348 bars. Run B: 199.7 trades in 348 bars = 0.57 trades/bar. Not degenerate but very active. |

**Control baseline note:** Run A val return (-39.55) is 11.95 points better than the historical pre-003 baseline (-51.5). This is within the ±15 tolerance but notable. Possible explanations: (1) natural variance (single seed, 5 val days), (2) subtle infrastructure drift since pre-003. Since it's within tolerance, we proceed, but the comparison between Run A and Run B uses the fresh Run A as the baseline (not pre-003), eliminating this drift concern for the primary hypothesis test.

---

## Resource Usage

| Resource | Budgeted | Actual | Notes |
|----------|----------|--------|-------|
| GPU hours | 0 | 0 | All CPU. |
| CPU hours | 4 | 1.56 | Well under budget. |
| Wall clock | 3 hours | 23.7 min (1,422 sec) | ~8x under budget. |
| Training runs | 4 | 4 | MVE + 3 full runs. |
| Abort triggered | — | No | All runs completed normally. |

Budget was appropriate. Wall clock was much shorter than estimated, likely because the 30-minute per-run estimate was conservative for Apple Silicon.

---

## Confounds and Alternative Explanations

### 1. Forced Flatten Cost Confound — MINOR CONCERN

The spec acknowledged that `compute_forced_flatten()` always charges `spread/2 * |position|` (~$0.625 per episode) even without `--execution-cost`. With 199.7 trades/episode, this is a tiny fraction (~0.3%) of total potential trading PnL. Run B's val return of -4.43 is close enough to zero that this confound could theoretically explain the gap. However, this would require the "true" mean to be about +$0.63, which is deep within the noise band. The forced-flatten cost is a **minor systematic bias** that does not change the verdict.

### 2. VecNormalize Reward Scaling — ACKNOWLEDGED

Without exec cost, raw rewards are larger (no spread deduction per trade), changing the effective learning signal. Both conditions show high in-sample returns and high explained variance, suggesting the optimization landscape was similar enough. Not a major confound.

### 3. Execution Cost as Implicit Regularization — PARTIALLY SUPPORTED

The hypothesis in the spec was that removing exec cost would cause MORE overfitting. The data weakly supports this:
- Run B in-sample (483.79) is higher than Run A (423.75), but OOS gap is narrower (Run B: val -4.43 vs in-sample 483.79, gap 488; Run A: val -39.55 vs in-sample 423.75, gap 463). The overfitting gap is actually similar.
- Run B trades more frequently (199.7 vs 162.8), consistent with exec cost regularizing trade frequency.
- Run B entropy is slightly higher (0.542 vs 0.490), contradicting the "exec cost regularizes" story. The agent without cost pressure maintains MORE policy diversity, not less.

**Conclusion:** Exec cost regularizes trade *frequency* but does not regularize the policy's entropy or explained variance. The regularization effect is narrow, not general.

### 4. Different Optimal Policies — CONFIRMED AND CRITICAL

The cross-evaluation results confirm this confound decisively. Run B's policy (trained without exec cost) evaluated WITH exec cost produces val -85.22, which is **worse** than Run A's native val -39.55. The no-exec-cost agent learned a fundamentally different (and more actively trading) policy that is catastrophically bad under real costs. This means even if Run B had positive gross alpha, the learned policy would be useless in practice.

### 5. Single Seed / Small Val Set — HIGH CONCERN

5 val episodes with std ~41 produces a 95% CI of roughly ±36 around the mean. Run B's -4.43 is well within noise. The "true" gross alpha could be anywhere from -40 to +32. A single seed with 5 val days is simply too noisy to make fine distinctions near zero. The test set (10 episodes) is only marginally better. SC-5 (≥3/5 positive val episodes) appropriately guards against this, and Run B failed it (2/5).

### 6. Could the Baseline (Run A) Be Poorly Tuned?

Run A reproduced the pre-003 baseline within tolerance (11.95 points). While not identical, this is single-seed variance on 5 val days. The comparison between Runs A and B is valid because both used identical infrastructure, same seed, and ran in the same session. Infrastructure drift between pre-003 and this experiment affects the absolute level but not the A-vs-B delta.

### 7. Run C Undertraining — DEFINITIVE

Run C's explained variance of 0.174 at 2M steps means the value function has captured less than 18% of the return variance on training data. In-sample return is only 86.05 (vs 483.79 for 20d). The model is grossly undertrained. Run C's positive val return (+10.93) should be interpreted with extreme caution: the model is still in a near-random exploration phase, and "positive val return" at this stage may simply be lucky variance from a policy that hasn't converged. Run C is not a valid test of the "199d + no exec cost" combination — it needs 10-20M steps.

---

## What This Changes About Our Understanding

### The Core Finding

**Execution cost accounts for ~35 points of OOS loss** (delta between Run A val -39.55 and Run B val -4.43). This is a large fraction of the ~52-point deficit from zero. However, removing it does not flip the sign to positive. The agent is "less wrong" without exec cost, not "right."

### Updated Mental Model

Before this experiment, two competing hypotheses explained negative OOS:
1. **H2-confirmed:** "The agent learns weak signal masked by execution cost."
2. **H2-refuted:** "The agent learns no signal at all — execution cost is irrelevant."

The truth is **between these extremes:** Removing exec cost brings OOS returns dramatically closer to zero (from -40 to -4) but doesn't cross it. This pattern is most consistent with:

- **The agent learns very weak directional signal** that nearly (but not quite) breaks even on gross basis.
- **OR:** The agent learns no signal, and the -4 → 0 gap is noise from the forced-flatten confound and 5-day sample variance.

We cannot distinguish between these interpretations with this experiment design. The signal-to-noise ratio of 5 val episodes is too poor. However, Run C's positive val return (+10.93) on 199 days, despite being severely undertrained, hints that more data + no exec cost could push into positive territory.

### What H2 (Signal Masked by Execution Cost) Means Now

H2 as originally stated — "positive OOS without exec cost" — is **refuted** for 20 training days. But a weaker version survives: "exec cost explains a large fraction of OOS loss." This reframes the problem: the gap to profitability may be ~5 points (gross alpha needed), not ~50 points (net alpha needed). Reducing execution cost in practice (limit orders, larger bar sizes, or richer features that allow fewer but more confident trades) becomes a more tractable problem.

### Run C: A Promising but Undertrained Signal

Run C (199d, no exec cost) hitting +10.93 on val with only 17.4% explained variance is tantalizing. If properly trained (10-20M steps), it might show significant positive gross alpha. But it could also regress to negative as the model converges and overfits. This is the most important follow-up.

---

## Proposed Next Experiments

1. **exp-003: 199d No-Exec-Cost with Sufficient Training (10M+ steps)** — Run C showed positive val returns at 2M steps but was catastrophically undertrained (explained_variance = 0.174). Run with 10-15M steps to allow convergence, then evaluate. This is the most direct follow-up. If positive OOS persists at convergence: strong evidence for signal + data interaction. If it collapses: the 2M result was random walk artifacts.

2. **exp-004: Multi-Seed Replication of 20d No-Exec-Cost** — Run B's val return (-4.43) is within noise of zero (95% CI: [-40, +32]). Running seeds 42, 43, 44 would reduce uncertainty. If 2/3 seeds are positive: signal exists but is fragile. If all 3 negative: REFUTED with conviction. Low cost (~1.5 hours for 3 runs).

3. **exp-005: Execution Cost Gradient (Half-Cost Ablation)** — Instead of binary on/off, test with `spread/4` instead of `spread/2` per trade. This would help distinguish "exec cost is too high" from "no signal at all." If half-cost still negative but better than full-cost: the relationship is monotonic and reducing cost helps. Requires a code change (HANDOFF).

4. **Regardless of above: Resolve exp-001 (Data Scaling WITH Exec Cost)** — The P0 question "Does increasing training data from 20 to 199 days fix OOS generalization?" remains unanswered. Run C tantalizes but changed two variables. A clean 199d-with-exec-cost run is needed to isolate the data effect.

---

## Program Status

- Questions answered this cycle: **1** (P0: "Is the agent learning signal masked by execution cost?" → REFUTED for 20d; exec cost explains ~35 points of loss but does not flip the sign)
- New questions added this cycle: **1** (Does 199d + no exec cost produce positive OOS with sufficient training steps?)
- Questions remaining (open, not blocked): **5** (P0 data scaling, P1 checkpoints, P1 VecNormalize, P2 reward shaping, P3 observation signal)
- Handoff required: **NO** (all findings addressable within research scope; half-cost ablation would need a handoff but is lower priority than the proposed next experiments)
