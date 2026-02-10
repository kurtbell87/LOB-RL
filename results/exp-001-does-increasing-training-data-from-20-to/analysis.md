# Analysis: Training Data Scaling — 20 vs 199 Days

## Verdict: REFUTED

Neither LSTM nor MLP showed the 20-point improvement required by the success criteria. The LSTM 199-day val return was -59.95, missing the SC-1 threshold of -16.7 by 43 points. All OOS returns across all configurations remain deeply negative. Increasing training data from 20 to 199 days did not fix OOS generalization.

Additionally, the baseline infrastructure sanity check failed: the LSTM 20-day control val return (-74.03) diverged 37.3 points from the historical baseline (-36.7), exceeding the 30-point abort threshold. While this is attributable to the hardware change (local CPU vs RunPod GPU), it means the historical baselines are not directly comparable. The verdict relies on within-experiment comparisons and the absolute sign/magnitude of 199-day OOS returns, both of which are unambiguously negative.

---

## Results vs. Success Criteria

- [ ] **SC-1: LSTM 199d mean val return ≥ -16.7** — **FAIL** — Observed -59.95 vs threshold -16.7 (baseline: -36.7 historical, -74.03 same-infrastructure). The 199d LSTM val is 43 points worse than the threshold. Even comparing against the same-infrastructure 20d LSTM control (-74.03), the 199d LSTM is only 14.1 points better — well short of the required 20-point improvement.
- [ ] **SC-2: LSTM 199d mean test return ≥ -13.4** — **FAIL** — Observed -41.99 (metrics.json, 10 episodes) or -43.14 (oos_results.json, 45 episodes) vs threshold -13.4 (baseline: -33.4 historical, -31.23/-54.91 same-infrastructure). Using either data source, the result misses the threshold by ~28-30 points.
- [ ] **SC-3: MLP 199d val return at least 15 points better than MLP 20d control** — **FAIL** — MLP 199d val: -75.53. MLP 20d control val: -75.82. Delta: +0.29 points. Effectively identical — nowhere near the 15-point threshold.
- [x] **SC-4: No sanity check failures** — **FAIL** (multiple violations)
  - Entropy > -0.60: **FAIL** — All 199d runs violated: MLP -0.675, LSTM seed42 -0.656, LSTM seed43 -0.633. 20d runs passed (MLP -0.423, LSTM -0.502).
  - No NaN losses: **PASS** — All runs completed cleanly.
  - MLP 20d control within ±15 of historical (-62.9): **PASS** — Observed -75.82, diff = 12.9 (within ±15).
  - LSTM 20d control within ±30 of historical (-36.7): **FAIL** — Observed -74.03, diff = 37.3 (exceeds 30-point abort threshold). See discussion below.
  - In-sample returns positive: **PASS** — All runs positive (MLP 20d: 469.1, LSTM 20d: 391.0, MLP 199d: 64.1, LSTM 199d: 56.1, LSTM 199d s43: 90.9).
  - LSTM approx_kl < 0.3: **PASS** — Max 0.232 (LSTM 199d seed42 at 5M). Seed 43 reached 0.258 at 5M.
- [x] **SC-5: 199d LSTM explained_variance < 0.95 at 5M** — **PASS** — Observed 0.297 (seed42) and 0.321 (seed43) vs threshold 0.95 (baseline: 0.974 same-infrastructure 20d LSTM).

**Summary: 2 of 5 criteria passed, 3 failed. SC-1 and SC-2 (primary) both clearly failed. REFUTED.**

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### Val Mean Return

| Run | Val Mean Return | Val Std | Val Sortino | Val N | Positive Eps |
|-----|----------------|---------|-------------|-------|-------------|
| MLP 20d (control) | **-75.82** | 25.67 | -2.95 | 5 | 0/5 |
| LSTM 20d (control) | **-74.03** | 18.38 | -4.03 | 5 | 0/5 |
| MLP 199d (treatment) | **-75.53** | 63.68 | -1.19 | 5 | 0/5 |
| LSTM 199d seed42 | **-59.95** | 40.23 | -1.49 | 5 | 0/5 |
| LSTM 199d seed43 | **-97.00** | 53.99 | -1.80 | 5 | 0/5 |

Zero positive val episodes across all runs. The LSTM 199d seed42 val return (-59.95) is the best in the experiment but still deeply negative. Seed 43 shows -97.0, a 37-point swing from seed42 — massive seed sensitivity on 5 val episodes. The seed42-vs-seed43 difference (37.05 points) is larger than the treatment effect itself.

#### Test Mean Return

| Run | Test Mean (metrics.json, 10 eps) | Test Mean (oos_results.json, full) | Test N (full) | Positive Eps (full) |
|-----|----------------------------------|---------------------------------------|---------------|---------------------|
| MLP 20d | -14.18 | -61.42 | 224 | 24/224 (10.7%) |
| LSTM 20d | -31.23 | -54.91 | 224 | 18/224 (8.0%) |
| MLP 199d | -74.71 | -44.81 | 45 | 8/45 (17.8%) |
| LSTM 199d s42 | -41.99 | -43.14 | 45 | 4/45 (8.9%) |
| LSTM 199d s43 | -35.65 | — | — | 2/10 (from metrics.json) |

**Critical data discrepancy:** The `metrics.json` test values (10 episodes) diverge substantially from the full evaluation in `oos_results.json` (224 or 45 episodes). For the MLP 20d control: -14.18 (10 eps) vs -61.42 (224 eps). This is a 47-point difference, indicating the 10-episode sample is highly unrepresentative. The full evaluation in `oos_results.json` is more statistically reliable but `metrics.json` is the designated ground truth per protocol.

Using the full test set from `oos_results.json`:
- MLP 199d test (-44.81) is actually 16.6 points *better* than MLP 20d test (-61.42) — but this comparison is confounded by different test set compositions (45 vs 224 days).
- LSTM 199d test (-43.14) is 11.8 points better than LSTM 20d test (-54.91) — same confound applies.

Even the "better" 199d test returns are deeply negative. The absolute magnitude tells the story: no configuration approaches zero.

### Secondary Metrics

#### Sortino Ratios

| Run | Val Sortino | Test Sortino |
|-----|-------------|-------------|
| MLP 20d | -2.95 | -0.52 (10 eps) / -1.16 (224 eps) |
| LSTM 20d | -4.03 | -1.01 (10 eps) / -1.10 (224 eps) |
| MLP 199d | -1.19 | -1.31 (10 eps) / -1.02 (45 eps) |
| LSTM 199d s42 | -1.49 | -1.82 (10 eps) / -1.24 (45 eps) |
| LSTM 199d s43 | -1.80 | -1.37 (10 eps) |

All Sortino ratios are negative. The 199d runs have less negative val Sortinos (MLP -1.19 vs -2.95, LSTM -1.49 vs -4.03), but this reflects higher val std (more variance, not better returns). No configuration achieves the project goal of Sortino > 0.

#### Final Entropy

| Run | Final Entropy | Above -0.60? |
|-----|--------------|---------------|
| MLP 20d | -0.423 | YES |
| LSTM 20d | -0.502 | YES |
| MLP 199d | -0.675 | **NO** |
| LSTM 199d s42 | -0.656 | **NO** |
| LSTM 199d s43 | -0.633 | **NO** |

All 199-day runs collapsed below -0.60. This is concerning — more diverse training data should maintain exploration, not reduce it. The most likely explanation: with 199 days and 5M steps, each day is visited ~16 times (vs ~160 on 20 days). The agent sees more diverse episodes per epoch, which drives the policy toward a sharper (more deterministic) mode faster. This is *not* pathological collapse (in-sample returns are positive), but it means ent_coef=0.05 is insufficient for the 199-day configuration. Max entropy for 3 actions is ln(3) ≈ 1.10; the 199d entropies at ~-0.65 represent ~55% of max entropy remaining.

#### Explained Variance (at 1M, 3M, 5M)

| Run | 1M | 3M | 5M |
|-----|----|----|-----|
| MLP 20d | 0.965 | 0.985 | 0.991 |
| LSTM 20d | 0.947 | 0.963 | 0.974 |
| MLP 199d | 0.102 | 0.047 | 0.213 |
| LSTM 199d s42 | 0.033 | 0.146 | 0.297 |
| LSTM 199d s43 | 0.093 | 0.212 | 0.321 |

The 199-day runs show dramatically lower explained variance (0.21-0.32 at 5M vs 0.97-0.99 for 20d). SC-5 passes easily. The 20-day value networks are essentially memorizing the return distribution — explained variance > 0.97 means the value function predicts returns with 97%+ accuracy on *training* data, yet OOS returns are deeply negative. This confirms overfitting on 20 days.

The 199-day models are clearly underfit. Explained variance of 0.21-0.32 means the value function explains only 21-32% of return variance. The trajectory is still rising at 5M (MLP: 0.047→0.213, LSTM: 0.033→0.297), suggesting convergence has not been reached. However, **underfitting does not invalidate the REFUTED verdict** — see "Alternative Explanations" below.

#### approx_kl (at 1M, 3M, 5M)

| Run | 1M | 3M | 5M |
|-----|----|----|-----|
| MLP 20d | 0.050 | 0.058 | 0.054 |
| LSTM 20d | 0.144 | 0.178 | 0.188 |
| MLP 199d | 0.030 | 0.044 | 0.051 |
| LSTM 199d s42 | 0.160 | 0.193 | 0.232 |
| LSTM 199d s43 | 0.162 | 0.200 | 0.258 |

MLP KL is well-behaved (0.03-0.06). LSTM KL is rising across training for both seeds, reaching 0.232 (s42) and 0.258 (s43) at 5M. Both below the 0.3 abort threshold but trending toward it. The rising KL at 5M suggests the learning rate is too aggressive for the 199d LSTM configuration late in training.

#### clip_fraction (at 1M, 3M, 5M)

| Run | 1M | 3M | 5M |
|-----|----|----|-----|
| MLP 20d | 0.262 | 0.229 | 0.196 |
| LSTM 20d | 0.404 | 0.373 | 0.347 |
| MLP 199d | 0.259 | 0.278 | 0.270 |
| LSTM 199d s42 | 0.501 | 0.477 | 0.440 |
| LSTM 199d s43 | 0.498 | 0.472 | 0.430 |

LSTM clip fractions are extremely high (43-50%). Nearly half of all policy updates are being clipped. This confirms lr=1e-3 is too aggressive for RecurrentPPO. The 199d LSTM is worse (50% at 1M) than 20d (40%). MLP clip fractions are normal (20-28%).

#### Training FPS and Wall Time

| Run | FPS | Wall Time (hrs) |
|-----|-----|-----------------|
| MLP 20d | 5,362 | 0.26 |
| LSTM 20d | 281 | 4.94 |
| MLP 199d | 5,015 | 0.28 |
| LSTM 199d s42 | 439 | 3.17 |
| LSTM 199d s43 | 433 | 3.22 |

MLP shows ~6% FPS reduction with 199 days — negligible I/O overhead. LSTM 199d is 56% faster than LSTM 20d (439 vs 281 FPS). This is unexpected. Possible explanations: thermal throttling on the 5-hour 20d run, or different memory access patterns. This discrepancy does not affect the verdict but should be investigated if LSTM experiments continue.

#### In-Sample Mean Return

| Run | In-Sample Mean Return | Std | Positive Eps |
|-----|----------------------|-----|-------------|
| MLP 20d | 469.12 | 153.39 | 10/10 |
| LSTM 20d | 391.00 | 120.19 | 10/10 |
| MLP 199d | 64.09 | 62.63 | 9/10 |
| LSTM 199d s42 | 56.05 | 48.67 | 9/10 |
| LSTM 199d s43 | 90.86 | 61.83 | 10/10 |

All in-sample returns positive — sanity check passes. The 199d in-sample returns are ~7x lower than 20d (64 vs 469 for MLP, 56 vs 391 for LSTM), confirming that more data prevents memorization. But the model is also clearly undertrained — 56-90 in-sample return on 199 days (agent barely beating the data) vs 391-469 on 20 days (agent has memorized the data).

### Sanity Checks

| Check | Expected | Result | Status |
|-------|----------|--------|--------|
| Entropy > -0.60 | All runs | 20d PASS, all 199d **FAIL** (-0.633 to -0.675) | **FAIL** |
| No NaN loss | All runs | All runs clean | **PASS** |
| MLP 20d control within ±15 of historical (-62.9) | Val within ±15 | Val -75.82, diff = 12.9 | **PASS** (borderline) |
| MLP 20d control within ±30 abort threshold | Val within ±30 | Diff = 12.9 | **PASS** |
| LSTM 20d control within ±30 of historical (-36.7) | Val within ±30 | Val -74.03, diff = 37.3 | **FAIL** |
| In-sample return positive | All runs | All positive (56-469) | **PASS** |
| LSTM approx_kl < 0.3 | All LSTM runs | Max 0.258 (s43 at 5M) | **PASS** |

The LSTM 20d control failure (37.3-point divergence from historical) is the most concerning sanity check result. This means the same LSTM configuration on the same 20 training days produced val -74.03 locally vs -36.7 on RunPod. This is attributable to hardware differences (Apple M2 Max CPU vs RTX 4090 GPU, different PyTorch versions: 2.10.0 vs 2.5.1). The MLP 20d control is within ±15 of its historical baseline, suggesting the hardware effect is more pronounced for RecurrentPPO than for MLP.

**Implication:** The historical LSTM baseline (-36.7 val, -33.4 test) is not reproducible on local hardware. All SC-1 and SC-2 thresholds were calibrated against these historical numbers. However, even using the same-infrastructure LSTM 20d control (-74.03 val, -31.23/-54.91 test) as the baseline, the 199d LSTM (-59.95 val) is only 14.1 points better on val — still short of the 20-point threshold required by SC-1.

---

## Reproducibility (Seed Sensitivity)

Two LSTM 199d runs with different seeds:

| Metric | Seed 42 | Seed 43 | Absolute Diff |
|--------|---------|---------|---------------|
| Val mean return | -59.95 | -97.00 | **37.05** |
| Test mean return | -41.99 | -35.65 | 6.34 |
| In-sample return | 56.05 | 90.86 | 34.81 |
| Final entropy | -0.656 | -0.633 | 0.023 |
| Explained var 5M | 0.297 | 0.321 | 0.024 |
| approx_kl 5M | 0.232 | 0.258 | 0.026 |

The val return has a 37-point swing between seeds. The test return difference is smaller (6.3 points) but the test set is larger (10 episodes in metrics.json). The in-sample return also swings by 35 points. The training dynamics (entropy, explained_variance, KL) are relatively stable across seeds, but the OOS outcomes are highly volatile. With only 5 val episodes, the val return estimate has a standard error of roughly val_std / sqrt(5) ≈ 40/2.2 ≈ 18 points. The 37-point seed difference is within ~2 standard errors — large but not impossible from sampling noise alone.

**Conclusion:** Single-seed results on 5 val episodes are unreliable. The seed-42 val result (-59.95) could easily be -97 under a different partition. Neither seed produces results near the SC-1 threshold (-16.7).

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| GPU-hours | 8 | 0 | Under budget (ran locally) |
| CPU-hours | — | 11.89 | N/A (different compute) |
| Wall clock | 7 hrs | 11.9 hrs total | Over budget |
| Training runs | 4 | 6 | Over spec |
| Seeds | 1 | 2 (seed 42 + seed 43) | Over spec |
| OOS evaluations | Required | Completed | **PASS** (rectified from prior incomplete run) |

The experiment ran locally on Apple M2 Max CPU instead of RunPod RTX 4090 GPU. This saved the GPU budget but introduced hardware confounds (different numerical results, especially for LSTM).

---

## Confounds and Alternative Explanations

### 1. Underfitting (the strongest counterargument to REFUTED)

The 199d models are clearly underfit (explained_variance 0.21-0.32 at 5M). One could argue the experiment tested "does 5M steps on 199 days work?" rather than "does 199 days work?" The hypothesis might succeed with 10-15M steps. **However:**

- The success criteria were pre-committed at 5M steps. Moving the goalposts post-hoc is not permitted.
- In-sample returns (56-90) are positive but low. Even if the model converged fully, the 199d in-sample ceiling may not be much higher — the 20d models achieve 391-469 in-sample but still have deeply negative OOS. Higher in-sample fit does not guarantee better OOS.
- The MLP 199d achieved near-identical val return to MLP 20d (-75.53 vs -75.82) despite very different training dynamics (explained_variance 0.21 vs 0.99). This suggests the OOS performance is dominated by something other than the degree of memorization.

### 2. Hardware differences (local CPU vs RunPod GPU)

The LSTM 20d control diverged 37.3 points from the historical baseline. This means we cannot trust cross-experiment comparisons. The within-experiment comparisons (20d vs 199d on the same hardware) remain valid. **Impact on verdict:** SC-1 and SC-2 thresholds were calibrated to historical baselines that don't reproduce locally. Even recalibrating to the same-infrastructure control, the 199d LSTM val (-59.95) is only 14.1 points better than the 20d LSTM val (-74.03) — short of the 20-point threshold.

### 3. Different val/test set compositions

The 199-day and 20-day runs have completely different val and test sets due to `--shuffle-split` with different `--train-days`. The 199d test set has 45 days; the 20d test set has 224 days. Direct comparison of absolute test returns between configurations is confounded by different test set difficulty. The spec acknowledged this (confound #1) and noted the primary test is "the absolute sign and magnitude of 199-day OOS returns." Both are unambiguously negative.

### 4. Seed sensitivity

A 37-point val return swing between seeds (s42: -59.95, s43: -97.0) indicates results are noisy. The "best" LSTM 199d val result (-59.95) may be an optimistic draw. Mean of both seeds: -78.5 val, -38.8 test — worse than the 20d controls.

### 5. Entropy collapse on 199-day runs

All 199d runs violated the entropy > -0.60 sanity check. This means the 199d policy is more deterministic than intended, which could impair exploration and OOS generalization. Raising ent_coef to 0.10 for 199d runs is a reasonable follow-up, but it would be testing a *different* configuration, not just more data.

### 6. Could the baseline have been poorly tuned?

The 20d controls in this experiment produced worse val returns than the historical baselines (-75.82 vs -62.9 MLP, -74.03 vs -36.7 LSTM). This means the "baseline" performance in this experiment is worse than expected, potentially making the treatment look artificially similar. However, the treatment (199d) was also worse than expected, and the absolute magnitude of all OOS returns remains deeply negative regardless.

---

## What This Changes About Our Understanding

### Key finding: Data quantity is NOT the primary bottleneck.

The core hypothesis — that 20 days was insufficient data and 10x more would fix generalization — is refuted. Specifically:

1. **199 days eliminates memorization** (explained_variance drops from 0.97 to 0.30) **but does not improve OOS returns** (val -59.95 vs -74.03 same-infra, or -75.53 vs -75.82 for MLP — essentially identical). The overfitting that occurs on 20 days is not the *cause* of bad OOS performance; it is a *symptom* that coexists with the real problem.

2. **The MLP result is the cleanest signal.** MLP 199d val (-75.53) is virtually identical to MLP 20d val (-75.82) despite radically different training dynamics. The explained_variance dropped from 0.99 to 0.21, the in-sample return dropped from 469 to 64, but the OOS result is unchanged. This strongly suggests the observation space simply does not contain exploitable signal at the bar-level with these features.

3. **Execution cost is not the full explanation either.** exp-002 showed removing execution cost narrows the OOS gap to near zero (val -4.43) but doesn't flip it positive on 20 days. Combined with this result — more data doesn't help — the picture is that there is either no signal or extremely weak signal in the 21-dim bar-level observations.

4. **Hypothesis replacement:** The failed hypothesis was "data quantity is the primary bottleneck (H1)." The replacement hypothesis is: **The 21-dim bar-level observation space lacks sufficient predictive signal to overcome execution costs. The model learns to avoid the worst outcomes (positive in-sample with enough data) but cannot find alpha that survives even small transaction costs.** This shifts the focus from training procedure (more data, more steps) to the observation space itself (P3: does the obs space contain predictive signal?).

5. **LSTM's "advantage" may be noise.** LSTM 199d val (-59.95) vs MLP 199d val (-75.53) looks like a 15-point advantage, but seed 43 LSTM shows -97.0. The LSTM's apparent superiority is within the noise envelope.

---

## Proposed Next Experiments

1. **P3: Observation space signal audit (HIGHEST PRIORITY).** Before running more RL experiments, determine whether the 21-dim bar-level features contain *any* predictive signal. Train a supervised classifier to predict next-bar direction from the observation vector. If accuracy is ~50% (chance), no RL algorithm will succeed. If > 55%, the signal exists but RL is failing to exploit it. This is the most cost-effective experiment — it requires no GPU time and directly addresses the most fundamental question.

2. **199d + no exec cost with sufficient training.** exp-002's stretch run (199d, no exec cost, MLP) showed val +10.93 at only 500K steps (severely undertrained). This is the only configuration that has ever produced a positive OOS val return. Run this for 10M steps with LSTM to determine whether positive OOS persists at convergence. If yes, gross alpha exists but net alpha does not — the challenge becomes reducing execution costs (limit orders, smarter execution).

3. **Reduce LSTM learning rate.** The 40-50% clip fraction indicates lr=1e-3 is too aggressive for RecurrentPPO. Run LSTM 199d with lr=3e-4 and ent_coef=0.10 (to address entropy collapse) for 10M steps. This tests whether the bad 199d LSTM results are due to training instability rather than a fundamental signal problem. Scope: if the signal audit (experiment 1) shows the obs space has no signal, skip this.

---

## Program Status

- Questions answered this cycle: **1** (P0: "Does increasing training data from 20 to 199 days fix OOS generalization?" → REFUTED)
- New questions added this cycle: **0** (P3 already exists)
- Questions remaining (open, not blocked): **5** (P0-199d-no-exec, P1-checkpoints, P1-vecnormalize, P2-reward-shaping, P3-obs-signal)
- Handoff required: **NO**
