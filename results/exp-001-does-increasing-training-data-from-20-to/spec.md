# Experiment: Training Data Scaling — 20 vs 199 Days

## Hypothesis

Increasing training data from 20 to 199 days (10x) will reduce overfitting and produce meaningfully less negative OOS returns for the LSTM agent:

1. **LSTM 199-day mean val return will be ≥ -16.7** (at least 20 points better than the historical 20-day LSTM baseline of -36.7).
2. **LSTM 199-day mean test return will be ≥ -13.4** (at least 20 points better than the historical 20-day LSTM baseline of -33.4).

The mechanism: with 199 training days, each day is visited ~16 times per 5M steps (vs ~160 times on 20 days). The model cannot memorize day-specific artifacts and must learn generalizable patterns. If the 21-dim observation space contains exploitable signal, 199 days should be sufficient to find it.

**Null hypothesis:** 199-day OOS returns remain within ±10 points of the 20-day baselines (-36.7 val, -33.4 test), indicating the problem is not data quantity but rather the observation space lacking exploitable signal or execution cost exceeding available alpha.

## Independent Variables

| Variable | Values | Notes |
|----------|--------|-------|
| `--train-days` | **199** (treatment) | The only variable that changes vs historical baselines |

Two architectures are tested:

| Config | Architecture | Why | GPU-hours |
|--------|-------------|-----|-----------|
| LSTM | `--recurrent` (RecurrentPPO, MlpLstmPolicy) | Best OOS performer in all prior experiments | ~6.3 |
| MLP | PPO 256x256 ReLU | Cheap sanity check — confirms data effect is not architecture-specific | ~1.2 |

**Why no 20-day re-run controls:** The 8 GPU-hour budget is binding. A single LSTM re-run control costs 6.1 GPU-hours — 76% of the budget — and would leave insufficient budget for the treatment runs that actually test the hypothesis. Instead, we rely on the historical baselines from pre-005 (LSTM: val -36.7, test -33.4) and pre-006 (MLP: val -62.9, test -44.0), with a cheap MLP 20-day control (1.1 GPU-hours) as an infrastructure drift sanity check. If the MLP control reproduces within ±15 of the historical baseline, we trust the infrastructure has not changed.

**Critical confound acknowledged:** Changing `--train-days` from 20 to 199 changes the val/test set composition (different shuffled days). The 199-day runs are evaluated on *different* held-out days than the 20-day baselines. Therefore, the primary test is the **absolute sign and magnitude** of 199-day OOS returns, not the delta vs 20-day baselines. The 20-day baselines establish "how bad things were" as a reference point.

## Controls

All of the following are held constant across all runs:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--bar-size` | 1000 | Best granularity (pre-001) |
| `--ent-coef` | 0.05 | Prevents entropy collapse (pre-001) |
| `--learning-rate` | 0.001 | Best convergence speed (pre-001) |
| `--policy-arch` | 256,256 | Standard architecture |
| `--activation` | relu | Standard activation |
| `--execution-cost` | enabled | Realistic trading conditions |
| `--shuffle-split` | enabled | Eliminates regime-shift confound |
| `--seed` | 42 | Reproducibility |
| `--total-timesteps` | 5,000,000 | Matches GPU baselines (pre-005, pre-006) |
| `--checkpoint-freq` | 1,000,000 | Saves at 1M, 2M, 3M, 4M, 5M for overfitting diagnosis |
| Val set size | 5 days | Hardcoded in train.py |
| Docker image | `kurtbell87/lob-rl:latest` | Same container as pre-005/006/007 |
| Cache | `cache/mes/` (249 .npz files) | Pre-uploaded to RunPod volume |

**Why both architectures:** If only LSTM improves, the data effect could be LSTM-specific (e.g., temporal patterns only emerge with enough data). If both improve, the effect is about data quantity generically. If neither improves, the problem is not data quantity.

## Metrics (ALL must be reported)

### Primary

1. **Mean val return** — Directly tests whether more data improves OOS generalization. Computed via `evaluate_sortino()` over the 5 val episodes.
2. **Mean test return** — Confirms the val result on the independent test hold-out (45 days for 199-day runs).

### Secondary

| Metric | Purpose |
|--------|---------|
| Val Sortino ratio | Risk-adjusted return (penalizes downside volatility) |
| Test Sortino ratio | Risk-adjusted return on test set |
| Positive val episodes (count/total) | Distribution of outcomes — are some days profitable? |
| Positive test episodes (count/total) | Same for test set |
| Val std return | Variance of OOS performance |
| Test std return | Variance of OOS performance |
| Final entropy | Policy determinism at end of training. Higher = more exploration retained |
| Explained variance (at 1M, 3M, 5M) | Overfitting diagnostic. Near 1.0 = memorization. Should be lower with 199 days |
| approx_kl (at 1M, 3M, 5M) | Training stability. LSTM showed 0.148 at 5M on 20 days — watch for change |
| clip_fraction (at 1M, 3M, 5M) | Policy update magnitude. Correlated with KL |
| Training FPS (steady-state) | Whether 199 days changes throughput due to I/O patterns |
| Wall time (total) | Resource consumption |
| In-sample mean return | Calibration. Expect lower in-sample return with 199 days (harder to memorize) |

### Sanity Checks

| Check | Expected | If Violated |
|-------|----------|-------------|
| Entropy stays above -0.60 throughout training | ent_coef=0.05 prevents collapse | If entropy < -0.60, ent_coef may be too low for 199 days |
| Loss does not diverge (no NaN) | Standard PPO stability | Abort if NaN detected |
| MLP 20-day control val return within ±15 of pre-006 baseline (-62.9) | Reproduces prior result on same infra | Infrastructure drift — invalidates all comparisons to historical baselines |
| In-sample return is positive for all runs | Agent can at least fit training data | If negative in-sample, something is broken in the pipeline |
| LSTM approx_kl stays below 0.3 | Stable policy updates | If sustained > 0.3, LR is too high for 199-day data distribution |

## Baselines

| Baseline | Source | Val Return | Test Return | Notes |
|----------|--------|------------|-------------|-------|
| LSTM 20-day | pre-005-gpu-lstm | -36.7 | -33.4 | Historical. Same hardware class (RTX 4090), same Docker image, same hyperparams. |
| MLP 20-day | pre-006-gpu-mlp | -62.9 | -44.0 | Historical. **Re-run as infrastructure sanity check** (Run A). |

The MLP 20-day control (Run A) is cheap (1.1 GPU-hours) and serves as an infrastructure drift detector. If Run A reproduces pre-006 within ±15 points, we trust the historical LSTM baseline from pre-005 without re-running it.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1:** LSTM 199-day mean val return ≥ -16.7 (at least 20 points better than historical baseline -36.7)
- [ ] **SC-2:** LSTM 199-day mean test return ≥ -13.4 (at least 20 points better than historical baseline -33.4)
- [ ] **SC-3:** MLP 199-day mean val return is at least 15 points better than MLP 20-day control (Run A) from this experiment
- [ ] **SC-4:** No sanity check failures: entropy > -0.60, no NaN losses, MLP control reproduces within ±15 of historical, in-sample returns positive
- [ ] **SC-5:** 199-day LSTM explains variance < 0.95 at 5M steps (reduced memorization compared to 20-day baseline's 0.983)

**Interpretation guide:**
- SC-1 + SC-2 + SC-3 pass → **CONFIRMED**: Data quantity is a primary bottleneck. Scale up training data for future experiments.
- SC-1 + SC-2 pass but SC-3 fails → **PARTIALLY CONFIRMED**: Data helps LSTM but not MLP. LSTM's temporal modeling benefits specifically from more diverse data.
- SC-1 or SC-2 fails, but 199-day LSTM returns are still meaningfully better (>10 points) than historical baselines → **INCONCLUSIVE**: Data helps but not enough. Pivot to P0-b (execution cost ablation) as the binding constraint.
- Both SC-1 and SC-2 fail, 199-day LSTM returns within ±10 of historical baselines → **REFUTED**: Problem is not data quantity. Pivot to P0-b (execution cost ablation) or P3 (observation signal audit).

## Minimum Viable Experiment

Before running the full protocol, execute a quick validation on RunPod:

1. **Run MLP with `--train-days 199` for 500K steps** (~7 min on RTX 4090).
2. **Verify:**
   - Training starts without errors (all 199 .npz files load correctly)
   - In-sample return is positive by 500K steps
   - FPS is within 50% of the 20-day MLP FPS (1,275) — no severe I/O degradation from 199 files
   - VecNormalize computes stats without errors over 199 days
   - Checkpoints save correctly at the expected intervals
3. **If MVE fails:** Diagnose before burning the full budget. Likely causes: memory pressure from 199 lazy-loaded files, I/O bottleneck from cycling through more files, or VecNormalize instability with more diverse data.

**MVE success gate:** Training runs for 500K steps without error, FPS > 600, and in-sample return is positive. Only proceed to full protocol if MVE passes.

## Full Protocol

### Phase 1: Infrastructure Validation (MVE)

1. Launch one RunPod RTX 4090 pod via `./runpod/launch.sh`.
2. Run MLP with `--train-days 199 --total-timesteps 500000 --checkpoint-freq 100000`.
3. Verify per MVE criteria above. If fails, stop and diagnose. If passes, proceed.

### Phase 2: MLP 20-Day Control (infrastructure sanity check)

4. **Run A (MLP 20-day control):** Verifies infrastructure matches pre-006 baseline.

### Phase 3: 199-Day Treatment Runs

5. **Run B (MLP 199-day):** Cheap treatment run, tests data effect on MLP.
6. **Run C (LSTM 199-day):** Primary treatment run, tests the core hypothesis.

Runs B and C can launch in parallel on separate pods (MLP completes in ~1.2 hrs while LSTM takes ~6.3 hrs).

### Phase 4: Collect and Compare

7. For each run, extract from TensorBoard logs and `evaluate_sortino()` output:
   - Val/test mean return, std, Sortino, positive episode count
   - Entropy at 1M, 2M, 3M, 4M, 5M steps
   - Explained variance at 1M, 3M, 5M
   - approx_kl and clip_fraction at 1M, 3M, 5M
   - In-sample return (final training return)
   - FPS, wall time
8. Compare Run A (MLP 20d) vs pre-006 historical baseline (infrastructure sanity check).
9. Compare Run B (MLP 199d) vs Run A (MLP 20d) — same-infrastructure MLP data effect.
10. Compare Run C (LSTM 199d) vs pre-005 historical baseline (primary hypothesis test).
11. Evaluate all success criteria.

### Commands

All commands use `./runpod/launch.sh` which handles `--cache-dir` and `--output-dir` automatically.

**MVE (MLP 199-day, 500K steps):**
```bash
EXP_NAME=exp001-mve ./runpod/launch.sh \
  --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --train-days 199 --total-timesteps 500000 \
  --checkpoint-freq 100000
```

**Run A (MLP 20-day control):**
```bash
EXP_NAME=exp001-run-a-mlp-20d ./runpod/launch.sh \
  --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --train-days 20 --total-timesteps 5000000 \
  --checkpoint-freq 1000000
```

**Run B (MLP 199-day treatment):**
```bash
EXP_NAME=exp001-run-b-mlp-199d ./runpod/launch.sh \
  --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --train-days 199 --total-timesteps 5000000 \
  --checkpoint-freq 1000000
```

**Run C (LSTM 199-day treatment):**
```bash
EXP_NAME=exp001-run-c-lstm-199d ./runpod/launch.sh \
  --recurrent --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --train-days 199 --total-timesteps 5000000 \
  --checkpoint-freq 1000000
```

## Resource Budget

- Max GPU-hours: **8**
- Max wall-clock time: **7 hours** (critical path = Run C at ~6.3 hrs)
- Max training runs: **4** (MVE + 3 full runs)
- Max seeds per configuration: **1** (seed 42 only — budget does not allow reproducibility run)

| Run | Architecture | Train Days | Est. GPU-hours | Est. Wall Time | Est. Cost |
|-----|-------------|------------|----------------|----------------|-----------|
| MVE | MLP | 199 | 0.12 | ~7 min | ~$0.07 |
| A (MLP 20d) | MLP | 20 | 1.1 | ~1.1 hrs | ~$0.65 |
| B (MLP 199d) | MLP | 199 | 1.2 | ~1.2 hrs | ~$0.71 |
| C (LSTM 199d) | LSTM | 199 | 6.3 | ~6.3 hrs | ~$3.72 |
| **Total** | | | **8.7** | | **~$5.15** |

**Budget note:** The total of 8.7 GPU-hours slightly exceeds the 8-hour budget. Mitigation: Run A and B can share a pod sequentially (A finishes in 1.1 hrs, B takes 1.2 hrs on the same pod = 2.3 hrs total). If Run C takes longer than 6.3 hrs, kill at 8 hrs and evaluate the latest checkpoint. Parallelism: launch Run A+B pod and Run C pod simultaneously — wall-clock time = max(2.3, 6.3) = 6.3 hrs.

**Why no LSTM 20-day re-run or seed 43 reproducibility run:** Each LSTM 5M-step run costs ~6.3 GPU-hours. Adding either would push total to 15+ GPU-hours, nearly double the budget. The MLP 20-day control (Run A) is the cheapest way to verify infrastructure consistency. The single-seed limitation is documented as a confound.

## Compute Target

**Compute:** `runpod`
**GPU type:** NVIDIA GeForce RTX 4090

LSTM at ~228 FPS requires ~6.1 hrs for 5M steps — too slow for local execution. MLP runs could theoretically run locally but are dispatched to RunPod for infrastructure consistency with Run C. All commands use `./runpod/launch.sh`. Runs B and C SHOULD be launched as parallel pods.

## Abort Criteria

| Condition | Action |
|-----------|--------|
| Loss diverges (NaN or inf) for any run | Abort that run. Record as infrastructure failure. Do not count toward hypothesis evaluation. |
| Training FPS < 100 for LSTM or < 500 for MLP (sustained > 100K steps) | Abort — I/O or memory issue. Diagnose before continuing. |
| MLP 20-day control (Run A) val return differs from pre-006 historical baseline (-62.9) by > 30 points | Abort all runs — infrastructure has changed materially. Results are not comparable to historical baselines. |
| LSTM approx_kl > 0.3 sustained for > 500K steps | Abort Run C — LR is too high for the 199-day data distribution. Record as a finding (LR needs tuning for larger datasets). |
| After 2.5M steps (50% budget), Run C in-sample return is negative AND entropy < -0.55 | Consider early abort. The model is both underfitting the data and collapsing its policy — a pathological state. |
| Any run exceeds 8 hours wall time | Kill to stay within budget. Evaluate the latest checkpoint. |

## Confounds to Watch For

1. **Different val/test compositions.** With `--seed 42 --shuffle-split`, changing `--train-days` from 20 to 199 moves 179 days from test→train. The val set is `all_files[train_days:train_days+5]`, so the 20-day and 199-day experiments have **completely different val and test sets**. The primary comparison is the absolute sign and magnitude of 199-day OOS returns, not a point-for-point delta against 20-day baselines. The 20-day results establish "how bad things were" as context.

2. **VecNormalize statistics.** With 199 training days, the running mean/variance in VecNormalize is computed from a more representative sample. This could improve generalization independently of what the policy learns. **Mitigation:** Report the final VecNormalize running mean and variance. If OOS improves, this confound must be investigated in a follow-up (P1 question).

3. **Fewer passes per day (underfitting risk).** At 5M steps with 199 days, each day is visited ~16 times (vs ~160 times on 20 days). The model may underfit — not learning enough from each day rather than overfitting less. **Mitigation:** Monitor in-sample return and explained variance. If in-sample return is much lower than the 20-day runs AND OOS doesn't improve, the model needs more total steps (10M+). This would be INCONCLUSIVE, not REFUTED.

4. **2022 bear market homogeneity.** All 249 days are from one year with a strong directional bias (down ~20%). Even with 199 training days, regime diversity is limited. **Mitigation:** This experiment only tests within-year generalization. Cross-year testing is out of scope.

5. **Single seed.** Budget constraints limit us to seed 42 only. A lucky or unlucky partition could dominate the result. **Mitigation:** The 5-day val set and 45-day test set provide some statistical averaging. If the experiment produces a strong signal (SC-1 or SC-2 passes with margin), seed sensitivity is less likely to explain it. If marginal, flag as a limitation.

6. **Test set size asymmetry.** The 199-day test set has only 45 days (vs 224 for the historical 20-day baseline). The 199-day test mean has higher variance (~1.5x standard error assuming same per-episode std). **Mitigation:** Report std and per-episode counts. Consider the standard error of the mean when interpreting test set differences.

7. **Checkpoint evaluation timing.** If the agent overfits late in training, the 5M checkpoint may not be the best. **Mitigation:** Checkpoints saved at 1M, 2M, 3M, 4M, 5M. The READ agent should compare the 3M, 4M, and 5M checkpoints to detect late-stage overfitting. This also provides data for the P1 early-stopping question.

8. **No LSTM infrastructure sanity check.** We verify infrastructure via the MLP 20-day control only. If there is an LSTM-specific regression (e.g., RecurrentPPO version change), we would not detect it. **Mitigation:** This is accepted as a budget trade-off. If Run C produces unexpected results (e.g., much worse than -36.7), investigate LSTM-specific infrastructure issues before concluding.
