# Experiment: Execution Cost Ablation — Gross vs Net Alpha Decomposition

## Hypothesis

Removing execution cost from the reward function will produce **positive mean OOS returns** for the MLP agent, demonstrating that the observation space contains exploitable signal that is currently masked by transaction costs:

1. **No-exec-cost MLP (20 days) mean val return > 0.0** — the agent learns signal when not penalized for trading.
2. **No-exec-cost MLP (20 days) mean test return > 0.0** — the signal generalizes out-of-sample.

The mechanism: execution cost of $1.25 round-trip (1 tick) may exceed the per-trade alpha available in bar_size=1000 observations. Without this cost, the agent's reward gradient points toward genuine price prediction rather than "don't trade." If the agent can predict direction better than chance, removing the cost penalty should reveal positive returns.

**Null hypothesis:** No-exec-cost OOS returns remain ≤ 0.0, indicating the agent is not learning directionally useful signal from the 21-dim observation space — the problem is not execution cost but rather absence of predictive features or insufficient training data.

**Alternative failure mode:** No-exec-cost in-sample returns are very high (>200) but OOS returns remain ≤ 0.0. This would indicate the agent overfits more aggressively without the implicit regularization that execution cost provides (cost penalizes frequent position changes, acting as a trade-frequency regularizer).

## Independent Variables

| Variable | Control Value | Treatment Value | Notes |
|----------|--------------|-----------------|-------|
| `--execution-cost` | Present (enabled) | Omitted (disabled) | The ONLY variable that changes between control and treatment |

Two training-day configurations are tested to assess interaction effects:

| Run | Train Days | Exec Cost | Architecture | Purpose |
|-----|-----------|-----------|-------------|---------|
| A | 20 | YES | MLP | Control — reproduces historical baseline (pre-003/pre-006) |
| B | 20 | NO | MLP | Primary treatment — isolates exec cost effect |
| C | 199 | NO | MLP | Stretch — tests if data + no-exec-cost interacts |

**Why 20 days as primary:** All historical baselines (pre-003: val -51.5, pre-006: val -62.9) used 20 days with exec cost. Using 20 days isolates the exec-cost variable against a well-characterized baseline. 199-day runs change two variables simultaneously (data + cost) and are harder to interpret without a 199-day-with-cost baseline from exp-001.

**Why MLP only (no LSTM):** This is a diagnostic experiment — does any signal exist? MLP at 20 days is the cheapest, fastest way to answer the question. If signal is found, LSTM follow-up is a separate experiment. LSTM adds 6+ GPU-hours per run, which would consume the entire budget for a diagnostic that MLP can answer in ~1 hour.

**Why 2M steps (not 5M):** Pre-006 showed that 5M steps on 20 days causes MORE overfitting than 2M (val -51.5 → -62.9). The 2M baseline (pre-003: val -51.5) is the least-overfit MLP result. Using 2M also keeps runs short (~30 min), allowing all 3 runs within budget.

## Controls

All of the following are held constant across all runs:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--bar-size` | 1000 | Best granularity (pre-001) |
| `--ent-coef` | 0.05 | Prevents entropy collapse — critical here since prior no-exec-cost run collapsed at 0.01 |
| `--learning-rate` | 0.001 | Best convergence speed (pre-001) |
| `--policy-arch` | 256,256 | Standard architecture |
| `--activation` | relu | Standard activation |
| `--shuffle-split` | enabled | Eliminates regime-shift confound |
| `--seed` | 42 | Reproducibility. Identical train/val/test split as pre-003. |
| `--total-timesteps` | 2,000,000 | Matches pre-003 baseline (least overfit at 20 days) |
| `--checkpoint-freq` | 500,000 | Saves at 500K, 1M, 1.5M, 2M for overfitting trajectory |
| Architecture | MLP (PPO, MlpPolicy) | CPU-optimal, cheapest diagnostic |
| Val set size | 5 days | Hardcoded in train.py |
| Software | Same SB3/PyTorch versions as all prior experiments | No version changes |
| Hardware | Local (Apple Silicon) | MLP is CPU-bound per DOMAIN_PRIORS.md |

**Forced flatten cost note:** `compute_forced_flatten()` always charges `spread/2 * |position|` on the terminal step regardless of the `--execution-cost` flag. This means ~$0.625 per episode closing cost exists even in the no-exec-cost condition. With ~23 bars per episode at bar_size=1000, this is 1 trade out of potentially 5-15 position changes — a minor but systematic negative bias. We accept this confound for this experiment rather than requiring a code change. If results are marginal (close to 0), this confound must be investigated in a follow-up.

## Metrics (ALL must be reported)

### Primary

1. **Mean val return (no exec cost eval)** — Directly tests whether the agent learns directionally useful signal. Evaluated on the trained model without execution cost.
2. **Mean test return (no exec cost eval)** — Confirms the val result on the independent test hold-out.

### Secondary

| Metric | Purpose |
|--------|---------|
| Mean val return (WITH exec cost eval) | Practical test: would the learned signal survive real trading costs? Cross-eval the no-exec-cost model using `execution_cost=True`. |
| Mean test return (WITH exec cost eval) | Same for test set. |
| Val Sortino ratio (no exec cost) | Risk-adjusted return |
| Test Sortino ratio (no exec cost) | Risk-adjusted return on test set |
| Positive val episodes (count/5) | Distribution of outcomes |
| Positive test episodes (count/total) | Distribution of outcomes |
| Val std return | Variance — high std with ~0 mean suggests noise trading |
| Test std return | Variance |
| Final entropy | Higher than -0.60 expected. Compare to exec-cost control (~-0.48). If much lower → exec cost was regularizing. |
| Entropy trajectory (at 500K, 1M, 1.5M, 2M) | Detects entropy collapse timing |
| Mean episode trade count (in-sample) | Number of position changes per episode. If >> 15 without exec cost, agent is noise-trading. |
| In-sample mean return | Calibration. Expect higher than exec-cost control. |
| Explained variance (at 1M, 2M) | Overfitting diagnostic |
| approx_kl (at 1M, 2M) | Training stability |
| Training FPS | Should be similar to exec-cost runs (same architecture) |

### Sanity Checks

| Check | Expected | If Violated |
|-------|----------|-------------|
| Entropy stays above -0.60 throughout training | ent_coef=0.05 prevents collapse | If entropy < -0.60, the agent is exploiting a degenerate strategy. The result may be invalid. |
| Loss does not diverge (no NaN) | Standard PPO stability | Abort run. |
| Control (Run A) val return within ±15 of pre-003 historical baseline (-51.5) | Reproduces prior result | Infrastructure drift — cannot trust comparisons. |
| In-sample return is positive for all runs | Agent can at least fit training data | If negative in-sample without exec cost, something is fundamentally broken. |
| Mean trade count per episode < 40 (no-exec-cost runs) | Agent isn't flipping every bar | If > 40 out of ~23 bars (multiple round-trips per bar), the agent learned a degenerate high-frequency strategy. Return may be positive from noise trading, not signal. |

## Baselines

| Baseline | Source | Val Return | Test Return | Notes |
|----------|--------|------------|-------------|-------|
| MLP 20d + exec cost (2M steps) | pre-003-local-mlp | -51.5 | -62.5 | Primary comparison. Same config except exec cost. |
| MLP 20d + exec cost (5M steps) | pre-006-gpu-mlp | -62.9 | -44.0 | Secondary reference. More steps = more overfit. |

**Run A (MLP 20d + exec cost, 2M steps)** reproduces the pre-003 baseline as an infrastructure sanity check. The delta between Run A and Run B (no exec cost) is the primary treatment effect.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1:** No-exec-cost MLP 20d (Run B) mean val return > 0.0 (positive gross alpha on val set)
- [ ] **SC-2:** No-exec-cost MLP 20d (Run B) mean test return > 0.0 (positive gross alpha on test set)
- [ ] **SC-3:** No-exec-cost MLP 20d (Run B) mean val return is at least 30 points better than exec-cost control (Run A) — the exec cost removal has a material effect
- [ ] **SC-4:** No sanity check failures: entropy > -0.60, no NaN, control reproduces within ±15 of historical, in-sample positive, trade count < 40/episode
- [ ] **SC-5:** At least 3 out of 5 val episodes have positive return in Run B — signal is not driven by 1-2 outlier days

**Interpretation guide:**

- SC-1 + SC-2 + SC-3 + SC-5 pass → **CONFIRMED**: The observation space contains exploitable signal masked by execution cost. The agent can predict direction but not with enough magnitude to overcome the 1-tick round-trip cost. Actionable: reduce trading frequency (larger bar sizes, position-change penalty), seek stronger signals (richer features, more data), or reduce costs (limit orders instead of market orders).
- SC-1 passes but SC-2 fails → **INCONCLUSIVE**: Signal on val but not test. Possible overfitting or small val sample (5 days). Need more seeds/data to distinguish signal from noise.
- SC-1 fails (val ≤ 0.0) → **REFUTED**: Even without execution cost, the agent does not learn positive OOS signal from the 21-dim observation space. The problem is deeper than transaction costs — the features may lack predictive power, or 20 training days is insufficient. Pivot to P3 (observation signal audit via supervised learning) and/or re-test with 199 days (Run C provides preliminary data).
- SC-4 fails (entropy collapse or degenerate trading) → **INVALID**: The result cannot be interpreted. The agent found an exploit rather than learning genuine signal. Increase ent_coef and re-run.

## Minimum Viable Experiment

Before the full protocol, run a quick validation:

1. **Run MLP with no `--execution-cost` for 200K steps** (~4 min locally).
2. **Verify:**
   - Training starts without errors
   - In-sample return is positive by 200K steps
   - Entropy remains above -0.60
   - FPS is comparable to exec-cost runs (±20%)
   - No NaN in loss
3. **If MVE fails:** Diagnose before running full protocol. Most likely: entropy collapse (increase ent_coef) or degenerate strategy (agent flips every bar).

**MVE success gate:** Training runs for 200K steps without error, entropy > -0.60, in-sample return > 0. Only proceed to full protocol if MVE passes.

## Full Protocol

### Phase 1: Minimum Viable Experiment

1. Run MLP with no `--execution-cost` for 200K steps locally.
2. Check: entropy > -0.60, in-sample return > 0, no errors, FPS reasonable.
3. If passes → proceed. If fails → diagnose and adjust.

### Phase 2: Control Run (Run A)

4. **Run A (MLP 20d, exec cost, 2M steps):** Infrastructure sanity check.

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --bar-size 1000 --execution-cost \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --total-timesteps 2000000 \
  --checkpoint-freq 500000
```

5. Verify Run A val return is within ±15 of pre-003 baseline (-51.5). If not → abort.
6. Record all metrics. Save model and VecNormalize.

### Phase 3: Treatment Runs (Runs B and C)

Runs B and C can execute in parallel (both are MLP on CPU).

7. **Run B (MLP 20d, NO exec cost, 2M steps):** Primary treatment.

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --bar-size 1000 \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --total-timesteps 2000000 \
  --checkpoint-freq 500000
```

Note: `--execution-cost` is **omitted** — this is the only difference from Run A.

8. **Run C (MLP 199d, NO exec cost, 2M steps):** Stretch run — tests data x cost interaction.

```bash
cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py \
  --cache-dir ../cache/mes/ --bar-size 1000 \
  --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 \
  --shuffle-split --seed 42 --train-days 199 --total-timesteps 2000000 \
  --checkpoint-freq 500000
```

### Phase 4: Cross-Evaluation

9. **Re-evaluate Run B's saved model with `execution_cost=True`:**
   - Load Run B's `ppo_lob.zip` and `vec_normalize.pkl`
   - Call `evaluate_sortino()` on val and test sets with `execution_cost=True`
   - This answers: "Would the no-exec-cost policy survive real trading costs?"

10. **Re-evaluate Run C's saved model with `execution_cost=True`:** Same cross-eval for the 199-day run.

### Phase 5: Collect and Compare

11. For each run, extract from TensorBoard logs and `evaluate_sortino()` output:
    - Val/test mean return, std, Sortino, positive episode count (both with and without exec cost for Runs B and C)
    - Entropy at 500K, 1M, 1.5M, 2M steps
    - Explained variance at 1M, 2M
    - approx_kl at 1M, 2M
    - In-sample return (final training return)
    - Mean trade count per episode (if available from logs, otherwise count position changes in eval episodes)
    - FPS, wall time
12. Compare Run A (exec cost) vs pre-003 historical baseline (infrastructure sanity).
13. Compare Run B (no exec cost) vs Run A (exec cost) — **primary hypothesis test**.
14. Compare Run C (199d no exec cost) vs Run B (20d no exec cost) — data x cost interaction.
15. Report cross-eval results (Run B model evaluated WITH exec cost) — practical significance.
16. Evaluate all success criteria.

## Resource Budget

- Max GPU-hours: **0** (all MLP, CPU-only)
- Max CPU-hours: **4** (estimated)
- Max wall-clock time: **3 hours** (Runs B and C can overlap with Run A sequential)
- Max training runs: **4** (MVE + 3 full runs)
- Max seeds per configuration: **1** (seed 42 only)

| Run | Architecture | Train Days | Exec Cost | Est. Time | Purpose |
|-----|-------------|------------|-----------|-----------|---------|
| MVE | MLP | 20 | NO | ~4 min | Infrastructure validation |
| A | MLP | 20 | YES | ~30 min | Control baseline |
| B | MLP | 20 | NO | ~30 min | Primary treatment |
| C | MLP | 199 | NO | ~45 min | Stretch: data x cost interaction |
| **Total** | | | | **~1.8 hrs** | |

**Budget is conservative.** All runs are local MLP at 2M steps. Even if runs take 2x longer than estimated, total is well under the 8 GPU-hour budget. No GPU cost incurred.

## Compute Target

**Compute:** `local`

MLP 256x256 with PPO is CPU-bound (per `DOMAIN_PRIORS.md`: "SB3 MlpPolicy doesn't benefit from GPU"). All runs complete in under 1 hour each locally. No need for AWS.

## Abort Criteria

| Condition | Action |
|-----------|--------|
| Loss diverges (NaN or inf) for any run | Abort that run. Record as infrastructure failure. |
| Entropy drops below -0.60 before 1M steps in Run B or C | Abort — entropy collapse without exec cost. Record as a finding: exec cost was acting as implicit regularization. Increase ent_coef to 0.1 and re-run if budget remains. |
| Run A val return differs from pre-003 baseline (-51.5) by > 20 points | Infrastructure has changed. Investigate before proceeding. |
| Mean trade count > 40 per episode AND mean return ≈ 0 in Run B | Agent learned degenerate noise-trading. Result is INVALID for the hypothesis. Record as confound. |
| Any single run exceeds 2 hours wall time | Kill to stay within budget. Evaluate latest checkpoint. |

## Confounds to Watch For

1. **Forced flatten cost asymmetry.** `compute_forced_flatten()` always charges `spread/2 * |position|` on the terminal step, even without `--execution-cost`. This creates a small systematic negative bias (~$0.625 per episode) in the no-exec-cost condition. With ~23 bars per episode, this is ~4% of a typical episode's trade count. If no-exec-cost OOS returns are slightly negative (e.g., -5 to 0), this confound may explain the result. **Mitigation:** Report per-episode forced-flatten cost. If results are marginal, create a HANDOFF to TDD to make `compute_forced_flatten` respect the `execution_cost` flag, then re-run.

2. **VecNormalize reward scaling.** Without exec cost, raw reward magnitudes change (no spread deduction per trade). VecNormalize's running reward variance will adapt, but the effective learning signal differs. This is inherent to the ablation — we are changing the reward function, which necessarily changes the reward distribution. **Mitigation:** Report the final VecNormalize reward running mean and variance for both conditions.

3. **Execution cost as implicit regularization.** Execution cost penalizes position changes, acting like an L1 penalty on trade frequency. Without it, the agent may trade more aggressively, amplifying noise exposure. Positive OOS returns without exec cost could reflect lucky noise trading rather than genuine signal. **Mitigation:** SC-5 requires ≥3/5 positive val episodes (not just 1-2 outliers). Also monitor trade frequency — if dramatically higher without exec cost (>3x), interpret results cautiously.

4. **Different optimal policies.** The exec-cost and no-exec-cost environments have fundamentally different optimal policies. The no-exec-cost optimal policy in a random walk is to trade every bar (zero expected cost). Comparing returns between these conditions does not measure the "same strategy minus cost" — it measures two different learned strategies. **Mitigation:** The cross-eval (Run B model evaluated WITH exec cost) tests whether the no-exec-cost policy is also viable under real costs. If cross-eval returns are much worse than Run A, the no-exec-cost policy is impractical regardless of gross alpha.

5. **Single seed / small val set.** 5 val days with seed 42 may not be representative. A lucky split could produce positive val returns by chance. **Mitigation:** SC-5 requires ≥3/5 positive val episodes, reducing the chance of one outlier driving the result. The test set (224 days for 20-day training) provides a larger sample. If budget allows and results are promising, a seed-43 replication would strengthen confidence.

6. **Interaction with exp-001 (data scaling).** If exp-001 shows that 199 days improves OOS with exec cost, Run C's result becomes harder to interpret (two variables changed). **Mitigation:** Run B (20 days, no exec cost) is the primary comparison against Run A (20 days, exec cost). Run C is a stretch goal that probes the interaction but is not needed for the primary hypothesis test.

7. **Historical baseline staleness.** Pre-003 ran months ago. If any environment code, VecNormalize behavior, or data processing has changed since then, Run A may not reproduce the historical baseline. **Mitigation:** Run A is explicitly an infrastructure sanity check. If it deviates by >20 points, investigate before proceeding.
