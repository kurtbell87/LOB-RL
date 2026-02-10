# Orchestrator Directive: PPO Barrier-Hit Agent for /MES

## Mission

Execute the attached specification (`ppo_barrier_hit_agent_mes_spec_v1.md`) to completion. The spec defines a PPO agent that learns entry timing on /MES futures using barrier-hit events as the reward signal. Your job is to decompose this into parallelizable work streams, assign infrastructure, enforce TDD discipline, and gate progression on explicit pass/fail criteria.

**Read the full spec before planning. Every section is referenced below by number.**

---

## Infrastructure Routing

All work runs on one of two targets. No local execution.

| Target  | Use for                                                        | Examples                                                             |
|---------|----------------------------------------------------------------|----------------------------------------------------------------------|
| **AWS** | CPU-bound data pipelines, synthetic validation, environment logic, unit tests, feature engineering, data analysis | Bar construction, label pipeline, Gambler's ruin, regime-switch validation, reward accounting, env step logic, feature extraction, hyperparameter sweep analysis |
| **RunPod** | GPU experiments: any neural network training or inference       | Supervised diagnostic (MLP classifier), PPO training, behavioral inspection, hyperparameter sweeps involving RL, out-of-sample evaluation with trained policies |

**Rule:** If the task does not require `torch`, `jax`, or GPU-accelerated libraries, it runs on AWS. If it trains or runs inference on a neural network, it runs on RunPod. When in doubt, AWS.

---

## Execution Model

### TDD Protocol

Every implementation task follows this cycle:

```
1. Write tests that encode the spec's pass/fail criteria
2. Run tests — confirm they FAIL (red)
3. Implement until tests PASS (green)
4. Refactor if needed — tests must still pass
5. Report: {task_id, status, test_results, artifacts}
```

Do not write implementation code before tests exist. Do not skip the red-green confirmation. If a test passes before implementation, the test is wrong — it's not testing anything.

### Gate Enforcement

The spec defines a strict ordering: "Do not proceed to step N+1 until step N passes." However, some steps have no mutual dependency and can run in parallel. The dependency graph below is authoritative. **Do not start a task until all of its dependencies have status PASSED.**

### Failure Protocol

If a task fails its gate criteria after implementation:

1. Diagnose root cause. Log it.
2. If the failure is in a dependency (e.g., label pipeline bug surfaced during supervised diagnostic), go back to the dependency task and fix it. Re-run all downstream gates.
3. If the failure is in the current task (e.g., MLP can't beat baseline), investigate per the spec's guidance (the spec tells you what each failure mode means — read it).
4. Do not work around a failure by relaxing the gate criteria. The criteria exist for a reason.

---

## Dependency Graph and Task Decomposition

```
                    ┌──────────┐
                    │  T1: Bar │
                    │  Pipeline│
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          │
        ┌──────────┐ ┌──────────┐  │
        │ T2: Label│ │ T3: Feat │  │
        │ Pipeline │ │ Extract  │  │
        └────┬─────┘ └────┬─────┘  │
             │             │        │
        ┌────┼────┐        │        │
        │         │        │        │
        ▼         ▼        │        │
  ┌──────────┐ ┌──────────┐│        │
  │T4: Gambler│ │T5: Regime││        │
  │Ruin Valid.│ │Switch Val││        │
  └────┬─────┘ └────┬─────┘│        │
       │             │      │        │
       └──────┬──────┘      │        │
              │             │        │
              ▼             ▼        │
        ┌─────────────────────┐     │
        │  T6: Supervised     │     │
        │  Diagnostic (GPU)   │     │
        └────────┬────────────┘     │
                 │                  │
                 ▼                  ▼
        ┌─────────────────────────────┐
        │  T7: Reward Accounting      │
        │  Unit Test                  │
        └────────┬────────────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  T8: Environment    │
        │  Implementation     │
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  T9: PPO Training   │
        │  (GPU)              │
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  T10: Behavioral    │
        │  Inspection (GPU)   │
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  T11: Hyperparam    │
        │  Sweep (GPU + CPU)  │
        └────────┬────────────┘
                 │
                 ▼
        ┌─────────────────────┐
        │  T12: Out-of-Sample │
        │  Eval (GPU)         │
        └────────┬────────────┘
                 │
                 ▼
              [DONE]
```

**Parallelism opportunities:**

- **T2 and T3** can run in parallel once T1 passes. T2 (labels) and T3 (features) both depend only on bar data.
- **T4 and T5** can run in parallel once T2 passes. Both validate the label pipeline on synthetic data.
- **T6** requires T4, T5, T2, and T3 all passed (needs validated labels + validated features).
- **Everything from T7 onward is sequential.** Each step's output is the next step's input.

Launch parallel tasks aggressively where the graph allows. Do not serialize work that can be parallelized.

---

## Task Specifications

### T1: Bar Construction Pipeline
**Infra:** AWS  
**Spec reference:** Section 1  
**Input:** Raw Databento MBO files for /MES, 2022-01-01 to 2023-12-31  
**Tests to write first:**
- Bar with exactly N trades produces correct OHLCV (hand-computed from known trade sequence)
- Bar does not straddle RTH session boundaries (8:30 AM / 3:00 PM CT)
- Bar does not straddle Globex maintenance window
- Incomplete bar at session end is discarded
- Bars across the full dataset have correct trade counts (sum of V_k = total matched trades)
- `t_start_k < t_end_k` for all bars
- `t_end_{k} <= t_start_{k+1}` for all consecutive bars within a session
- VWAP is bounded by [L_k, H_k] for all bars

**Gate:** All tests pass. Spot-check 10 random bars against a known charting platform (TradingView or Sierra Chart) for OHLCV correctness.  
**Artifacts:** Bar dataset (parquet or HDF5), trade sequence sidecar indexed by bar number.

---

### T2: Label Construction Pipeline
**Infra:** AWS  
**Spec reference:** Section 2  
**Depends on:** T1 PASSED  
**Input:** Bar dataset + trade sequence sidecar from T1  
**Tests to write first:**
- For a hand-crafted bar sequence where upper barrier is hit on bar j=5, label is +1 and τ⁺=5
- For a hand-crafted bar sequence where lower barrier is hit on bar j=3, label is -1 and τ⁻=3
- For a bar sequence where neither barrier is hit within T_max, label is 0
- Dual-barrier breach within a single bar resolves correctly via intrabar trade scan (construct a synthetic bar where both H >= U and L <= D, with known trade order)
- Gap-through edge case: first trade of bar exceeds both barriers, resolved by gap direction
- Short-direction barriers are correctly mirrored (profit barrier below entry, stop barrier above)
- T_max calibration procedure: with T_max=∞, compute P95 of winner distribution, verify it's a reasonable integer
- Tiebreak frequency is reported and computed correctly

**Gate:** All tests pass. Tiebreak frequency < 5% on real data with default parameters. Label distribution (p⁺, p⁻, p⁰) is reported and plausible (p⁺ < p⁻ for 2:1 barriers under roughly driftless conditions).  
**Artifacts:** Labeled dataset {(bar_index, H_k, τ, resolution_type)} for all bars. T_max calibration report with histograms.

---

### T3: Feature Extraction
**Infra:** AWS  
**Spec reference:** Section 3  
**Depends on:** T1 PASSED  
**Input:** Bar dataset from T1, raw MBO data (for LOB snapshots and cancel events)  
**Tests to write first:**
- Trade flow imbalance is in [-1, +1] for all bars
- BBO imbalance is in [0, 1] for all bars
- Depth imbalance is in [0, 1] for all bars
- Bar range is non-negative for all bars
- Body/range ratio is in [-1, +1] for all bars (0 when range=0)
- VWAP displacement is in [-1, +1] for all bars (0 when range=0)
- Volume is finite and positive for all bars
- Trailing realized vol uses exactly 20 bars of lookback (NaN for first 19 bars of each session — handle appropriately)
- Normalized session time is in [0, 1] and monotonically non-decreasing within a session
- Session age starts at 0 and saturates at 1.0 after 20 bars
- Cancel rate asymmetry is in [-1, +1]
- Mean spread is positive for all bars
- Z-score normalization: after normalization, mean ≈ 0 and std ≈ 1 over trailing window
- Clipping: no values outside [-5, +5] after normalization
- No NaN or Inf in the final feature matrix (after handling warm-up period)
- Lookback window assembly: Z_k correctly stacks h consecutive feature vectors

**Gate:** All tests pass. Feature distributions plotted and visually inspected (no pathological spikes, no constant features, no features that are >99% one value).  
**Artifacts:** Feature matrix (parquet or HDF5), normalization statistics, distribution plots.

---

### T4: Gambler's Ruin Validation
**Infra:** AWS  
**Spec reference:** Section 6.1  
**Depends on:** T2 PASSED  
**Tests to write first:**
- Synthetic random walk generator produces correct tick-level prices (no off-by-one, correct drift parameterization)
- At p=0.5, empirical P(upper hit) is within 2 SE of 0.3333 (n >= 10,000 bars)
- At p=0.505, empirical P(upper hit) is within 2 SE of ~0.388
- At p=0.510, empirical P(upper hit) is within 2 SE of ~0.445
- At p=0.490, empirical P(upper hit) is within 2 SE of ~0.280
- At p=0.485, empirical P(upper hit) is within 2 SE of ~0.232
- Analytic formula implementation matches known values (unit test the formula itself)

**Gate:** All 5 drift levels pass within 2 SE. If zero-drift passes but non-zero fails, investigate sign convention — do not proceed.  
**Artifacts:** Validation report with empirical vs analytic frequencies, standard errors, and pass/fail per drift level.

---

### T5: Regime-Switch Validation
**Infra:** AWS  
**Spec reference:** Section 6.1b  
**Depends on:** T2 PASSED, T3 PASSED  
**Tests to write first:**
- Timeout rate in low-vol segment is significantly higher than in high-vol segment (define "significantly" as >2x ratio)
- Mean time-to-resolution in low-vol > mean time-to-resolution in high-vol
- Feature distributions show a visible shift at the regime boundary (test: KS test on feature distributions from 500 bars before vs 500 bars after boundary, p < 0.01 for at least bar range and realized vol features)
- Label distribution (p⁺, p⁻, p⁰) differs between segments (chi-squared test, p < 0.01)
- Trailing normalization does not create a >500-bar dead zone at the boundary where normalized features are stale (inspect normalized realized vol — it should reflect the new regime within ~100 bars, not 2000)

**Gate:** All tests pass. Diagnostic plots of timeout rate, resolution time, and normalized features across the regime boundary.  
**Artifacts:** Regime-switch validation report with plots and statistical test results.

---

### T6: Supervised Diagnostic
**Infra:** RunPod (GPU)  
**Spec reference:** Section 6.2  
**Depends on:** T2 PASSED, T3 PASSED, T4 PASSED, T5 PASSED  
**Input:** Feature matrix from T3, label dataset from T2  
**Tests to write first:**
- Overfitting test: MLP achieves >95% train accuracy on a batch of 256 samples within 500 epochs. If this fails, stop — architecture or optimizer is broken.
- MLP validation accuracy exceeds majority-class baseline on held-out Q4 2023 data
- Random forest baseline on same features: compute and report (not a gate, but diagnostic)
- Class-balanced accuracy is reported (not just raw accuracy — matters if label distribution is skewed)
- Confusion matrix is logged: check that the model isn't just predicting the majority class

**Gate:** Overfitting test passes AND validation accuracy > majority-class baseline. If overfitting passes but validation fails, the signal may be weak — report this but proceed cautiously (the spec says "if it can't beat baseline, fix features first", so investigate feature quality before moving on).  
**Artifacts:** Trained MLP checkpoint (for architecture validation only — not used downstream), validation metrics, confusion matrix, comparison to random forest baseline.

---

### T7: Reward Accounting Unit Test
**Infra:** AWS  
**Spec reference:** Section 8, step 7  
**Depends on:** T6 PASSED  
**Input:** Bar dataset from T1, label dataset from T2  
**Tests to write first:**
- Select 5 specific bars from real data where barrier hit is known from T2 labels
- For each, manually compute the full reward sequence for a long entry: entry → hold → hold → ... → barrier hit → reward
- For each, manually compute the full reward sequence for a short entry (barrier directions reversed)
- For at least 1 bar, manually compute MTM reward at timeout (verify `/b` normalization)
- Verify transaction cost C is deducted exactly once per round-trip
- Verify position state transitions: flat → long → flat, flat → short → flat
- Verify unrealized PnL computation for both long and short positions at intermediate bars
- Verify action masking: when position != 0, only hold is valid

**Gate:** All hand-computed reward sequences match environment logic output exactly (to floating point precision).  
**Artifacts:** Test cases with hand computations documented (these become regression tests).

---

### T8: Environment Implementation
**Infra:** AWS  
**Spec reference:** Section 4  
**Depends on:** T7 PASSED  
**Input:** Bar dataset, label dataset, feature matrix  
**Tests to write first:**
- Gymnasium API compliance: env.reset() returns valid observation, env.step() returns (obs, reward, terminated, truncated, info)
- Observation shape is correct: (132,) = 13 features × 10 lookback + 2 position state
- Action space is Discrete(4) with correct masking via action_masks()
- Episode terminates at last bar of session
- Random agent runs for 100 episodes without crashing
- Random agent's mean episode reward is approximately -0.20 per trade (within ±0.10, given variance)
- Position state transitions are correct across a full episode
- Barrier exits fire on correct bars (cross-reference with T2 labels)
- Force-close at session end applies MTM correctly
- Action masking prevents entry actions when holding, prevents hold action when flat
- No reward leakage: sum of rewards in episode equals realized PnL minus costs

**Gate:** All tests pass. Random agent baseline reward is in the expected range from Section 4.6.  
**Artifacts:** Gymnasium environment class, integration test suite, random agent baseline metrics.

---

### T9: PPO Training
**Infra:** RunPod (GPU)  
**Spec reference:** Section 5  
**Depends on:** T8 PASSED  
**Input:** Environment from T8  
**Tests to write first (these are monitoring assertions, checked during training):**
- Entropy on flat-state steps starts near 1.1 and does not collapse below 0.3 in first 100 updates
- Value loss has a decreasing trend (moving average over 50 updates)
- Episode reward mean exceeds -0.20 (random baseline) within 500 updates
- Flat action rate stays in [10%, 90%] throughout training
- No NaN in losses or gradients at any point

**Training protocol:**
1. Run PPO with exact hyperparameters from Section 5.2
2. Train for a minimum of 1000 updates (or until diagnostics plateau)
3. Log all metrics from Section 5.3 every update
4. Checkpoint every 100 updates

**Gate:** Training completes without NaN. Episode reward exceeds random baseline. Entropy does not collapse. If any diagnostic is in the "red flag" column from Section 5.3, stop training, diagnose, and report before proceeding.  
**Artifacts:** Trained policy checkpoint, full training log (all metrics from Section 5.3), learning curves plotted.

---

### T10: Behavioral Inspection
**Infra:** RunPod (GPU)  
**Spec reference:** Section 6.3, in-sample  
**Depends on:** T9 PASSED  
**Input:** Trained policy from T9, full 2022-2023 bar dataset  
**Tests to write first:**
- Flat rate in low-vol months (Jun-Dec 2023, VIX ~15) > flat rate in high-vol months (Jan-Jun 2022, VIX ~25) — quantify the difference
- Entry direction has statistically significant correlation with trade flow imbalance (sign test or chi-squared, p < 0.05)
- Flat rate in last 2 bars of session > flat rate in mid-session (agent learns to avoid late entries)
- After 3 consecutive losses, flat rate on next opportunity > overall flat rate (agent learns caution after streaks)
- Trade count per session is reasonable (not 0, not every bar — between 2 and 30 per session on average)

**Gate:** At least 3 of the 5 behavioral checks pass. If fewer than 3 pass, the agent has not learned meaningful structure — report findings but do not proceed to hyperparameter sweep (return to T9 with adjusted hyperparameters or to T6 to investigate feature signal).  
**Artifacts:** Behavioral analysis report with per-check statistics, entry heatmap by session time, flat rate by volatility regime plot.

---

### T11: Hyperparameter Sweep
**Infra:** AWS for Priority 1-3 (supervised diagnostic + label analysis), RunPod for Priority 4-6 (RL training)  
**Spec reference:** Section 7  
**Depends on:** T10 PASSED  
**Input:** Full pipeline from T1-T8  

**Sweep protocol:**
1. **Priority 1 (N):** Run supervised diagnostic (T6 protocol) at N ∈ {200, 500, 1000, 2000}. Report classifier accuracy per N. Select best N. *[AWS for label/feature recomputation, RunPod for MLP training]*
2. **Priority 2 (a,b):** At best N, recompute labels for (a,b) ∈ {(10,5), (20,10), (40,20)}. Report timeout rate and tiebreak rate. Eliminate any (a,b) with tiebreak > 5% or timeout > 30%. Run supervised diagnostic on survivors. *[AWS]*
3. **Priority 3 (T_max):** Calibrate T_max per Section 2.4 for each surviving (N, a, b) combination. *[AWS]*
4. **Priority 4 (h):** Run PPO at h ∈ {5, 10, 20} with best (N, a, b, T_max). Compare episode reward convergence. *[RunPod]*
5. **Priority 5 (LR):** Run PPO at LR ∈ {3e-4, 1e-4, 5e-5} with best settings from above. *[RunPod]*
6. **Priority 6 (Entropy):** Run PPO at entropy coeff ∈ {0.001, 0.01, 0.05} with best settings from above. *[RunPod]*

**Parallelism within sweep:** Priority 1 runs all 4 N values in parallel. Priority 2 runs all 3 (a,b) values in parallel. Priority 4-6 each run all values in parallel. Priorities themselves are sequential (each uses the winner from the previous).

**Gate:** Best configuration identified. Report full sweep results as a table.  
**Artifacts:** Sweep results table, best hyperparameter configuration, all training logs.

---

### T12: Out-of-Sample Evaluation
**Infra:** RunPod (GPU)  
**Spec reference:** Section 6.3, out-of-sample  
**Depends on:** T11 PASSED  
**Input:** Best policy from T11, held-out data (Q4 2023 or early 2024)  

**Tests:**
- Compute all metrics from Section 6.3 evaluation table
- Compare against minimum viability thresholds:
  - Win rate > 36%
  - Profit factor > 1.0
  - Per-trade Sharpe > 0.05
  - Flat rate between 20% and 80%

**Gate:** Report all metrics. Passing the minimum viability thresholds is not a gate for completion — the project is complete either way. The thresholds determine whether the result is "agent learned something" vs "agent did not learn". Report honestly.  
**Artifacts:** Final evaluation report with all metrics, equity curve, trade log, per-session P&L breakdown.

---

## Reporting

After each task completes (pass or fail), report:

```
Task: T{N}
Status: PASSED | FAILED
Infra: AWS | RunPod
Duration: {wall clock time}
Tests: {passed}/{total}
Gate: {met | not met — reason}
Artifacts: {list of output files with paths}
Notes: {anything unexpected, diagnostic findings, or recommendations}
```

After all tasks complete, produce a final summary:

```
Pipeline Status: COMPLETE | BLOCKED at T{N}
Best Configuration: {N, a, b, T_max, h, LR, entropy_coeff}
Out-of-Sample Results: {win_rate, profit_factor, sharpe, flat_rate, trades_per_session, max_drawdown}
Viability: VIABLE | NOT VIABLE (per Section 6.3 thresholds)
Total Wall Clock: {time}
Total Compute Cost: {AWS: $X, RunPod: $Y}
```

---

## Spec Document

The full specification is attached as `ppo_barrier_hit_agent_mes_spec_v1.md`. It is the single source of truth for all implementation details, parameter values, formulas, and pass/fail criteria. If this orchestrator directive and the spec conflict, the spec wins.
