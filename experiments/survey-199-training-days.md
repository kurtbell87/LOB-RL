# Survey: Does increasing training data from 20 to 199 days fix OOS generalization?

## Prior Internal Experiments

Seven experiments have been run (pre-001 through pre-007), all documented in `RESEARCH_LOG.md`. **No experiment has ever used more than 20 training days.** This is the fundamental gap.

| Experiment | Architecture | Steps | Train Days | Val Return | Test Return | Key Takeaway |
|------------|-------------|-------|------------|------------|-------------|--------------|
| pre-001 | MLP 256x256 | 2M | 20 (chrono) | -53.8 | -36.6 | First OOS eval — massive overfit |
| pre-003 | MLP 256x256 | 2M | 20 (shuffle) | -51.5 | -62.5 | Shuffle-split doesn't fix OOS |
| pre-004 | MLP+FrameStack4 | 2M | 20 (shuffle) | -48.4 | -50.2 | Frame-stack marginally better |
| pre-005 | LSTM | 5M | 20 (shuffle) | -36.7 | -33.4 | **Best OOS** but still negative |
| pre-006 | MLP 256x256 | 5M | 20 (shuffle) | -62.9 | -44.0 | More steps made val **worse** |
| pre-007 | MLP+FrameStack4 | 5M | 20 (shuffle) | -82.3 | -49.4 | More steps made val **much worse** |

**Critical pattern:** More steps on 20 days made MLP and frame-stack *worse* (pre-003->pre-006, pre-004->pre-007). Classic overfitting to a small training set. The model memorizes day-specific patterns that don't transfer.

**Training diagnostics at 5M steps (20 days):**
- `explained_variance > 0.95` for all architectures — value function has memorized training returns
- `approx_kl = 0.148` for LSTM (vs 0.019 MLP) — policy updating aggressively, LR may be too high
- `clip_fraction = 0.311` for LSTM — correlated with high KL
- All entropy declining monotonically but above -0.60 (ent_coef=0.05 prevents collapse)

## Existing Experiment Spec (exp-001)

**An experiment spec already exists:** `experiments/exp-001-does-increasing-training-data-from-20-to.md`. It was designed by a prior FRAME agent and includes:
- 5 runs: MVE, Run A (MLP 20d), Run B (LSTM 20d), Run C (MLP 199d), Run D (LSTM 199d), Run E (LSTM 199d seed 43, conditional)
- Pre-committed success criteria (SC-1 through SC-5) with clear CONFIRMED/REFUTED/INCONCLUSIVE interpretation
- Abort criteria for FPS, entropy collapse, NaN, and control reproduction
- Resource budget: ~$12.47 GPU cost, 8 GPU-hour budget

**Partial execution has occurred locally (not on RunPod):**
- **MVE (MLP 199d, 500K steps):** Completed. Model saved (`ppo_lob.zip`, `vec_normalize.pkl`).
- **Run A (MLP 20d, 5M steps):** Completed. Model saved.
- **Run B (LSTM 20d, 5M steps):** Started but interrupted. Only empty checkpoints and TB log dir exist.
- Runs C, D, E: Not started.
- **No `metrics.json` exists.** The run script (`scripts/run_exp001.py`) was never completed.
- **Runs were local (Apple M2 Max CPU)**, not on RunPod RTX 4090 as specified. LSTM at ~422 FPS locally vs 228 FPS on GPU — local LSTM is actually faster but the prior session killed it at 15% because it was too slow for a 5M run.

**The experiment needs to be restarted on RunPod** for the LSTM runs to complete in reasonable time (6.1 hrs on GPU vs potentially 3+ hrs on M2 for 5M steps MLP, much longer for LSTM).

## Current Infrastructure

### Data Pipeline
- **249 cached `.npz` files** in `cache/mes/` (~300 MB lazy-loaded, 18 GB on RunPod volume)
- Each file = one MES trading day from 2022 (bear market, S&P 500 down ~20%)
- Cache is ready — no rebuild needed
- RunPod volume `4w2m8hek66` has cache pre-uploaded and verified

### Split Mechanism
`train.py:251-256`: `train_files = all_files[:train_days]`, `val = [train_days:train_days+5]`, `test = [train_days+5:]`

With `--shuffle-split --seed 42`:
- `--train-days 20`: 20 train / 5 val / 224 test (8% / 2% / 90% — extremely unusual split)
- `--train-days 199`: 199 train / 5 val / 45 test (80% / 2% / 18% — standard split)

**The `--train-days` flag already exists and is trivially configurable. No code changes needed.**

### Training Pipeline
- `SubprocVecEnv` with 8 parallel workers, each running `MultiDayEnv`
- `MultiDayEnv` lazy-loads `.npz` on `reset()` — memory-safe even with 199 days (one file in memory per worker at a time)
- `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)` wraps the environment
- LSTM via `RecurrentPPO('MlpLstmPolicy')` from sb3-contrib
- Checkpointing at configurable intervals (`--checkpoint-freq`)
- Evaluation via `evaluate_sortino()` at training end: deterministic rollouts, computes mean/std return, Sortino, positive episodes

### Compute
- RunPod RTX 4090 ($0.59/hr, 24 GB VRAM), region US-NC-1
- LSTM at 228 FPS on GPU -> 5M steps = ~6.1 hrs = ~$3.60
- MLP at 1,275 FPS on GPU -> 5M steps = ~1.1 hrs = ~$0.65
- RunPod dispatch protocol exists in `.claude/prompts/run.md`

### Execution Script
`scripts/run_exp001.py` exists — a standalone Python script that orchestrates all 5 runs sequentially, parses output, reads TensorBoard metrics, checks abort criteria, and writes `metrics.json`. This bypasses the `experiment.sh` pipeline and runs locally. **For the RUN agent, this script can be adapted or the runs can be dispatched individually to RunPod.**

## Known Failure Modes

1. **More steps = more overfitting (on 20 days).** Empirically established. With 199 days, each day is visited ~16 times (vs ~160 times on 20 days). This should reduce memorization, but the model may also need more total steps to converge on the larger dataset.

2. **VecNormalize cross-day leakage (unverified).** Running statistics are computed across all training days. With 199 days, normalization stats will be more representative of the full distribution. This is a potential confound — improvement could come from better normalization, not better learning. P1 question in QUESTIONS.md.

3. **LSTM high KL divergence.** `approx_kl = 0.148` and `clip_fraction = 0.311` on 20 days at 5M steps. With 199 days, gradient diversity increases, which may stabilize or destabilize training. Watch this metric.

4. **Val/test composition changes.** With `--seed 42 --shuffle-split`, changing `--train-days` from 20 to 199 means 179 days move from test->train. **The 20-day and 199-day experiments have completely different val and test sets.** Magnitude comparisons are noisy. Focus on the sign (positive vs negative) not exact point differences.

5. **Test set shrinkage.** 199-day test has only 45 days (vs 224 for 20-day). Higher variance on the test mean. Report per-episode returns and standard error.

6. **Val set is always 5 days.** Very noisy — a single outlier day can dominate the mean.

7. **Underfitting risk.** At 5M steps with 199 days, each day is visited ~16 times. This is 10x fewer passes per day. The model may not converge. Monitor in-sample return — if it's much lower than the 20-day run, consider 10M steps.

8. **RunPod auto-fetch failures.** Prior GPU experiments had SSH-based fetch fail. Pipeline now uses S3 sync — should work but hasn't been battle-tested with the `experiment.sh` pipeline.

9. **Local run interruption.** The prior attempt ran locally and was interrupted during Run B (LSTM). The partial results (MVE, Run A) used local hardware, not RunPod. These should be discarded for infrastructure consistency — re-run everything on RunPod.

## Key Codebase Entry Points

| File | Line(s) | Role |
|------|---------|------|
| `scripts/train.py:185` | — | `--train-days` argument (default 20) |
| `scripts/train.py:251-256` | — | Shuffle-split logic (train/val/test slicing) |
| `scripts/train.py:268-281` | — | `SubprocVecEnv` creation with `cache_files` |
| `scripts/train.py:86-178` | — | `evaluate_sortino()` — OOS evaluation |
| `scripts/train.py:336-352` | — | `RecurrentPPO` model creation |
| `scripts/train.py:371-403` | — | Checkpoint callback setup |
| `python/lob_rl/multi_day_env.py:35-121` | — | `MultiDayEnv` — lazy-loads `.npz` per `reset()` |
| `python/lob_rl/bar_level_env.py` | — | `BarLevelEnv` — 21-dim bar-level obs |
| `python/lob_rl/_reward.py` | — | Shared reward: PnL delta + execution cost |
| `scripts/run_exp001.py` | — | Standalone experiment executor (sequential, local) |
| `runpod/launch.sh` | — | Pod launcher — passes `--cache-dir` and args |
| `runpod/fetch-results.sh` | — | S3-based result download (no running pod needed) |
| `.claude/prompts/run.md` | — | RUN agent prompt with RunPod dispatch protocol |

## Architectural Priors

Per DOMAIN_PRIORS.md:

- **LSTM is the default architecture.** Least overfit in all prior experiments (val -36.7 vs MLP -62.9, frame-stack -82.3). Advantage may be implicit regularization rather than temporal learning.
- **MLP 256x256 ReLU is the baseline.** 5.6x faster to train. Useful as a comparison point.
- **Frame-stacking is abandoned.** Actively hurts generalization.
- **Best hyperparameters are fixed:** `bar_size=1000, ent_coef=0.05, lr=1e-3, policy_arch=256,256, activation=relu, shuffle-split, seed=42`.

For the data quantity question specifically:
- The 21-dim observation space is low-dimensional. An MLP can easily memorize 20 days. The capacity is not the bottleneck — data volume is.
- With 199 training days and 8 SubprocVecEnv workers, each worker sees ~25 days per epoch. At 5M steps with ~200 bars/day average, each day is visited ~16 times. On 20 days, each day was visited ~160 times — a **10x reduction in repetition per day**.
- The 8/2/90 train/val/test split is extremely unusual in ML. Moving to 80/2/18 brings the split closer to standard practice.

## External Context

The data quantity hypothesis is well-supported by general ML practice:
- **Standard splits:** 70-80% train / 10-15% val / 10-15% test. The current 8/2/90 split is extreme.
- **RL-specific:** RL is notoriously sample-hungry. DRL for finance literature consistently emphasizes diverse training environments. 20 days is akin to training an image classifier on 20 images.
- **Bear market 2022:** All 249 days are from a single year. Even with 199 days, the agent only sees one market regime. This limits the generalization we can expect, but 199 days should capture intra-year variation (trend, consolidation, volatility spikes, FOMC events, etc.).
- **Expected outcome from literature:** Increasing training data typically improves generalization when the model is overfitting. However, if the observation space lacks predictive signal (P3 question), no amount of data will help. This experiment disambiguates between "too little data" and "no signal."
- **Execution cost as a binding constraint:** Per-bar price moves in MES are often < 1 tick ($1.25). The round-trip cost is $2.50 (2 ticks). The agent needs signals worth > 2 ticks to profit. Even with perfect generalization, if the signal-to-cost ratio is too low, OOS will remain negative. This is why P0-b (execution cost ablation) is equally important.

## Constraints and Considerations

1. **Compute cost is modest.** Full protocol (5 runs + conditional 6th) is ~$12.47 on RunPod. Even with 20% overhead, well within budget.

2. **Wall time.** LSTM 5M steps = ~6.1 hrs on RTX 4090. With parallel pod dispatch (A+B, then C+D), critical path is ~6.3 hrs. Feasible within a single day.

3. **The existing spec is well-designed.** `exp-001` has pre-committed success criteria, abort conditions, confound analysis, and resource budget. It does not need redesign — it needs execution.

4. **Partial local results should be discarded.** The MVE and Run A completed on Apple M2 Max (CPU). For infrastructure consistency, all runs should be on the same hardware (RunPod RTX 4090). The partial results are useful only as a rough sanity check.

5. **`--eval-only` flag does not exist.** The eval command in the system prompt (`--eval-only`) is not implemented in `train.py`. Evaluation happens automatically after training completes.

6. **Seed sensitivity.** A single seed could produce a lucky or unlucky partition. Run E (seed 43) is designed to bound this risk. If seed 42 and 43 produce results within ±20 of each other, the finding is robust.

7. **Underfitting vs overfitting tradeoff.** With 10x more data at the same step count, the model may underfit rather than overfit. Key diagnostic: if in-sample return is much lower than 20-day run AND OOS doesn't improve, the model needs more total steps (10M+).

## Recommendation

The FRAME agent should **not redesign this experiment** — the existing spec (`exp-001`) is thorough, well-controlled, and ready to execute. The FRAME agent should either:

1. **Adopt the existing spec as-is**, or
2. **Make minimal updates** only if needed (e.g., confirm `compute: runpod` is declared, ensure the spec format matches the current template).

**Focus areas for the experiment:**

1. **Execute on RunPod, not locally.** The prior attempt ran locally and was interrupted. LSTM needs GPU for reasonable wall time.

2. **Primary question:** Do 199 training days produce positive (or meaningfully less negative) OOS returns? The pre-committed gate from QUESTIONS.md: "If positive OOS: scale up. If still negative: problem is not data quantity."

3. **Key diagnostics to watch:**
   - Explained variance trajectory (should be < 0.95 if overfitting is reduced)
   - Entropy trajectory (should stay higher with more diverse data)
   - `approx_kl` for LSTM (stability check with 199 days)
   - In-sample return (if much lower → underfitting → need more steps)
   - Checkpoint comparison: 3M, 4M, 5M for late-stage overfitting detection

4. **If REFUTED:** The problem is not data quantity. Pivot to P0-b (execution cost ablation) to test whether the agent learns signal that's masked by spread cost. If that also fails, escalate to P3 (observation signal audit via supervised classifier).

5. **If CONFIRMED:** Scale up. Consider multi-year data, LSTM with reduced LR (for KL stability), and reward shaping experiments on the larger training set.
