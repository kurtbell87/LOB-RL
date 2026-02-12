# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Conditional signal detection or accept null and pivot.**

exp-008 REFUTED bar-size sweep: B ∈ {200,500,1000,2000} × R ∈ {1x,2x,3x} — best BSS = 0.0023 (B=200/R=4/long), 2.3× below the 0.005 meaningful threshold. 20/24 cells negative. BSS degrades monotonically with bar size. Phase 2b is complete.

**Three axes now exhausted:**
- **Architecture:** LR, GBT, MLP, RF, LSTM, Transformer — all fail (exp-006, exp-007)
- **Scale:** B ∈ {200,500,1000,2000} × R ∈ {1x,2x,3x} — all fail (exp-008)
- **Data quantity:** 20d vs 199d — no effect (exp-001)

**Remaining options:**

1. **Conditional signal detection (regime-filtered):** Stratify by realized_vol quartile or trend/mean-reversion regime. Run LR on each subset at B=500/R=6. Signal may exist conditionally but average to zero unconditionally. This is the cheapest experiment that could reveal hidden signal.

2. **Alternative target formulation:** Test 1:1 symmetric barriers (ȳ ≈ 1/2 instead of 1/3). The 2:1 asymmetry makes the constant predictor hard to beat. Symmetric barriers expose different feature-outcome relationships.

3. **Feature pivot — external context:** Add session-level features (time of day, daily vol, overnight gap, macro regime). The 22 current features are all intra-bar/lookback. Conditioning information may be missing.

4. **Accept null and pivot to execution-timing RL:** If directional signal is absent, an RL agent can potentially learn execution-quality policies (when to enter/exit based on spread dynamics) without directional prediction.

**Research plan:** `experiments/Asymmetric First-Passage Trading.md`

**Label formulation (IMPORTANT):**
- Y_long=1: price hits entry + 2R before entry - R (reward:risk = 2:1)
- Y_short=1: price hits entry - 2R before entry + R
- These are NOT disjoint — both can be 0 (both stopped), or under certain conditions both 1
- Under martingale null with 2:1 asymmetry: E[Y] ≈ 1/3 (from gambler's ruin)
- The plan explicitly models these as two independent binary predictions, NOT one three-class problem

**Training script ready:** `scripts/train_barrier.py` with CLI args, Section 5.2 hyperparameters, MaskablePPO with net_arch=[256,256,dict(pi=[64],vf=[64])].

### Roll calendar

`data/mes/roll_calendar.json` maps each date to the front-month instrument_id. `precompute_cache.py --roll-calendar` uses this to filter each day's `.dbn.zst` to exactly ONE contract (no spreads, no back months).

| Period | Contract | Instrument ID |
|--------|----------|--------------|
| Jan 1 – Mar 11 | MESH2 | 11355 |
| Mar 12 – Jun 10 | MESM2 | 13615 |
| Jun 11 – Sep 9 | MESU2 | 10039 |
| Sep 10 – Dec 31 | MESZ2 | 10299 |

Source: `data/symbology.json` from Databento download. Roll dates are ~1 week before CME 3rd-Friday expiry.

### What was just completed

**exp-008 bar-size sweep REFUTED (2026-02-11).** Swept 4 bar sizes × 3 risk calibrations = 12 (B, R) configurations (24 cells with both labels). No cell achieves BSS ≥ 0.005. Best: B=200/R=4/long BSS = 0.0023 (p=0.0 Bonferroni) — statistically significant due to 1.15M samples but practically negligible (0.23% of variance). 20/24 cells negative. BSS degrades monotonically with bar size. All 12 null calibration gates passed. Wall time 143.5 min. Scripts at `scripts/run_exp008_bar_size_sweep.py`, results at `results/exp-008-bar-size-sweep/`.

**exp-007 sequence model signal detection REFUTED (2026-02-11).** LSTM and Transformer on full-session bar sequences (22 features per bar, ~1000-4400 bars per session) both fail to beat constant Brier. All 8 (model, label, split) BSS values negative. Best: transformer/short BSS = -0.0004 (p=0.791). LSTM is *worse* than flat LR (BSS -0.0173 vs -0.0003). Transformer collapses to near-constant predictions (p̂_std=0.014). Key finding: temporal ordering contains no additional calibrated signal beyond flat features. MPS-accelerated locally (~26 min total). PR #49.

**Per-bar feature cache DONE (2026-02-11).** C++ `barrier_precompute()` now outputs `bar_features` (n_trimmed, 22) per-bar z-score normalized features. `load_session_features()` loader in `first_passage_analysis.py`. Barrier cache rebuilt (248 sessions, 221 MB). PR #47.

**Sequence model infrastructure DONE (2026-02-11).** `sequence_models.py` with `BarrierLSTM`, `BarrierTransformer`, swappable `BarEmbedding` interface, causal masking, masked BCEWithLogits loss, AdamW+cosine, early stopping, Brier evaluation. 49 tests. PR #48.

**exp-006 signal detection REFUTED (2026-02-11).** LR and GBT both fail to beat constant Brier on Y_long/Y_short. All 4 (model, label) pairs have negative BSS. Best: logistic/short BSS = -0.0003. GBT is *worse* than LR (overfits). Key finding: T6's +5pp accuracy does not translate to calibrated probabilities — models are well-calibrated near ȳ≈0.32 but miscalibrated at tails. Information content < 0.1% of outcome variance. PR #46.

**exp-005 null calibration CONFIRMED (2026-02-11).** ȳ_long = 0.320, ȳ_short = 0.322, both ≈ 1/3 (gambler's ruin prediction). ȳ_long + ȳ_short = 0.642 ≈ 2/3 (non-complementarity confirmed). Temporal stability: 20-session rolling mean within [0.27, 0.38]. Joint distribution: {both_stop: 52.1%, long_only: 16.7%, short_only: 17.5%, both_win: 13.7%}. PR #45.

**exp-004 quick diagnostic (2026-02-11).** 22-feature vs 9-feature signal detection on 50K subsample. RF results:
- Set A (all 22 features): 49.6% balanced accuracy (2 seeds)
- Set B (original 9 features): 47.5% balanced accuracy (2 seeds)
- Delta: +2.1pp (A > B). Both far above majority baseline (38.1%).
- Set B jumped from T6's 40.5% → 47.5% because cols 0,1,2,11 (previously dead) are now active in C++ cache.
- Full experiment aborted (Python process killed by SIGPIPE after parent experiment.sh was terminated). Quick results are directionally useful but not statistically rigorous.
- Metrics at `results/exp-004-22-feature-supervised-diagnostic/metrics.json` (quick tier only).

**Asymmetric First-Passage Trading plan written (2026-02-11).** Research plan at `experiments/Asymmetric First-Passage Trading.md`. 5 phases: data pipeline → null calibration → signal detection → feature engineering → sequence model. Grounded in probability theory (martingale null, Brier decomposition, information ceiling bounds).

**Barrier cache re-precomputed (2026-02-10).** 248 sessions processed via C++ backend in 570s (~9.5 min) with 8 workers. 64 skipped (insufficient data, mostly weekends/holidays). 461,108 total bars, 454,164 usable rows. 186 MB cache at `cache/barrier/`. Zero errors. N_FEATURES=22, all 22 features active.

**C++ Barrier Precompute Pipeline — 5 TDD Cycles (2026-02-10).** Moved the entire barrier precompute pipeline from Python to C++ for ~50x speedup:

- **Cycle 1 (PR #37):** Book Extensions. Added `total_bid_depth(n)`, `total_ask_depth(n)`, `weighted_mid()`, `vamp(n)` to existing `Book` class.
- **Cycle 2 (PR #38):** BarBuilder. Single-pass MBO stream → `TradeBar` + `BarBookAccum` vectors. `DbnFileSource` 'R' action mapped to Cancel. 51 tests.
- **Cycle 3 (PR #39):** Feature Computation. `compute_bar_features()` (22 features), `normalize_features()` (z-score, [-5,5] clip), `assemble_lookback()` (sliding window). O(n) realized vol. 76 tests.
- **Cycle 4 (PR #40):** Label Computation. `compute_labels()` — triple-barrier with intrabar tiebreaking via trade sequence scan. 36 tests.
- **Cycle 5 (PR #41):** Pipeline Integration + pybind11. `barrier_precompute()` wires all components end-to-end. `lob_rl_core.barrier_precompute()` Python binding returns dict compatible with `np.savez_compressed()`. `precompute_barrier_cache.py` updated to use C++ backend. 23 C++ + 35 Python tests.

Total: 217 new C++ tests (635 total), 8 new Python binding tests (2070 total).

**Microstructure Features — 3 TDD Cycles (2026-02-10).** Expanded barrier feature set from 13 to 22 features:

- **Cycle 1 (PR #34):** LOB Reconstructor + Fix Dead Features. Created `OrderBook` class in `lob_reconstructor.py`, added `extract_all_mbo()` to `bar_pipeline.py`, wired MBO data through `_compute_book_features()` to activate the 4 dead book features (BBO imbalance, depth imbalance, cancel asymmetry, mean spread). Introduced `N_FEATURES` constant. Added `n_features` version check to precompute cache. 67 tests.
- **Cycle 2 (PR #35):** Phase 1 Microstructure Features (cols 13-16). OFI (order flow imbalance), multi-level depth ratio, weighted mid-price displacement, spread dynamics (std). Expanded `_compute_book_features` from (n,4) to (n,8). `N_FEATURES` 13→17. 55 tests.
- **Cycle 3 (PR #36):** Phase 2 Microstructure Features (cols 17-21). VAMP displacement, aggressor imbalance, trade arrival rate, cancel-to-trade ratio, price impact per trade. Added `OrderBook.vamp(n)` method. Expanded `_compute_book_features` from (n,8) to (n,13). `N_FEATURES` 17→22. Refactored: `_BOOK_COL_MAP` for column assignment, `stride_tricks` in `assemble_lookback`, extracted MBO helpers to `conftest.py`. 78 tests.

Total: 200 new tests across 3 test files. 2062 Python tests pass (was 1858).

**T6 Supervised Diagnostic run (2026-02-10).** Result: **CONFIRMED — weak signal.**

v1 (long-only, misleading): MLP 61.6% vs 67.4% baseline on imbalanced 67/33/0% classes. Appeared to show no signal. This framing was wrong — it tests gambler's ruin, not directional predictability.

v2 (bidirectional, correct): Classifies {long_profitable, short_profitable, flat}. Key insight: τ_{+} and τ_{-} are mutually exclusive (long profit at +20 precludes short profit; short stops at +10 first). Distribution is balanced (33/33/35%).

| Model | Shuffle Acc | Chrono Acc | Baseline | Signal |
|-------|-----------|-----------|----------|--------|
| MLP [256,256] | 39.3% | 39.1% | 34.5% | +4.8pp |
| RF (100 trees) | 40.5% | 39.6% | 34.5% | +5.9pp |

- Overfit test PASSED (100% on 256 samples)
- Signal is weak but real, consistent across both splits
- Massive overfitting gap: 90% train → 39% test (regularization matters)
- RF slightly outperforms MLP (architecture matters)
- 4/13 features dead (book features) — activating these is P1
- Top features: trade flow imbalance (0.128), bar range, volume, vwap displacement
- New files: `scripts/run_barrier_diagnostic.py` (v1), `scripts/run_barrier_diagnostic_v2.py` (v2), `cache/t6_diagnostic_results.json`, `cache/t6_diagnostic_v2_results.json`
- Updated: RESEARCH_LOG.md, QUESTIONS.md, DOMAIN_PRIORS.md, CLAUDE.md, LAST_TOUCH.md

**PPO Barrier-Hit Agent pipeline T1-T9b complete (2026-02-09).** All data pipeline + environment + training infra tasks completed via strict TDD (`./tdd.sh`):
- **T1: Bar Construction Pipeline** — PR #20. `TradeBar`, `build_bars_from_trades()`, RTH filtering, dataset builder. 59 tests.
- **T2: Label Construction Pipeline** — PR #21. `BarrierLabel`, `compute_labels()`, intrabar tiebreaking, T_max calibration. 65 tests.
- **T3: Feature Extraction** — PR #22. 13 bar-level features, z-score normalization, lookback assembly. 92 tests.
- **T4: Gambler's Ruin Validation** — PR #23. Analytic formula validation at 5 drift levels (p=0.485..0.510). 81 tests.
- **T5: Regime-Switch Validation** — PR #24. Low-vol/high-vol synthetic regime switch, KS tests, chi-squared, normalization adaptation. 51 tests.
- **T6: Supervised Diagnostic** — PR #25. MLP classifier + random forest baseline for barrier label prediction. `_train_loop()` extracted, vectorized `compute_segment_stats()`, cached RTH boundaries. 56 tests.
- **T7: Reward Accounting** — PR #26. `RewardConfig`, `PositionState`, `compute_entry()`, `compute_hold_reward()`, `compute_unrealized_pnl()`, `compute_reward_sequence()`, `get_action_mask()`. Hand-computed reward sequences for all exit types. 46 tests.
- **T8: Barrier Environment** — PR #27. `BarrierEnv(gymnasium.Env)` — 132-dim obs, Discrete(4) action space, action masking, force-close at session end. Refactor: extracted `compute_mtm_reward()` and `classify_exit()` into reward_accounting.py. 41 tests.
- **T9: PPO Training Infrastructure** — PR #28. `MultiSessionBarrierEnv` multi-session wrapper, `barrier_vec_env` SB3 helpers, `BarrierDiagnosticCallback` with Section 5.3 metrics, `_sb3_compat` shim, `linear_schedule`. 41 tests.
- **T9b: Training Script** — PR #29. `scripts/train_barrier.py` CLI entry point with `parse_args()`, `split_sessions()`, `build_model()`, `main()`. Section 5.2 hyperparameters. 22 tests.

All code in `python/lob_rl/barrier/` + `scripts/` with tests in `python/tests/barrier/`. 1858 Python tests total (1308 core + 550 barrier).

**Prior: AWS EC2 Spot migration (2026-02-09).** Replaces RunPod with AWS EC2 Spot for all remote training. Six new files in `aws/`, five existing files modified (~700 lines total):

- **`aws/setup.sh`:** One-time infrastructure setup — S3 bucket, ECR repo, IAM role (least-privilege: S3 read/write, ECR pull, self-terminate), security group. Idempotent. Prints env var exports.
- **`aws/launch.sh`:** Launch EC2 Spot instance. Drop-in replacement for `runpod/launch.sh`. Auto-detects instance type: `--recurrent` → `g5.xlarge` (GPU, A10G, ~$0.24/hr), else → `c7a.4xlarge` (CPU, 16 vCPU, ~$0.39/hr). Generates user-data script that: starts spot interruption monitor, downloads cache from S3, pulls Docker from ECR, runs training, uploads results to S3, self-terminates on success.
- **`aws/fetch-results.sh`:** Download results from S3. Simpler than RunPod version (no `--profile runpod --endpoint-url`).
- **`aws/upload-cache.sh`:** One-time `aws s3 sync cache/mes/` to S3 bucket.
- **`aws/monitor.sh`:** Discovers lob-rl instances via Project tag, polls every 5 min, auto-fetches when terminated/stopped.
- **`aws/README.md`:** Full setup guide, env vars, instance types, S3 layout, cost comparison, spot interruption handling.
- **`scripts/start.sh`:** Provider-agnostic `RUN_ID` env var (AWS passes instance ID, RunPod falls back to `RUNPOD_POD_ID`).
- **`.dockerignore`:** Added `aws/` exclusion.
- **`experiment.sh`:** Added `AWS_S3_BUCKET`, `AWS_ECR_REPO`, `AWS_INSTANCE_TYPE`, `AWS_REGION` config vars. Exports in `run_run()`. Added to RUN agent system prompt context. Updated help text. RunPod vars marked deprecated.
- **`.claude/prompts/run.md`:** Added full AWS Dispatch Protocol (~80 lines): env validation, instance launching, parallel instances, polling, fetching, spot interruption handling, local eval, cleanup, error handling. Existing RunPod protocol marked deprecated.
- **`.claude/prompts/frame.md`:** Changed `compute: runpod` → `compute: aws`. Instance type field: `g5.xlarge` (GPU) or `c7a.4xlarge` (CPU). Quality standard: remote experiments MUST use `compute: aws`.

**Cost savings:** g5.xlarge spot ~$0.24/hr vs RunPod RTX 4090 $0.59/hr (60% cheaper GPU). c7a.4xlarge ~$0.39/hr for CPU-only MLP workloads (no GPU idle cost).

**Prior: RunPod compute backend for experiment.sh (2026-02-09).** Three files changed (~120 lines):

- **`experiment.sh`:** Added `RUNPOD_VOLUME_ID`, `DOCKERHUB_USER`, `RUNPOD_GPU_TYPE` config vars. Exports them in `run_run()`. Passes RunPod context (volume ID, Docker user, GPU type, script paths) to the RUN agent's system prompt. Updated help text.
- **`.claude/prompts/frame.md`:** Added `## Compute Target` section to the spec template (between Resource Budget and Abort Criteria). FRAME agent now declares `compute: local|runpod` and GPU type. Added planning step for compute target decision. Added quality standard: LSTM experiments MUST use `compute: runpod`.
- **`.claude/prompts/run.md`:** Added full `## RunPod Dispatch Protocol` section (~80 lines) covering: env var validation, pod launching (`EXP_NAME=<label> ./runpod/launch.sh <args>`), parallel pod management, polling (300s intervals), fetching (`./runpod/fetch-results.sh`), merging into experiment results, local eval for val/test metrics, cleanup (`runpodctl remove pod`), and error handling. Updated process steps for RunPod flow.

**Prior: Installed claude-research-kit (2026-02-09).** Two-tower architecture: TDD for engineering (`./tdd.sh`), Research for experiments (`./experiment.sh`). Merged pre-tool-use hook dispatches on `TDD_PHASE` vs `EXP_PHASE`. Populated `QUESTIONS.md` (6 open + 4 answered), `DOMAIN_PRIORS.md`, `RESEARCH_LOG.md` (7 pre-experiments backfilled). Updated CLAUDE.md with research workflow section and "when to use which" guide.

**Prior: Removed SSH dependency from RunPod infrastructure (2026-02-09).** All three GPU experiments had their auto-fetch fail because `monitor.sh` and `fetch-results.sh` relied on SSH to running pods. Fixed the entire flow:

- **`start.sh`:** Exits 0 on training success (pod auto-stops, billing stops). Only keeps container alive via `sleep infinity` on failure for SSH debugging.
- **`fetch-results.sh`:** Rewritten to use `aws s3 sync` from the network volume. Takes run dir name (e.g., `lstm_abc123`) not pod ID. No running pod required.
- **`monitor.sh`:** Detects completion via `runpodctl get pod` status (EXITED/not found) instead of SSH. Fetches via S3. Verifies `train.log` + `ppo_lob.zip` exist before removing pod.
- **`launch.sh`:** Removed `--ports "22/tcp"` and `--startSSH`. Updated printed instructions for S3-based fetch.
- **`runpod/README.md`:** Updated for new auto-stop + S3 fetch flow.
- **Dockerfile unchanged** — `openssh-server`+`rsync` kept for debugging failed pods.

**Prior: GPU experiment analysis completed (2026-02-09).** All three 5M-step experiments finished on RunPod. Results were stranded on the persistent volume (auto-fetch failed due to SSH issues). Recovery: launched a recovery pod, rsync'd results locally. Report: `research/experiment_report.md`.

Key finding: **More training steps made things worse, not better.** MLP val went from -51.5 (2M) to -62.9 (5M). Frame-stack val went from -48.4 to -82.3. The agent is memorizing the 20 training days. LSTM is least overfit but still negative. The problem is likely the 8%/2%/90% train/val/test split — 20 training days is far too few.

**Prior: Launched 3 parallel GPU experiments on RunPod.** LSTM, MLP, and frame-stack all running 5M steps with checkpointing every 500k. Key fixes in this session:

- **Per-experiment output dirs:** `launch.sh` auto-detects experiment name from args (`--recurrent` → lstm, `--frame-stack` → framestack, else mlp). `start.sh` constructs `/workspace/runs/{exp_name}_{RUNPOD_POD_ID}/` using RunPod-injected env vars. No collisions when multiple pods share a volume.
- **Volume mount fix:** Added `--volumePath /workspace` to `runpodctl create pod` (default was `/runpod/`, causing "no training files" errors).
- **AWS profile for S3:** Switched from inline `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env vars to `--profile runpod` (via `aws configure --profile runpod`).
- **Automated monitoring:** `research/monitor.sh` polls pods hourly, fetches results, removes pods, then launches Claude Code to synthesize `research/experiment_report.md`.

**Prior: RunPod deployment fixes and cache upload.** Fixed three issues blocking GPU training:

- **ARM→amd64 image fix:** Docker image was being built for ARM (Apple Silicon default). RunPod GPUs are x86_64. Now uses `docker buildx build --platform linux/amd64`.
- **Container entrypoint rewrite:** Old `ENTRYPOINT ["python", "train.py"]` caused crash-loops (container exits when script finishes/fails, killing SSH). New `scripts/start.sh` starts sshd, runs training in background, keeps container alive via `sleep infinity`. SSH auth picks up RunPod's `PUBLIC_KEY` env var.
- **fetch-results.sh rewrite:** Was using `runpodctl receive` (peer-to-peer transfer code), which doesn't work for pod downloads. Now uses rsync over SSH (same pattern as upload-cache.sh).
- **Cache upload via S3:** Uploading 249 `.npz` files (~13GB) to volume `4w2m8hek66` via `aws s3 sync --profile runpod`.

**Prior: RunPod GPU training infrastructure (PR #17 + infrastructure files).** Two deliverables:

- **Checkpointing (PR #17):** `--checkpoint-freq N` and `--resume PATH` on `train.py`. Periodic model + VecNormalize saves via `CheckpointCallback` + custom `VecNormalizeSaveCallback`. Resume with `PPO.load()`/`RecurrentPPO.load()` and `reset_num_timesteps=False`. 72 new tests.
- **RunPod infrastructure:** `Dockerfile` (PyTorch 2.5.1 + CUDA 12.4, pure Python, ~7GB), `.dockerignore`, `runpod/upload-cache.sh` (one-time cache upload), `runpod/launch.sh` (pod launcher), `runpod/fetch-results.sh` (result download). Pattern: persistent network volume (data) + ephemeral GPU pods (compute) + Docker image (code).
- **Refactoring:** Extracted duplicate `TRAIN_SCRIPT`/`load_train_source()`/`extract_main_body()` into `conftest.py`. Extracted obs layout constants into `_obs_layout.py`. Deduplicated `make_train_env` branches in `train.py`.

**Local comparative experiments (2M steps, shuffle-split, seed 42).** Both completed runs are negative OOS:

| Model | Val Return | Test Return | Val Sortino | Test Sortino | Notes |
|-------|-----------|-------------|-------------|--------------|-------|
| Baseline MLP | -51.5 | -62.5 | -2.09 | -1.51 | 0/10 positive test episodes |
| Frame Stack (4) | -48.4 | -50.2 | -1.82 | -1.08 | 2/10 positive test episodes |
| LSTM | — | — | — | — | Killed at 15% — 422 fps too slow for local CPU |

**Key finding:** Shuffle-split doesn't fix OOS. The negative results aren't just chronological regime shift — the agent genuinely isn't generalizing at 2M steps.

**Shuffle split, frame stacking, and RecurrentPPO (prior session, PRs #14–16).** Three features to improve OOS evaluation and give the agent temporal context:

- **Shuffle split (PR #14):** `--shuffle-split` and `--seed 42` on `train.py`. Reproducible random train/val/test splits instead of chronological. Valid because episodes (days) are independent — no cross-day state. 42 new tests.
- **Frame stacking (PR #15):** `--frame-stack N` on `train.py`. `VecFrameStack` wrapper between SubprocVecEnv and VecNormalize. Eval also wraps with VecFrameStack. bar_size=1000 + frame_stack=4 → 84-dim obs. 40 new tests.
- **RecurrentPPO (PR #16):** `--recurrent` flag on `train.py`. `RecurrentPPO('MlpLstmPolicy')` from sb3-contrib. LSTM state tracking in eval (lstm_states, episode_start). Mutually exclusive with `--frame-stack > 1`. 59 new tests.
- **Shared reward module:** Extracted `compute_forced_flatten()` and `compute_step_reward()` into `python/lob_rl/_reward.py`.
- **Refactoring:** `conftest.py` shared helpers (`make_tick_data`, `save_cache_with_instrument_id`), `test_helpers.h` flag constants, `SyntheticSource` named constants, `precompute.cpp` extracted `process_rth_legacy()` / `process_rth_flag_aware()`.
- 418 C++ + 1304 Python = 1722 tests pass.

**Prior session context:** Model trained on 170 days (5M steps, winning config) showed return 139.5 in-sample but -53.8 val / -36.6 test OOS on chronological split. Hypothesis: chronological split conflates overfitting with regime shift (2022 H1 vs H2 were different markets).

**Lazy loading (PR #13).** OOM fix for 200+ day training.

**Contract boundary guard (PR #12).** Forced flatten on terminal step.

**Hyperparameter sweep.** Best: bar=1000, ent=0.05, lr=1e-3 → return 139.5.

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k, ent_coef=0.0, 1 env | Sortino -1.05. Entropy collapsed. |
| v2 + exec cost | 2M, ent_coef=0.01, 8 envs, exec_cost | Entropy stable (0.70). Agent stays flat. |
| v2 no exec cost | 2M, ent_coef=0.01, 8 envs | Entropy collapsed (0.09). Sortino -1.05. |
| v2 + participation bonus | 2M, participation_bonus=0.01 | Sortino -0.91. Wrong direction. |
| Post-fix + exec cost + bonus | 500k, exec_cost, bonus=0.01 | Entropy collapsed (0.01). Exploits bonus. |
| Step-interval + exec cost | 2M, exec_cost, step_interval=10 | Entropy collapsed (0.03). Stays flat. |
| Bar-level v1 | 2M, bar_size=500, 256x256 ReLU, exec_cost | Return 67.5. Entropy 0.20. First positive result. |
| Sweep: ent_coef=0.05 | 2M, bar_size=500, ent=0.05 | Return 111.0. Entropy -0.55. |
| Sweep: ent_coef=0.1 | 2M, bar_size=500, ent=0.1 | Return 101.0. Entropy -0.77. |
| Sweep: bar_size=200 | 2M, bar_size=200 | Return 3.75. Too noisy. |
| Sweep: bar_size=1000 | 2M, bar_size=1000 | Return 128.0. Entropy -0.16. |
| Sweep: lr=1e-3 | 2M, bar_size=500, lr=1e-3 | Return 116.25. Entropy -0.30. |
| Sweep: lr=1e-4 | 2M, bar_size=500, lr=1e-4 | Return 18.0. Too slow. |
| **Combined best** | **2M, bar=1000, ent=0.05, lr=1e-3** | **Return 139.5. Entropy -0.48. Best result.** |
| **Full dataset (chrono)** | **5M, 170 train / 40 val / 40 test** | **139.5 in-sample, -53.8 val, -36.6 test** |
| **MLP shuffle-split** | **2M, 170 train, shuffle, seed 42** | **Val -51.5, Test -62.5. Sortino -1.51.** |
| **Frame-stack shuffle** | **2M, frame_stack=4, shuffle, seed 42** | **Val -48.4, Test -50.2. Sortino -1.08.** |
| **LSTM shuffle** | **2M, recurrent, shuffle, seed 42** | **Killed at 15% — too slow locally (422 fps).** |
| **MLP GPU** | **5M, shuffle, seed 42, RTX 4090** | **Val -62.9, Test -44.0. Sortino -2.22. Worse than 2M on val.** |
| **Frame-stack GPU** | **5M, frame_stack=4, shuffle, RTX 4090** | **Val -82.3, Test -49.4. Sortino -1.10. Much worse on val.** |
| **LSTM GPU** | **5M, recurrent, shuffle, RTX 4090** | **Val -36.7, Test -33.4. Sortino -1.06. Best OOS but still negative.** |

## Key files for current task

| File | Role |
|---|---|
| `experiments/Asymmetric First-Passage Trading.md` | **THE PLAN.** 5-phase research plan. Read this first. |
| `results/exp-004-22-feature-supervised-diagnostic/metrics.json` | Quick exp-004 results (22-feature vs 9-feature RF on 50K subsample) |
| `scripts/run_barrier_diagnostic.py` | T6 supervised diagnostic runner — loads barrier cache, runs overfit/MLP/RF tests |
| `scripts/precompute_barrier_cache.py` | Barrier cache builder — uses C++ `lob_rl_core.barrier_precompute()` backend. ~10 min for 312 files with `--workers 8`. |
| `cache/barrier/` | 248 barrier `.npz` files (220-dim features + per-bar features, N_FEATURES=22). **FRESH** — rebuilt with `bar_features` key. 461K bars, 456K usable, 221 MB. |
| `python/lob_rl/barrier/sequence_models.py` | Sequence model infrastructure — `BarrierLSTM`, `BarrierTransformer`, `BarEmbedding`, training/eval/predict. |
| `results/exp-008-bar-size-sweep/` | exp-008 results — metrics.json, analysis.md. REFUTED. Bar-size sweep, 24 cells. |
| `results/exp-007-sequence-model-signal-detection/` | exp-007 results — metrics.json, analysis.md, config.json. REFUTED. |
| `python/lob_rl/barrier/lob_reconstructor.py` | `OrderBook` class — LOB reconstruction from MBO messages. `vamp(n)` method. |
| `cache/t6_diagnostic_results.json` | Full T6 diagnostic results in JSON format |
| `python/lob_rl/barrier/supervised_diagnostic.py` | MLP classifier + RF baseline functions for barrier label prediction |
| `python/lob_rl/barrier/feature_pipeline.py` | Feature extraction — `compute_bar_features(bars, mbo_data=None)`. 22 features. Neutral defaults when mbo_data=None. |
| `experiment.sh` | Research experiment orchestrator — survey, frame, run, read, log, program |
| `aws/setup.sh` | One-time AWS infra setup: S3, ECR, IAM, security group |
| `aws/launch.sh` | Launch EC2 Spot instance: `EXP_NAME=<label> ./aws/launch.sh <train.py args>` |
| `aws/fetch-results.sh` | Download results from S3: `./aws/fetch-results.sh <run_dir>` |
| `aws/upload-cache.sh` | One-time cache upload to S3: `./aws/upload-cache.sh` |
| `aws/monitor.sh` | Poll EC2 instances, auto-fetch results on termination |
| `aws/README.md` | AWS setup guide, env vars, instance types, S3 layout, cost comparison |
| `QUESTIONS.md` | Research agenda — 6 open questions, 4 answered |
| `DOMAIN_PRIORS.md` | LOB-RL domain knowledge for experiment agents |
| `RESEARCH_LOG.md` | Cumulative experiment findings (7 pre-experiments) |
| `experiments/` | Experiment spec files (created by FRAME phase) |
| `results/` | Experiment output directories (metrics.json, analysis.md) |
| `.claude/prompts/{survey,frame,run,read,synthesize}.md` | Phase-specific agent prompts (frame.md has Compute Target, run.md has AWS Dispatch Protocol) |
| `scripts/train.py` | Training entry point — `--bar-size`, `--cache-dir`, `--output-dir`, `--train-days`, `--shuffle-split`, `--seed`, `--frame-stack`, `--recurrent`, `--checkpoint-freq`, `--resume` |
| `scripts/start.sh` | Container entrypoint. Provider-agnostic via `RUN_ID` env var. Runs train.py; exits 0 on success, sleep infinity on failure. |
| `Dockerfile` | Training container image. PyTorch 2.5.1 + CUDA 12.4 + sshd. Entrypoint: `start.sh`. Must build with `--platform linux/amd64`. Push to ECR. |
| `runpod/` | **Deprecated.** RunPod scripts kept for backward compatibility. Use `aws/` instead. |
| `scripts/precompute_cache.py` | CLI tool to build `.npz` cache. Use `--roll-calendar` or `--instrument-id`. |
| `data/mes/roll_calendar.json` | Maps each date → front-month instrument_id. Used by `--roll-calendar`. |
| `data/mes/*.mbo.dbn.zst` | 312 daily files, Jan–Dec 2022, 57GB. All /MES instruments per file. |
| `python/lob_rl/bar_level_env.py` | `BarLevelEnv` — bar-level gymnasium env (21-dim obs) |
| `python/lob_rl/precomputed_env.py` | `PrecomputedEnv` — tick-level env (54-dim obs) |
| `python/lob_rl/multi_day_env.py` | `MultiDayEnv` — wraps multiple days, lazy-loads `.npz` |
| `python/lob_rl/_reward.py` | Shared `compute_forced_flatten()`, `compute_step_reward()` |

## Don't waste time on

- **Build verification** — `build-release/` is current, 650 C++ + 2221 Python = 2871 tests pass.
- **Dependency checks** — SB3, sb3-contrib, gymnasium, numpy, tensorboard, torch, databento-cpp all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating lookahead/leakage** — fully audited, all clean.
- **Hyperparameter sweep** — done, best config identified.
- **Shuffle split, frame stacking, RecurrentPPO** — all done (PRs #14–16).
- **Lazy loading, contract boundary guard, native DBN source** — all done (PRs #11–13).
- **Precompute fix, spread verification, precompute cache, bar-level env** — all done.

## Architecture overview

```
data/mes/*.mbo.dbn.zst  →  precompute_cache.py --roll-calendar  →  cache/mes/*.npz
                                  (DbnFileSource + map_mbo_to_message)
                                                   ↓
                              ┌─── bar_size=0: PrecomputedEnv (54-dim, tick-level)
                              │
         MultiDayEnv ─────────┤
                              │
                              └─── bar_size>0: BarLevelEnv (21-dim, bar-level)
                                       ↑
                                  aggregate_bars() + cross-bar temporal
                                                   ↓
                           SubprocVecEnv → [VecFrameStack] → VecNormalize
                                                   ↓
                                  SB3 PPO / RecurrentPPO (scripts/train.py)
```

## Test coverage

- **650 C++ tests** — `cd build-release && ./lob_tests` (15 skipped: need `.dbn.zst` fixture)
- **2221 Python tests** (1308 core + 913 barrier) — `PYTHONPATH=build-release:python uv run --with pytest --with pandas --with scipy --with scikit-learn --with torch --with sb3-contrib pytest python/tests/` (4 core + 59 barrier skipped: fixture-dependent)
- **2871 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| ~~Hyperparameter sweep~~ | ~~Done~~ | Best: bar=1000, ent=0.05, lr=1e-3, return 139.5. |
| ~~Native DBN source~~ | ~~Done~~ | PR #11. |
| ~~Extract new data~~ | ~~Done~~ | 312 files in `data/mes/`, 57GB. |
| ~~Contract boundary guard~~ | ~~Done~~ | PR #12. |
| ~~Lazy loading~~ | ~~Done~~ | PR #13. |
| ~~Shuffle split~~ | ~~Done~~ | PR #14. `--shuffle-split --seed 42`. |
| ~~Frame stacking~~ | ~~Done~~ | PR #15. `--frame-stack N`. |
| ~~RecurrentPPO~~ | ~~Done~~ | PR #16. `--recurrent`. |
| ~~Precompute cache~~ | ~~Done~~ | 249 `.npz` in `cache/mes/`, built with `--roll-calendar`. Use `--workers 8` if rebuilding. |
| ~~Comparative experiments~~ | ~~Done (partial)~~ | MLP and frame-stack done locally (both negative OOS). LSTM killed — needs GPU. |
| ~~Proper OOS validation~~ | ~~Done~~ | Shuffle-split also negative. Not just regime shift — agent doesn't generalize at 2M steps. |
| ~~RunPod GPU training~~ | ~~Done~~ | PR #17 (checkpointing) + infrastructure files (Dockerfile, runpod/ scripts). |
| ~~Run experiments on RunPod~~ | ~~Done~~ | All 3 completed 5M steps. LSTM best (val -36.7), all negative OOS. See `research/experiment_report.md`. |
| ~~Install research kit~~ | ~~Done~~ | claude-research-kit installed, configured, research files populated. |
| ~~Investigate negative OOS~~ | ~~Done (P0s REFUTED)~~ | exp-001 (data scaling) REFUTED, exp-002 (exec cost) REFUTED. Both P0 hypotheses eliminated. |
| **Asymmetric First-Passage Trading plan** | **P0** | 5-phase plan at `experiments/Asymmetric First-Passage Trading.md`. Phase 0 mostly done. Phase 1-2 are next. |
| **T10-T12 barrier GPU training** | **P1** | Signal confirmed. Train barrier PPO via `./experiment.sh`. Deferred until Phase 2 signal detection passes. |
| ~~Fix dead book features~~ | ~~Done~~ | PR #34. LOB Reconstructor + MBO wiring. All book features active. |
| ~~Phase 1 microstructure features~~ | ~~Done~~ | PR #35. OFI, depth ratio, wmid displacement, spread std. N_FEATURES 13→17. |
| ~~Phase 2 microstructure features~~ | ~~Done~~ | PR #36. VAMP, aggressor imbalance, trade arrival, cancel-to-trade, price impact. N_FEATURES 17→22. |
| ~~Re-precompute barrier cache~~ | ~~Done~~ | 248 sessions, N_FEATURES=22, 186 MB. C++ backend: 570s vs 8+ hours Python. |
| ~~Architecture comparison~~ | ~~CLOSED~~ | exp-007 REFUTED. Six model families (LR, GBT, MLP, RF, LSTM, Transformer) all fail to beat constant Brier. Architecture cannot amplify absent signal. |
| **199d no-exec-cost 10M+ steps** | **P1** | Only positive OOS ever (exp-002 Run C val +10.93, undertrained). Needs GPU. |
| Checkpoint early stopping | P1 | eval 4M vs 5M checkpoints. |
| VecNormalize audit | P1 | Check for cross-day information leakage. |
| More data (second year) | Low | 2023-2024 as complement if 2022 generalizes well. |

---

For project history, see `git log`.
