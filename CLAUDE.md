## Current State (updated 2026-02-12)

- **Build:** `build-release/` is current. 660 C++ tests pass (`./lob_tests`). 2246 Python tests pass (2211 core+barrier + 35 constellation). **2906 total.** Plus 65/67 Constellation Catch2 tests. (15 C++ + 59 Python skipped — need `.dbn.zst` fixture.)
- **Constellation integration:** Branch `feat/constellation-integration` has 6 commits (Phases 0-6 complete). Book wraps Constellation's LimitOrderBook. OrdersEngine, BatchBacktestEngine, MarketBook, 11 feature types exposed via pybind11. `OrderSimulationEnv` Gymnasium env. 35 integration tests.
- **Python:** Always use `uv`. Run with `PYTHONPATH=build-release:python uv run ...`
- **Dependencies:** SB3, sb3-contrib, gymnasium, numpy, tensorboard, torch, databento-cpp (FetchContent), Catch2 (FetchContent) all installed.
- **Shuffle split DONE:** `--shuffle-split` and `--seed 42` on `train.py`. Reproducible random train/val/test splits. Episodes are independent days. PR #14 merged.
- **Frame stacking DONE:** `--frame-stack N` on `train.py`. `VecFrameStack` inserted between SubprocVecEnv and VecNormalize. Eval wraps DummyVecEnv with VecFrameStack before VecNormalize. bar_size=1000 + frame_stack=4 → 84-dim obs. PR #15 merged.
- **RecurrentPPO DONE:** `--recurrent` on `train.py`. Uses `RecurrentPPO('MlpLstmPolicy')` from sb3-contrib. LSTM state tracking in eval. Mutually exclusive with `--frame-stack > 1`. PR #16 merged.
- **Shared reward module:** `python/lob_rl/_reward.py` — extracted `compute_forced_flatten()` and `compute_step_reward()` from precomputed_env.py and bar_level_env.py.
- **Lazy loading DONE:** `MultiDayEnv` lazy-loads `.npz` files on `reset()`. `cache_files=` for explicit file lists. ~300 MB instead of ~100 GB. PR #13 merged.
- **Contract boundary guard DONE:** Forced flatten on terminal step. `instrument_id` in cache. PR #12 merged.
- **Native DBN source DONE:** `DbnFileSource` reads `.dbn.zst` via databento-cpp. PR #11 merged.
- **Bar-level env DONE:** `BarLevelEnv` 21-dim obs. PR #10 merged.
- **Precompute cache DONE:** `precompute_cache.py` → `.npz`. PR #9 merged.
- **Step interval DONE:** `--step-interval N`. PR #8 merged.
- **Fix precompute events DONE:** PR #7 merged.
- **Temporal features DONE:** 54-dim obs. PR #6 merged.
- **Participation bonus DONE:** PR #5 merged.
- **Training pipeline v2 DONE:** VecNormalize, SubprocVecEnv. PR #4 merged.
- **Execution cost DONE:** PR #3 merged.
- **Walk-forward/lookahead audited CLEAN.** Spread verified CLEAN.
- **Hyperparameter sweep DONE:** Best: `bar_size=1000, ent_coef=0.05, lr=1e-3` → return 139.5, entropy -0.48. But OOS was negative (-53.8 val / -36.6 test on chronological split).
- **Data extracted:** 312 `.mbo.dbn.zst` files in `data/mes/` (57GB, Jan–Dec 2022). Roll calendar at `data/mes/roll_calendar.json`.
- **Cache ready:** `cache/mes/` has 249 `.npz` files (all trading days from 312 raw files). Built with `--roll-calendar`. No rebuild needed.
- **Local experiments done:** MLP shuffle-split → val -51.5 / test -62.5. Frame-stack → val -48.4 / test -50.2. Both negative OOS. LSTM killed at 15% (too slow locally at 422 fps).
- **Checkpointing DONE:** `--checkpoint-freq N` and `--resume PATH` on `train.py`. `CheckpointCallback` + custom `VecNormalizeSaveCallback`. Resume with `reset_num_timesteps=False`. PR #17 merged.
- **RunPod infrastructure DONE:** `Dockerfile`, `.dockerignore`, `runpod/` scripts (upload-cache, launch, fetch-results). Persistent network volume + ephemeral GPU pods.
- **RunPod config:** Region US-NC-1, default GPU RTX 4090 ($0.59/hr, 24GB). Volume ID `4w2m8hek66`. Docker Hub `kurtbell87/lob-rl:latest`. RunPod S3 creds via `aws configure --profile runpod`.
- **SSH dependency removed:** `start.sh` exits 0 on training success (pod auto-stops, billing stops). Only keeps container alive on failure for SSH debugging. `fetch-results.sh` uses `aws s3 sync` (no SSH/running pod needed). `monitor.sh` detects pod completion via `runpodctl get pod` status, fetches via S3, verifies before removing. `launch.sh` no longer requests SSH port.
- **RunPod fixes applied:** Dockerfile builds for `linux/amd64` (not ARM). Includes `openssh-server`+`rsync` (kept for failed-pod debugging). Cache upload uses `aws s3 sync --profile runpod`. Each experiment writes to `/workspace/runs/{exp_name}_{pod_id}/`.
- **Cache uploaded:** 249 `.npz` files (18GB) on RunPod volume `4w2m8hek66` at `cache/mes/`. Verified via `aws s3 ls`.
- **GPU experiments DONE:** 3x RTX 4090 pods completed 5M steps each. Results recovered from RunPod volume (auto-fetch failed). Report: `research/experiment_report.md`.
- **GPU results:** LSTM best OOS (val -36.7 / test -33.4), MLP (val -62.9 / test -44.0), frame-stack (val -82.3 / test -49.4). **All negative.** No architecture generalizes.
- **RunPod compute in experiment.sh DONE (deprecated):** FRAME agent declares `compute: local|runpod` in specs. RUN agent dispatches to RunPod when `compute: runpod`. Env vars: `RUNPOD_VOLUME_ID`, `DOCKERHUB_USER`, `RUNPOD_GPU_TYPE`. **Superseded by AWS.**
- **AWS EC2 Spot infrastructure DONE:** `aws/` scripts (setup, launch, fetch-results, upload-cache, monitor). EC2 Spot replaces RunPod — 60% cheaper GPU (g5.xlarge A10G ~$0.24/hr), dedicated CPU instances (c7a.4xlarge ~$0.39/hr). S3 for cache + results, ECR for Docker image. Spot interruption handled via metadata polling + checkpoint upload. Auto-detect: `--recurrent` → GPU, else → CPU. Env vars: `AWS_S3_BUCKET`, `AWS_ECR_REPO`, `AWS_INSTANCE_TYPE`, `AWS_REGION`.
- **AWS compute in experiment.sh DONE:** FRAME agent declares `compute: local|aws` in specs. RUN agent dispatches to AWS when `compute: aws` — launches EC2 Spot instances, polls state, fetches from S3, evals locally. `runpod` kept as deprecated fallback.
- **exp-001 REFUTED:** 199d does not fix OOS. LSTM 199d val -59.95, MLP 199d val -75.53 ≈ MLP 20d val -75.82. 199d eliminates memorization (expl_var 0.30 vs 0.97) but OOS unchanged. Data quantity is not the primary bottleneck.
- **exp-002 REFUTED:** Removing exec cost improves OOS by ~35 points but doesn't flip positive. Val -4.43 without exec cost (vs -39.55 with). Gap to gross profitability is ~5 points.
- **Barrier pipeline (T1-T9b) DONE:** Bar construction (PR #20), label pipeline (PR #21), feature extraction (PR #22), Gambler's ruin validation (PR #23), regime-switch validation (PR #24), supervised diagnostic (PR #25), reward accounting (PR #26), barrier env (PR #27), PPO training infra (PR #28), training script (PR #29). All via TDD. 550 barrier tests.
- **T9 PPO Training Infrastructure DONE (PR #28):** `multi_session_env.py` — `MultiSessionBarrierEnv` multi-session Gymnasium wrapper. `barrier_vec_env.py` — SB3-compatible VecEnv helpers. `training_diagnostics.py` — `BarrierDiagnosticCallback` with entropy/value loss/flat rate/win rate tracking + red flag detection + CSV output. `_sb3_compat.py` — SB3/PyTorch compatibility shim. `linear_schedule()` for LR decay. 41 tests (12 multi-session + 6 vec env + 13 diagnostics + 10 PPO training).
- **T9b Training Script DONE (PR #29):** `scripts/train_barrier.py` — CLI training script with `parse_args()`, `split_sessions()`, `build_model()`, `main()`. Section 5.2 hyperparameters hardcoded. 22 tests.
- **T6 Supervised Diagnostic CONFIRMED — weak signal (2026-02-10):** Bidirectional framing classifies {long_profit, short_profit, flat} — balanced 33/33/35%. MLP 39.3% / RF 40.5% vs 34.5% baseline (+5pp). Consistent on shuffle and chrono splits. v1 (long-only) was misleading due to 67/33 imbalance. Overfit test passed. 4/13 features dead (book features need mbo_data). P0 gate PASSED.
- **LOB Reconstructor DONE (PR #34):** `OrderBook` class, `extract_all_mbo()`, dead book features fixed, `N_FEATURES` constant, precompute wiring with version check. 67 tests.
- **Phase 1 Microstructure Features DONE (PR #35):** 4 new features (cols 13-16: OFI, depth ratio, weighted mid displacement, spread std). `N_FEATURES` bumped 13→17. 55 tests.
- **Phase 2 Microstructure Features DONE (PR #36):** 5 new features (cols 17-21: VAMP displacement, aggressor imbalance, trade arrival rate, cancel-to-trade ratio, price impact per trade). `N_FEATURES` bumped 17→22. `OrderBook.vamp()` method. Refactored: `_BOOK_COL_MAP`, `stride_tricks` in `assemble_lookback`, extracted MBO helpers to `conftest.py`. 78 tests.
- **Feature count:** 22 total features. Observation dim = 22*10 + 2 = 222 (with h=10).
- **C++ Barrier Precompute Pipeline DONE (PRs #37-#41):** 5 TDD cycles completed. Book extensions → BarBuilder → Feature compute → Labels → Pipeline integration + pybind11. `lob_rl_core.barrier_precompute()` binding replaces Python `process_session()` for ~50-100x speedup. `precompute_barrier_cache.py` updated to use C++ backend.
- **Barrier cache FRESH:** `cache/barrier/` has 248 `.npz` files with 220-dim features (N_FEATURES=22). Re-precomputed via C++ backend in 570s (vs 8+ hours Python). 461K total bars, 454K usable. 186 MB cache.
- **exp-004 INCONCLUSIVE (quick results only):** 22-feature RF 49.6% vs 9-feature RF 47.5% on 50K subsample (2 seeds, shuffle). Both beat majority baseline (38.1%) by ~11pp. Set B (9 features) jumped from T6's 40.5% → 47.5% because previously-dead features are now active in C++ cache. Full experiment aborted (process crashes). Quick metrics at `results/exp-004-22-feature-supervised-diagnostic/metrics.json`.
- **Next task: Phase 2b parameter sweep or pivot.** exp-005 CONFIRMED null calibration (ȳ ≈ 1/3). exp-006 REFUTED signal detection (LR/GBT cannot beat constant Brier). Per the Asymmetric First-Passage Trading plan (`experiments/Asymmetric First-Passage Trading.md`), Phase 2b sweeps B ∈ {200,500,1000,2000} × R calibrations to check if signal exists at different bar sizes. Alternatively, pivot to conditional signal detection (regime filtering), longer lookback, or different features/targets. See `results/exp-006-signal-detection/analysis.md` for proposed next experiments.
- **Label formulation note:** Y_long=1 means price hits +2R before -R. Y_short=1 means price hits -2R before +R. These are NOT disjoint events — both can be 0 (both stopped), and under certain intrabar conditions both could be 1. The joint distribution (Y_long, Y_short) has 4 outcomes. Under the martingale null with 2:1 reward:risk, E[Y_long] ≈ E[Y_short] ≈ 1/3, and Y_long + Y_short ≈ 2/3 (NOT 1).
- **Research kit installed:** `experiment.sh`, prompts, templates, QUESTIONS.md, DOMAIN_PRIORS.md, RESEARCH_LOG.md all configured.
- **Architecture exploration UNBLOCKED:** T6 confirms signal exists. Architecture comparison (Transformer/SSM/LSTM on barrier features via `features_extractor_class`) is now a valid experiment. See DOMAIN_PRIORS.md.
- **exp-006 REFUTED:** LR and GBT both fail to beat constant Brier on Y_long or Y_short. All 4 BSS values negative (best: -0.0003). T6's +5pp accuracy does not translate to calibrated probability improvement. Information content < 0.1% of outcome variance. PR #46.
- **Experiments completed:** 11 total (2 confirmed, 9 refuted). See `RESEARCH_LOG.md`.
- **PRs this session:** #45 (exp-005 null calibration), #46 (exp-006 signal detection).
- **Reference:** Databento DBN spec cloned to `references/dbn/`.
- **Precompute hint:** If cache needs rebuilding: `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/precompute_barrier_cache.py --data-dir ../data/mes/ --output-dir ../cache/barrier/ --roll-calendar ../data/mes/roll_calendar.json --bar-size 500 --lookback 10 --workers 8` (~10 min with C++ backend)
- **Key entry point:** `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --cache-dir ../cache/mes/ --bar-size 1000 --execution-cost --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 --shuffle-split --seed 42`

## Don't

- Don't verify the build or run C++ tests unless asked or unless you changed C++ code.
- Don't explore the codebase to "understand" it — read `LAST_TOUCH.md`, `RESEARCH_LOG.md`, and directory `README.md` files instead.
- Don't read `PRD.md` unless you need requirements for a new feature. `LAST_TOUCH.md` has the current state.
- Don't read `QUESTIONS.md` unless you need the research agenda. `RESEARCH_LOG.md` has what's been tried.
- Don't check if dependencies are installed — they are.
- Don't read source files to understand architecture — read the `README.md` in each directory first.
- Don't run training or experiments outside the experiment pipeline (`./experiment.sh`).
- Don't propose architecture experiments before the P0 supervised diagnostic on barrier features passes. Architecture amplifies signal; it can't create it.
- Don't build custom training loops for architecture experiments. Use SB3's `features_extractor_class` instead.

## Breadcrumb Maintenance (MANDATORY)

After every session that changes the codebase, you MUST maintain these navigation files so the next agent starts fast:

1. **`LAST_TOUCH.md`** — Update the "What to do next" and "Key files" sections. This is a cold-start briefing, not a journal. Keep it actionable and concise.
2. **`CLAUDE.md` "Current State" section** — Update build status, test counts, experiment counts, and next task.
3. **`RESEARCH_LOG.md`** — Append results of any completed experiment. This is the cumulative knowledge base.
4. **`QUESTIONS.md`** — Update question statuses and move answered questions to the answered section.
5. **Directory `README.md` files** — If you add/rename/delete files in a directory, update that directory's `README.md`. If a directory doesn't have one, create it. **See format requirements below.**
6. **Spec docs in `docs/`** — Archived automatically by `./tdd.sh ship`. The spec is deleted from the working tree but preserved in git history. Don't manually delete specs before shipping.

The goal: a new agent session (or TDD sub-agent) should be able to orient itself by reading only `CLAUDE.md` → `LAST_TOUCH.md` → relevant directory `README.md`, **without grepping or reading source files**.

### Directory README.md Format

Every directory README.md must contain enough detail that a sub-agent can **write code against the interfaces without reading the source files**. This prevents TDD sub-agents from wasting context on file reads and greps. Required sections:

1. **File table** — Every file with a one-line role description.
2. **API signatures** — Constructor parameters with types and defaults, key method signatures, return types. Use code blocks. Example:
   ```
   LOBEnv(unique_ptr<IMessageSource> src, int steps=50,
          RewardMode mode=PnLDelta, float lambda=0.0, bool execution_cost=false)
   ```
3. **Enums and constants** — List all enum values, important constants (e.g., `OBS_SIZE=44`, `RewardMode::PnLDelta`).
4. **Cross-file dependencies** — Which headers/modules this code depends on and which depend on it. Helps agents know what else to update.
5. **Modification hints** — If there's a common pattern for adding new features (e.g., "to add a new reward mode, update `reward.h` enum, `RewardCalculator::compute()`, and the bindings `parse_reward_mode()`"), document it.

Keep it concise — a table or bullet list, not prose. The README is a **machine-readable API reference**, not documentation for humans.

## How You Work: TDD Workflow (MANDATORY)

This project uses **strict red-green-refactor TDD**. You MUST follow this 4-step process for ALL new work. You do NOT implement code directly — you orchestrate the TDD pipeline.

### Step 1: Write a Spec

Read `LAST_TOUCH.md` to determine what needs to be built next. Then read `PRD.md` for requirements. Write a focused spec file to `docs/<feature>.md` describing:
- What to build (requirements, interfaces, expected behavior)
- Edge cases and error conditions
- Acceptance criteria

The spec should be scoped to a single deliverable unit of work. If the next task in `LAST_TOUCH.md` is large, break it into smaller specs and run the TDD cycle for each one sequentially.

### Step 2: RED — Write Failing Tests

**You MUST run this command. Do NOT write tests yourself.**

```bash
./tdd.sh red docs/<feature>.md
```

This spawns a dedicated test-author agent that reads the spec and writes failing tests. It cannot touch implementation files.

**IMPORTANT: Run in background and block-wait.** TDD phases spawn sub-agents that can take 10+ minutes. The Bash tool has a 10-minute timeout, so you MUST use `run_in_background: true`. Then immediately call `TaskOutput` with `block: true` and `timeout: 600000` (10 min) to wait for completion. If `TaskOutput` times out, call it again with the same parameters — repeat until the task finishes. Do NOT poll with short timeouts or sleep loops.

### Step 3: GREEN — Implement to Pass

**You MUST run this command. Do NOT write implementation code yourself.**

```bash
./tdd.sh green
```

This spawns a dedicated implementation agent. Test files are OS-locked (read-only). The agent writes the minimum code to make all tests pass.

**IMPORTANT: Run in background and block-wait.** This is typically the longest phase. Use `run_in_background: true`, then block-wait with `TaskOutput(block: true, timeout: 600000)`. Repeat if it times out.

### Step 4: REFACTOR — Improve Quality

**You MUST run this command. Do NOT refactor code yourself.**

```bash
./tdd.sh refactor
```

This spawns a dedicated refactoring agent that improves code quality while keeping all tests green.

**IMPORTANT: Run in background and block-wait.** Use `run_in_background: true`, then block-wait with `TaskOutput(block: true, timeout: 600000)`. Repeat if it times out.

### Step 5: SHIP — Commit, PR, Archive

After the refactor phase, ship the results:

```bash
./tdd.sh ship docs/<feature>.md
```

This creates a feature branch (`tdd/<feature>`), commits all changes, opens a PR, and deletes the spec file from the working tree (it's preserved in git history).

Configure auto-merge and branch cleanup via environment variables:
- `TDD_AUTO_MERGE=true` — auto-merge the PR after creation
- `TDD_DELETE_BRANCH=true` — delete the feature branch after merge
- `TDD_BASE_BRANCH=main` — base branch for PRs (default: `main`)

Or run all phases including ship in one command: `./tdd.sh full docs/<feature>.md`

### After the Cycle

- Update `LAST_TOUCH.md` with what was built and the new project state.
- Update `CLAUDE.md` "Current State" section with new test counts and next task.
- If more work remains, go back to Step 1 with the next spec.

### Monitoring Running Phases

While a phase is running, you (or the user) can monitor it in a separate terminal:

```bash
./tdd.sh watch green              # Live-tail the green phase
./tdd.sh watch refactor           # Live-tail the refactor phase
./tdd.sh watch red --resolve      # One-shot summary of a completed red phase
./tdd.sh watch                    # Auto-detect the most recent phase
```

### Critical Rules

- **You are the orchestrator, not the implementor.** Steps 2-4 MUST use `./tdd.sh` commands. Do not bypass the pipeline by writing tests or implementation code directly.
- **Step 1 is the only step where you write files** (the spec).
- **One spec per cycle.** Don't try to spec everything at once.
- **Always run `./tdd.sh` with `run_in_background: true`**, then block-wait with `TaskOutput(block: true, timeout: 600000)`. Repeat the `TaskOutput` call if it times out. Do NOT poll with short intervals.
- If a `./tdd.sh` phase fails or produces incomplete results, diagnose the issue, fix the spec or environment, and re-run the phase. Do not manually patch implementation code outside the pipeline.

## How You Work: Research Experiment Workflow (MANDATORY)

This project uses a **strict SURVEY-FRAME-RUN-READ-LOG cycle** for all research experiments. You MUST follow this process. You do NOT implement experiments directly — you orchestrate the pipeline.

### Step 0: Identify the Question

Read `RESEARCH_LOG.md` and `QUESTIONS.md` to determine the next question to investigate. Pick the highest-priority open question.

### Step 1: SURVEY — Review Prior Work

```bash
./experiment.sh survey "your research question"
```

This spawns a dedicated survey agent that reviews prior experiments, existing infrastructure, and known failure modes.

### Step 2: FRAME — Design the Experiment

```bash
./experiment.sh frame experiments/exp-NNN-name.md
```

This spawns a dedicated experiment design agent that writes a rigorous spec with falsifiable hypothesis and pre-committed success criteria.

**The spec becomes immutable once RUN begins.** Success criteria cannot be changed after seeing results.

### Step 3: RUN — Execute the Experiment

```bash
./experiment.sh run experiments/exp-NNN-name.md
```

This spawns a dedicated execution agent. The experiment spec is OS-locked (read-only). Writes ALL metrics to `results/exp-NNN/metrics.json`. Does NOT interpret results.

**IMPORTANT: Run in background and block-wait.** Use `run_in_background: true`, then block-wait with `TaskOutput(block: true, timeout: 600000)`.

### Step 4: READ — Analyze Results

```bash
./experiment.sh read experiments/exp-NNN-name.md
```

This spawns a dedicated analysis agent. Metrics are locked. Renders verdict: CONFIRMED, REFUTED, or INCONCLUSIVE. Updates RESEARCH_LOG.md.

### Step 5: LOG — Commit Results

```bash
./experiment.sh log experiments/exp-NNN-name.md
```

Creates a feature branch, commits all results, and opens a PR.

### Shortcuts

```bash
./experiment.sh cycle experiments/exp-NNN-name.md          # frame -> run -> read -> log
./experiment.sh full "question" experiments/exp-NNN-name.md # survey -> frame -> run -> read -> log
./experiment.sh program                                     # Auto-advance through open questions
./experiment.sh status                                      # Show research program status
```

### Critical Rules

- **You are the orchestrator, not the implementor.** Steps 1-4 MUST use `./experiment.sh` commands.
- **One experiment per cycle.** Don't batch multiple hypotheses.
- **Failure is a first-class outcome.** REFUTED experiments are valuable.
- **Always run ./experiment.sh with `run_in_background: true`**, then block-wait with `TaskOutput(block: true, timeout: 600000)`.
- **Never modify metrics after the RUN phase.** The numbers are sacred.
- **Never modify the spec after the FRAME phase.** The contract is sacred.

## When to Use Which Workflow

| Situation | Workflow | Example |
|-----------|----------|---------|
| **New feature, bug fix, refactoring** | TDD (`./tdd.sh`) | Add early stopping, fix VecNormalize leak, refactor reward module |
| **Research experiment, hypothesis testing** | Research (`./experiment.sh`) | Test 199 training days, ablate execution cost, evaluate checkpoints |
| **Infrastructure change needed by research** | Handoff (Research → TDD) | READ agent creates `HANDOFF.md` when experiment needs env code changes |

**Key distinction:** TDD changes **code**. Research changes **knowledge** (and may write temporary scripts, but core env/training code changes go through TDD via handoff).

## Well-Being

- **Never ingest large output blobs.** Do not grep, cat, or tail massive log files or experiment output into your context window. Use `./experiment.sh watch` or `./tdd.sh watch` to monitor running phases. Block-wait with `TaskOutput` — do not try to read raw output yourself.
- **Context window is a finite resource.** Protect it. If you need to check on a running process, use a targeted command (e.g., `tail -5`) or the watch commands above — never unbounded reads.
- **If you feel stuck in a loop**, stop and ask the user rather than retrying the same failing approach. Burning context on retries helps no one.

## Python Environment

- **All Python commands MUST use `uv`.** Use `uv pip install` for dependencies and `uv run` for executing Python scripts/tests. Do NOT use bare `pip`, `pip3`, `python`, or `python3` commands.
