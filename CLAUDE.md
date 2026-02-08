## Current State (updated 2026-02-08)

- **Build:** `build-release/` is current. 418 C++ tests pass (`./lob_tests`). 1304 Python tests pass. **1722 total.** (15 C++ + 4 Python skipped — need `.dbn.zst` fixture.)
- **Python:** Always use `uv`. Run with `PYTHONPATH=build-release:python uv run ...`
- **Dependencies:** SB3, sb3-contrib, gymnasium, numpy, tensorboard, torch, databento-cpp (FetchContent) all installed.
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
- **RunPod fixes applied:** Dockerfile now builds for `linux/amd64` (not ARM). Includes `openssh-server`+`rsync`. Entrypoint is `scripts/start.sh` (starts sshd, runs training in background, keeps container alive). `fetch-results.sh` rewritten to use rsync over SSH. Cache upload uses `aws s3 sync --profile runpod`. Each experiment writes to `/workspace/runs/{exp_name}_{pod_id}/`.
- **Cache uploaded:** 249 `.npz` files (18GB) on RunPod volume `4w2m8hek66` at `cache/mes/`. Verified via `aws s3 ls`.
- **GPU experiments RUNNING:** 3 parallel pods on RTX 4090s — LSTM (`6kwbf810ribiza`), MLP (`yvag35jcok2egk`), frame-stack (`0w3gtzsu1h3nhl`). 5M steps each, checkpoint every 500k. `research/monitor.sh` polling hourly, auto-fetches results, auto-removes pods, then launches Claude Code to write `research/experiment_report.md`.
- **Next task:** Check `research/experiment_report.md` in the morning. Then investigate negative OOS results.
- **Reference:** Databento DBN spec cloned to `references/dbn/`.
- **Precompute hint:** If cache needs rebuilding: `precompute_cache.py --roll-calendar ... --workers 8` (script supports `--workers N` via `ProcessPoolExecutor`).
- **Key entry point:** `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --cache-dir ../cache/mes/ --bar-size 1000 --execution-cost --policy-arch 256,256 --activation relu --ent-coef 0.05 --learning-rate 0.001 --shuffle-split --seed 42`

## Don't

- Don't verify the build or run C++ tests unless asked or unless you changed C++ code.
- Don't explore the codebase to "understand" it — read `LAST_TOUCH.md` and directory `README.md` files instead.
- Don't read `PRD.md` unless you need requirements for a new feature. `LAST_TOUCH.md` has the current state.
- Don't check if dependencies are installed — they are.
- Don't read source files to understand architecture — read the `README.md` in each directory first.

## Breadcrumb Maintenance (MANDATORY)

After every session that changes the codebase, you MUST maintain these navigation files so the next agent starts fast:

1. **`LAST_TOUCH.md`** — Update the "What to do next" and "Key files" sections. This is a cold-start briefing, not a journal. Keep it actionable and concise.
2. **`CLAUDE.md` "Current State" section** — Update build status, test counts, and next task.
3. **Directory `README.md` files** — If you add/rename/delete files in a directory, update that directory's `README.md`. If a directory doesn't have one, create it. **See format requirements below.**
4. **Spec docs in `docs/`** — Archived automatically by `./tdd.sh ship`. The spec is deleted from the working tree but preserved in git history. Don't manually delete specs before shipping.

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

## Python Environment

- **All Python commands MUST use `uv`.** Use `uv pip install` for dependencies and `uv run` for executing Python scripts/tests. Do NOT use bare `pip`, `pip3`, `python`, or `python3` commands.
