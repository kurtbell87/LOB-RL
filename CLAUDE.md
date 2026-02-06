## Current State (updated 2026-02-06)

- **Build:** `build-release/` is current. 404 C++ tests pass (`./lob_tests`). 463 Python tests pass.
- **Python:** Always use `uv`. Run with `PYTHONPATH=build-release:python uv run ...`
- **Dependencies:** SB3, gymnasium, numpy, tensorboard all installed in uv environment.
- **Next task:** Step 4c — run training (`scripts/train.py`). All infrastructure is complete. No code changes needed.
- **Data:** 27 days of /MES MBO data in `data/mes/`, manifest at `data/mes/manifest.json`.
- **Key entry point:** `cd build-release && PYTHONPATH=.:../python uv run python ../scripts/train.py --data-dir ../data/mes`

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
3. **Directory `README.md` files** — If you add/rename/delete files in a directory, update that directory's `README.md`. If a directory doesn't have one, create it.
4. **Spec docs in `docs/`** — After completing a TDD cycle, the spec stays as historical record. Don't delete them.

The goal: a new agent session should be able to orient itself by reading only `CLAUDE.md` → `LAST_TOUCH.md` → relevant directory `README.md`, without grepping or exploring.

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

### After the Cycle

- Verify all tests pass
- Update `LAST_TOUCH.md` with what was built and the new project state.
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
