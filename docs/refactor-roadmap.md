# LOB-RL Refactor Roadmap

## Purpose

This document proposes a concrete refactor plan for turning this repository into a high-velocity, evidence-driven R&D platform for /MES RL trading research. The goal is not cosmetic cleanup; it is to increase experiment truthfulness, reduce false positives, and improve iteration speed.

## Current-State Forensics (Condensed)

### Structural strengths to preserve
- **Performance split is correct:** C++ replay and market mechanics in `src/` + Python experimentation and training in `python/` and `scripts/`.
- **Cache-first workflow is correct:** DBN-to-NPZ precompute dramatically lowers training-time IO overhead.
- **Research orchestration exists:** `experiment.sh` and experiment specs show an explicit scientific loop.
- **Multi-environment direction exists:** classic event-level and bar/barrier pathways are both represented.

### Structural pain points to address
1. **Experiment rigor is partially encoded in conventions instead of code.** Split/eval assumptions are spread across scripts and docs.
2. **Entrypoint fragmentation.** `train.py` and `train_barrier.py` overlap in behavior but diverge in APIs and outputs.
3. **Packaging and import fragility.** Operational scripts rely on runtime path insertion and implicit environment setup.
4. **Metrics contract inconsistency.** Some runs produce rich artifacts, others produce sparse results with different field names.
5. **Documentation drift risk.** Fast-moving capabilities and counts can diverge from implementation reality.

---

## Refactor Objectives

### Objective A — Reproducibility by default
Every experiment should be replayable from a single typed config + immutable code revision + deterministic data split seed.

### Objective B — Evidence quality gates
No result should be labeled “promising” unless it passes predefined out-of-sample and stability gates.

### Objective C — One training framework, many policies
Unify training loop infrastructure and treat env/algorithm/reward variants as plug-ins.

### Objective D — Portable execution
Local, AWS Spot, and any future worker backend should share the same run contract and artifact schema.

### Objective E — Honest observability
Test and experiment dashboards should be generated from executable outputs, not manually curated claims.

---

## Target Architecture

## 1) Package layout and boundaries

Proposed package structure:

```text
python/lob_rl/
  config/
    schemas.py
    loaders.py
  data/
    cache_index.py
    roll_calendar.py
  envs/
    event_env.py
    bar_env.py
    barrier_env.py
    wrappers.py
  training/
    trainer.py
    algorithms.py
    callbacks.py
    evaluation.py
  orchestration/
    run_manifest.py
    artifact_writer.py
    backends/
      local.py
      aws.py
  reporting/
    metrics_schema.py
    summarize.py
  cli/
    train.py
    evaluate.py
    precompute.py
```

Design principles:
- `scripts/` becomes thin wrappers (or is retired) once `python -m lob_rl.cli.*` commands are stable.
- Env definitions live only in `envs/`; training code never imports low-level source/parsing internals directly.
- All side-effecting code (filesystem, cloud, process launch) is isolated in `orchestration/`.

## 2) Unified run manifest

Introduce `run_manifest.yaml` (saved alongside outputs):
- git SHA
- config hash
- data snapshot ID / cache file list hash
- split policy and seed
- algorithm + policy + hyperparameters
- hardware/backend metadata
- expected artifact schema version

This becomes the canonical provenance object for all runs.

## 3) Metrics schema versioning

Define a strict JSON schema (`metrics.v1.json`) for:
- train/val/test returns
- downside deviation and Sortino (with sampling details)
- turnover and trade count
- cost-on vs cost-off ablations (if enabled)
- confidence intervals / bootstrap metadata

All runners must emit valid schema-compliant output.

---

## Refactor Workstreams

## Workstream 1 — Configuration hardening

### Deliverables
- Typed config models (Pydantic/dataclass + validation).
- One config file path accepted by all entrypoints.
- Explicit split strategies: chronological, shuffle, walk-forward.
- Explicit eval budget: episodes, seeds, confidence method.

### Migration strategy
1. Add parser for new config while preserving CLI flags.
2. Deprecate direct flags in stages with warning windows.
3. Lock experiments to config files in `experiments/`.

### Success criteria
- Two identical configs + seeds on same cache produce statistically identical rollout metrics.

## Workstream 2 — Trainer unification

### Deliverables
- `Trainer` core that manages vectorization, normalization, callbacks, checkpointing, and evaluation.
- Algorithm adapters (`PPO`, `RecurrentPPO`, future algorithms) behind common interface.
- Environment factory layer with typed signatures.

### Migration strategy
1. Extract common logic from `train.py` and `train_barrier.py`.
2. Keep both scripts as compatibility frontends calling shared trainer.
3. Remove duplicated callback/eval/serialization code.

### Success criteria
- `train.py` and `train_barrier.py` become thin launchers with <15% unique logic.

## Workstream 3 — Artifact and reporting contract

### Deliverables
- Stable output tree:
  - `manifest.yaml`
  - `metrics.json`
  - `model/`
  - `vecnormalize/`
  - `logs/`
  - `plots/` (optional)
- Validation command: `lob_rl.cli.evaluate --validate-only`.
- Automatic experiment summary markdown generation.

### Success criteria
- Every run folder passes schema validation in CI.

## Workstream 4 — Test system integrity

### Deliverables
- CI checks that detect unregistered C++ tests or stale test-count claims in docs.
- Golden tests for metrics schema compatibility.
- Integration test: smoke train (small timesteps) + eval + artifact validation.

### Success criteria
- Any mismatch between docs test-count badges and real runner output fails CI.

## Workstream 5 — Execution backend abstraction

### Deliverables
- Backend interface: `submit()`, `status()`, `fetch_artifacts()`, `cancel()`.
- AWS and local implementations behind shared contract.
- Backend chosen by config, not branch-specific scripts.

### Success criteria
- Same experiment config can run local or AWS with identical artifact schema.

---

## Phased Implementation Plan

## Phase 0 (1 week): Baseline and guardrails
- Freeze baseline branch and capture benchmark runs.
- Add metrics schema draft + validator.
- Add run manifest writer.

## Phase 1 (1–2 weeks): Config + manifest integration
- Introduce typed config loader.
- Wire trainer to consume config object.
- Backfill legacy CLI flags into config translator.

## Phase 2 (1–2 weeks): Trainer consolidation
- Extract shared train loop.
- Factor env factories and algorithm adapters.
- Ensure checkpoint/resume parity across env types.

## Phase 3 (1 week): Artifact/reporting standardization
- Normalize run directory layout.
- Add automated markdown and JSON summaries.
- Add schema validation CLI.

## Phase 4 (1 week): CI truthfulness + backend abstraction
- Add CI contract checks (schema, docs drift hooks, integration smoke).
- Implement backend interface and adapt AWS/local launchers.

## Phase 5 (ongoing): Research governance hardening
- Adopt pre-commit experiment gate templates (minimum OOS coverage, required metrics).
- Add statistical significance defaults to experiment specs.

---

## Risk Register

1. **Regression risk from trainer extraction**  
   Mitigation: snapshot baseline metrics and run side-by-side A/B comparisons before deleting old paths.

2. **Config migration friction**  
   Mitigation: keep backward-compatible flag translation for at least two release cycles.

3. **Cloud/backend drift**  
   Mitigation: enforce artifact schema at fetch time; fail fast on nonconforming outputs.

4. **Long-running experiment cost during transition**  
   Mitigation: add mandatory smoke-run tier before any full run submission.

---

## Definition of Done (Refactor)

Refactor is considered complete when:
1. A single config-driven command can launch any supported training mode.
2. Every run emits validated `manifest.yaml` + schema-compliant `metrics.json`.
3. Legacy training scripts are reduced to compatibility wrappers or retired.
4. CI catches doc/test truth drift automatically.
5. Local and AWS runs are interchangeable at the configuration layer.
6. Research reports are generated from structured artifacts, not manual transcription.

---

## Immediate Next Actions

1. Draft `metrics.v1.json` and implement validator.
2. Implement `RunManifest` object and write it for current `train.py` runs.
3. Extract common training/eval/callback flow into `python/lob_rl/training/trainer.py`.
4. Create one canonical experiment config template and convert one existing experiment to it.
5. Add CI smoke pipeline that trains for a tiny step budget and validates artifacts.

This sequence gives maximum leverage quickly: provenance first, then consolidation, then enforcement.
