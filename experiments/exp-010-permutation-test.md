# Experiment: Permutation Test — Is T6's +5pp Accuracy Real Signal or Label-Structure Artifact?

## Hypothesis

The +5pp accuracy gain observed in T6 (RF 40.5% vs 34.5% baseline on 3-class bidirectional labels) survives a permutation test: the real-label accuracy exceeds the 95th percentile of the shuffled-label accuracy distribution.

**Direction:** real accuracy > permuted accuracy distribution.
**Magnitude:** p < 0.05 (real accuracy exceeds 95% of permuted runs).

**Rationale:** T6 found +5pp accuracy above majority baseline, but exp-006 through exp-009 found BSS < 0 everywhere (calibrated probability is no better than constant). Three hypotheses explain this:

1. **H1 (no signal):** T6's +5pp is an artifact of label structure (e.g., timeout rate correlates with features). Permuting labels destroys this structure and accuracy should drop to baseline.
2. **H2 (real but weak signal):** Features contain genuine discriminative information about barrier outcomes, but it's too weak to improve calibrated probabilities. Permuting labels should destroy the accuracy gain.
3. **H3 (pipeline artifact):** T6's accuracy gain comes from data leakage or evaluation bug. Permuting labels would NOT destroy it.

H1 and H2 both predict the permutation test passes (real > permuted). The test discriminates {H1 or H2} from H3. To further distinguish H1 from H2, we add a BSS permutation test: if BSS under permuted labels is *worse* than BSS under real labels (even though real BSS < 0), that confirms real signal exists but is too weak for calibration.

## Independent Variables

**Label permutation** (2 levels): real labels vs. shuffled labels.
**N_permutations:** 100 (sufficient for p < 0.01 detection).

## Controls

- **Data:** B=500 barrier cache (`cache/barrier/`, 248 sessions, 220-dim features from C++ backend).
- **Model:** Random Forest (n_estimators=100, seed=42). Consistent with T6v2.
- **Target:** 3-class bidirectional {long_profit, short_profit, flat} from Y_long and Y_short binary labels.
- **Split:** Shuffle 80/20 (consistent with T6v2). Also chronological.
- **BSS test:** Logistic regression on Y_long (same as exp-006) under real vs. permuted labels.
- **Seed:** 42 for real run. Seeds 1..100 for permutations (each shuffles labels with a different seed).

## Metrics (ALL must be reported)

### Primary

1. **Real accuracy** (RF, shuffle-split and chronological).
2. **Permuted accuracy distribution** (100 permutations, RF, shuffle-split).
3. **Permutation p-value:** fraction of permuted runs with accuracy >= real accuracy.

### Secondary

- **Real BSS** (LR on Y_long, chronological split) — should match exp-006.
- **Permuted BSS distribution** (100 permutations, LR on Y_long).
- **BSS permutation p-value:** fraction of permuted runs with BSS >= real BSS.
- Mean and std of permuted accuracy distribution.
- Mean and std of permuted BSS distribution.

## Success Criteria (immutable once RUN begins)

- [ ] **C1 — Accuracy signal is real:** Permutation p-value < 0.05 for RF accuracy (real accuracy exceeds 95th percentile of permuted distribution).
- [ ] **C2 — BSS signal confirmed weak:** Real BSS > mean permuted BSS (even if both negative). This confirms features carry *some* information, just insufficient for calibration.
- [ ] **C3 — No pipeline artifact:** Real accuracy on permuted labels drops to within ±1pp of majority baseline.

**Verdict mapping:**
- C1 + C2 + C3 → **CONFIRMED** — signal is real but too weak for calibrated prediction. H2 confirmed. The problem is feature informativeness, not pipeline or methodology.
- C1 fails → **REFUTED** — T6's accuracy was an artifact. H3 confirmed. Past experiments may have fundamental evaluation issues.
- C1 passes but C2 fails → **INCONCLUSIVE** — accuracy signal is real but doesn't translate to any BSS advantage. Unusual; may indicate non-linear signal that LR cannot capture.

## Resource Budget

**Tier:** Quick
- Max wall-clock time: 30 minutes
- Max GPU-hours: 0 (CPU only)
- 100 permutation runs × ~2s each ≈ ~4 min for accuracy
- 100 permutation runs × ~3s each ≈ ~5 min for BSS
- No precompute needed (uses existing cache/barrier/)

## Compute Target

**Compute:** `local`
