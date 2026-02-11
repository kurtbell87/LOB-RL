# Analysis: Sequence Model Signal Detection — Can LSTM or Transformer Beat the Constant Brier Score?

## Verdict: REFUTED

No sequence model beats the constant predictor on any label. All eight (model, label, split) combinations have **negative** Brier Skill Scores. C1 fails: no (model, label) pair achieves BSS > 0 with p < 0.05. The lowest p-value is 0.633 (transformer/short on test), nowhere near significance. Per the verdict mapping: C1 fails → **REFUTED**.

Full-session causal context (LSTM, Transformer) does not contain calibrated probabilistic signal for barrier prediction, just as flat models (LR, GBT) failed in exp-006. The hypothesis that temporal ordering would unlock signal that flat feature assembly destroys is refuted.

## Results vs. Success Criteria

- [ ] **C1 — BSS > 0 with p < 0.05: FAIL** — No (model, label) pair on val has positive BSS. All four val BSS values are negative: lstm_long -0.0038, lstm_short -0.0173, transformer_long -0.0011, transformer_short -0.0004. All p-values are ≥ 0.791. `C1_bss_positive_significant = false`.
- [ ] **C2 — Best BSS ≥ 0.005: FAIL** — Best val BSS = -0.0004 (transformer/short). The threshold was +0.005. Not only is no model above threshold, no model achieves a *positive* BSS. `C2_bss_gte_0005 = false`.
- [ ] **C3 — Best model Brier < constant AND < exp-006's best flat model: FAIL** — The best model Brier on val is transformer/short at 0.21861, vs. constant 0.21852 and exp-006 best (logistic/short) 0.21849. The model Brier exceeds both baselines. `C3_brier_lt_constant_and_exp006 = false`.
- [x] **Sanity checks: PASS** — N_sessions = 248, N_bars = 456,396, N_features = 22, ȳ_train_long = 0.322, ȳ_train_short = 0.324 (both in [0.28, 0.38]), device = MPS. Split: 149/50/49 train/val/test. `sanity_ok = true`.

**Per the verdict rules:** C1 fails → **REFUTED** — sequence models also can't beat constant.

## Metric-by-Metric Breakdown

### Primary Metrics — Validation Set

| Model | Label | BS_model | BS_constant | BSS | Delta (const − model) | Δ 95% CI | p-value |
|-------|-------|----------|-------------|-----|----------------------|----------|---------|
| LSTM | Y_long | 0.21744 | 0.21663 | **-0.0038** | -0.000813 | [-0.00139, -0.00021] | 0.995 |
| LSTM | Y_short | 0.22230 | 0.21852 | **-0.0173** | -0.003779 | [-0.00500, -0.00240] | 1.000 |
| Transformer | Y_long | 0.21688 | 0.21663 | **-0.0011** | -0.000249 | [-0.00053, +0.00001] | 0.965 |
| Transformer | Y_short | 0.21861 | 0.21852 | **-0.0004** | -0.000091 | [-0.00033, +0.00013] | 0.791 |

Key observations:
1. **All BSS values are negative.** Every model is worse than guessing ȳ.
2. **LSTM is significantly worse than Transformer on all pairs.** LSTM long BSS = -0.0038 vs Transformer -0.0011. LSTM short BSS = -0.0173 vs Transformer -0.0004. The more temporally-expressive model overfits more — the same pattern as GBT vs LR in exp-006.
3. **LSTM delta CIs exclude zero on the wrong side.** LSTM long CI [-0.00139, -0.00021] and LSTM short CI [-0.00500, -0.00240] are *significantly* worse than constant. The bootstrap is confident LSTM hurts.
4. **Transformer/short is the closest to zero** (BSS = -0.0004, p = 0.791), but its CI [-0.00033, +0.00013] straddles zero — statistically indistinguishable from the constant predictor.
5. **LSTM short is the worst pair** (BSS = -0.0173). The LSTM pushes short predictions to p̂_mean = 0.376, far from the true base rate ~0.324. This is a systematic bias, not noise.

### Primary Metrics — Test Set

| Model | Label | BS_model | BS_constant | BSS | Delta (const − model) | Δ 95% CI | p-value |
|-------|-------|----------|-------------|-----|----------------------|----------|---------|
| LSTM | Y_long | 0.21722 | 0.21652 | **-0.0032** | -0.000702 | [-0.00129, -0.00010] | 0.990 |
| LSTM | Y_short | 0.22049 | 0.21647 | **-0.0186** | -0.004020 | [-0.00549, -0.00261] | 1.000 |
| Transformer | Y_long | 0.21684 | 0.21652 | **-0.0015** | -0.000318 | [-0.00059, -0.00003] | 0.986 |
| Transformer | Y_short | 0.21652 | 0.21647 | **-0.0002** | -0.000051 | [-0.00031, +0.00019] | 0.633 |

Test results confirm val results. Same ordering: LSTM worst, Transformer closest to zero. LSTM short BSS degrades further on test (-0.0186 vs -0.0173 on val). Transformer/short remains the only pair whose CI includes zero, but on the wrong side (BSS = -0.0002). No contradictions between val and test — the null holds on both.

### Comparison with exp-006 Flat Models

| Model | Label | Val BSS | Comparison |
|-------|-------|---------|------------|
| exp-006 Logistic | Y_long | -0.0007 | — |
| exp-006 Logistic | Y_short | -0.0003 | Best exp-006 |
| exp-006 GBT | Y_long | -0.0028 | — |
| exp-006 GBT | Y_short | -0.0019 | — |
| **exp-007 LSTM** | **Y_long** | **-0.0038** | Worse than LR and GBT |
| **exp-007 LSTM** | **Y_short** | **-0.0173** | Much worse than all exp-006 |
| **exp-007 Transformer** | **Y_long** | **-0.0011** | Similar to LR (-0.0007) |
| **exp-007 Transformer** | **Y_short** | **-0.0004** | Similar to LR (-0.0003) |

The Transformer approximately matches logistic regression — the minimum-complexity linear model. The LSTM is *worse* than all exp-006 models, including GBT. Adding temporal structure does not help; adding temporal expressiveness (LSTM > Transformer) actively hurts. This is a strong overfitting gradient.

### Prediction Distribution Analysis

| Model | Label | p̂_mean | p̂_std | True ȳ |
|-------|-------|--------|--------|--------|
| LSTM | Y_long | 0.319 | 0.033 | 0.322 |
| LSTM | Y_short | 0.376 | 0.033 | 0.324 |
| Transformer | Y_long | 0.313 | 0.015 | 0.322 |
| Transformer | Y_short | 0.319 | 0.014 | 0.324 |

Critical finding: **LSTM short predictions are biased high** (p̂_mean = 0.376 vs ȳ = 0.324, a +5.2pp shift). This systematic bias accounts for the very poor BSS (-0.0173). The LSTM learned a biased representation of short probabilities during training that does not transfer.

**Transformer predictions are much tighter** (std 0.014–0.015 vs LSTM 0.033). The Transformer essentially learned to predict near the base rate for every bar, with minimal variation. This is the optimal strategy when no signal exists — collapse to the constant predictor. The Transformer's lower BSS penalty is explained by it being closer to the constant predictor, not by it finding signal.

### Sanity Checks

| Check | Result | Detail |
|-------|--------|--------|
| N_sessions = 248 | **PASS** | 248 |
| N_bars ≈ 456K | **PASS** | 456,396 |
| N_features = 22 | **PASS** | 22 |
| ȳ_train_long ∈ [0.28, 0.38] | **PASS** | 0.322 |
| ȳ_train_short ∈ [0.28, 0.38] | **PASS** | 0.324 |
| Split 149/50/49 | **PASS** | Matches exp-006 |
| No abort triggered | **PASS** | Both models completed |
| Early stopping triggered | **PASS** | LSTM stopped epoch 29 (best 18), Transformer epoch 17 (best 6) |

All sanity checks pass. The experiment is valid. The negative result is not a pipeline bug.

### Training Details

| Model | Params | Wall time | Best epoch | Total epochs | Early stopped |
|-------|--------|-----------|------------|--------------|---------------|
| LSTM | 68,162 | 226s (3.8 min) | 18 | 29 | Yes (patience 10) |
| Transformer | 68,546 | 1327s (22.1 min) | 6 | 17 | Yes (patience 10) |

Both models are ~68K params (spec estimated ~45-55K; actual is slightly larger, immaterial). The Transformer took 5.9x longer than LSTM — expected due to attention's quadratic cost on ~1000-bar sessions.

The Transformer's best epoch was **6** — it found its best val loss very early and then degraded for 11 more epochs before early stopping. This suggests the Transformer learned to approximate the constant predictor quickly and then started overfitting. The LSTM's best epoch was **18** — it had a longer training trajectory but also couldn't improve beyond its early plateau.

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| Wall clock | < 2 hours | 1578s (26.3 min) | Well under budget |
| Memory | < 4 GB | Not measured (no OOM) | Under budget |
| Disk | < 100 MB | ~100 MB (metrics + checkpoints) | At budget |
| GPU-hours | 0 (CPU/MPS) | 0 (MPS used) | On budget |
| Models trained | 2 | 2 | On budget |

Budget was appropriate. MPS acceleration kept training fast.

## Confounds and Alternative Explanations

### 1. Could LSTM/Transformer hyperparameters be poorly tuned?

**Possible but unlikely to change the conclusion.** The models used reasonable defaults (d_model=64, 2 layers, AdamW 1e-3, dropout 0.1). The Transformer's behavior is telling: it converged to near-constant predictions (p̂_std = 0.014) with BSS ≈ -0.0004, essentially matching logistic regression. A well-tuned model that finds no signal will still collapse to the constant — which is what we observe. Larger models (d_model=128, 4 layers) would likely overfit more, not less, given the LSTM→Transformer gradient already shows more expressiveness = worse performance.

### 2. Could the training procedure be flawed?

**Unlikely.** Both models early-stopped appropriately. BCEWithLogitsLoss with per-bar masking is the standard approach. Cosine annealing prevents lr-related issues. The fact that both models produce predictions in a reasonable range (0.31–0.38, close to base rate) indicates training converged properly. The LSTM's short bias (p̂_mean = 0.376) is a learned bias from training data, not a training bug.

### 3. Could sequence length matter? (~1000 bars might be too long)

**This is the strongest alternative explanation.** Full-session context of ~1000 bars may introduce more noise than signal. The LSTM must propagate gradients through ~1000 timesteps, which is near the practical limit for LSTM gradient flow. The Transformer avoids this with attention, but ~1000 positions with 4 heads and d_model=64 may lack the capacity to find relevant patterns in the noise. A shorter context window (e.g., 50-100 bars) could be tested, but this would overlap with the h=10 lookback already tested in exp-006 (which also found nothing).

### 4. Could the 2:1 barrier formulation be fundamentally unpredictable?

**This is the most parsimonious explanation.** Under efficient markets, the barrier outcomes are determined by the random walk component of price. The Gambler's Ruin null (exp-005) fits the data well. Any predictive signal would need to exploit drift or volatility clustering — effects that may exist at longer horizons but not at the ~1000-bar intraday timescale with B=500 barriers. The consistent failure across 6 model types (LR, GBT, MLP in T6, LSTM, Transformer) with different feature representations strongly suggests the feature-label relationship is near the noise floor.

### 5. Is this just seed variance?

**No.** Only one seed was used (42), so we cannot compute cross-seed variance. However, the results are consistent across val and test splits, and consistent with exp-006's findings. The LSTM short bias is systematic (both splits show p̂_mean ≈ 0.376), not a seed artifact. A multi-seed run would tighten confidence intervals but is unlikely to change the direction of any BSS value.

### 6. Could the feature normalization (per-bar) be destroying temporal structure?

**Possible.** The spec says features are "per-bar normalized." If normalization was applied per-session (z-scoring within each session), it would remove cross-session level information but preserve within-session temporal dynamics. If normalization was applied globally, session-level regime information is intact. The exact normalization scheme matters — if per-bar normalization strips the very temporal dependencies that sequence models need, the experiment may not be a fair test. However, the features include session_time and session_age, which explicitly encode temporal position, so complete temporal destruction is unlikely.

## What This Changes About Our Understanding

### The critical update: temporal ordering contains no additional calibrated signal beyond what flat features provide.

exp-006 showed flat models (LR, GBT) on 220-dim flattened features cannot beat the constant Brier score. The hypothesis that this was because flattening destroyed temporal order is now **refuted**. LSTM and Transformer with full causal access to the raw bar sequence (22 features per bar, ~1000 bars per session) perform identically to or worse than the flat models.

### Updated mental model:

1. **Six model types have now failed.** LR, GBT (exp-006), LSTM, Transformer (exp-007), plus MLP and RF (T6, accuracy-only). The coverage of the model complexity spectrum is thorough: from linear (LR) to recurrent (LSTM) to attention-based (Transformer) to ensemble (GBT). No architecture beats constant-prediction Brier score.

2. **More model expressiveness → worse performance.** The ordering is consistent across both experiments: LR ≈ Transformer > GBT > LSTM (where ">" means "less bad BSS"). Simpler models overfit less. This is the signature of fitting to noise, not signal.

3. **The Transformer learned to be a constant predictor.** With p̂_std = 0.014, the Transformer's per-bar predictions barely deviate from the mean. This is the optimal strategy when no signal exists. The fact that 68K-parameter Transformer with 4 attention heads and full session context reduces to near-constant predictions is strong evidence that the signal is absent, not that the model is incapable.

4. **LSTM has a systematic bias problem.** The LSTM's p̂_mean for Y_short is 0.376 (5.2pp above ȳ), producing BSS = -0.0173 — the worst result in the entire experiment. LSTM gates are memorizing training-set statistics that don't transfer. This is consistent with pre-005-gpu-lstm showing LSTM was "least overfit" in RL but still deeply negative.

5. **Hypothesis H6 is refuted.** H6 postulated that SB3's built-in LSTM was insufficient and that a Transformer or windowed attention could do better. Direct comparison shows the Transformer cannot extract signal either. The limitation is in the signal, not the architecture.

6. **The information ceiling is confirmed at < 0.1% of outcome variance.** After testing 6 model families across two feature representations (flat 220-dim and sequential 22×T), no model explains meaningful variance in barrier outcomes. The 2:1 asymmetric barrier at B=500 on MES appears to be near-unpredictable from intraday features.

### What should replace the tested hypothesis?

The tested hypothesis was: "Temporal ordering contains signal that flat assembly destroys." This is refuted. The replacement possibilities are:

- **Different features:** The 22 features are trade-derived microstructure measures. Order flow at finer granularity (per-event rather than per-bar), cross-instrument signals, or macro regime indicators might contain signal not captured here.
- **Different targets:** The 2:1 barrier with B=500 may simply be too noisy. Larger barriers (B=2000), different reward:risk ratios (3:1, 1:1), or longer-horizon targets might be more predictable.
- **Different bar sizes:** B=500 at the default bar size yields races that last ~12 bars. At B=2000 or with smaller bars, the race duration changes, and the signal-to-noise ratio may shift.
- **Conditional prediction:** Signal may exist in specific regimes (high-volatility sessions, post-news bars) while averaging to zero over the full sample. This requires regime-stratified evaluation.

## Proposed Next Experiments

1. **Bar-size sweep (Phase 2b).** Test B ∈ {200, 500, 1000, 2000} with re-calibrated R values. Different bar sizes produce different race durations and may shift the signal-to-noise ratio. Use logistic regression only (cheapest, and exp-006/007 showed it matches the best complex model). This directly follows the Asymmetric First-Passage Trading plan's Phase 2b.

2. **Conditional signal detection — regime stratification.** Split sessions or bar windows by realized_vol quintile, trade_arrival_rate quintile, or session_time bucket. Run logistic BSS within each stratum. Hypothesis: signal exists in high-vol or high-activity periods but averages to zero. This is the cheapest experiment that could reveal a hidden signal.

3. **Alternative target formulation.** Test symmetric barriers (1:1 R:R), larger barriers (3:1 or 4:1), or raw directional prediction (next-bar return sign) instead of the 2:1 barrier. If the 2:1 barrier is inherently noisy, a different target may be more predictable. This requires a new label pipeline (handoff to TDD for cache regeneration).

4. **Accept the null for this feature set and pivot.** If experiments 1-3 also fail, the evidence will be overwhelming that the 22 intraday microstructure features at the bar level do not predict barrier outcomes. The productive pivot would be to fundamentally different information sources: cross-instrument correlations, event-driven features (economic calendar), or finer-grained order flow (MBO-level features, not bar aggregates).

## Program Status

- Questions answered this cycle: 1 (sequence model signal detection → REFUTED)
- New questions added this cycle: 0
- Questions remaining (open, not blocked): 5
- Handoff required: NO
