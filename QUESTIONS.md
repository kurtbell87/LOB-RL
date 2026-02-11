# Research Questions -- LOB-RL

## 1. Goal

Determine whether a reinforcement learning agent can profitably trade MES (Micro E-mini S&P 500) futures using limit order book data, and identify the key factors preventing out-of-sample generalization.

**Success looks like:** Positive mean episode return on both val and test sets with execution cost enabled. Sortino ratio > 0 on OOS data.

---

## 2. Constraints

| Constraint | Decision |
|------------|----------|
| Framework  | SB3 (PPO) / sb3-contrib (RecurrentPPO), PyTorch backend |
| Compute    | RunPod RTX 4090 ($0.59/hr, 24GB VRAM). Local Apple Silicon for MLP/frame-stack. |
| Timeline   | Ongoing |
| Baselines  | In-sample return 139.5 (bar=1000, ent=0.05, lr=1e-3). OOS baseline: all negative. |

---

## 3. Non-Goals (This Phase)

- Multi-instrument trading (only MES)
- Live/paper trading integration
- Alternative RL algorithms beyond PPO family (SAC, DQN, etc.)
- Multi-year data (focus on 2022 first)
- Custom neural architectures beyond SB3's built-in policies BEFORE the supervised diagnostic (T6) confirms signal in the 132-dim barrier feature set. Once signal is confirmed, architecture comparison is a first-class experiment variable.

---

## 4. Open Questions

| Priority | Question | Status | Parent | Blocker | Decision Gate |
|----------|----------|--------|--------|---------|---------------|
| P0 | Does the observation space contain any predictive signal? | **CONFIRMED (weak)** | — | — | T6 v2 (bidirectional framing): MLP 39.3% / RF 40.5% vs 34.5% baseline on balanced {long, short, flat}. Signal is +5pp above chance, consistent on chrono split. v1 (long-only) was misleading due to imbalanced 67/33/0 class distribution. 4/13 features still dead — book features may add more signal. Gate passed: proceed to RL. |
| P0 | Does 199d + no exec cost produce positive OOS with sufficient training (10M+ steps)? | Not started | — | — | If positive OOS persists at convergence: strong signal + data interaction. If collapses: 2M result was noise. |
| P1 | Are 4M-step checkpoints better than 5M (late-stage overfitting)? | Not started | — | — | If 4M > 5M OOS: implement early stopping. If equal: overfitting is gradual, not sudden. |
| P1 | Does VecNormalize leak cross-day information? | Not started | — | — | If leak found: fix normalizer. If clean: rule out this confounder. |
| P2 | Can reward shaping improve OOS returns? | Not started | P0, P0 | Resolve P0 questions first | If improved OOS: adopt new reward. If not: problem is deeper than reward design. |
| P1 | Does model architecture (Transformer / SSM / LSTM) significantly affect OOS on the 132-dim barrier features? | Not started | — | P0 supervised diagnostic must confirm signal | If architecture X beats LSTM OOS by >5pp: adopt X. If all within noise: LSTM is fine, problem is elsewhere. Determines whether to invest in custom policy infrastructure. |

---

## 5. Answered Questions

| Question | Answer Type | Answer | Evidence |
|----------|-------------|--------|----------|
| Does chronological vs shuffle split explain OOS failure? | REFUTED | No — shuffle-split also negative (val -51.5, test -62.5 for MLP). Agent genuinely isn't generalizing. | pre-003-local-mlp in RESEARCH_LOG.md |
| Do more training steps (2M→5M) help? | REFUTED | No — made things worse for MLP (val -51.5→-62.9) and frame-stack (-48.4→-82.3). More steps = more memorization on 20 days. | pre-005 through pre-007 in RESEARCH_LOG.md |
| Which architecture generalizes best? | CONFIRMED | LSTM (RecurrentPPO) is least overfit (val -36.7, 2/5 positive val eps) but still deeply negative. | pre-005-gpu-lstm in RESEARCH_LOG.md |
| Does frame-stacking help OOS? | REFUTED | No — worst val performance (-82.3), fastest entropy collapse. Memorizes rather than generalizes. | pre-007-gpu-framestack in RESEARCH_LOG.md |
| Is the agent learning signal masked by execution cost? | REFUTED | No — even without exec cost, OOS returns are slightly negative (val -4.43, test -5.03) on 20 training days. Exec cost accounts for ~35 points of OOS loss (val -39.55 → -4.43) but doesn't flip the sign. The gap to profitability is ~5 points gross, not ~50 net. Cross-eval shows no-cost policy is impractical under real costs (val -85.22). | exp-002-execution-cost-ablation in RESEARCH_LOG.md |
| Does increasing training data from 20 to 199 days fix OOS generalization? | REFUTED | No — LSTM 199d val -59.95 (threshold -16.7), MLP 199d val -75.53 ≈ MLP 20d val -75.82. 199 days eliminates memorization (expl_var 0.30 vs 0.97) but OOS is unchanged. Data quantity is not the primary bottleneck. Seed sensitivity extreme (37-point val swing). | exp-001 in RESEARCH_LOG.md |
| Is ȳ_long ≈ ȳ_short ≈ 1/3 under the asymmetric barrier null? | CONFIRMED | Yes — ȳ_long = 0.320, ȳ_short = 0.322, both within [0.28, 0.38]. sum_ȳ = 0.643 ≈ 2/3. P(1,1) = 0. Labels are independent and well-calibrated. Constant Brier score baseline = 0.218. | exp-005-null-calibration in RESEARCH_LOG.md |

---

## 6. Working Hypotheses

- **H1: ~~Data quantity is the primary bottleneck.~~ REFUTED (exp-001).** 199 days eliminates memorization but does not improve OOS. MLP val is identical at 20d and 199d despite 5x lower explained_variance. Replaced by H5.
- **H2: ~~The agent learns weak signal masked by execution cost.~~ REFUTED (exp-002).** Exec cost explains ~35 points of OOS loss but removing it doesn't produce positive returns at 20d. Updated: the gap to gross profitability is ~5 points, not ~50. Data + no-exec-cost interaction (199d, Run C val +10.93 undertrained) is the most promising lead.
- **H3: LSTM's advantage is regularization, not temporal learning.** LSTM retains more entropy and distributes learning across more parameters, acting as implicit regularization.
- **H4: Late-stage training is harmful.** The 4M checkpoint may outperform the 5M checkpoint, suggesting early stopping would help.
- **H5: ~~The trade-derived features lack predictive signal.~~ PARTIALLY REFUTED (T6 v2).** Bidirectional framing shows weak but real signal: MLP 39.3% / RF 40.5% vs 34.5% baseline. The v1 diagnostic (long-only) was misleading — proper framing reveals +5pp above chance. Signal is small and regularization gap is massive (90% train → 39% test). 4/13 features still dead. Updated: features have SOME signal, but activating book features and improving generalization are priorities.
- **H6: SB3's built-in LSTM is insufficient for the 132-dim barrier feature set.** The 13-feature × 10-lookback observation has spatial structure across the lookback window (features at position k-1 are semantically different from k-5) that a flat LSTM processes sequentially but a Transformer or windowed attention could process in parallel. This is untested — gated on supervised diagnostic.
