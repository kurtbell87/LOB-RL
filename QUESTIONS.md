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
- Custom neural architectures beyond SB3's built-in policies

---

## 4. Open Questions

| Priority | Question | Status | Parent | Blocker | Decision Gate |
|----------|----------|--------|--------|---------|---------------|
| P0 | Does increasing training data from 20 to 199 days fix OOS generalization? | Not started | — | — | If positive OOS: scale up. If still negative: problem is not data quantity. |
| P0 | Is the agent learning signal that is masked by execution cost? | Not started | — | — | If OOS positive without exec cost: agent learns signal but can't overcome spread. If still negative: no signal learned. |
| P1 | Are 4M-step checkpoints better than 5M (late-stage overfitting)? | Not started | — | — | If 4M > 5M OOS: implement early stopping. If equal: overfitting is gradual, not sudden. |
| P1 | Does VecNormalize leak cross-day information? | Not started | — | — | If leak found: fix normalizer. If clean: rule out this confounder. |
| P2 | Can reward shaping improve OOS returns? | Not started | P0, P0 | Resolve P0 questions first | If improved OOS: adopt new reward. If not: problem is deeper than reward design. |
| P3 | Does the observation space contain any predictive signal? | Not started | — | — | If supervised classifier > 55% accuracy: signal exists, RL is failing to exploit. If ~50%: features lack predictive power. |

---

## 5. Answered Questions

| Question | Answer Type | Answer | Evidence |
|----------|-------------|--------|----------|
| Does chronological vs shuffle split explain OOS failure? | REFUTED | No — shuffle-split also negative (val -51.5, test -62.5 for MLP). Agent genuinely isn't generalizing. | pre-003-local-mlp in RESEARCH_LOG.md |
| Do more training steps (2M→5M) help? | REFUTED | No — made things worse for MLP (val -51.5→-62.9) and frame-stack (-48.4→-82.3). More steps = more memorization on 20 days. | pre-005 through pre-007 in RESEARCH_LOG.md |
| Which architecture generalizes best? | CONFIRMED | LSTM (RecurrentPPO) is least overfit (val -36.7, 2/5 positive val eps) but still deeply negative. | pre-005-gpu-lstm in RESEARCH_LOG.md |
| Does frame-stacking help OOS? | REFUTED | No — worst val performance (-82.3), fastest entropy collapse. Memorizes rather than generalizes. | pre-007-gpu-framestack in RESEARCH_LOG.md |

---

## 6. Working Hypotheses

- **H1: Data quantity is the primary bottleneck.** 20 training days (8% of 249) is far too few. Increasing to 199 days should reduce overfitting.
- **H2: The agent learns weak signal masked by execution cost.** OOS returns might be positive without execution cost, indicating the spread is too large relative to the signal strength.
- **H3: LSTM's advantage is regularization, not temporal learning.** LSTM retains more entropy and distributes learning across more parameters, acting as implicit regularization.
- **H4: Late-stage training is harmful.** The 4M checkpoint may outperform the 5M checkpoint, suggesting early stopping would help.
