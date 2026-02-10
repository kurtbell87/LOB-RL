# GPU Experiment Report: 5M-Step Architecture Comparison

**Date:** 2026-02-09
**Hardware:** 3x NVIDIA RTX 4090 (RunPod, US-NC-1, $0.59/hr each)
**Data:** 249 cached days (MES 2022), shuffle-split seed 42 → 20 train / 5 val / 224 test

## Summary

All three models completed 5M timesteps. All produced **negative OOS returns** — no architecture generalizes.

| Metric | LSTM (RecurrentPPO) | MLP (PPO 256x256) | Frame-Stack (PPO + VecFrameStack 4) |
|--------|--------------------|--------------------|--------------------------------------|
| **Val mean_return** | **-36.7** | -62.9 | -82.3 |
| **Test mean_return** | **-33.4** | -44.0 | -49.4 |
| **Val Sortino** | **-1.06** | -1.67 | -1.37 |
| **Test Sortino** | -1.48 | -2.22 | **-1.10** |
| **Val positive eps** | **2/5** | 0/5 | 0/5 |
| **Test positive eps** | **2/10** | 1/10 | 1/10 |
| Final entropy | -0.445 | -0.347 | -0.331 |
| Steady-state FPS | 228 | 1,275 | 1,641 |
| Wall time | 6.1 hrs | 1.1 hrs | 0.85 hrs |
| GPU cost | ~$3.60 | ~$0.65 | ~$0.50 |

**Best OOS model: LSTM** — least negative returns, only model with positive val episodes.

## OOS Performance

### LSTM wins on generalization

LSTM is the clear winner on OOS metrics despite being the slowest to train:
- **Val return -36.7 vs MLP -62.9 and frame-stack -82.3** — nearly 2x better than alternatives
- **2/5 positive val episodes** — the only model to produce any positive val days
- **Test return -33.4** — best of the three, though all are deeply negative

### All models are negative OOS

Despite 5M steps (2.5x the local 2M-step runs), no model produces positive expected returns OOS. The Sortino ratios are all well below zero (-1.06 to -2.22), confirming the agent is consistently losing money on unseen days.

## Comparison with Prior 2M-Step Local Results

| Model | Steps | Val Return | Test Return | Val Sortino | Test Sortino |
|-------|-------|-----------|-------------|-------------|--------------|
| MLP (local) | 2M | -51.5 | -62.5 | -2.09 | -1.51 |
| **MLP (GPU)** | **5M** | **-62.9** | **-44.0** | **-1.67** | **-2.22** |
| Frame-stack (local) | 2M | -48.4 | -50.2 | -1.82 | -1.08 |
| **Frame-stack (GPU)** | **5M** | **-82.3** | **-49.4** | **-1.37** | **-1.10** |
| LSTM (local) | 2M | — | — | — | — |
| **LSTM (GPU)** | **5M** | **-36.7** | **-33.4** | **-1.06** | **-1.48** |

Key observations:
- **MLP got worse on val (-51.5 → -62.9) but better on test (-62.5 → -44.0)** with 2.5x more steps. This suggests instability, not improvement — the model is oscillating, not converging.
- **Frame-stack got worse on val (-48.4 → -82.3) and similar on test (-50.2 → -49.4)**. More training made val performance significantly worse, indicating overfitting.
- **LSTM completed for the first time** (was killed at 15% locally). At -36.7 val / -33.4 test, it's the best-performing architecture but still far from profitable.

## Training Efficiency

| Model | Steady FPS | Wall Time | Speedup vs LSTM |
|-------|-----------|-----------|-----------------|
| LSTM | 228 | 6.1 hrs | 1x |
| MLP | 1,275 | 1.1 hrs | 5.6x |
| Frame-stack | 1,641 | 0.85 hrs | 7.2x |

- **MLP/Frame-stack are GPU-underutilized** — SB3 warns that MlpPolicy doesn't benefit from GPU. The FPS gains come from the RTX 4090's fast CPU side, not the GPU cores. Future runs should use `device='cpu'` to avoid paying for unused GPU.
- **LSTM is genuinely GPU-bound** — RecurrentPPO's sequential LSTM forward pass uses the GPU but at only 228 FPS, well below the 1,600+ FPS of the feedforward models.
- **Total GPU cost: ~$4.75** for all three experiments combined.

## Entropy Trajectory

All models start at -1.09 (max entropy for the 3-action space: hold/buy/sell).

| Timesteps | LSTM | MLP | Frame-stack |
|-----------|------|-----|-------------|
| 0 (start) | -1.09 | -1.09 | -1.09 |
| ~800K | -0.60 | -0.68 | -0.55 |
| ~1.6M | -0.53 | -0.52 | -0.45 |
| ~2.5M | -0.50 | -0.44 | -0.40 |
| ~3.3M | -0.49 | -0.39 | -0.36 |
| ~4.1M | -0.46 | -0.36 | -0.36 |
| 5M (end) | -0.45 | -0.35 | -0.33 |

- All models show monotonic entropy decrease (increasing policy determinism).
- **LSTM retains the most entropy (-0.45)** — it explores more throughout training.
- **Frame-stack collapses fastest** — it becomes the most deterministic policy.
- None show entropy collapse to near-zero (ent_coef=0.05 is preventing that), but the policies are becoming quite peaked.

## Diagnosis: Why Nothing Generalizes

1. **Overfitting on 20 training days.** The shuffle-split uses only 20 days for training (8% of 249). The model memorizes patterns from these specific 20 days that don't transfer to the other 229 days. More steps = more memorization, which explains why MLP/frame-stack got *worse* on val with 5M steps.

2. **LSTM's advantage is likely regularization, not temporal learning.** LSTM retains more entropy and has more parameters to distribute learning across, which acts as implicit regularization. It's less overfit, not more generalizable.

3. **The in-sample return (~139.5 at 2M steps) is illusory.** The agent learns day-specific patterns (e.g., "buy at bar 50 on day X") that are artifacts of the training set, not tradeable signals.

4. **Execution cost may be too punitive.** All returns are negative — the agent can't overcome the bid-ask spread consistently. Even random trading with execution cost would produce negative returns.

## Recommendations

### Immediate next steps

1. **Increase training days.** The 20-day training set is too small. Test with 80% train (199 days) to see if more data helps generalization. The current 8/2/90 split is extreme.

2. **Ablate execution cost.** Run one experiment without `--execution-cost` to isolate whether the agent is learning meaningful signals that are just unprofitable after costs, or learning nothing at all.

3. **Evaluate the 4M-step checkpoints.** All three models saved checkpoints at 4M steps. Compare 4M vs 5M OOS performance — if 4M is better, the model is overfitting in the last 1M steps and early stopping would help.

4. **Use CPU for MLP/frame-stack.** Add `device='cpu'` to avoid paying for GPU. Only LSTM benefits from GPU, and even then poorly.

### Medium-term investigation

5. **Reward shaping.** The current PnL-delta reward with execution cost produces sparse, noisy signal. Consider:
   - Reducing execution cost coefficient
   - Adding intermediate rewards for favorable position changes
   - Normalizing rewards per-day to reduce variance

6. **Observation audit.** Verify VecNormalize statistics aren't leaking cross-day information. Check if the running mean/variance is computed across all days or per-day.

7. **Action space redesign.** The 3-action discrete space (hold/buy/sell) may be too coarse. Consider a continuous position target or a 5-action space (strong buy/buy/hold/sell/strong sell).

### If nothing works

8. **Supervised pre-training.** Train a classifier to predict next-bar price direction as a diagnostic. If the classifier also fails, the observation space lacks predictive signal and the problem is with the features, not the RL algorithm.

## Infrastructure Note

The automated result fetch (`monitor.sh` → `fetch-results.sh`) failed for all three pods due to an SSH connectivity issue. Results were recovered manually from the RunPod persistent volume (`4w2m8hek66`) using a recovery pod. The fetch script should be hardened to retry on SSH failure and avoid removing pods before confirming successful download.

## Raw Data

### Final training diagnostics (iteration 306, ~5M steps)

| Metric | LSTM | MLP | Frame-stack |
|--------|------|-----|-------------|
| approx_kl | 0.148 | 0.019 | 0.037 |
| clip_fraction | 0.311 | 0.118 | 0.146 |
| explained_variance | 0.983 | 0.953 | 0.989 |
| value_loss | 0.0097 | 0.033 | 0.0074 |
| policy_gradient_loss | -0.042 | -0.012 | -0.017 |

- **LSTM has high KL divergence (0.148) and clip fraction (0.311)** — the policy is changing aggressively between updates. This is concerning and suggests the learning rate may be too high for LSTM.
- **All models have very high explained variance (>0.95)** — the value function fits the training data extremely well, consistent with overfitting.
- **Value loss is near zero** — the critic has memorized the training returns.
