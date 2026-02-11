# exp-007: Sequence Model Signal Detection

## Question

Do sequence-aware models (LSTM, Transformer) with full-session causal context achieve positive Brier Skill Score on barrier prediction, where flat models (LR, GBT) in exp-006 failed?

## Hypothesis

At least one sequence model (LSTM, Transformer) with full-session causal context achieves BSS > 0 on Y_long or Y_short, using the same temporal split and Brier evaluation framework as exp-006.

## Rationale

exp-006 REFUTED flat models (LR, GBT) on 220-dim lookback-assembled features — all BSS negative. But:
- T6 confirmed weak discriminative signal (+5pp accuracy)
- GPU LSTM was best OOS in RL experiments (val -36.7 vs MLP -62.9)
- The 220-dim features flatten temporal ordering; sequence models preserve it
- Full-session context (~1000 bars) may capture dependencies that h=10 lookback misses

## Setup

### Data
- **Source:** `cache/barrier/` — 248 sessions with `bar_features` key
- **Features:** Per-bar normalized features `(T_i, 22)` float32 per session
- **Labels:** Y_long = (label_values == 1), Y_short = (short_label_values == -1)
- **Split:** 60/20/20 temporal (same as exp-006): ~149 train / 50 val / 49 test

### Models
1. **BarrierLSTM:** 2-layer LSTM, d_model=64, dropout=0.1. ~45K params.
2. **BarrierTransformer:** 2-layer causal encoder, 4 heads, d_model=64, ff=128, dropout=0.1. ~55K params.

### Training
- Epochs: 50 (with early stopping, patience=10)
- Batch size: 4 sessions
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- LR schedule: Cosine annealing
- Loss: BCEWithLogitsLoss per bar, masked for padding, summed over Y_long + Y_short
- Seed: 42

### Evaluation
- Brier score on val and test splits
- BSS = 1 - Brier(model) / Brier(constant)
- Paired bootstrap CI (1000 resamples) for significance
- Same metrics format as exp-006

## Success Criteria (pre-committed)

- **C1:** At least one (model, label) pair on val has BSS > 0 with bootstrap p < 0.05
- **C2:** Best BSS on val >= 0.005
- **C3:** Best model's val Brier < constant Brier AND < exp-006's best flat model Brier

## Verdict Rules

- C1 + C2 + C3 → **CONFIRMED** — temporal structure contains calibrated signal
- C1 pass, C2 fail → **INCONCLUSIVE** — significant but tiny
- C1 fail → **REFUTED** — sequence models also can't beat constant

## Compute

compute: local
instance_type: cpu (MPS if available)

Models are ~50K params. Sessions are ~1000 bars. Estimated ~10-20 min per model. Total < 1.5 hours.

## Resource Budget

- Time: < 2 hours wall clock
- Memory: < 4 GB
- Disk: < 100 MB (metrics + model checkpoints)

## Abort Criteria

- Any model takes > 45 min → abort that model, report what completed
- OOM → reduce batch_size to 2, retry once
- All models produce NaN loss → abort experiment, report infrastructure issue
