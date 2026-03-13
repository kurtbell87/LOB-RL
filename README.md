# LOB-RL: Microstructure Signal Research for Futures Markets

An end-to-end research platform for investigating whether intraday microstructure features extracted from tick-level order book data carry exploitable predictive signal in CME futures (MES / Micro E-mini S&P 500). Spans the full pipeline from raw market-by-order (MBO) data ingestion and LOB reconstruction through feature engineering, signal detection across linear-to-deep-learning models, and RL-based strategy backtesting.

## Motivation

The central question isn't "which model works best" — it's **"is there signal at all?"** Most quantitative trading research jumps straight to model fitting and evaluates on in-sample returns, making it impossible to separate learned signal from overfitting. This project inverts the order: rigorously test for predictive signal using proper scoring rules and information-theoretic bounds *before* committing to any modeling approach.

The result is a systematic narrowing of the hypothesis space — 13 controlled experiments, each with pre-committed success criteria and immutable metrics, producing a clear map of where signal does and does not exist in MES tick-level microstructure.

## Research Framework: Asymmetric First-Passage Trading

The prediction target is framed as a **first-passage problem**: given the current bar of aggregated MBO events, will price hit a reward barrier (+2R) before a risk barrier (-R)?

Under the martingale null with 2:1 reward-to-risk, Gambler's Ruin gives a closed-form base rate of **P = 1/3**. This provides:

- **A testable null** — empirically confirmed at P = 0.320 / 0.322 on 249 trading days (454K bars)
- **A proper scoring rule** — Brier Skill Score against the constant predictor, avoiding the pitfalls of accuracy on imbalanced classes
- **An information ceiling** — the variance decomposition Var(P) = Var(Y) - E[P(1-P)] bounds the maximum achievable improvement
- **A direct profitability gate** — for a trade to overcome round-trip costs C at risk unit R, the model must predict p > (1 + C/R) / 3

### Research Phases

```
Phase 0: Data Pipeline        ── raw MBO → LOB reconstruction → tick bars → first-passage labels
Phase 1: Null Calibration     ── verify ȳ ≈ 1/3                                   ✓ CONFIRMED
Phase 2: Signal Detection     ── do features beat constant Brier?                  ✓ COMPLETE (6 model families)
Phase 2b: Parameter Sweep     ── bar size × risk calibration grid                  ← CURRENT
Phase 3: Feature Engineering   ── information ceiling estimation, group ablation
Phase 4: Sequence Modeling     ── temporal context via Transformer / SSM
Phase 5: Ceiling Analysis      ── Var(f̂) / Var(P) ratio → ship, iterate, or stop
```

## Key Findings

### Signal Detection: Linear through Deep Learning

The core Phase 2 question — **do microstructure features carry calibrated predictive signal?** — was tested across six model families spanning the full complexity spectrum:

| Model | Type | Brier Skill Score | Verdict |
|-------|------|------------------|---------|
| Logistic Regression | Linear | -0.0003 (best) | No signal |
| Gradient-Boosted Trees | Nonlinear, tabular | -0.0028 | Worse than constant |
| Random Forest | Ensemble | +5pp balanced accuracy, but BSS < 0 | Classification gain ≠ calibration |
| MLP [256, 256] | Deep, feedforward | BSS < 0 | No signal |
| LSTM (RecurrentPPO) | Deep, sequential | BSS -0.0173 | Temporal context hurts |
| Transformer (causal) | Deep, attention | BSS -0.0004 | Collapses to near-constant |

**Key insight:** Random Forest achieves +5pp balanced accuracy (40.5% vs 34.5% baseline) — the features weakly separate classes — but this does not translate to calibrated probability improvement under Brier scoring. The information content is < 0.1% of outcome variance. More expressive models perform *worse* (noise fitting).

### Full Experiment Log (13 experiments, 2 confirmed / 11 refuted)

| # | Experiment | Hypothesis | Result |
|---|-----------|-----------|--------|
| 1 | Null calibration | ȳ ≈ 1/3 under martingale null | **Confirmed** (0.320 / 0.322) |
| 2 | Supervised diagnostic | 22 features contain directional signal | **Confirmed** (+5pp accuracy) |
| 3 | Brier signal detection | LR / GBT beat constant Brier | **Refuted** (all BSS negative) |
| 4 | Sequence models | LSTM / Transformer with causal context achieves BSS > 0 | **Refuted** |
| 5 | 10x training data | 199 days (vs 20) reduces OOS gap | **Refuted** (eliminates memorization, OOS unchanged) |
| 6 | Execution cost ablation | Positive gross returns masked by transaction costs | **Refuted** (gross still negative) |
| 7 | Architecture sweep | LSTM > MLP > frame-stack on 222-dim features | **Refuted** (expressiveness → noise fitting) |
| 8 | LOB fix impact | 4-bug LOB correction changes MES features | **Refuted** (CME MBO has 0 IsTob / 0 partial cancels) |
| 9-13 | RL agent variations | Various PPO configurations achieve positive OOS | **All refuted** |

11 of 13 hypotheses were refuted. Each refutation is a concrete finding about the structure of MES microstructure at tick-bar resolution:

- **Data quantity is not the bottleneck** — 10x more training data eliminates memorization but doesn't improve OOS
- **Transaction costs explain ~70% of OOS loss** — but even gross returns are slightly negative
- **Temporal ordering adds nothing** — sequential models (LSTM, Transformer) perform no better than logistic regression
- **CME futures MBO is clean** — unlike equity exchanges, CME/GLBX has no IsTob flags and no partial cancels, so common LOB reconstruction bugs are irrelevant

## End-to-End Data Pipeline

### Raw Data → Features

```
312 .mbo.dbn.zst files (57 GB, MES 2022)
         │
         ▼
┌─────────────────────────────────────────────────┐
│  C++ LOB Reconstruction (Constellation engine)   │
│  Full MBO replay: Add, Cancel, Modify, Trade     │
│  Validated: 0 mismatches on 7.5M BBO checks     │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Tick Bars    Book State   First-Passage
    (B events    snapshots    Labels (±R
     per bar)    per bar      barrier race)
         │           │           │
         └───────────┼───────────┘
                     ▼
         22 microstructure features × h-bar lookback
         → 222-dim observation vector per bar
                     │
         ┌───────────┼───────────────┐
         ▼           ▼               ▼
    Supervised   RL Training     Experiment
    Diagnostics  (PPO/LSTM)      Pipeline
    (BSS, cal.)                  (immutable specs)
```

### Futures-Specific Details

- **Instrument:** MES (Micro E-mini S&P 500), CME Globex, $1.25 tick size
- **Data period:** Full 2022 (249 trading days, bear market — S&P dropped ~20%)
- **Contract rolls:** Quarterly (MESH2 → MESM2 → MESU2 → MESZ2), front-month filtered via roll calendar
- **Bar aggregation:** Deterministic tick bars (B ∈ {200, 500, 1000, 2000} MBO events per bar)
- **Session handling:** Pre-market warmup → RTH trading → forced position flatten at close. Each day is an independent episode.
- **Execution economics:** 1 tick ($1.25) per trade, round-trip $2.50. At B=500, typical per-bar moves are < 1 tick — the cost structure is the central challenge.

## Tick-Data Feature Engineering

22 features per bar, extracted from full-depth MBO replay:

| Feature | Source | What It Captures |
|---------|--------|-----------------|
| OHLCV + signed body | Trade tape | Intra-bar price dynamics and direction |
| bar_range, realized_vol | Trade tape | Intra-bar volatility at tick resolution |
| vwap_displacement | Trade tape | (VWAP - mid) / tick — volume-weighted directional pressure |
| trade_flow_imbalance | Trade tape (Lee-Ready) | Signed aggressor flow: (buy_vol - sell_vol) / total_vol |
| body_range_ratio | Trade tape | \|body\| / range — conviction measure |
| OFI | Book depth | Order flow imbalance: net change in bid vs ask depth |
| depth_ratio | Book depth | Best-level depth asymmetry |
| weighted_mid_displacement | Book depth | Depth-weighted mid vs raw mid — supply/demand tilt |
| VAMP_displacement | Book depth | Volume-adjusted mid-price divergence from mid |
| spread_std | Book depth | Intra-bar spread volatility — liquidity stability |
| aggressor_imbalance | Trade tape | Net aggressive order flow direction |
| trade_arrival_rate | Trade tape | Trades per unit time — activity intensity |
| cancel_to_trade_ratio | Book + Trade | Order cancellation rate relative to execution — toxicity proxy |
| price_impact_per_trade | Trade tape | Average price move per trade — market impact |
| session_time, session_age | Metadata | Temporal position in session (intraday seasonality) |

Features are stacked over a lookback window (default h=10 bars), yielding 22h + 2 = 222 dimensions. All feature computation runs in the C++ backend (~50-100x faster than the Python equivalent).

## LOB Reconstruction Engine

Full limit order book rebuilt from raw MBO messages via Constellation's `LimitOrderBook`, with four correctness fixes validated against Databento's reference implementation:

1. **IsTob flag handling** — clears the full side and inserts a synthetic level (equity exchange convention)
2. **Partial cancel** — subtracts the cancel message's size, not the original order's full size
3. **Modify-as-add** — routes modify messages to the add handler when order ID is unknown (exchange sequence gaps)
4. **Trade/Fill as no-ops** — trade messages don't modify book state; fills are reconciled by subsequent cancel/modify messages

Validated against Databento reference: **0 mismatches** on 161K BBO checks (DBEQ equities) and **7.5M BBO checks** (MES futures). The validation uncovered that CME/GLBX MBO data contains 0 IsTob adds and 0 partial cancels across 8.4M messages — these are equity exchange conventions that don't appear in futures data.

## Technical Implementation

| Layer | Stack | Details |
|-------|-------|---------|
| **LOB engine** | C++17 | Full-depth order book from MBO, handles Add/Cancel/Modify/Trade. Constellation CQRS engine. |
| **Bar + feature pipeline** | C++17 | Deterministic tick bars, 22-feature extraction, first-passage labeling. Processes 249 days in ~10 min (8 workers). |
| **Python bindings** | pybind11 | `lob_rl_core`: Book, OrdersEngine, BatchBacktestEngine, MarketBook, feature compute, barrier precompute. |
| **Signal diagnostics** | scikit-learn | LR, RF, GBT. Brier score, BSS, calibration curves, permutation importance, bootstrap significance. |
| **RL environments** | Gymnasium | `BarLevelEnv` (21-dim), `BarrierEnv` (222-dim), `OrderSimulationEnv`. Lazy-loading multi-session wrapper. |
| **RL training** | Stable-Baselines3 | PPO, RecurrentPPO (LSTM). VecNormalize, SubprocVecEnv, checkpointing, shuffle-split. |
| **Compute** | AWS EC2 Spot | GPU (g5.xlarge A10G) for LSTM, CPU (c7a.4xlarge) for MLP. S3 cache/results, ECR images. Spot interruption → checkpoint upload. |
| **Testing** | Catch2 + pytest | **2,906 tests** (660 C++ + 2,246 Python). Strict TDD — all features built via red-green-refactor cycle. |

## Experiment Methodology

Every experiment follows a strict **SURVEY → FRAME → RUN → READ → LOG** protocol:

1. **SURVEY** — review prior experiments and known failure modes
2. **FRAME** — spec with falsifiable hypothesis and pre-committed success criteria (**spec becomes immutable**)
3. **RUN** — execute and write metrics to `results/` (**metrics become immutable**)
4. **READ** — render verdict: CONFIRMED, REFUTED, or INCONCLUSIVE
5. **LOG** — commit to research log with full provenance

This prevents the most common failure mode in quantitative research: adjusting hypotheses or success criteria after seeing results. The experiment log is an honest record — 11 of 13 hypotheses were refuted, and each negative result constrained the next experiment's design.

## Repository Structure

```
src/
├── barrier/          # C++ barrier pipeline: bar construction, feature extraction, labeling
├── constellation/    # Constellation order book engine (LimitOrderBook, OrdersEngine, CQRS)
├── engine/           # Book class wrapping Constellation's LOB
├── env/              # C++ RL environment (LOBEnv, reward calculation)
├── data/             # Data sources: DbnFileSource (.dbn.zst), BinaryFileSource, SyntheticSource
└── bindings/         # pybind11 bindings exposing C++ pipeline to Python
python/lob_rl/
├── barrier/          # Python barrier pipeline: labels, features, env, training diagnostics
├── config/           # Environment and training configuration
├── orchestration/    # Multi-session wrappers, VecEnv helpers, SB3 compatibility
├── bar_level_env.py  # 21-dim bar-level Gymnasium env
├── order_sim_env.py  # Constellation-backed order simulation env
└── _reward.py        # Shared reward computation (PnL, execution cost, forced flatten)
scripts/
├── train.py          # CLI training: PPO / RecurrentPPO with full config
├── train_barrier.py  # Barrier-specific training entry point
└── precompute_*.py   # Cache generation from raw .dbn.zst
experiments/          # Experiment specs (immutable after FRAME)
results/              # Experiment metrics and analysis (immutable after RUN)
```

## Current Status and Open Directions

**Phase 2 complete.** 13 experiments across 6 model families establish that single-bar microstructure features at B=500 do not carry calibrated predictive signal for MES first-passage outcomes. The research frontier is:

1. **Bar-size sweep** — B ∈ {200, 1000, 2000} with R recalibrated per bar scale. Signal may exist at different aggregation horizons.
2. **Conditional signal detection** — stratify by realized volatility regime, session time, or spread state. Signal may be present in specific market conditions even if unconditionally absent.
3. **Extended features** — cross-bar momentum, full depth profile, arrival-rate dynamics, cross-instrument signals.
4. **Alternative targets** — direct mid-price return prediction, regime classification, execution quality metrics rather than first-passage outcomes.
5. **Cross-instrument validation** — extend to ES, NQ, or FX futures to test whether findings are MES-specific or reflect broader CME microstructure properties.

See the full [research plan](experiments/Asymmetric%20First-Passage%20Trading.md) for the complete phase specification and decision tree.
