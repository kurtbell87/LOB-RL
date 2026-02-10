# Domain Priors -- LOB-RL

Knowledge injected by the research lead. The SURVEY and FRAME agents
MUST read this file and incorporate these priors into experiment design.

## Problem Structure

- **Instrument:** MES (Micro E-mini S&P 500 futures), CME Globex
- **Tick size:** $1.25 per tick (0.25 index points x $5 multiplier)
- **Spread:** Typically 1 tick ($1.25) during RTH; widens in overnight/low-volume
- **Contract rolls:** Quarterly. Roll calendar at `data/mes/roll_calendar.json`
- **Episodes:** Each trading day is one independent episode. No cross-day position carry.
- **Data:** 249 trading days from 2022 (bear market, S&P 500 dropped ~20%)
- **Bar aggregation:** bar_size=1000 MBO events per bar. Empirically best granularity.
- **Observation:** 21-dim bar-level (OHLCV, spread, imbalance, position, PnL, temporal)
- **Action space:** Discrete 3-action {hold, buy, sell}. Position clipped to [-1, +1].
- **Execution cost:** 1 tick ($1.25) per trade. Round-trip cost = $2.50.

## Known Architecture-Problem Mappings

- **LSTM > MLP > frame-stack** within SB3's default policy classes on the **21-dim** BarLevelEnv observation space (empirically established, pre-005 through pre-007). This ranking has NOT been validated on the 132-dim barrier feature set.
- LSTM retains more entropy, overfits less. Advantage may be regularization, not temporal learning.
- Frame-stacking actively hurts — fastest entropy collapse and worst OOS. Do not revisit.
- SB3 MlpPolicy doesn't benefit from GPU (CPU-bound). Only RecurrentPPO uses GPU.
- Best hyperparameters (21-dim system): bar_size=1000, ent_coef=0.05, lr=1e-3, policy_arch=256x256 ReLU. These may not transfer to the barrier system.

### Architecture exploration (gated on P0 supervised diagnostic)

Once the supervised diagnostic confirms signal in the 132-dim barrier features, architecture becomes a first-class experiment variable. Candidates:

1. **LSTM baseline** (SB3 RecurrentPPO MlpLstmPolicy) — current default. Processes lookback sequentially.
2. **Transformer encoder** — Self-attention over the h=10 lookback window. Can attend to any bar in the window without sequential bottleneck. Natural fit for the 13×10 feature matrix structure. Implement as a custom `features_extractor_class` in SB3.
3. **S4 / Mamba SSM** — Linear-time sequence model. Strong on long sequences but the h=10 lookback is short enough that the advantage over LSTM is unclear. Worth testing if Transformer overhead is problematic.
4. **Windowed attention + positional encoding** — Lightweight attention that encodes bar position (k, k-1, ..., k-9) explicitly. May capture "how stale is this feature" better than LSTM's implicit recency bias.

**Implementation path:** SB3's `FeaturesExtractorMixin` allows plugging a custom PyTorch `nn.Module` as the policy backbone without abandoning SB3's PPO training loop, diagnostics, checkpointing, or VecEnv infrastructure. The custom extractor receives the raw 132-dim observation and outputs a latent vector that feeds into SB3's standard actor-critic heads. This is the lowest-friction path — no need to rewrite training infrastructure.

**Anti-pattern:** Do NOT tune hyperparameters within a single architecture (e.g., LSTM hidden size, number of layers) before comparing architecture families. Architecture class is likely a higher-information-value variable than architecture hyperparameters.

## T6 Supervised Diagnostic Results (2026-02-10)

Ran on 157,040 samples from 247 sessions. 130-dim barrier features (13 × 10 lookback, 4 dead). Two diagnostic framings:

### v2 — Bidirectional framing (CORRECT)

Target: {long_profitable, short_profitable, flat}. Key insight: τ_{+} and τ_{-} are mutually exclusive (long profit precludes short profit). Distribution: 33/33/35% — nearly balanced.

| Metric | MLP (shuffle) | RF (shuffle) | MLP (chrono) | RF (chrono) |
|--------|--------------|-------------|-------------|------------|
| Accuracy | 0.393 | 0.405 | 0.391 | 0.396 |
| Balanced accuracy | 0.394 | 0.405 | 0.392 | 0.397 |
| Baseline | 0.346 | 0.346 | 0.350 | 0.350 |
| Beats baseline | **Yes (+4.8pp)** | **Yes (+5.9pp)** | **Yes (+4.2pp)** | **Yes (+4.6pp)** |

**Verdict: CONFIRMED — weak signal.** +5pp above chance, consistent across splits. P0 gate passed.

### v1 — Long-only framing (misleading, for reference)

Target: {stop, timeout, profit} from long perspective only. Distribution: 67/0.04/33% — heavily imbalanced.
MLP 61.6% vs 67.4% baseline — appeared to show no signal. This framing tests gambler's ruin outcomes, not directional predictability. Discard.

### Feature importance (RF, v2)
trade_flow_imbalance (0.128) > bar_range (0.122) > volume_log (0.121) > vwap_displacement (0.121) > body_range_ratio (0.117) > bar_body (0.115) > realized_vol (0.114) > session_time (0.089) > session_age (0.074). Dead features: 0.000.

**4/13 features dead:** bbo_imbal, depth_imbal, cancel_asym, mean_spread (need mbo_data fix in precompute). Activating these is P1.

## Anti-Patterns to Avoid

- **Don't chase in-sample returns.** Return 139.5 on 20 training days is meaningless. Only OOS matters.
- **Don't increase steps without increasing data.** 2M→5M on 20 days made things worse, not better.
- **Don't use frame-stacking.** It memorizes rather than generalizes on this problem.
- **Don't run MLP/frame-stack on GPU.** SB3 warns it's wasted. Use CPU.
- **Don't tune hyperparameters with only 20 training days.** Any result on 20 days is unreliable.
- **Don't use chronological split alone.** Shuffle-split confirmed the problem isn't regime shift — combine with chronological for robustness.
- **Don't conclude "architecture doesn't matter" from the 21-dim experiments.** The 21-dim BarLevelEnv obs is a flat vector with minimal temporal structure. The 132-dim barrier obs (13 features × 10 lookback) has explicit spatial-temporal structure that different architectures handle differently. Results don't transfer.
- **Don't build custom training infrastructure for architecture experiments.** Use SB3's `features_extractor_class` to swap backbones while keeping the existing training loop.

## Domain-Specific Guidance

- **2022 bear market regime:** Strong directional bias (down). Trend-following would have been profitable. The agent must discover this or handle regime-dependent behavior.
- **Execution cost is the central challenge.** Per-bar price moves are often < 1 tick. The agent needs to find signals worth > 2 ticks (round-trip) to be profitable.
- **VecNormalize audit needed.** Running statistics computed across training days may leak distributional information. Verify per-day vs cross-day normalization.
- **20 training days is the likely root cause.** 8% of data for training is extreme. Standard ML practice suggests 70-80% for training. The immediate priority is testing with 199 training days.
- **Reward shaping is a second-order concern.** Fix the data quantity problem first. Reward shaping on 20 days will just find a different way to overfit.
- **Python environment:** All commands use `uv`. Run with `PYTHONPATH=build-release:python uv run ...`
