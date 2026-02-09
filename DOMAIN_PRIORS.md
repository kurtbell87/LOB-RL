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

- **LSTM > MLP > frame-stack** for this problem (empirically established, pre-005 through pre-007)
- LSTM retains more entropy, overfits less. Advantage may be regularization, not temporal learning.
- Frame-stacking actively hurts — fastest entropy collapse and worst OOS.
- SB3 MlpPolicy doesn't benefit from GPU (CPU-bound). Only RecurrentPPO uses GPU.
- Best hyperparameters: bar_size=1000, ent_coef=0.05, lr=1e-3, policy_arch=256x256 ReLU

## Anti-Patterns to Avoid

- **Don't chase in-sample returns.** Return 139.5 on 20 training days is meaningless. Only OOS matters.
- **Don't increase steps without increasing data.** 2M→5M on 20 days made things worse, not better.
- **Don't use frame-stacking.** It memorizes rather than generalizes on this problem.
- **Don't run MLP/frame-stack on GPU.** SB3 warns it's wasted. Use CPU.
- **Don't tune hyperparameters with only 20 training days.** Any result on 20 days is unreliable.
- **Don't use chronological split alone.** Shuffle-split confirmed the problem isn't regime shift — combine with chronological for robustness.

## Domain-Specific Guidance

- **2022 bear market regime:** Strong directional bias (down). Trend-following would have been profitable. The agent must discover this or handle regime-dependent behavior.
- **Execution cost is the central challenge.** Per-bar price moves are often < 1 tick. The agent needs to find signals worth > 2 ticks (round-trip) to be profitable.
- **VecNormalize audit needed.** Running statistics computed across training days may leak distributional information. Verify per-day vs cross-day normalization.
- **20 training days is the likely root cause.** 8% of data for training is extreme. Standard ML practice suggests 70-80% for training. The immediate priority is testing with 199 training days.
- **Reward shaping is a second-order concern.** Fix the data quantity problem first. Reward shaping on 20 days will just find a different way to overfit.
- **Python environment:** All commands use `uv`. Run with `PYTHONPATH=build-release:python uv run ...`
