# Market Microstructure Features for Barrier-Hitting Probability Estimation

## Problem Framing

You're training a model as a function approximator to P(tau_plus) and P(tau_minus) -- the probabilities that price hits an upper or lower barrier at a 2:1 R:R ratio within some time horizon, conditioned on the current microstructure state. This is fundamentally a **first-passage-time problem with two absorbing barriers**, and the key insight is that the microstructure state at bar formation time carries substantial information about the *asymmetry* of these hitting probabilities -- far more than trade-level features alone.

Your current feature set (trade features, VWAP, etc.) captures what *has happened*. The LOB features below capture what *is about to happen* -- the latent supply/demand structure that governs the path-dependent dynamics between your barriers.

---

## Tier 1: High-Value LOB Features (Implement First)

### 1. Order Flow Imbalance (OFI) -- Multi-Level

The single most predictive microstructure feature in the literature. Cont, Kukanov & Stoikov (2014) established a near-linear relationship between OFI and short-horizon price changes, with R^2 ~65% for short intervals. Critically, **OFI subsumes trade imbalance** -- when both are included as regressors, trade imbalance becomes redundant.

**Computation per bar:**

For consecutive LOB snapshots n-1 and n, compute bid/ask order flows:

```
OFI_n = delta_V_bid(n) - delta_V_ask(n)
```

Where delta_V captures volume additions, cancellations, and executions at each level. Aggregate across your bar:

```
OFI_bar = sum OFI_n  (over all updates in the bar)
```

**Multi-level extension (Kolm et al. 2023):** Compute OFI at each of your 10 depth levels separately. Kolm et al. showed that models trained on OFI significantly outperform models trained directly on raw order book states. Their key finding: the effective prediction horizon is approximately **two average price changes** -- relevant for calibrating your tau horizons.

**Dimensionality reduction:** The first principal component of multi-level OFIs explains >89% of total variance. Consider using PCA-integrated OFI rather than all 10 levels raw to avoid overfitting while preserving the information.

### 2. Static Order Book Imbalance (OBI) -- L1 through L10

```
rho_t = (V_bid - V_ask) / (V_bid + V_ask),  rho in [-1, 1]
```

Compute at L1 (BBO only) and across all 10 levels. This is the "worst-kept secret of high-frequency trading" per Pulido & Lehalle -- when rho -> 1, mid-price is likely to jump up; when rho -> -1, likely down.

**Per-bar aggregation options:**
- Snapshot at bar close
- Time-weighted average across bar
- EMA-smoothed (reduces spoofing noise)
- Min/max/range within bar (captures transient extremes)

**For your tau problem specifically:** The asymmetry in OBI maps almost directly to asymmetry in barrier-hitting probability. A strongly positive OBI means the ask side is thin relative to the bid -- price has less resistance traveling upward. This is structural information about *potential future trades*, not lagged.

### 3. Micro-Price (Stoikov 2017)

The micro-price is constructed as the limit of expected future mid-prices conditional on the current book state (spread + imbalance). It is a **martingale by construction**, unlike the mid-price or weighted mid-price.

```
Micro-price = f(spread, imbalance) adjusted mid-price
```

Where f is estimated from historical data using a Markov chain on (spread, imbalance) states.

**Why this matters for tau_plus/tau_minus:** The deviation of micro-price from mid-price gives you a directional bias estimate that's theoretically grounded. If micro-price > mid-price, the book structure implies upward pressure -- P(tau_plus) should be elevated.

**Features to extract:**
- Micro-price - mid-price (signed deviation)
- Micro-price - VWAP (deviation from volume-weighted fair value)
- Rate of change of micro-price within bar

### 4. Volume-Adjusted Mid Price (VAMP)

```
VAMP_bbo = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)
```

Extend to N levels:
```
VAMP_N = (sum P_bid_i * Q_ask_i + sum P_ask_i * Q_bid_i) / (sum Q_bid_i + sum Q_ask_i)
```

Note the cross-multiplication -- bid prices are weighted by *ask* quantities and vice versa. This captures where price would move if the *opposing* side were consumed.

**For barrier estimation:** VAMP_N at progressively deeper levels (3, 5, 10) gives you a "cost of traversal" curve -- how much volume needs to be consumed to move price through each depth stratum. The shape of this curve directly informs how far price can travel before encountering resistance.

---

## Tier 2: Liquidity & Friction Features

### 5. Depth Profile Features

Beyond simple imbalance, the *shape* of the depth profile matters:

- **Cumulative depth ratio at K ticks:** `sum Q_bid(1..K) / sum Q_ask(1..K)` for K = 1, 3, 5, 10
- **Depth concentration:** What fraction of total visible liquidity sits at L1? High concentration = fragile book, large moves possible
- **Depth gradient:** Is liquidity increasing or decreasing away from BBO? Increasing = strong support/resistance; decreasing = vulnerable to sweep
- **Liquidity holes:** Gaps in the book (empty price levels) create potential for rapid price jumps

**Barrier relevance:** Your upper barrier (2R) requires price to traverse some number of ticks. The cumulative depth on the ask side between current price and that barrier is a direct estimate of "volume required to reach target." Similarly for the bid side and your stop (R).

### 6. Spread Dynamics

- **Spread (ticks or bps):** Wider spread -> higher friction, affects both barriers symmetrically
- **Spread volatility within bar:** Unstable spreads indicate uncertain liquidity conditions
- **Relative spread:** Spread / recent average spread -- identifies regime shifts
- **Effective spread:** Based on actual trades vs. mid-price, captures hidden costs

### 7. Trade Arrival & Aggression Features

These complement your existing trade features with temporal structure:

- **Trade arrival rate (lambda):** Trades per unit time, captures activity intensity
- **Buy/sell trade arrival rate ratio:** lambda_buy / lambda_sell -- directional aggression
- **Trade size distribution moments:** Mean, variance, skewness of trade sizes within bar
- **Aggressor imbalance:** (Buy-initiated volume - Sell-initiated volume) / Total volume
- **Large trade indicator:** Count/volume of trades > K * median trade size -- proxy for informed flow

---

## Tier 3: Information Asymmetry Features

### 8. VPIN (Volume-Synchronized Probability of Informed Trading)

Developed by Easley, Lopez de Prado & O'Hara. Measures the probability that order flow is dominated by informed traders adversely selecting market makers.

**Computation:**
1. Partition trading volume into equal-sized buckets (not time buckets -- volume buckets)
2. Within each bucket, classify buy/sell volume using Bulk Volume Classification (BVC): use the normalized price change within the bucket to probabilistically assign volume
3. VPIN = mean(|V_buy - V_sell|) / V_bucket over a rolling window of N buckets

**Key nuances:**
- Use volume bars (which you already have as trade bars), not time bars
- BVC is preferred over tick-rule classification at high frequency
- The bucket size is a critical hyperparameter -- too small = noise, too large = lag

**For tau_plus/tau_minus:** High VPIN indicates informed traders are active. This has two effects: (a) increased volatility, making *both* barriers more reachable, and (b) directional information, making one barrier more likely. VPIN predicts toxicity-induced volatility, not direction per se, but combined with OFI it becomes powerful. Research shows VPIN's predictive capacity spikes during already-volatile markets.

### 9. Kyle's Lambda (Price Impact Coefficient)

```
delta_P = lambda * signed_volume + epsilon
```

Estimated via rolling regression within your bar or over recent bars. Higher lambda = less liquidity = more price impact per unit volume = larger expected price excursions in both directions.

**Features:**
- lambda magnitude (overall illiquidity)
- lambda asymmetry: estimate separately for buy-initiated and sell-initiated trades. Asymmetric lambda implies directional information -- if buy impact > sell impact, buyers are more informed
- lambda trend: increasing lambda may predict regime change

### 10. Hasbrouck's Information Share

Decomposes price variance into components attributable to trades vs. quotes. The trade-information component indicates how much of price discovery is happening via aggressive orders rather than passive quote updates. Higher information share from trades suggests more informed activity -- relevant for assessing whether current flow is "smart" or "noise."

---

## Tier 4: Temporal & Structural Features

### 11. Book Resilience / Recovery Rate

After a trade consumes liquidity at the BBO, how quickly does the book replenish?

- **Time to refill:** Seconds until consumed level is restored to pre-trade depth
- **Refill ratio:** Depth at BBO 100ms/500ms/1s after a trade vs. pre-trade depth
- **Asymmetric resilience:** Does the bid side recover faster than the ask after a sell, or vice versa?

Fast resilience = strong support, price less likely to continue moving through that side. This directly informs barrier-hitting asymmetry.

### 12. Quote-to-Trade Ratio & Cancellation Rate

- **QTR:** Number of order book updates / number of trades in bar
- **Cancellation rate:** Cancelled volume / total volume submitted

High QTR with high cancellation = "flickering" quotes, possibly spoofing or HFT probing. The literature suggests filtering these out improves OBI signal quality. Low QTR = genuine, stable liquidity.

### 13. Uncertainty Zones (Robert & Rosenbaum 2011)

The "uncertainty zone" parameter eta captures mean-reversion tendency:

```
eta = N_continuations / (N_continuations + N_alternations)
```

Where continuations = mid-price moves in same direction as previous move, alternations = reversals.

- eta < 0.5: Mean-reverting (alternation-dominant) -- makes barriers harder to reach
- eta > 0.5: Trending (continuation-dominant) -- makes barriers easier to reach
- This is directly relevant to your 2:1 R:R calculation: in a trending regime, P(tau_plus) and P(tau_minus) are both elevated but the directional one dominates; in mean-reverting, both are suppressed

---

## Feature Engineering: Connecting to tau_plus and tau_minus

### Asymmetry Features (Most Directly Predictive)

The core question is: *what makes P(tau_plus) != P(tau_minus)?* Build explicit asymmetry features:

1. **Depth-weighted path resistance:**
   - `R_up = sum Q_ask(i) * (P_ask(i) - mid)` for levels up to your target
   - `R_down = sum Q_bid(i) * (mid - P_bid(i))` for levels down to your stop
   - Feature: `log(R_up / R_down)` -- negative means less resistance upward

2. **Asymmetric OFI:** Decompose OFI into positive and negative components, track their magnitudes separately

3. **Directional VAMP gradient:** How does VAMP change as you go deeper into the book? Compute `VAMP_5 - VAMP_1` and `VAMP_10 - VAMP_5` -- this captures whether deeper liquidity supports or undermines the BBO signal

4. **Barrier-aware depth:** Specifically compute cumulative depth from current price to your exact barrier levels, not just at generic level counts

### Interaction Features

- **OBI * Spread:** Imbalance is more informative when spread is tight (implies committed liquidity)
- **VPIN * OFI:** Informed flow + directional bias = strongest signal
- **Depth concentration * Trade arrival rate:** Concentrated fragile book + high activity = high probability of barrier breach
- **Kyle's lambda asymmetry * OBI:** Asymmetric impact + asymmetric book = strong directional signal

### Normalization & Stationarity

Critical for ML: raw LOB features are non-stationary across instruments and time.

- **Z-score against rolling window:** For all volume-based features, normalize by recent mean/std (e.g., 500-bar rolling)
- **Rank normalization:** Convert to percentile within recent history
- **Volume representation (Lucchese et al. 2024):** Represent order book as cumulative volume at each price level rather than price/volume pairs -- shown to be more robust and have practical advantages for DL models
- **Log-transform asymmetric features:** log(bid_depth / ask_depth) rather than raw difference

---

## Architecture Considerations

Given your problem structure (probabilistic outputs for two barrier events), consider:

1. **DeepLOB-style architecture** (Zhang et al. 2019): CNN for spatial LOB structure -> LSTM for temporal dynamics. This handles the 10-level depth naturally as a spatial dimension.

2. **Multi-task output:** Predict P(tau_plus) and P(tau_minus) jointly -- they share most of the feature representation but diverge at the output layer. A shared encoder with two heads.

3. **Calibrated probabilities:** Use temperature scaling or Platt scaling post-training. Your outputs need to be well-calibrated probabilities, not just rankings. Consider proper scoring rules (Brier score, log loss) rather than accuracy metrics.

4. **Bayesian extension:** DeepLOB with dropout variational inference (Bayesian BDLOB) provides uncertainty estimates on P(tau_plus) and P(tau_minus) -- useful for position sizing and for knowing when the model is uncertain.

---

## Key References

- **Cont, Kukanov & Stoikov (2014)** -- OFI and price impact. Foundational.
- **Kolm, Turiel & Westray (2023)** -- "Deep Order Flow Imbalance." Multi-level OFI + DL. State of the art.
- **Stoikov (2017)** -- Micro-price construction. Martingale fair price estimator.
- **Easley, Lopez de Prado & O'Hara (2012)** -- VPIN and flow toxicity.
- **Zhang, Zohren & Roberts (2019)** -- DeepLOB architecture.
- **Robert & Rosenbaum (2011)** -- Uncertainty zones and mean-reversion estimation.
- **Lucchese et al. (2024)** -- Volume representation for LOB, robustness advantages.
- **Briola et al. (2024)** -- LOBFrame: links microstructural properties to DL forecasting success.
- **Pulido & Lehalle (2023)** -- Endogenous price-imbalance connection via market making models.

---

## Implementation Priority

Given you're working with trade/tick bars and have depth to 10 levels:

**Phase 1 (immediate lift):**
- Multi-level OBI (snapshot at bar close + EMA-smoothed average)
- OFI aggregated across bar (L1 + integrated via PCA across 10 levels)
- Spread (absolute + relative to recent average)
- Depth-weighted path resistance to actual barrier levels

**Phase 2 (strong incremental value):**
- Micro-price deviation from mid and VWAP
- VAMP at levels 1, 5, 10
- Aggressor imbalance + large trade indicators
- Depth concentration and gradient features

**Phase 3 (diminishing but real returns):**
- VPIN (requires careful bucket calibration)
- Kyle's lambda and its asymmetry
- Book resilience metrics
- Uncertainty zone parameter eta
- Cancellation rates and QTR

The features in Phase 1 alone should meaningfully improve your model's ability to discriminate between P(tau_plus) and P(tau_minus), because they capture the structural asymmetry in the LOB that trade features alone cannot see.
