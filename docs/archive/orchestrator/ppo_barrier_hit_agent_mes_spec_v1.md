# PPO Barrier-Hit Agent for /MES: Formal Specification

**Instrument:** Micro E-mini S&P 500 (/MES) — CME Globex  
**Data source:** Databento MBO, 2022–2023  
**Training window:** 2022-01-01 to 2023-12-31 (RTH only)  
**Tick size:** 0.25 index points ($1.25 per tick), uniform across all price levels  
**Session hours:** RTH 8:30 AM – 3:00 PM CT (exclude Globex maintenance 4:00–5:00 PM CT)

-----

## 1. Data Pipeline

### 1.1 Trade Bar Construction

Raw MBO messages from Databento are consumed and matched trades are extracted. Trades are aggregated into fixed-count bars.

**Bar definition:** A bar `k` completes when `N` matched trades have accumulated since the previous bar closed. `N` is a hyperparameter.

| Field       | Definition                                 |
|-------------|--------------------------------------------|
| `O_k`       | Price of the first trade in bar `k`        |
| `H_k`       | Maximum trade price in bar `k`             |
| `L_k`       | Minimum trade price in bar `k`             |
| `C_k`       | Price of the last trade in bar `k`         |
| `V_k`       | Total volume (contracts) traded in bar `k` |
| `VWAP_k`    | Volume-weighted average price over bar `k` |
| `t_start_k` | Timestamp of first trade in bar `k`        |
| `t_end_k`   | Timestamp of last trade in bar `k`         |

**Session boundary rule:** A bar must not straddle RTH open/close or the Globex maintenance window. If bar `k` is incomplete at 3:00 PM CT, discard it. A new bar begins fresh at 8:30 AM CT the next session.

**Default parameterization:** `N = 500` trades/bar. Sweep range: `N ∈ {200, 500, 1000, 2000}`.

### 1.2 Trade Sequence Retention

Although the agent observes bar-level aggregates, the underlying trade sequence within each bar must be retained for label construction (Section 2.2, intrabar tiebreaking). Store the ordered vector of trade prices `(p_1, p_2, ..., p_N)` for each bar in a sidecar structure, indexed by bar number. This data is used offline during labeling only — never at inference time.

-----

## 2. Label Construction (Barrier Events)

### 2.1 Barrier Definition

At bar `k` (the candidate entry bar), define the reference price as `C_k` (close of bar `k`). Set symmetric-ratio barriers:

```
Upper barrier:  U = C_k + a    (target, in ticks)
Lower barrier:  D = C_k - b    (stop, in ticks)
Timeout:        T_max bars     (maximum holding period)
```

**Default parameterization:** `a = 20` ticks (5.0 index points), `b = 10` ticks (2.5 index points), `T_max = 40` bars. The 2:1 target/stop ratio implies a base-rate win probability of `b / (a + b) = 1/3` under a driftless random walk.

### 2.2 Barrier-Hit Detection (High/Low with Intrabar Tiebreaking)

For each future bar `j > k`, check the bar high and low against the barriers:

```
Upper hit on bar j:  H_j >= U    (i.e., H_j - C_k >= a)
Lower hit on bar j:  L_j <= D    (i.e., L_j - C_k <= -b)
```

Define stopping times:

```
τ⁺ = inf{j > k : H_j >= U}
τ⁻ = inf{j > k : L_j <= D}
τ  = min(τ⁺, τ⁻, k + T_max)
```

**Tiebreaking rule (dual barrier hit within a single bar):**

If on some bar `j`, both `H_j >= U` and `L_j <= D`, then both barriers were breached within the same bar. Resolve by inspecting the retained intrabar trade sequence `(p_1, ..., p_N)` for bar `j`:

1. Scan trades in order.
2. The first trade that crosses either barrier determines which barrier was hit first.
3. **Gap-through edge case:** If the first trade of the bar already exceeds both barriers (possible on session opens after overnight moves if extending beyond RTH, effectively impossible within RTH on /MES), resolve by gap direction: gap up from previous close → upper hit, gap down → lower hit. Do not label as timeout — the gap itself is directional information.

This is exact because Databento MBO provides the full trade sequence with ordering.

**Tiebreak frequency diagnostic:** After labeling, compute the fraction of labels that required tiebreaking. If this exceeds 5%, the barriers are too tight relative to bar width `N`. Either widen barriers or increase `N`.

### 2.3 Three-Outcome Label

```
H_k =  +1   if τ⁺ < τ⁻  and  τ⁺ <= k + T_max     (upper barrier hit first)
H_k =  -1   if τ⁻ < τ⁺  and  τ⁻ <= k + T_max     (lower barrier hit first)
H_k =   0   if min(τ⁺, τ⁻) > k + T_max             (timeout — neither hit)
```

The map `φ_k: Ω → {+1, -1, 0}` is `G_{k+T_max}`-measurable. It depends on future bars but is computed offline. It is a label, not an observation.

### 2.4 T_max Calibration

`T_max` should be set to the 95th percentile of the **winner** distribution (bars where `H_k = +1`), not the overall distribution. Rationale: losses resolve faster than wins due to asymmetric barrier distances (`b < a`). Setting `T_max` to the overall 95th percentile penalizes patient winning trades that need more time to reach the wider target.

**Procedure:**

1. Run label construction on the full training set with `T_max = ∞` (no timeout).
2. Compute the empirical distribution of `τ⁺ - k` for all bars where `H_k = +1`.
3. Set `T_max = ceil(P95 of this distribution)`.
4. Re-run label construction with the calibrated `T_max`.

Plot the histograms of time-to-resolution separately for wins (`H_k = +1`) and losses (`H_k = -1`). The loss distribution should be concentrated well below `T_max`.

-----

## 3. Observation Space (Bar-Level Features)

The agent's observation at bar `k` is a feature vector `Z_k ∈ R^d`. Features are computed from bar `k` and a lookback window of `h` bars. All features are normalized to zero mean and unit variance using rolling statistics from the training set.

### 3.1 Feature Definitions

**Priority 1 — Trade flow (highest expected signal for short-horizon /MES prediction):**

| # | Feature              | Definition                                                                                                                                             | Dim |
|---|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| 1 | Trade flow imbalance | `(aggressive_buy_volume - aggressive_sell_volume) / total_volume` over bar `k`. Aggressive side determined by trade initiator flag from Databento MBO. | 1   |

**Priority 2 — Order book state:**

| # | Feature                 | Definition                                                                               | Dim |
|---|-------------------------|------------------------------------------------------------------------------------------|-----|
| 2 | BBO imbalance           | `bid_size_L1 / (bid_size_L1 + ask_size_L1)` sampled at bar close                        | 1   |
| 3 | Depth imbalance (top 5) | `sum(bid_size_L1:L5) / (sum(bid_size_L1:L5) + sum(ask_size_L1:L5))` sampled at bar close | 1   |

**Priority 3 — Bar structure:**

| # | Feature           | Definition                                          | Dim |
|---|-------------------|-----------------------------------------------------|-----|
| 4 | Bar range         | `(H_k - L_k)` in ticks                              | 1   |
| 5 | Bar body          | `(C_k - O_k)` in ticks (signed)                     | 1   |
| 6 | Body/range ratio  | `(C_k - O_k) / (H_k - L_k)` if range > 0, else 0   | 1   |
| 7 | VWAP displacement | `(C_k - VWAP_k) / (H_k - L_k)` if range > 0, else 0 | 1   |
| 8 | Volume            | `V_k` (log-transformed)                              | 1   |

**Priority 4 — Regime and context:**

| #  | Feature                 | Definition                                                                      | Dim |
|----|-------------------------|---------------------------------------------------------------------------------|-----|
| 9  | Trailing realized vol   | `std(log(C_j / C_{j-1}))` for `j ∈ [k-19, k]` (20-bar rolling)                | 1   |
| 10 | Normalized session time | `(t_end_k - RTH_open) / (RTH_close - RTH_open)`, range [0, 1]                  | 1   |

**Priority 5 — Book dynamics:**

| #  | Feature               | Definition                                                                     | Dim |
|----|-----------------------|--------------------------------------------------------------------------------|-----|
| 11 | Cancel rate asymmetry | `(bid_cancels - ask_cancels) / total_cancels` over bar `k`                     | 1   |
| 12 | Mean spread           | Average `(best_ask - best_bid)` in ticks, sampled at each trade within bar `k` | 1   |

**Priority 6 — Session boundary:**

| #  | Feature     | Definition                                                     | Dim |
|----|-------------|----------------------------------------------------------------|-----|
| 13 | Session age | `min(bars_since_session_open / 20, 1.0)`, continuous in [0, 1] | 1   |

**Total observation dimension:** `d = 13` per bar. With a lookback window of `h` bars, the full observation is `Z_k = (features_k, features_{k-1}, ..., features_{k-h+1})` with dimension `13 * h`.

**Default parameterization:** `h = 10` (lookback of 10 bars), giving `d_total = 130` (plus 2 for position state = 132).

**v2 features (explicitly deferred):** Order lifetime tracking, queue position dynamics, inter-market signals (e.g., /ES lead-lag). These require cross-bar state management and are not worth the implementation/debug cost for v1.

### 3.2 Normalization

All features are z-score normalized using a 2000-bar trailing window of means and standard deviations, computed from training data only. At inference time, use the most recent 2000 bars of live data for normalization statistics.

Clip normalized features to `[-5, +5]` to prevent outlier contamination (e.g., fat-finger trades, CME halt/resume events).

**Session boundary normalization leak:** At session open, the trailing window includes bars from prior session(s). Overnight regime changes (e.g., post-hours Fed announcement shifting /MES 50 points) make those statistics stale. The first 10-20 bars of the new session will be normalized against the wrong distribution. Rather than resetting statistics (which discards useful data), the session age feature (#13) lets the agent learn to discount early-session observations without discarding them.

-----

## 4. Environment Contract

### 4.1 Episode Structure

One episode = one RTH trading session (8:30 AM – 3:00 PM CT). The episode terminates when the last bar of the session completes. If the agent holds a position at episode end, force-close at mark-to-market (treat as timeout).

### 4.2 State

The environment state at step `k` consists of:

```
s_k = (Z_k, Z_{k-1}, ..., Z_{k-h+1},  position_k,  unrealized_pnl_k)
```

Where:

- `Z_k` = bar-level feature vector (Section 3)
- `position_k ∈ {-1, 0, +1}` = current position (short, flat, long)
- `unrealized_pnl_k` = `position_k * (C_k - entry_price)` in ticks, or 0 if flat

### 4.3 Action Space and Masking

Discrete action space `A = {long, short, flat, hold}`, indexed as `{0, 1, 2, 3}`.

The 4-action space is deliberately chosen over reinterpreting `flat` as `hold` when in a position. A 3-action space where `flat` means different things depending on position state is a bug factory — any refactor of the masking logic risks accidentally letting `flat` trigger a voluntary exit. The extra action index costs nothing and makes the semantics unambiguous: `flat` always means "choose not to enter", `hold` always means "maintain current position until barrier resolution".

**Masking rules:**

| Current position | Valid actions       | Rationale                                             |
|------------------|---------------------|-------------------------------------------------------|
| Flat (0)         | {long, short, flat} | Agent decides whether and in which direction to enter |
| Long (+1)        | {hold}              | Exit is mechanical — barrier or timeout only          |
| Short (-1)       | {hold}              | Exit is mechanical — barrier or timeout only          |

### 4.4 Transition and Reward

**Entry (from flat):**

When `position_k == 0` and the agent selects `a_k ∈ {long, short}`:

```
position_{k+1} = +1 if long, -1 if short
entry_price = C_k
hold_counter = 0
reward_k = 0  (no immediate reward on entry)

For long entry at C_k:
    profit_barrier = C_k + a    (price goes UP to win)
    stop_barrier   = C_k - b    (price goes DOWN to lose)
    Win check:  H_j >= profit_barrier
    Loss check: L_j <= stop_barrier

For short entry at C_k:
    profit_barrier = C_k - a    (price goes DOWN to win)
    stop_barrier   = C_k + b    (price goes UP to lose)
    Win check:  L_j <= profit_barrier
    Loss check: H_j >= stop_barrier
```

Note: `unrealized_pnl = position * (C_k - entry_price)` is correct for both directions because `position = -1` for shorts, which flips the sign automatically. Verify the implementation doesn't double-negate.

**Holding:**

At each bar `j` while holding:

```
hold_counter += 1

Check barrier breach using H_j, L_j:

If profit barrier hit:
    reward_j = +G - C
    position_{j+1} = 0

If stop barrier hit:
    reward_j = -L - C
    position_{j+1} = 0

If both hit (dual breach):
    Resolve via intrabar trade sequence (Section 2.2)

If hold_counter == T_max:
    reward_j = position * (C_j - entry_price) / b - C    (mark-to-market in barrier units)
    position_{j+1} = 0

Otherwise:
    reward_j = 0
    position unchanged
```

**Staying flat:**

When `position_k == 0` and agent selects `flat`:

```
reward_k = 0
position_{k+1} = 0
```

### 4.5 Reward Definitions

| Symbol | Definition                                      | Default                     |
|--------|-------------------------------------------------|-----------------------------|
| `G`    | Profit reward (normalized)                      | `+2.0`                      |
| `L`    | Loss penalty (normalized)                       | `+1.0`                      |
| `C`    | Total transaction cost (normalized)             | `0.2` (see breakdown below) |
| `MTM`  | `position * (C_τ - entry_price) / b` at timeout | Varies                      |

Rewards are normalized to barrier units: `G = a/b = 2.0`, `L = b/b = 1.0`. This makes the reward scale invariant to the absolute barrier sizes in ticks.

**Transaction cost breakdown:** Round-trip cost on /MES consists of spread cost (~1 tick = $1.25) plus commissions (~$0.50-0.70/side, so ~$1.00-1.40 round-trip). Total ≈ $2.25-2.65 ≈ 2 ticks. With `b = 10` ticks: `C = 2/10 = 0.2` in barrier units. Parameterize separately for clarity:

```
C = C_spread + C_commission
C_spread = 1 tick / b = 0.1       (market observable, from mean spread feature)
C_commission = 1 tick / b = 0.1   (broker-dependent, update per your actual rate)
```

### 4.6 Expected Value Sanity Check

Under a random (uniform) entry policy with 2:1 barriers, the full expected reward per trade is:

```
E[R] = p⁺(G - C) + p⁻(-L - C) + p⁰(E[MTM] - C)
```

Under a driftless walk, `E[MTM] ≈ 0` (symmetric). Timeouts add a pure cost term.

**With negligible timeout rate (high-vol regime):**

```
E[R] ≈ (1/3)(2.0 - 0.2) + (2/3)(-1.0 - 0.2)
     = (1/3)(1.8) + (2/3)(-1.2)
     = 0.60 - 0.80
     = -0.20
```

**With 20% timeout rate (realistic for low-vol 2023 periods):**

The non-timeout probability redistributes: `p⁺ ≈ 0.267`, `p⁻ ≈ 0.533`, `p⁰ = 0.20`.

```
E[R] ≈ 0.267(1.8) + 0.533(-1.2) + 0.20(-0.2)
     = 0.481 - 0.640 - 0.040
     = -0.199
```

Both scenarios give approximately **-0.20 per trade in barrier units** as the random baseline. This is the number to beat. An agent performing worse than -0.20 has a bug; an agent at -0.20 hasn't learned anything; an agent above -0.20 is extracting signal.

-----

## 5. PPO Configuration

### 5.1 Network Architecture

Use a shared trunk with separate policy and value heads. The features that predict barrier-hit probabilities (value function's job) are the same features that determine entry quality (policy's job). A shared representation lets both objectives shape the same learned features, and halves parameter count for the shared layers — important given the dataset size (~25k-50k bars).

| Component    | Architecture                                                                          |
|--------------|---------------------------------------------------------------------------------------|
| Shared trunk | MLP: `[256, 256]`, ReLU activation                                                   |
| Policy head  | Linear: `[64] → 4` (long, short, flat, hold) — masked per Section 4.3                |
| Value head   | Linear: `[64] → 1`                                                                   |
| Input dim    | `d_total = 132` (13 features × 10-bar lookback + 2 position state)                   |

SB3 configuration:

```python
policy_kwargs = dict(
    net_arch=[256, 256, dict(pi=[64], vf=[64])],
    activation_fn=nn.ReLU,
)
```

This gives `[256, 256]` shared, then `[64]` separate for each head. Do not use `shared_layers` — it is not a valid SB3 `policy_kwargs` key.

### 5.2 Hyperparameters

| Parameter                  | Value                    | Notes                                                     |
|----------------------------|--------------------------|-----------------------------------------------------------|
| Learning rate              | `1e-4` with linear decay | PPO-sensitive; start conservative                         |
| Batch size                 | 2048 steps               | ~4-10 episodes depending on session length                |
| Mini-batch size            | 256                      |                                                           |
| Epochs per update          | 10                       |                                                           |
| Gamma (discount)           | 0.99                     | Episodes are finite (one session), so discounting is mild |
| GAE lambda                 | 0.95                     |                                                           |
| Clip range                 | 0.2                      | Standard                                                  |
| Entropy coefficient        | 0.01                     | Monitor entropy — if it collapses, increase this          |
| Value function coefficient | 0.5                      | Standard                                                  |
| Max grad norm              | 0.5                      |                                                           |

### 5.3 Training Diagnostics

Monitor the following during training. If any of these are wrong, stop and debug before proceeding.

| Metric                | Healthy range                                                                                                                                       | Red flag                                                    |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| Entropy               | When flat: slow decay from ~1.1 (uniform over 3 valid actions) toward ~0.5–0.8. When holding: 0 (only hold is valid). Average depends on flat rate. | Collapse to <0.3 on flat-state steps in first 100 updates   |
| Value loss            | Decreasing trend                                                                                                                                    | Increasing or oscillating wildly                            |
| Policy loss           | Small magnitude, stable                                                                                                                             | Large swings                                                |
| Episode reward (mean) | Above -0.20 (random baseline) and improving                                                                                                        | Persistently below random baseline                          |
| Flat action rate      | 30–70%                                                                                                                                              | <10% (agent never abstains) or >90% (agent always abstains) |
| Trade win rate        | Converging above 33% (base rate)                                                                                                                    | Persistently at or below 33%                                |
| Tiebreak rate         | <5% of labeled bars                                                                                                                                 | >5% — widen barriers or increase N                          |
| Timeout rate          | <15% of resolved trades                                                                                                                             | >30% — increase T_max or widen barriers                     |

-----

## 6. Validation Protocol

### 6.1 Unit Test: Gambler's Ruin Oracle

Before any agent training, validate the label construction pipeline against known analytic results.

**Procedure:**

1. Generate synthetic trade prices from a discrete random walk on the tick grid:

   ```
   p_{t+1} = p_t + ε_t,   ε_t ∈ {+1, -1} with P(+1) = p, P(-1) = q = 1-p
   ```

2. Aggregate synthetic trades into bars of size `N`.
3. Run the full label construction pipeline (Section 2) on synthetic data.
4. Compare empirical barrier-hit frequencies against the Gambler's ruin closed form:

   ```
   P(hit upper first) = (1 - (q/p)^b) / (1 - (q/p)^(a+b))
   ```

   For `p = q = 0.5` (zero drift): `P(upper) = b / (a + b)`.

5. **Run at multiple drift values:** `p ∈ {0.500, 0.505, 0.510, 0.490, 0.485}` corresponding to zero drift and ±0.5σ, ±1σ approximately.
6. For each drift value, verify that the empirical frequency is within 2 standard errors of the analytic value (using at least 10,000 labeled bars).

**Expected results for default barriers (a=20, b=10):**

| Drift (p) | Analytic P(upper hit) | Notes                            |
|-----------|-----------------------|----------------------------------|
| 0.500     | 0.3333                | Zero drift — tests barrier logic |
| 0.505     | ~0.388                | Mild upward drift                |
| 0.510     | ~0.445                | Moderate upward drift            |
| 0.490     | ~0.280                | Mild downward drift              |
| 0.485     | ~0.232                | Moderate downward drift          |

If zero-drift results are correct but non-zero drift results are wrong, you likely have a sign convention bug in your barrier direction or trade price differencing.

### 6.1b Synthetic Regime-Switch Validation

Generate synthetic data with a known volatility regime switch: 5000 bars of low-vol (tick increments drawn from `{-1, +1}` with `p = 0.5`) followed by 5000 bars of high-vol (tick increments drawn from `{-3, -2, -1, +1, +2, +3}` uniform). Run the full label + normalization pipeline and verify:

1. Timeout rate is significantly higher in the low-vol segment and drops at the regime boundary.
2. Time-to-resolution is longer in the low-vol segment.
3. The trailing normalization does not smooth over the transition pathologically — features should show a visible shift at the boundary, not a gradual blend over 2000 bars.
4. The label distribution (p⁺, p⁻, p⁰) shifts appropriately at the boundary.

This tests that the pipeline preserves regime information and that normalization lag doesn't create a dead zone around regime transitions.

### 6.2 Supervised Diagnostic (Pre-RL Validation)

Before training PPO, verify that the observation features contain learnable signal about barrier outcomes.

**Procedure:**

1. Construct the full labeled dataset: `{(Z_k, H_k)}` for all bars `k` in the training window.
2. Train a standalone MLP classifier (same architecture as the shared trunk: `[256, 256]`, ReLU) to predict `H_k ∈ {+1, -1, 0}` from `Z_k` using cross-entropy loss.
3. Evaluate on a held-out validation set (e.g., last 3 months of 2023).

**Baselines:**

| Baseline                       | Expected accuracy                                    | If your model can't beat this               |
|--------------------------------|------------------------------------------------------|---------------------------------------------|
| Majority class                 | ~33% (if balanced) or whatever the empirical mode is | Features contain zero signal about barriers |
| Random forest on same features | Varies                                               | MLP architecture issue, not feature issue   |

If the supervised classifier can predict barrier outcomes above baseline, the signal exists in the features and PPO has something to learn. If it can't, adding RL on top won't help — fix the features first.

**Overfitting test:** The MLP must be able to overfit a batch of 256 samples to >95% train accuracy. If it can't, the architecture is too small or the optimization is broken. This test takes <1 minute and should be run first.

### 6.3 Agent Evaluation

**In-sample (2022-2023):** Replay the agent's learned policy through the training data. This is not a performance estimate — it's a sanity check that the agent learned coherent behavior.

Check for:

- Does the agent go flat more often in low-vol regimes (mid-2023) than high-vol regimes (mid-2022)?
- Does the agent's entry direction correlate with trade flow imbalance?
- Does the agent avoid entries in the last 15 minutes of RTH?
- Does the agent abstain after consecutive losses?

If the answer to most of these is no, the agent probably hasn't learned meaningful structure.

**Out-of-sample:** Hold out a contiguous block (e.g., Q4 2023 or early 2024 if you acquire the data). Never touch this until final evaluation. Report:

| Metric                   | Definition                                      |
|--------------------------|-------------------------------------------------|
| Win rate                 | Fraction of resolved trades where `H_k = +1`    |
| Profit factor            | `(sum of wins) / (sum of losses)` in ticks      |
| Sharpe ratio (per trade) | `mean(trade_pnl) / std(trade_pnl)`              |
| Trades per session       | Average number of entries per RTH session        |
| Max drawdown             | Largest peak-to-trough in cumulative PnL         |
| Flat rate                | Fraction of bars where agent chose not to enter  |

**Minimum viability thresholds (agent is "working"):**

- Win rate > 36% (3 percentage points above 33% base rate with 2:1 barriers)
- Profit factor > 1.0
- Per-trade Sharpe > 0.05
- Flat rate between 20% and 80%

These are low bars. An agent that clears them has learned *something*. Whether that something is economically meaningful after realistic costs is a separate question.

-----

## 7. Hyperparameter Sweep Plan

Sweep the following parameters in the order listed. Each sweep fixes all other parameters at their defaults and varies one dimension.

| Priority | Parameter        | Sweep values               | Default  | Evaluation metric              |
|----------|------------------|----------------------------|----------|--------------------------------|
| 1        | `N` (trades/bar) | {200, 500, 1000, 2000}     | 500      | Supervised classifier accuracy |
| 2        | `(a, b)` (ticks) | {(10,5), (20,10), (40,20)} | (20, 10) | Timeout rate, tiebreak rate    |
| 3        | `T_max` (bars)   | Calibrate per Section 2.4  | 40       | Timeout rate                   |
| 4        | `h` (lookback)   | {5, 10, 20}                | 10       | Supervised classifier accuracy |
| 5        | Learning rate    | {3e-4, 1e-4, 5e-5}         | 1e-4     | Episode reward convergence     |
| 6        | Entropy coeff    | {0.001, 0.01, 0.05}        | 0.01     | Entropy trajectory             |

Priority 1 and 2 are swept with the supervised diagnostic (fast, no RL needed). Priority 3 is computed from the label distribution. Priorities 4-6 require RL training runs.

-----

## 8. Implementation Sequence

Ordered checklist. Do not proceed to step N+1 until step N passes.

1. **Bar construction pipeline.** Ingest Databento MBO, aggregate into trade bars, verify OHLCV correctness by spot-checking against a known charting platform.
2. **Label construction pipeline.** Implement barrier-hit detection with high/low + intrabar tiebreaking. Retain trade sequences per bar for tiebreaking.
3. **Gambler's ruin validation (Section 6.1).** Run on synthetic data. Must pass at all drift levels within 2 standard errors.
4. **Regime-switch validation (Section 6.1b).** Run on synthetic regime-switch data. Verify timeout rate shift and normalization behavior at boundary.
5. **Feature extraction.** Compute all 13 features (Section 3.1). Sanity check: plot feature distributions, verify no NaN/Inf, confirm normalization.
6. **Supervised diagnostic (Section 6.2).** Train MLP classifier on `(Z_k, H_k)`. Must beat majority-class baseline on validation set.
7. **Reward accounting unit test.** Select 5-10 specific bars from real data where you can hand-compute which barrier gets hit and when. Trace through the environment logic manually for both long and short entries. Verify the reward sequence matches hand calculations exactly. This catches bugs in: barrier reversal for shorts, MTM calculation at timeout, cost accounting, and position state transitions. Budget 30 minutes; it can save days.
8. **Environment implementation.** Build the Gymnasium-compatible environment per Section 4. Verify with a random agent that: (a) episode lengths are correct, (b) barrier exits fire correctly, (c) reward accounting matches hand calculations on a few specific bars.
9. **PPO training.** Run with default hyperparameters (Section 5.2). Monitor diagnostics (Section 5.3).
10. **Behavioral inspection.** Check in-sample behavior (Section 6.3). Does the agent exhibit regime-adaptive behavior?
11. **Hyperparameter sweep.** Sweep per Section 7.
12. **Out-of-sample evaluation.** Run on held-out data. Report metrics from Section 6.3.

-----

## 9. v2 Signpost: Adaptive Barrier Sizing

This spec assumes fixed barrier parameters `(a, b)` across all market conditions. This is almost certainly suboptimal — a 20-tick target in a 50-point range day is conservative; in a 15-point range day it may never hit. The optimal barrier widths are themselves a function of the current volatility regime.

The v2 formulation would have the agent output barrier sizes as part of its action space (e.g., `a ∈ {10, 20, 40}` ticks), or condition barriers on a separate regime-classification module that maps realized vol to barrier parameters. This turns the system from an entry-timing agent into a full trade-structuring agent.

Do not invest heavily in sweeping `(a, b)` in v1. If the agent learns meaningful entry timing with fixed barriers, the next step is making barriers adaptive — not finding the single best fixed barrier via grid search.
