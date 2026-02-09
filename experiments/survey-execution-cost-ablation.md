# Survey: Does removing execution cost improve OOS generalization?

## Prior Internal Experiments

### Direct evidence: one early no-exec-cost run exists

From the training history in `LAST_TOUCH.md`:

| Run | Config | Result |
|-----|--------|--------|
| v2 no exec cost | 2M, ent_coef=0.01, 8 envs | **Entropy collapsed (0.09). Sortino -1.05.** |
| v2 + exec cost | 2M, ent_coef=0.01, 8 envs, exec_cost | Entropy stable (0.70). Agent stays flat. |

**Critical finding:** The only prior no-exec-cost run (pre-hyperparameter-sweep, pre-bar-level, pre-shuffle-split) was *worse* than the exec-cost run. Entropy collapsed to 0.09 without execution cost — the agent exploited a degenerate strategy. However, this run used:
- ent_coef=0.01 (not 0.05 — the later sweep showed 0.05 is needed to prevent collapse)
- Tick-level env (not bar_size=1000)
- No shuffle-split
- Only 2M steps

**This early result should NOT be taken as conclusive.** The hyperparameters were suboptimal. The bar-level observation space, higher entropy coefficient, and shuffle-split all postdate this run. The question needs re-testing under the current best configuration.

### Indirect evidence: execution cost magnitude analysis

From `DOMAIN_PRIORS.md`:
- **MES tick size:** $1.25 per tick (0.25 index pts x $5 multiplier)
- **Spread:** Typically 1 tick ($1.25) during RTH
- **Execution cost formula:** `spread/2 * |position_change|` = $0.625 per position change
- **Round-trip cost:** $1.25 (enter + exit)

From `_reward.py` (line 34):
```python
reward -= spread_prev / 2.0 * abs(position - prev_position)
```

The reward is `position * (mid_now - mid_prev)` — the PnL delta per bar. With bar_size=1000, each bar aggregates 1000 MBO events. The typical mid-price movement per bar is on the order of a few ticks. **If per-bar moves are frequently < 1 tick, execution cost consumes most or all of the alpha.**

From `DOMAIN_PRIORS.md` line 39: *"Execution cost is the central challenge. Per-bar price moves are often < 1 tick. The agent needs to find signals worth > 2 ticks (round-trip) to be profitable."*

### All 7 prior experiments used execution cost

Every prior experiment (pre-001 through pre-007, plus exp-001) has had `--execution-cost` enabled. **There is no OOS evaluation without execution cost under the current best hyperparameters.** This is the gap this experiment would fill.

### Exp-001 status (data scaling) — partially complete

The exp-001 results directory shows runs completed locally: MVE (MLP 199d), Run A (MLP 20d), Run B (LSTM 20d), Run C (MLP 199d), and Run D (LSTM 199d) appears to be in progress. Note: the runs diverge slightly from the original spec (Run B is LSTM 20d instead of MLP 199d). These results have not been analyzed yet (no `metrics.json` found). Regardless of exp-001's outcome, the execution cost question is independently valuable — it's a P0 question in `QUESTIONS.md`.

## Current Infrastructure

### Training pipeline (fully supports this experiment)

The `--execution-cost` flag is already a boolean toggle on `train.py` (line 189). Removing it is trivial: just omit the flag. No code changes needed.

**Key:** The `evaluate_sortino()` function (line 86) also takes `execution_cost` as a parameter and passes it through to the env. When training without exec cost, evaluation must ALSO omit exec cost (the agent learns a different policy). This is already the default behavior — `train.py` passes `args.execution_cost` to both training and eval.

**Important design consideration for the experiment:** There are TWO distinct questions:
1. **Train without exec cost, eval without exec cost** — does the agent learn *any* signal?
2. **Train without exec cost, eval WITH exec cost** — would the signal survive real trading costs?

Question 1 is the diagnostic. Question 2 is the practical question. Both can be answered from a single training run by re-evaluating the saved model with `execution_cost=True` vs `False`.

### Reward computation

From `_reward.py`:
- Without `execution_cost`: `reward = position * (mid_now - mid_prev)` (pure PnL delta)
- With `execution_cost`: `reward -= spread_prev / 2.0 * abs(position - prev_position)` (PnL delta minus half-spread per trade)
- Forced flatten (terminal step): `reward = -spread / 2.0 * abs(prev_position)` (closing cost — this applies regardless of `execution_cost` flag!)

**Subtle point:** The forced flatten cost at episode end (`compute_forced_flatten`) is *always* charged. It doesn't check the `execution_cost` flag. This means even without `--execution-cost`, the agent pays a closing cost on the last bar of each day. This is a smaller effect (1 trade per episode vs many intra-day trades) but should be noted.

### Eval infrastructure

`evaluate_sortino()` supports the `execution_cost` parameter. A trained model can be re-evaluated with different exec cost settings by loading the saved model and vec_normalize, then calling `evaluate_sortino()` with `execution_cost=True` or `False`.

### Compute

MLP runs take ~1.1 hours on local machine or cloud. This experiment does not require GPU (MLP is CPU-optimal per `DOMAIN_PRIORS.md`). LSTM would take ~6 hours on GPU but the MLP ablation alone answers the core question.

## Known Failure Modes

1. **Entropy collapse without execution cost.** The one prior no-exec-cost run (ent_coef=0.01) had entropy collapse to 0.09. Without the "tax" on trading, the agent may learn to trade aggressively on noise, cycling position rapidly. The entropy bonus (ent_coef=0.05) should mitigate this, but monitor entropy closely.

2. **Degenerate strategies.** Without execution cost, the optimal policy in a random walk is to trade every bar (there's no penalty for changing position). The agent might learn to flip position constantly, generating high variance returns with zero expected value. This would produce near-zero mean return with high std — technically "better" than negative returns but not meaningful signal.

3. **VecNormalize reward scaling.** VecNormalize normalizes rewards. Without execution cost, raw reward magnitude changes (no spread deduction). The running variance will adapt, but the first ~100k steps may have different scaling than exec-cost runs. This is a confound when comparing absolute return magnitudes.

4. **Forced flatten cost asymmetry.** As noted above, `compute_forced_flatten()` always charges a closing cost regardless of the `execution_cost` flag. This creates a small but systematic negative bias even in the no-exec-cost condition. The FRAME agent should decide whether to also remove this cost (which would require a code change — a HANDOFF to TDD).

## Key Codebase Entry Points

| File | Path | Relevance |
|------|------|-----------|
| Training script | `scripts/train.py` | `--execution-cost` flag (line 189-190). Passed to `make_train_env()` and `evaluate_sortino()`. |
| Reward computation | `python/lob_rl/_reward.py` | Lines 32-34: exec cost deduction. Line 11: forced flatten cost (always applied). |
| Bar-level env | `python/lob_rl/bar_level_env.py` | `execution_cost` parameter forwarded to `compute_step_reward()`. |
| Eval function | `scripts/train.py:86-178` | `evaluate_sortino()` — takes `execution_cost` param, constructs env with it. |
| Multi-day env | `python/lob_rl/multi_day_env.py` | `execution_cost` forwarded to inner env on each `reset()`. |

## Architectural Priors

This experiment does not change the architecture — it changes the reward signal. The MLP 256x256 ReLU with bar_size=1000 remains appropriate for the diagnostic question. If signal is found without exec cost, LSTM should be tested as a follow-up (LSTM was the best OOS performer in prior experiments).

The theoretical framing: execution cost is a **fixed per-trade cost** in a market-making/trading context. In microstructure theory, profitability requires signal strength > transaction cost. If the observation space contains signal worth 0.5 ticks per trade but the round-trip cost is 1 tick, the agent can never be profitable — but removing the cost would reveal the 0.5-tick signal.

## External Context

This is a well-studied phenomenon in quantitative trading:
- **Transaction cost erosion** is the primary reason many theoretical trading strategies fail in practice. Academic alpha signals often disappear after realistic costs.
- **Gross vs net alpha decomposition** is standard practice: first verify signal exists (gross), then assess if it survives costs (net). This project has only tested net alpha.
- **In RL for trading specifically**, execution cost in the reward function changes the optimal policy discontinuously — a signal worth 0.8 ticks has very different optimal behavior with 0 cost (trade) vs 1 tick cost (don't trade). The agent may learn "don't trade" as optimal and converge to a flat policy, which is exactly what was observed in early experiments ("agent stays flat").

## Constraints and Considerations

1. **Compute:** MLP runs ~1.1 hours locally. No GPU needed. Budget is minimal.
2. **Exp-001 may still be running.** The data-scaling experiment appears partially complete. The execution cost ablation is independent and can run in parallel.
3. **Two-phase eval needed.** Train without exec cost, then evaluate both with and without. The second eval requires re-running `evaluate_sortino()` with `execution_cost=True` on the no-exec-cost model.
4. **Forced flatten cost.** The terminal step always charges spread/2 even without `--execution-cost`. This is a minor confound (~1 trade per episode of ~23 bars at bar_size=1000). The FRAME agent should decide whether to address this (requires HANDOFF to TDD to modify `compute_forced_flatten` to respect the `execution_cost` flag).
5. **Comparison baseline.** The most relevant comparison is the MLP 20-day run from exp-001 Run A (or historical pre-006: val -62.9, test -44.0) — same config but WITH exec cost. The delta isolates the exec cost effect.
6. **Train days.** Should match the best available baseline. If exp-001 results are available with 199 days, compare against those. Otherwise, use 20 days to match the historical baselines.

## Recommendation

The FRAME agent should design a simple, cheap ablation experiment:

1. **Primary run:** MLP 256x256, bar_size=1000, ent_coef=0.05, lr=1e-3, shuffle-split, seed 42, 20 train days, 2M steps, **NO** `--execution-cost`. This matches the pre-003/pre-006 baselines except for exec cost removal.

2. **Dual eval:** After training, evaluate the saved model twice:
   - Without exec cost (matches training conditions) — answers "does the agent learn signal?"
   - With exec cost (realistic conditions) — answers "would the signal survive costs?"

3. **Success criteria should be pre-committed:**
   - If no-exec-cost OOS returns are **positive** → agent learns signal that is masked by costs. This is actionable: reduce trading frequency, find stronger signals, or reduce costs.
   - If no-exec-cost OOS returns are **still negative** → the agent is not learning any useful signal. The problem is deeper than execution cost.
   - If no-exec-cost in-sample is high but OOS is still negative → the agent overfits regardless of cost structure. Execution cost is not the binding constraint.

4. **Watch for:**
   - Entropy collapse (the one prior no-exec-cost run collapsed)
   - Degenerate high-frequency trading (position changing every bar)
   - The forced flatten cost creating a systematic negative bias

5. **Stretch goal (if budget allows):** Run the same ablation with LSTM to see if the architecture x cost interaction matters. LSTM was the best OOS performer, and removing exec cost might amplify its advantage.

6. **Consider whether to also ablate forced flatten cost.** If the forced flatten cost is non-trivial, its presence in the no-exec-cost condition is a confound. However, modifying this requires a code change (HANDOFF to TDD). The FRAME agent should assess whether the magnitude is large enough to matter (~$0.625 per episode vs ~$0.625 * N_trades per episode with exec cost, where N_trades could be 5-15 per episode).
