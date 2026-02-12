# LOB Bug Impact Assessment

**Date:** 2026-02-12
**Trigger:** OrderBook implementation found to be incorrect — does not account for Top-of-Book flags, produces wildly inconsistent results vs known working implementations.

---

## Blast Radius

### What's broken

The OrderBook (`include/lob/book.h`, `src/engine/book.cpp`, `python/lob_rl/barrier/lob_reconstructor.py`) is the foundation of 13 out of 22 barrier features. Every cached `.npz` file, every experiment that used those features, and every conclusion drawn from them is suspect.

### Contaminated features (13/22)

| Col | Feature | Why contaminated |
|-----|---------|-----------------|
| 1 | BBO imbalance | `best_bid_qty()`, `best_ask_qty()` |
| 2 | Depth(5) imbalance | `total_bid_depth()`, `total_ask_depth()` |
| 10 | Cancel asymmetry | Cancel tracking during book updates |
| 11 | Mean spread | `spread()` samples |
| 13 | OFI | Add msgs at BBO (requires correct book state) |
| 14 | Depth ratio | `total_bid_depth(3/10)`, `total_ask_depth(3/10)` |
| 15 | Weighted mid displacement | `weighted_mid()` |
| 16 | Spread std | `spread()` variance |
| 17 | VAMP displacement | `vamp(5)` |
| 18 | Aggressor imbalance | Trade side classification (uses book) |
| 19 | Trade arrival rate | Low risk — trade count only |
| 20 | Cancel-to-trade ratio | Cancel count depends on book update path |
| 21 | Price impact per trade | Low risk — trade count only |

### Clean features (9/22)

| Col | Feature | Why clean |
|-----|---------|----------|
| 0 | Trade flow imbalance | Tick rule on trade prices only |
| 3 | Bar range | Bar OHLC (trade-derived) |
| 4 | Bar body | Bar OHLC |
| 5 | Body/range ratio | Bar OHLC |
| 6 | VWAP displacement | Trade prices + sizes |
| 7 | Log volume | Trade sizes |
| 8 | Realized vol | Bar close prices |
| 9 | Session time | Timestamps |
| 12 | Session age | Bar index |

### Clean infrastructure (still valid)

- **Bar construction** (TradeBar OHLCV) — built from trades, not book state
- **Barrier labels** (Y_long, Y_short) — computed from bar OHLC, not book
- **Experiment pipeline** (`experiment.sh`, `tdd.sh`) — orchestration is fine
- **Scoring/evaluation** (Brier, BSS, bootstrap, temporal split) — pure math
- **Training infrastructure** (SB3 wrappers, diagnostics, checkpointing) — untouched
- **RL environments** (barrier env, multi-session env) — consume features, don't produce them

---

## Experiment Triage

### Experiments that MUST be re-run

| Experiment | Used 22-feature cache? | Verdict at risk? |
|-----------|----------------------|-----------------|
| exp-004 (22-feature diagnostic) | Yes | INCONCLUSIVE → might flip |
| exp-006 (LR/GBT signal detection) | Yes | REFUTED → **might flip to CONFIRMED** if book features were garbage-in |
| exp-007 (LSTM/Transformer sequence) | Yes | REFUTED → might flip |
| exp-008 (bar-size sweep) | Yes (all 12 configs) | REFUTED → might flip |
| exp-009 (realistic barrier sweep) | Yes (all 4 configs) | REFUTED → might flip |
| exp-010 (permutation test) | Yes (aborted but uses same cache) | Never completed |

### Experiments that are STILL VALID

| Experiment | Why valid |
|-----------|----------|
| pre-001 (hyperparam sweep) | Used 21-dim bar-level obs (no book features) |
| pre-002 through pre-007 (RL training) | Same 21-dim bar-level obs |
| exp-001 (199d data scale) | Same 21-dim bar-level obs |
| exp-002 (execution cost ablation) | Same 21-dim bar-level obs |
| exp-005 (null calibration) | Labels only, no features |
| T6 original (v1, long-only) | Used 130-dim but 4 book features were dead (zeros) — effectively 9-feature |

### T6v2 (bidirectional diagnostic) — PARTIALLY VALID

T6v2 used `cache/barrier/` which was built with the C++ backend including all 22 features. **However**, T6v2's feature importance showed trade-derived features dominated (trade_flow 0.128, bar_range 0.122, volume 0.121). The +5pp accuracy gain was driven primarily by clean features. The result is **likely still directionally correct** but should be re-verified.

### Key question: Were the book features just noise, or actively misleading?

If the buggy book produced **random garbage** for features 1,2,10,11,13-18,20: the experiments tested 9 clean features + 13 noise features. LR/RF would have learned to ignore the noise (L2 regularization downweights useless features). The REFUTED verdicts are probably still correct — we just had less effective signal.

If the buggy book produced **systematically wrong but correlated** values: the models may have learned spurious patterns that hurt generalization. Fixing the book could either help (correct features add signal) or have no effect (even correct book features carry no predictive info).

**Bottom line:** The most likely outcome is that correct book features add marginal signal, and most REFUTED experiments stay REFUTED. But we can't know until we re-run.

---

## Remediation Plan

### Phase 0: Fix the OrderBook

_Already in progress via separate agent._

Deliverables:
- [ ] C++ `Book` class handles Top-of-Book flags correctly
- [ ] Python `OrderBook` handles Top-of-Book flags correctly
- [ ] Both match reference implementation on known test sequences
- [ ] Existing C++ tests pass (650 tests)
- [ ] New validation tests against Databento reference data

### Phase 1: Rebuild caches

After the book is fixed:

```bash
# Delete ALL contaminated caches
rm -rf cache/barrier/
rm -rf cache/barrier-b*/

# Rebuild primary cache (B=500, default barriers)
cd build-release && PYTHONPATH=.:../python uv run python \
  ../scripts/precompute_barrier_cache.py \
  --data-dir ../data/mes/ --output-dir ../cache/barrier/ \
  --roll-calendar ../data/mes/roll_calendar.json \
  --bar-size 500 --lookback 10 --workers 8
```

Estimated time: ~10 min for primary cache. ~40 min for each sweep cache.

Verify:
- [ ] 248 `.npz` files produced
- [ ] Spot-check: book-derived features (cols 1,2,11,13-17) are non-zero and plausible
- [ ] Feature statistics (mean, std, range) look reasonable for spread, depth, OFI

### Phase 2: Quick diagnostic (T6 redux)

Before re-running the full experiment battery, run a quick supervised diagnostic to see if correct book features change anything:

1. Re-run T6v2 bidirectional RF on new cache
2. Compare: accuracy with 22 correct features vs. T6v2's 40.5% (which used 9 clean + 13 garbage)
3. If accuracy jumps significantly (>45%): book features matter, re-run everything
4. If accuracy is similar (~40%): book features were noise, REFUTED verdicts likely hold

This takes ~5 min and tells us whether the full re-run battery is worth the effort.

### Phase 3: Re-run critical experiments

Priority order (highest information value first):

1. **exp-006 redux** (LR/GBT signal detection) — the foundational BSS test. ~5 min.
2. **exp-004 redux** (22-feature RF diagnostic) — quantifies book feature contribution. ~5 min.
3. **exp-008 redux** (bar-size sweep) — only if exp-006 shows signal. ~2.5 hrs.
4. **exp-007 redux** (sequence models) — only if exp-006 shows signal. ~30 min.
5. **exp-010** (permutation test) — useful regardless. ~15 min.

### Phase 4: Update research log

- Mark all contaminated experiments with `[PRE-LOB-FIX]` tag
- Log new results alongside old for comparison
- Update CLAUDE.md current state
- Update QUESTIONS.md with revised status

---

## What this means for the research program

### Optimistic case (book features add real signal)

Correct book features (spread, depth imbalance, OFI, VAMP) are among the most informative microstructure signals in the literature. If our buggy implementation was producing garbage for these, we were effectively testing with half our feature set disabled. Fixing the book could push BSS above the 0.005 threshold, unlocking:
- Profitable barrier trading via calibrated probabilities
- RL training with meaningful features
- The full Phase 3 (RL on barrier features) roadmap

### Realistic case (book features are marginal)

Even correct book features may not help because:
- MES is a highly liquid derivative — spread is usually 1 tick, depth ratios are noisy
- At B=500+ bars (aggregating 500+ events), intra-bar book dynamics average out
- T6 feature importance showed trade-derived features dominate even with the buggy book

In this case, the REFUTED verdicts hold and we're back to the pivot decision: conditional signal, alternative targets, external features, or accept the null.

### What's definitely true regardless

- **Labels are clean.** Y_long and Y_short depend only on bar OHLC (trades).
- **The null calibration holds.** ȳ ≈ 1/3 (exp-005 is untouched).
- **The RL infrastructure works.** Training pipeline, environments, diagnostics are all valid.
- **6 model families fail on trade-only features.** The pre-book-fix experiments with 9 clean features all failed. Even if book features help, the marginal contribution needs to be large enough to flip BSS from -0.003 to +0.005 — an 8pp swing. That would be remarkable.

---

## Action Items (post-fix)

1. [ ] Rebuild `cache/barrier/` with fixed book
2. [ ] Spot-check: verify book features are non-zero and plausible
3. [ ] Run T6v2 quick diagnostic (~5 min) — decides if full re-run is needed
4. [ ] If T6 accuracy jumps: re-run exp-006, exp-008 in sequence
5. [ ] If T6 accuracy stable: document, update RESEARCH_LOG, move on
6. [ ] Update CLAUDE.md, LAST_TOUCH.md, RESEARCH_LOG.md
