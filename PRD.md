# LOB-RL: Limit Order Book RL Environment

## Product Requirements Document

---

## 1. Goal

Train RL agents on Databento MBO futures data and evaluate trading performance.

**Success metric:** Positive Sortino ratio on held-out test data.

---

## 2. Constraints

| Constraint | Decision |
|------------|----------|
| Strategy | Directional (not market-making) |
| Position | {-1, 0, +1} target (long, flat, short) |
| Orders | Market only, instant fill at BBO |
| Market impact | Zero (small size assumption, <10 contracts) |
| Holding period | Minutes to ~1 hour |
| Episode | One session, flat at open and close |
| Initial session | US RTH (13:30-20:00 UTC / 14:30-21:00 UTC DST) |
| Data | 30 days /MES MBO (Databento) |

---

## 3. Non-Goals (v1)

- Market making / limit orders
- Fill simulation / queue position modeling
- Multi-instrument
- GPU acceleration
- Live trading / broker connectivity

---

## 4. Architecture

```
┌─────────────────────────────────────┐
│  Python: Training / Evaluation      │
│  (Gym wrapper, SB3, metrics)        │
└──────────────┬──────────────────────┘
               │ pybind11
┌──────────────▼──────────────────────┐
│  C++ LOBEnv                         │
│  step(action) → {obs, reward, done} │
│  reset() → obs                      │
│                                     │
│  ┌───────────┐  ┌────────────────┐  │
│  │  Reward   │  │   Feature      │  │
│  │  Calc     │  │   Builder      │  │
│  └─────┬─────┘  └───────┬────────┘  │
│        └────────┬───────┘           │
│           ┌─────▼─────┐             │
│           │   Book    │             │
│           └─────┬─────┘             │
│           ┌─────▼─────┐             │
│           │ IMessage  │             │
│           │  Source   │             │
│           └───────────┘             │
└─────────────────────────────────────┘
```

---

## 5. Observation Space

44 floats (K=10 book depth):

| Index | Feature |
|-------|---------|
| 0-9 | Bid prices (relative to mid, normalized) |
| 10-19 | Bid sizes (normalized) |
| 20-29 | Ask prices (relative to mid, normalized) |
| 30-39 | Ask sizes (normalized) |
| 40 | Spread (normalized) |
| 41 | Imbalance [-1, 1] |
| 42 | Time remaining in session [0, 1] |
| 43 | Current position {-1, 0, 1} |

---

## 6. Action Space

Discrete, 3 actions:

| Action | Meaning |
|--------|---------|
| 0 | Target short (-1) |
| 1 | Target flat (0) |
| 2 | Target long (+1) |

Agent outputs target position. Environment executes market orders to reach target.

---

## 7. Reward

Configurable via enum. Initial options:

- **PnLDelta**: Change in mark-to-market PnL
- **PnLDeltaPenalized**: PnL - λ|position| (inventory penalty)

Reward function is C++ only (no Python callbacks).

---

## 8. Overfitting Mitigation

| Strategy | Implementation |
|----------|----------------|
| Temporal split | Days 1-20 train, 21-25 val, 26-30 test |
| Walk-forward | Train [1-20] test 21, train [2-21] test 22, ... |
| Feature simplicity | Start minimal, add features only if they help |
| Baseline | Must beat buy-and-hold and random on test set |

---

## 9. Build Order (MVP)

### Step 1: Vertical Slice
- Hardcoded SyntheticSource (100 messages)
- Minimal Book (best bid/ask only)
- Minimal LOBEnv (step/reset)
- Python binding (call step() from Python)
- **Deliverable:** Python can step through synthetic data

### Step 2: Real Data
- DatabentoSource
- Full Book (10 levels)
- Session boundaries
- **Deliverable:** Replay real /MES data

### Step 3: Training
- Full FeatureBuilder (44 floats)
- RewardCalculator
- Gym wrapper
- Train agent, evaluate Sortino
- **Deliverable:** Learning curve, test set metrics

---

## 10. Success Criteria

1. Replay Databento MBO data correctly (book state matches expected BBO)
2. Agent can step through episodes, receive observations and rewards
3. Training produces increasing reward over time
4. Positive Sortino on held-out test set (or clear explanation why not)
5. Clean, documented code suitable for portfolio presentation
