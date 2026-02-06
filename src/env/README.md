# src/env — RL Environment

Wraps the Book engine as a Gym-like RL environment.

## Files

| File | Role |
|---|---|
| `lob_env.cpp` | `LOBEnv` — step/reset interface. Actions: Hold(0), Buy(1), Sell(2). Manages position, PnL, episode length. Session-aware mode supports RTH warmup and position flattening. |
| `feature_builder.cpp` | `FeatureBuilder` — builds 44-float observation vector from Book state. 10 bid prices (relative to mid), 10 bid sizes (normalized), 10 ask prices, 10 ask sizes, spread/mid, imbalance, time_remaining, position. |
| `precompute.cpp` | `precompute(path, config, steps)` — runs full episode, returns all obs/rewards/dones as contiguous arrays. Used by Python `MultiDayEnv` for zero-C++ training. |

## Key interfaces (in `include/lob/`)

- `env.h` — `LOBEnv` class with multiple constructors (synthetic, file, session-aware)
- `feature_builder.h` — `FeatureBuilder::OBS_SIZE = 44`, `POSITION = 43`
- `reward.h` — `RewardCalculator` with `PnLDelta` and `PnLDeltaPenalized` modes
- `session.h` — `SessionConfig`, `SessionFilter` for RTH boundaries
- `precompute.h` — `precompute()` free function
