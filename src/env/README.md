# src/env — RL Environment

Wraps the Book engine as a Gym-like RL environment.

## Files

| File | Role |
|---|---|
| `lob_env.cpp` | `LOBEnv` — step/reset interface. Actions: Short(0), Flat(1), Long(2). Manages position, PnL, episode length. Session-aware mode supports RTH warmup and position flattening. |
| `feature_builder.cpp` | `FeatureBuilder` — builds 44-float observation vector from Book state. |
| `precompute.cpp` | `precompute(path, config)` — runs full episode, returns (obs, mid, spread, num_steps) as contiguous arrays. Used by Python `MultiDayEnv` for zero-C++ training. |

## API Signatures (headers in `include/lob/`)

### LOBEnv (`env.h`)

```cpp
LOBEnv(unique_ptr<IMessageSource> src, int steps_per_episode = 50,
       RewardMode mode = PnLDelta, float lambda = 0.0f,
       bool execution_cost = false);

LOBEnv(unique_ptr<IMessageSource> src, SessionConfig cfg, int steps_per_episode,
       RewardMode mode = PnLDelta, float lambda = 0.0f,
       bool execution_cost = false);

StepResult reset();           // returns {obs(44), reward=0, done}
StepResult step(int action);  // action 0=short, 1=flat, 2=long
int steps_per_episode() const;
```

### RewardCalculator (`reward.h`)

```cpp
enum class RewardMode { PnLDelta, PnLDeltaPenalized };

RewardCalculator();  // defaults to PnLDelta, lambda=0
RewardCalculator(RewardMode mode, float lambda = 0.0f);

float compute(float position, double current_mid, double prev_mid) const;
float flattening_penalty(float position, double spread) const;
float execution_cost(float old_pos, float new_pos, double spread) const;
// execution_cost = -|new_pos - old_pos| * spread/2
```

### FeatureBuilder (`feature_builder.h`)

```cpp
static constexpr int DEPTH = 10;
static constexpr int OBS_SIZE = 44;  // = 4*DEPTH + 4
// Layout: bid_price[0-9], bid_size[10-19], ask_price[20-29], ask_size[30-39],
//         spread[40], imbalance[41], time_left[42], position[43]

vector<float> build(const Book& book, float position, float time_remaining) const;
```

### precompute (`precompute.h`)

```cpp
PrecomputeResult precompute(const string& path, const SessionConfig& cfg);
// Returns: obs (N x 43 float), mid (N double), spread (N double), num_steps (int)
```

### SessionConfig / SessionFilter (`session.h`)

```cpp
struct SessionConfig {
    uint64_t rth_open_ns;   // e.g., 48_600_000_000_000 (13:30 UTC)
    uint64_t rth_close_ns;  // e.g., 72_000_000_000_000 (20:00 UTC)
    int warmup_messages = -1;  // -1 = all pre-market
};

SessionFilter(SessionConfig cfg);
Phase classify(uint64_t ts_ns) const;  // PreMarket, RTH, PostMarket
float session_progress(uint64_t ts_ns) const;  // 0.0 to 1.0
```

## Dependencies

- **Depends on:** `include/lob/book.h`, `include/lob/source.h`, `src/data/` sources
- **Depended on by:** `src/bindings/bindings.cpp` (pybind11), all C++ test files

## Modification hints

- **New reward mode:** Add to `RewardMode` enum in `reward.h`, handle in `RewardCalculator::compute()`, update `parse_reward_mode()` in `bindings.cpp`
- **New observation feature:** Increment `OBS_SIZE` in `feature_builder.h`, add to `build()` in `feature_builder.cpp`, update Python obs shape (44 → new size)
- **New env parameter:** Add to both `LOBEnv` constructors + header, update all 5 pybind11 overloads in `bindings.cpp`
