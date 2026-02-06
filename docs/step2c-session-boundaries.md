# Step 2c: Session Boundaries

## Goal

Add session-aware episode management so the LOBEnv can run episodes within US Regular Trading Hours (RTH) for /MES futures, with automatic book warmup from pre-market data.

## Background

Per the PRD:
- US RTH: 13:30-20:00 UTC (14:30-21:00 UTC during DST)
- Episodes start flat at session open, end flat at session close
- Agent must be flat at boundaries

/MES futures trade nearly 24 hours. A daily `.bin` file contains messages spanning midnight-to-midnight (roughly). We need to:
1. Fast-forward through pre-RTH messages to warm up the book
2. Run the RL episode during RTH only
3. Stop the episode at session close

## Components

### 1. SessionConfig

**File:** `include/lob/session.h`

```cpp
struct SessionConfig {
    // RTH boundaries in nanoseconds-since-midnight (UTC)
    uint64_t rth_open_ns;   // default: 13:30 UTC = 48600 * 1e9
    uint64_t rth_close_ns;  // default: 20:00 UTC = 72000 * 1e9

    // Number of messages to replay before RTH for book warmup
    // 0 = no warmup, replay starts from first RTH message
    // -1 = replay ALL pre-RTH messages (recommended)
    int warmup_messages;    // default: -1 (all pre-RTH)

    static SessionConfig default_rth();  // 13:30-20:00 UTC, warmup=-1
};
```

### 2. SessionFilter

**File:** `include/lob/session.h`, `src/env/session.cpp`

```cpp
class SessionFilter {
public:
    explicit SessionFilter(SessionConfig config = SessionConfig::default_rth());

    enum class Phase { PreMarket, RTH, PostMarket };

    // Classify a timestamp
    Phase classify(uint64_t ts_ns) const;

    // Extract nanoseconds-since-midnight from a Unix nanosecond timestamp
    static uint64_t time_of_day_ns(uint64_t unix_ns);

    // Fraction of RTH session elapsed [0.0, 1.0]
    // Returns 0.0 before open, 1.0 after close
    float session_progress(uint64_t ts_ns) const;

    uint64_t rth_open_ns() const;
    uint64_t rth_close_ns() const;
    uint64_t rth_duration_ns() const;
};
```

### 3. Updates to LOBEnv

The environment needs to support session-aware episodes when using real data. Add an optional `SessionConfig` to the constructor:

```cpp
class LOBEnv {
public:
    // Existing constructor (for synthetic data, no session filtering)
    explicit LOBEnv(std::unique_ptr<IMessageSource> source,
                    int steps_per_episode = 50);

    // New constructor (for real data with session boundaries)
    LOBEnv(std::unique_ptr<IMessageSource> source,
           SessionConfig session,
           int steps_per_episode = 0);  // 0 = run until session close
};
```

**New `reset()` behavior with SessionConfig:**
1. Reset the source
2. Replay messages until the first RTH message (warming up the book)
3. Return initial observation (with position=0)

**New `step()` behavior with SessionConfig:**
1. Advance messages as before
2. If the current message timestamp >= `rth_close_ns`, set `done=true`
3. When `steps_per_episode > 0`, also terminate at that limit (whichever comes first)
4. When `steps_per_episode == 0`, only terminate at session close or source exhaustion

**Position at session end:**
- When done=true due to session close, force position to 0 (flat)
- Final reward includes PnL from flattening (closing the position at last mid)

## Edge Cases

- Source has no RTH messages: `reset()` exhausts source, returns done=true immediately
- Source starts mid-session (no pre-market data): warmup is skipped, episode starts immediately
- All messages are pre-market: episode ends immediately with done=true
- `time_of_day_ns` handles midnight crossing correctly (not needed for RTH 13:30-20:00, but defensive)
- `session_progress` clamps to [0.0, 1.0]
- Default constructor (no SessionConfig): works exactly as before, no session filtering

## Acceptance Criteria

1. **SessionConfig tests:**
   - `default_rth()` returns 13:30-20:00 UTC in nanoseconds
   - Custom config stores correct values

2. **SessionFilter tests:**
   - `classify()` correctly identifies PreMarket, RTH, PostMarket
   - Boundary: timestamp exactly at open → RTH
   - Boundary: timestamp exactly at close → PostMarket
   - `time_of_day_ns()` correctly extracts time-of-day from Unix timestamp
   - `session_progress()` returns 0.0 at open, 1.0 at close, 0.5 at midpoint
   - `session_progress()` clamps to [0.0, 1.0] outside RTH

3. **LOBEnv session tests:**
   - With SessionConfig, `reset()` warms up book through pre-RTH messages
   - Episode runs only during RTH timestamps
   - Episode terminates at session close
   - Position is forced flat at session close
   - Reward at session close includes flattening PnL
   - `steps_per_episode=0` runs until session close
   - `steps_per_episode=N` terminates at min(N, session_close)
   - Default constructor (no session) behaves as before

4. **Integration:**
   - Use a test fixture with messages spanning pre-market → RTH → post-market
   - Verify warmup populates the book but doesn't count as episode steps
   - Verify episode boundaries align with RTH
