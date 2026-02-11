# src/bindings — Python Bindings

pybind11 module exposing C++ engine to Python as `lob_rl_core`.

## Files

| File | Role |
|---|---|
| `bindings.cpp` | Exposes: `LOBEnv` (5 constructor overloads), `SessionConfig`, `precompute()`, `barrier_precompute()`. Module name: `lob_rl_core`. |

## Python API (exposed by pybind11)

### LOBEnv — 5 constructor overloads

```python
# 1. Synthetic data (no file)
env = lob_rl_core.LOBEnv(steps=50, reward_mode="pnl_delta", lambda_=0.0,
                          execution_cost=False)

# 2. File-based
env = lob_rl_core.LOBEnv("path.bin", steps=50, reward_mode="pnl_delta",
                          lambda_=0.0, execution_cost=False)

# 3. File + session config
config = lob_rl_core.SessionConfig.default_rth()
env = lob_rl_core.LOBEnv("path.bin", config, steps=0,
                          reward_mode="pnl_delta", lambda_=0.0,
                          execution_cost=False)

obs_list = env.reset()       # returns list[float] (44 elements)
obs_list, reward, done = env.step(action)  # action: int 0/1/2
```

### SessionConfig

```python
config = lob_rl_core.SessionConfig.default_rth()
config.rth_open_ns    # uint64, default 48_600_000_000_000 (13:30 UTC)
config.rth_close_ns   # uint64, default 72_000_000_000_000 (20:00 UTC)
config.warmup_messages  # int, default -1 (all pre-market)
```

### precompute

```python
obs, mid, spread, num_steps = lob_rl_core.precompute(path, config)
# obs: ndarray (N, 43) float32
# mid: ndarray (N,) float64
# spread: ndarray (N,) float64
# num_steps: int
```

### barrier_precompute

```python
result = lob_rl_core.barrier_precompute(
    path, instrument_id,
    bar_size=500, lookback=10, a=20, b=10, t_max=40)
# Returns dict with keys: features, bar_open, bar_high, bar_low, bar_close,
#   bar_volume, bar_vwap, bar_t_start, bar_t_end, trade_prices, trade_sizes,
#   bar_trade_offsets, label_values, label_tau, label_resolution_bar,
#   bar_size, lookback, a, b, t_max, n_bars, n_usable, n_features
# Returns None if insufficient data (n_bars < lookback + 1)
```

## Dependencies

- **Depends on:** `lob_core` (static lib), pybind11
- **Depended on by:** All Python code (`python/lob_rl/`), `scripts/precompute_barrier_cache.py`

## Modification hints

- **New LOBEnv parameter:** Add `py::arg("param") = default` to ALL 5 `.def(py::init<...>())` overloads. The overloads are: (1) synthetic, (2) synthetic+steps, (3) file+steps, (4) file+session, (5) file+session+steps.
- **New reward mode:** Update `parse_reward_mode()` in bindings.cpp to handle the new string.
- **New binding function:** Add to the `PYBIND11_MODULE(lob_rl_core, m)` block at the end of the file.
