# src/bindings — Python Bindings

pybind11 module exposing C++ engine to Python as `lob_rl_core`.

## Files

| File | Role |
|---|---|
| `bindings.cpp` | Exposes: `LOBEnv` (all constructors), `SessionConfig` (with `default_rth()`), `precompute()`. Module name: `lob_rl_core`. |

## Usage from Python

```python
import lob_rl_core

# Synthetic
env = lob_rl_core.LOBEnv()

# Real data with session config
config = lob_rl_core.SessionConfig.default_rth()
env = lob_rl_core.LOBEnv("path/to/file.bin", config, 1000)

# Precompute for fast training
obs, rewards, dones = lob_rl_core.precompute("path/to/file.bin", config, 1000)
```
